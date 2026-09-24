"""Maven and Gradle providers for Java workspaces."""
from __future__ import annotations

import json
import os
import re
import shlex
import base64
import xml.etree.ElementTree as ET
from dataclasses import replace

from src.python_deps.depgraph.schema import Ecosystem
from src.ecosystems.base import (
    DepGraphFragment,
    ImportRef,
    PackageRecord,
    ProviderTarget,
    RequirementRecord,
    ResolutionResult,
    Workspace,
    file_digest,
    graph_from_resolution,
    in_workspace,
    read_text,
    run_native_resolver_probe,
    workspace_path,
)
from src.ecosystems.common import (
    find_files,
    find_source_files,
    language_requirement,
    numeric_version,
    parse_syntax_tree,
    syntax_text,
    walk_syntax_tree,
)


_MAVEN_TREE_MARKER = "__GRAPH2ENV_MAVEN_TREE_JSON__"
_GRADLE_ARTIFACT_MARKER = "__GRAPH2ENV_GRADLE_ARTIFACT__"
_GRADLE_EDGE_MARKER = "__GRAPH2ENV_GRADLE_EDGE__"
_JAVA_CLASS_MARKER = "__GRAPH2ENV_JAVA_CLASS__"


def _maven_wrapper_requires_unzip(
    repo_path: str,
    workspace: Workspace,
) -> bool:
    properties = os.path.join(
        workspace_path(repo_path, workspace),
        ".mvn",
        "wrapper",
        "maven-wrapper.properties",
    )
    for line in read_text(properties).splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "distributionUrl":
            return value.strip().split("?", 1)[0].endswith(".zip")
    return False


def _top_level_manifests(paths: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    roots: list[str] = []
    for path in sorted(paths, key=lambda item: (item.count("/"), item)):
        root = os.path.dirname(path) or "."
        if any(
            root != parent
            and (parent == "." or root.startswith(parent.rstrip("/") + "/"))
            for parent in roots
        ):
            continue
        roots.append(root)
        selected.append(path)
    return tuple(selected)


def _java_import_names(text: str) -> tuple[str, ...]:
    tree = parse_syntax_tree(text, "java")
    if tree is not None:
        source = text.encode("utf-8")
        imports = []
        for node in walk_syntax_tree(tree.root_node):
            if node.type == "import_declaration":
                declaration = syntax_text(source, node)
                raw = re.sub(r"^\s*import\s+(?:static\s+)?", "", declaration)
                raw = raw.rstrip().rstrip(";").rstrip()
                if raw.endswith(".*"):
                    raw = raw[:-2]
                if raw:
                    imports.append(raw)
            elif (
                node.type == "scoped_identifier"
                and (
                    node.parent is None
                    or node.parent.type not in {
                        "scoped_identifier",
                        "import_declaration",
                    }
                )
            ):
                raw = syntax_text(source, node)
                parts = raw.split(".")
                if (
                    len(parts) >= 3
                    and parts[0][:1].islower()
                    and any(part[:1].isupper() for part in parts[1:])
                ):
                    imports.append(raw)
        return tuple(imports)
    pattern = re.compile(
        r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][\w.]*)(?:\.\*)?\s*;"
    )
    return tuple(match.group(1) for match in pattern.finditer(text))


def _class_candidates(import_name: str) -> tuple[str, ...]:
    parts = import_name.split(".")
    candidates = []
    for end in range(len(parts), 2, -1):
        tail = parts[end - 1]
        if tail and (tail[0].isupper() or end == len(parts)):
            candidates.append(".".join(parts[:end]))
    return tuple(dict.fromkeys(candidates))


def _java_imports(
    repo_path: str,
    workspace: Workspace,
    class_index: dict[str, str] | None = None,
) -> tuple[ImportRef, ...]:
    refs: list[ImportRef] = []
    seen: set[tuple[str, str]] = set()
    class_index = class_index or {}
    for rel in find_source_files(repo_path, workspace.root, (".java",)):
        text = read_text(os.path.join(repo_path, rel))
        for raw in _java_import_names(text):
            key = (rel, raw)
            if key in seen:
                continue
            seen.add(key)
            matched_class = next(
                (candidate for candidate in _class_candidates(raw)
                 if candidate in class_index),
                None,
            )
            refs.append(ImportRef(
                language="java",
                workspace=workspace.root,
                raw_specifier=raw,
                source_file=rel,
                scope="test" if "/src/test/" in f"/{rel}" else "runtime",
                confidence="high" if matched_class else "medium",
                package_name=raw,
                package_locator=class_index.get(matched_class or ""),
            ))
    return tuple(refs)


def _maven_locator(item: dict) -> str:
    group = str(item.get("groupId") or "")
    artifact = str(item.get("artifactId") or "")
    packaging = str(item.get("type") or "jar")
    classifier = str(item.get("classifier") or "")
    version = str(item.get("version") or "managed")
    parts = [group, artifact, packaging]
    if classifier:
        parts.append(classifier)
    parts.append(version)
    return ":".join(parts)


def _maven_native_packages(output: str) -> tuple[PackageRecord, ...]:
    if _MAVEN_TREE_MARKER not in output:
        return ()
    payload = output.split(_MAVEN_TREE_MARKER, 1)[1]
    payload = payload.split(_JAVA_CLASS_MARKER, 1)[0].strip()
    try:
        root = json.loads(payload)
    except Exception:
        return ()
    records: dict[str, PackageRecord] = {}

    def visit(item: dict, *, include: bool) -> str | None:
        locator = _maven_locator(item)
        child_locators = tuple(
            child_locator for child in item.get("children") or ()
            if (child_locator := visit(child, include=True))
        )
        if include:
            name = f"{item.get('groupId')}:{item.get('artifactId')}"
            records[locator] = PackageRecord(
                name=name,
                version=str(item.get("version") or "") or None,
                locator=locator,
                dependencies=child_locators,
            )
            return locator
        return None

    for child in root.get("children") or ():
        visit(child, include=True)
    return tuple(records[key] for key in sorted(records))


def _class_index_from_jar_lines(
    output: str,
    packages: tuple[PackageRecord, ...],
) -> tuple[tuple[str, str], ...]:
    matches: list[tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or fields[0] != _JAVA_CLASS_MARKER:
            continue
        class_name, artifact_ref = fields[1], fields[2]
        if artifact_ref in {item.locator for item in packages}:
            locator = artifact_ref
        else:
            basename = os.path.basename(artifact_ref)
            locator = next((
                item.locator for item in packages
                if basename.startswith(
                    f"{item.name.split(':')[-1]}-{item.version or ''}"
                )
            ), None)
        if locator:
            matches.append((class_name, locator))
    return tuple(dict.fromkeys(matches))


def _jar_probe_shell(class_candidates: tuple[str, ...]) -> str:
    checks = []
    for class_name in class_candidates:
        entry = class_name.replace(".", "/") + ".class"
        checks.append(
            "if jar tf \"$jar\" "
            f"{shlex.quote(entry)} 2>/dev/null | grep -Fxq "
            f"{shlex.quote(entry)}; then printf "
            f"'{_JAVA_CLASS_MARKER}\\t%s\\t%s\\n' "
            f"{shlex.quote(class_name)} \"$jar\"; fi"
        )
    if not checks:
        return "true"
    return (
        "tr ':' '\\n' < /tmp/graph2env-classpath.txt "
        "| while IFS= read -r jar; do "
        + "; ".join(checks)
        + "; done"
    )


def _gradle_init_script(class_candidates: tuple[str, ...]) -> str:
    candidates = ", ".join(json.dumps(item) for item in class_candidates)
    return f"""
import java.util.zip.ZipFile
gradle.projectsEvaluated {{
  rootProject.tasks.register("graph2envResolveArtifacts") {{
    doLast {{
      def seenArtifacts = new HashSet()
      rootProject.allprojects.each {{ project ->
        project.configurations.findAll {{ it.canBeResolved }}.each {{ cfg ->
          try {{
            cfg.resolvedConfiguration.resolvedArtifacts.each {{ artifact ->
              def id = artifact.moduleVersion.id
              def coord = "${{id.group}}:${{artifact.name}}:${{id.version}}"
              if (seenArtifacts.add(coord + artifact.file.absolutePath)) {{
                println "{_GRADLE_ARTIFACT_MARKER}\\t${{coord}}\\t${{artifact.file.absolutePath}}"
              }}
              try {{
                def zip = new ZipFile(artifact.file)
                [{candidates}].each {{ className ->
                  def entry = className.replace(".", "/") + ".class"
                  if (zip.getEntry(entry) != null) {{
                    println "{_JAVA_CLASS_MARKER}\\t${{className}}\\t${{coord}}"
                  }}
                }}
                zip.close()
              }} catch (Exception ignored) {{}}
            }}
            def visit
            visit = {{ dep, parent ->
              def coord = "${{dep.moduleGroup}}:${{dep.moduleName}}:${{dep.moduleVersion}}"
              if (parent != null) {{
                println "{_GRADLE_EDGE_MARKER}\\t${{parent}}\\t${{coord}}"
              }}
              dep.children.each {{ child -> visit(child, coord) }}
            }}
            cfg.resolvedConfiguration.firstLevelModuleDependencies.each {{
              dep -> visit(dep, null)
            }}
          }} catch (Exception ignored) {{}}
        }}
      }}
    }}
  }}
}}
"""


def _gradle_native_packages(output: str) -> tuple[PackageRecord, ...]:
    coordinates: dict[str, tuple[str, str | None]] = {}
    dependencies: dict[str, list[str]] = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[0] == _GRADLE_ARTIFACT_MARKER:
            coord = fields[1]
            parts = coord.split(":")
            if len(parts) >= 3:
                coordinates[coord] = (":".join(parts[:-1]), parts[-1])
        elif len(fields) == 3 and fields[0] == _GRADLE_EDGE_MARKER:
            dependencies.setdefault(fields[1], []).append(fields[2])
    return tuple(PackageRecord(
        name=name,
        version=version,
        locator=coord,
        dependencies=tuple(
            item for item in dict.fromkeys(dependencies.get(coord, ()))
            if item in coordinates
        ),
    ) for coord, (name, version) in sorted(coordinates.items()))


class MavenProvider:
    ecosystems = (Ecosystem.MAVEN,)

    def detect_workspaces(self, repo_path, language_requirements):
        req = language_requirement(language_requirements, "java")
        if req is None:
            return ()
        return tuple(Workspace(
            language="java",
            ecosystem=Ecosystem.MAVEN,
            root=os.path.dirname(manifest) or ".",
            manifest_file=manifest,
            package_manager="maven",
            version_constraint=req.version_constraint,
            role=req.role,
            command_runner=(
                "./mvnw"
                if os.path.isfile(os.path.join(
                    repo_path,
                    os.path.dirname(manifest),
                    "mvnw",
                ))
                else "mvn"
            ),
        ) for manifest in _top_level_manifests(find_files(repo_path, {"pom.xml"})))

    def detect_runtime(self, workspace):
        return workspace.version_constraint

    def resolve(
        self,
        repo_path,
        workspace,
        target,
        sandbox=None,
        class_candidates: tuple[str, ...] = (),
    ):
        path = os.path.join(repo_path, workspace.manifest_file)
        try:
            root = ET.fromstring(read_text(path))
        except Exception as exc:
            return ResolutionResult(status="failed", error=f"invalid pom.xml: {exc}")
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}", 1)[0] + "}"
        requirements: list[RequirementRecord] = []
        packages: list[PackageRecord] = []
        for dep in root.findall(f".//{namespace}dependencies/{namespace}dependency"):
            group = dep.findtext(f"{namespace}groupId") or ""
            artifact = dep.findtext(f"{namespace}artifactId") or ""
            version = dep.findtext(f"{namespace}version") or ""
            scope = dep.findtext(f"{namespace}scope") or "runtime"
            if not group or not artifact:
                continue
            name = f"{group}:{artifact}"
            locator = f"{name}:{version or 'managed'}"
            requirements.append(RequirementRecord(
                name=name,
                constraint=version,
                scope=scope,
                source=workspace.manifest_file,
                package_locator=locator,
            ))
            packages.append(PackageRecord(
                name=name,
                version=version or None,
                locator=locator,
            ))
        has_wrapper = os.path.isfile(
            os.path.join(workspace_path(repo_path, workspace), "mvnw")
        )
        wrapper_requires_unzip = (
            has_wrapper and _maven_wrapper_requires_unzip(repo_path, workspace)
        )
        runner = "./mvnw" if has_wrapper else "mvn"
        base_has_java = target.base_image.startswith(
            ("eclipse-temurin:", "maven:", "gradle:")
        )
        major = numeric_version(workspace.version_constraint, "17").split(".", 1)[0]
        prepare_items = []
        if not base_has_java:
            prepare_items.extend((
                "apt-get update",
                f"apt-get install -y --no-install-recommends openjdk-{major}-jdk",
            ))
        if not has_wrapper and not target.base_image.startswith("maven:"):
            if not prepare_items:
                prepare_items.append("apt-get update")
            prepare_items.append(
                "apt-get install -y --no-install-recommends maven"
            )
        if wrapper_requires_unzip:
            if not prepare_items:
                prepare_items.append("apt-get update")
            prepare_items.append(
                "apt-get install -y --no-install-recommends unzip"
            )
        prepare = tuple(prepare_items)
        if has_wrapper:
            prepare += (in_workspace(workspace, "chmod +x ./mvnw"),)
        native = run_native_resolver_probe(
            sandbox,
            ecosystem=workspace.ecosystem,
            workspace=workspace,
            prepare_commands=prepare,
            command=in_workspace(
                workspace,
                f"{runner} -B -DskipTests dependency:tree -DoutputType=json "
                "-DoutputFile=/tmp/graph2env-maven-tree.json "
                f"&& {runner} -B -DskipTests dependency:build-classpath "
                "-Dmdep.outputFile=/tmp/graph2env-classpath.txt "
                f"&& printf '\\n{_MAVEN_TREE_MARKER}\\n' "
                "&& cat /tmp/graph2env-maven-tree.json "
                "&& printf '\\n' "
                f"&& {_jar_probe_shell(class_candidates)}",
            ),
        )
        native_packages = (
            _maven_native_packages(native.raw_output)
            if native and native.status == "resolved" else ()
        )
        selected_packages = native_packages or tuple(packages)
        if selected_packages:
            locators_by_name = {
                item.name.lower(): item.locator for item in selected_packages
            }
            requirements = [replace(
                item,
                package_locator=locators_by_name.get(item.name.lower())
                or item.package_locator,
            ) for item in requirements]
        class_index = (
            _class_index_from_jar_lines(native.raw_output, selected_packages)
            if native and native.status == "resolved" else ()
        )
        return ResolutionResult(
            requirements=tuple(requirements),
            packages=selected_packages,
            status="resolved" if selected_packages else "unresolved",
            error=(
                None if selected_packages
                else "no concrete Maven dependencies from pom.xml or resolver"
            ),
            lock_file=workspace.manifest_file,
            lock_digest=file_digest(path),
            resolution_source="resolver_sandbox" if native_packages else "pom.xml",
            reproducibility_confidence="medium",
            native_evidence=native,
            class_index=class_index,
        )

    def scan_imports(self, repo_path, workspace, target):
        return _java_imports(repo_path, workspace)

    def dependency_commands(self, workspace, resolution):
        runner = workspace.command_runner or "mvn"
        return (
            (in_workspace(workspace, f"{runner} -B -DskipTests dependency:go-offline"),),
            in_workspace(workspace, f"{runner} -B -o -DskipTests dependency:resolve"),
        )

    def project_commands(self, workspace):
        runner = workspace.command_runner or "mvn"
        return (
            (in_workspace(workspace, f"{runner} -B -DskipTests package"),),
            in_workspace(workspace, "test -d target"),
        )

    def test_commands(self, workspace):
        runner = workspace.command_runner or "mvn"
        return () if workspace.role == "build_tool" else (
            in_workspace(workspace, f"{runner} -B test"),
        )

    def build_fragment(self, repo_path, workspace, target, sandbox=None):
        has_wrapper = os.path.isfile(os.path.join(workspace_path(repo_path, workspace), "mvnw"))
        wrapper_requires_unzip = (
            has_wrapper and _maven_wrapper_requires_unzip(repo_path, workspace)
        )
        runner = "./mvnw" if has_wrapper else "mvn"
        initial_imports = self.scan_imports(repo_path, workspace, target)
        class_candidates = tuple(dict.fromkeys(
            candidate
            for item in initial_imports
            for candidate in _class_candidates(item.raw_specifier)
        ))
        resolution = self.resolve(
            repo_path,
            workspace,
            target,
            sandbox=sandbox,
            class_candidates=class_candidates,
        )
        base_has_java = target.base_image.startswith(("eclipse-temurin:", "maven:", "gradle:"))
        major = numeric_version(workspace.version_constraint, "17").split(".", 1)[0]
        runtime_setup = () if base_has_java else (
            f"apt-get update && apt-get install -y --no-install-recommends openjdk-{major}-jdk",
        )
        if wrapper_requires_unzip:
            tool_setup = (
                "apt-get update && apt-get install -y --no-install-recommends unzip",
            )
        elif has_wrapper or target.base_image.startswith("maven:"):
            tool_setup = ()
        else:
            tool_setup = (
                "apt-get update && apt-get install -y --no-install-recommends maven",
            )
        dep_commands = (
            in_workspace(workspace, f"{runner} -B -DskipTests dependency:go-offline"),
        )
        dep_check = in_workspace(workspace, f"{runner} -B -o -DskipTests dependency:resolve")
        project_commands = (in_workspace(workspace, f"{runner} -B -DskipTests package"),)
        tests = () if workspace.role == "build_tool" else (
            in_workspace(workspace, f"{runner} -B test"),
        )
        return graph_from_resolution(
            workspace=workspace,
            target=target,
            resolution=resolution,
            imports=_java_imports(
                repo_path,
                workspace,
                dict(resolution.class_index),
            ),
            runtime_setup=runtime_setup,
            runtime_check="java -version",
            tool_name="maven",
            tool_setup=tool_setup,
            tool_check=f"{runner} -version",
            dependency_commands=dep_commands,
            dependency_check=dep_check,
            project_commands=project_commands,
            project_check=in_workspace(workspace, "test -d target"),
            test_commands=tests,
        )


class GradleProvider:
    ecosystems = (Ecosystem.GRADLE,)

    def detect_workspaces(self, repo_path, language_requirements):
        req = language_requirement(language_requirements, "java")
        if req is None:
            return ()
        manifests = find_files(repo_path, {"build.gradle", "build.gradle.kts"})
        return tuple(Workspace(
            language="java",
            ecosystem=Ecosystem.GRADLE,
            root=os.path.dirname(manifest) or ".",
            manifest_file=manifest,
            package_manager="gradle",
            version_constraint=req.version_constraint,
            role=req.role,
            command_runner=(
                "./gradlew"
                if os.path.isfile(os.path.join(
                    repo_path,
                    os.path.dirname(manifest),
                    "gradlew",
                ))
                else "gradle"
            ),
        ) for manifest in _top_level_manifests(manifests))

    def detect_runtime(self, workspace):
        return workspace.version_constraint

    def resolve(
        self,
        repo_path,
        workspace,
        target,
        sandbox=None,
        class_candidates: tuple[str, ...] = (),
    ):
        path = os.path.join(repo_path, workspace.manifest_file)
        text = read_text(path)
        pattern = re.compile(
            r"""(implementation|api|compileOnly|runtimeOnly|testImplementation)"""
            r"""\s*\(?\s*["']([^:"']+):([^:"']+):([^"']+)["']"""
        )
        requirements: list[RequirementRecord] = []
        packages: list[PackageRecord] = []
        for configuration, group, artifact, version in pattern.findall(text):
            name = f"{group}:{artifact}"
            locator = f"{name}:{version}"
            requirements.append(RequirementRecord(
                name=name,
                constraint=version,
                scope="test" if configuration == "testImplementation" else "runtime",
                source=workspace.manifest_file,
                package_locator=locator,
            ))
            packages.append(PackageRecord(name=name, version=version, locator=locator))
        root = workspace_path(repo_path, workspace)
        has_wrapper = os.path.isfile(os.path.join(root, "gradlew"))
        runner = "./gradlew" if has_wrapper else "gradle"
        base_has_java = target.base_image.startswith(
            ("eclipse-temurin:", "maven:", "gradle:")
        )
        major = numeric_version(workspace.version_constraint, "17").split(".", 1)[0]
        prepare_items = []
        if not base_has_java:
            prepare_items.extend((
                "apt-get update",
                f"apt-get install -y --no-install-recommends openjdk-{major}-jdk",
            ))
        if not has_wrapper and not target.base_image.startswith("gradle:"):
            if not prepare_items:
                prepare_items.append("apt-get update")
            prepare_items.append(
                "apt-get install -y --no-install-recommends gradle"
            )
        prepare = tuple(prepare_items)
        if has_wrapper:
            prepare += (in_workspace(workspace, "chmod +x ./gradlew"),)
        init_script = _gradle_init_script(class_candidates)
        encoded_script = base64.b64encode(init_script.encode()).decode()
        native = run_native_resolver_probe(
            sandbox,
            ecosystem=workspace.ecosystem,
            workspace=workspace,
            prepare_commands=prepare,
            command=(
                f"printf %s {shlex.quote(encoded_script)} | base64 -d "
                "> /tmp/graph2env-resolver.gradle "
                "&& "
                + in_workspace(
                    workspace,
                    f"{runner} -I /tmp/graph2env-resolver.gradle "
                    "-q graph2envResolveArtifacts",
                )
            ),
        )
        native_packages = (
            _gradle_native_packages(native.raw_output)
            if native and native.status == "resolved" else ()
        )
        selected_packages = native_packages or tuple(packages)
        if selected_packages:
            locators_by_name = {
                item.name.lower(): item.locator for item in selected_packages
            }
            requirements = [replace(
                item,
                package_locator=locators_by_name.get(item.name.lower())
                or item.package_locator,
            ) for item in requirements]
        class_index = (
            _class_index_from_jar_lines(native.raw_output, selected_packages)
            if native and native.status == "resolved" else ()
        )
        return ResolutionResult(
            requirements=tuple(requirements),
            packages=selected_packages,
            status="resolved" if selected_packages else "unresolved",
            error=(
                None if selected_packages
                else "Gradle model did not expose resolvable dependencies"
            ),
            lock_file=workspace.manifest_file,
            lock_digest=file_digest(path),
            resolution_source=(
                "resolver_sandbox" if native_packages else "build.gradle"
            ),
            reproducibility_confidence=(
                "medium" if selected_packages else "low"
            ),
            native_evidence=native,
            class_index=class_index,
        )

    def scan_imports(self, repo_path, workspace, target):
        return _java_imports(repo_path, workspace)

    def dependency_commands(self, workspace, resolution):
        runner = workspace.command_runner or "gradle"
        return (
            (in_workspace(workspace, f"{runner} --no-daemon dependencies"),),
            in_workspace(workspace, f"{runner} --offline dependencies"),
        )

    def project_commands(self, workspace):
        runner = workspace.command_runner or "gradle"
        return (
            (in_workspace(workspace, f"{runner} --no-daemon -x test assemble"),),
            in_workspace(workspace, "test -d build"),
        )

    def test_commands(self, workspace):
        runner = workspace.command_runner or "gradle"
        return () if workspace.role == "build_tool" else (
            in_workspace(workspace, f"{runner} --no-daemon test"),
        )

    def build_fragment(self, repo_path, workspace, target, sandbox=None):
        root = workspace_path(repo_path, workspace)
        has_wrapper = os.path.isfile(os.path.join(root, "gradlew"))
        runner = "./gradlew" if has_wrapper else "gradle"
        initial_imports = self.scan_imports(repo_path, workspace, target)
        class_candidates = tuple(dict.fromkeys(
            candidate
            for item in initial_imports
            for candidate in _class_candidates(item.raw_specifier)
        ))
        resolution = self.resolve(
            repo_path,
            workspace,
            target,
            sandbox=sandbox,
            class_candidates=class_candidates,
        )
        base_has_java = target.base_image.startswith(("eclipse-temurin:", "maven:", "gradle:"))
        major = numeric_version(workspace.version_constraint, "17").split(".", 1)[0]
        runtime_setup = () if base_has_java else (
            f"apt-get update && apt-get install -y --no-install-recommends openjdk-{major}-jdk",
        )
        tool_setup = () if has_wrapper or target.base_image.startswith("gradle:") else (
            "apt-get update && apt-get install -y --no-install-recommends gradle",
        )
        return graph_from_resolution(
            workspace=workspace,
            target=target,
            resolution=resolution,
            imports=_java_imports(
                repo_path,
                workspace,
                dict(resolution.class_index),
            ),
            runtime_setup=runtime_setup,
            runtime_check="java -version",
            tool_name="gradle",
            tool_setup=tool_setup,
            tool_check=f"{runner} --version",
            dependency_commands=(in_workspace(workspace, f"{runner} --no-daemon dependencies"),),
            dependency_check=in_workspace(workspace, f"{runner} --offline dependencies"),
            project_commands=(in_workspace(workspace, f"{runner} --no-daemon -x test assemble"),),
            project_check=in_workspace(workspace, "test -d build"),
            test_commands=(
                () if workspace.role == "build_tool"
                else (in_workspace(workspace, f"{runner} --no-daemon test"),)
            ),
        )
