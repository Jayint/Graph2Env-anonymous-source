"""Go Modules provider."""
from __future__ import annotations

import json
import os
import re
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
    is_test_fixture_path,
    language_requirement,
    numeric_version,
)
from src.ecosystems.common import (
    parse_syntax_tree,
    quoted_string_value,
    syntax_text,
    walk_syntax_tree,
)


_REQUIRE_LINE = re.compile(r"^\s*([A-Za-z0-9._~+/-]+)\s+(v[^\s]+)")
_GOOS = frozenset({
    "aix", "android", "darwin", "dragonfly", "freebsd", "illumos", "ios",
    "js", "linux", "netbsd", "openbsd", "plan9", "solaris", "wasip1",
    "windows",
})
_GOARCH = frozenset({
    "386", "amd64", "arm", "arm64", "loong64", "mips", "mips64",
    "mips64le", "mipsle", "ppc64", "ppc64le", "riscv64", "s390x", "wasm",
})
_UNIX_GOOS = frozenset({
    "aix", "android", "darwin", "dragonfly", "freebsd", "illumos", "ios",
    "linux", "netbsd", "openbsd", "solaris",
})
_JSON_MARKER = "__GRAPH2ENV_GO_MODULES_JSON__"
_GRAPH_MARKER = "__GRAPH2ENV_GO_MODULE_GRAPH__"
_PACKAGES_MARKER = "__GRAPH2ENV_GO_PACKAGES_JSON__"


class _BuildExpression:
    def __init__(self, expression: str, tags: set[str]):
        self.tokens = re.findall(r"\|\||&&|!|\(|\)|[A-Za-z0-9_.]+", expression)
        self.position = 0
        self.tags = tags

    def evaluate(self) -> bool:
        value = self._or()
        if self.position != len(self.tokens):
            raise ValueError("trailing build expression tokens")
        return value

    def _or(self) -> bool:
        value = self._and()
        while self._take("||"):
            right = self._and()
            value = value or right
        return value

    def _and(self) -> bool:
        value = self._not()
        while self._take("&&"):
            right = self._not()
            value = value and right
        return value

    def _not(self) -> bool:
        if self._take("!"):
            return not self._not()
        if self._take("("):
            value = self._or()
            if not self._take(")"):
                raise ValueError("unclosed build expression")
            return value
        if self.position >= len(self.tokens):
            raise ValueError("missing build tag")
        token = self.tokens[self.position]
        self.position += 1
        return token in self.tags

    def _take(self, token: str) -> bool:
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False


def _target_go_tags(
    target: ProviderTarget,
    version_constraint: str | None,
) -> tuple[str, str, set[str]]:
    platform = (target.platform or "linux/amd64").removeprefix("docker://")
    parts = platform.split("/", 1)
    goos = parts[0] if parts and parts[0] in _GOOS else "linux"
    goarch = parts[1] if len(parts) == 2 and parts[1] in _GOARCH else "amd64"
    tags = {goos, goarch, "gc"}
    if goos in _UNIX_GOOS:
        tags.add("unix")
    version = numeric_version(version_constraint, "1.22")
    match = re.match(r"1\.(\d+)", version)
    if match:
        for minor in range(1, int(match.group(1)) + 1):
            tags.add(f"go1.{minor}")
    return goos, goarch, tags


def _filename_matches_target(path: str, goos: str, goarch: str) -> bool:
    stem = os.path.basename(path)
    if not stem.endswith(".go"):
        return False
    stem = stem[:-3]
    if stem.endswith("_test"):
        stem = stem[:-5]
    parts = stem.split("_")
    suffixes = parts[1:]
    os_tags = [part for part in suffixes if part in _GOOS]
    arch_tags = [part for part in suffixes if part in _GOARCH]
    return (not os_tags or goos in os_tags) and (
        not arch_tags or goarch in arch_tags
    )


def _legacy_build_line_matches(line: str, tags: set[str]) -> bool:
    alternatives = line.split()
    return any(
        all(
            (term[1:] not in tags) if term.startswith("!") else (term in tags)
            for term in option.split(",")
            if term
        )
        for option in alternatives
    )


def _source_matches_build_tags(
    path: str,
    text: str,
    *,
    goos: str,
    goarch: str,
    tags: set[str],
) -> bool:
    if not _filename_matches_target(path, goos, goarch):
        return False
    header = text.split("package", 1)[0]
    modern = re.search(r"(?m)^\s*//go:build\s+(.+?)\s*$", header)
    if modern:
        try:
            return _BuildExpression(modern.group(1), tags).evaluate()
        except ValueError:
            return True
    legacy = re.findall(r"(?m)^\s*//\s*\+build\s+(.+?)\s*$", header)
    return all(_legacy_build_line_matches(line, tags) for line in legacy)


def _go_imports_from_ast(text: str) -> tuple[str, ...]:
    tree = parse_syntax_tree(text, "go")
    if tree is None:
        return ()
    source = text.encode("utf-8")
    imports: list[str] = []
    for node in walk_syntax_tree(tree.root_node):
        if node.type != "import_spec":
            continue
        path = node.child_by_field_name("path")
        raw = quoted_string_value(syntax_text(source, path)) if path else None
        if raw:
            imports.append(raw)
    return tuple(imports)


def _go_work_modules(repo_path: str, workspace: Workspace) -> tuple[str, ...]:
    go_work = os.path.join(repo_path, "go.work")
    text = read_text(go_work)
    if not text:
        return ()
    paths: list[str] = []
    block = re.search(r"(?ms)^\s*use\s*\((.*?)^\s*\)", text)
    if block:
        paths.extend(
            line.split("//", 1)[0].strip()
            for line in block.group(1).splitlines()
            if line.split("//", 1)[0].strip()
        )
    paths.extend(re.findall(r"(?m)^\s*use\s+([^\s(][^\s]*)", text))
    modules: list[str] = []
    for path in paths:
        path = path.strip("\"'")
        module_file = os.path.join(repo_path, path, "go.mod")
        match = re.search(
            r"(?m)^\s*module\s+(\S+)",
            read_text(module_file),
        )
        if match:
            modules.append(match.group(1))
    return tuple(modules)


def _json_stream(text: str) -> tuple[dict, ...]:
    decoder = json.JSONDecoder()
    position = 0
    values: list[dict] = []
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if isinstance(value, dict):
            values.append(value)
    return tuple(values)


def _go_native_packages(output: str) -> tuple[PackageRecord, ...]:
    if _JSON_MARKER not in output:
        return ()
    payload = output.split(_JSON_MARKER, 1)[1]
    modules_text, _, graph_payload = payload.partition(_GRAPH_MARKER)
    graph_text = graph_payload.partition(_PACKAGES_MARKER)[0]
    modules = _json_stream(modules_text)
    locator_by_identity: dict[str, str] = {}
    raw_records: list[tuple[str, str | None, str]] = []
    for item in modules:
        if item.get("Main"):
            continue
        selected = item.get("Replace") if isinstance(item.get("Replace"), dict) else item
        name = str(item.get("Path") or "")
        selected_path = str(selected.get("Path") or name)
        version = str(selected.get("Version") or item.get("Version") or "") or None
        if not name:
            continue
        locator = f"{selected_path}@{version or 'local'}"
        raw_records.append((name, version, locator))
        original = f"{name}@{item.get('Version')}" if item.get("Version") else name
        locator_by_identity[original] = locator
        locator_by_identity[name] = locator
        locator_by_identity[locator] = locator

    dependencies: dict[str, list[str]] = {}
    for raw in graph_text.splitlines():
        parts = raw.split()
        if len(parts) != 2:
            continue
        src = locator_by_identity.get(parts[0])
        dst = locator_by_identity.get(parts[1])
        if src and dst:
            dependencies.setdefault(src, []).append(dst)
    return tuple(PackageRecord(
        name=name,
        version=version,
        locator=locator,
        dependencies=tuple(dict.fromkeys(dependencies.get(locator, ()))),
    ) for name, version, locator in raw_records)


def _go_native_imports(
    output: str,
    *,
    repo_path: str,
    workspace: Workspace,
    container_workdir: str = "/app",
) -> tuple[ImportRef, ...]:
    if _PACKAGES_MARKER not in output or _JSON_MARKER not in output:
        return ()
    modules_payload = output.split(_JSON_MARKER, 1)[1].split(_GRAPH_MARKER, 1)[0]
    modules = _json_stream(modules_payload)
    local_modules = tuple(
        str(item.get("Path") or "")
        for item in modules if item.get("Main") and item.get("Path")
    )
    external_modules = tuple(
        str(item.get("Path") or "")
        for item in modules if not item.get("Main") and item.get("Path")
    )
    packages = _json_stream(output.split(_PACKAGES_MARKER, 1)[1])
    refs = []
    seen = set()
    for package in packages:
        directory = str(package.get("Dir") or "")
        files = tuple(
            str(item)
            for key in ("GoFiles", "CgoFiles", "TestGoFiles", "XTestGoFiles")
            for item in package.get(key) or ()
        )
        source_file = files[0] if files else f"<go-list:{package.get('ImportPath')}>"
        if directory:
            try:
                source_file = os.path.relpath(
                    os.path.join(directory, source_file),
                    container_workdir,
                ).replace(os.sep, "/")
            except ValueError:
                pass
        for key, scope in (
            ("Imports", "runtime"),
            ("TestImports", "test"),
            ("XTestImports", "test"),
        ):
            for raw in package.get(key) or ():
                raw = str(raw)
                if "." not in raw.split("/", 1)[0]:
                    continue
                if any(
                    raw == module or raw.startswith(module + "/")
                    for module in local_modules
                ):
                    continue
                package_name = max(
                    (
                        module for module in external_modules
                        if raw == module or raw.startswith(module + "/")
                    ),
                    key=len,
                    default=raw,
                )
                identity = (source_file, raw, scope)
                if identity in seen:
                    continue
                seen.add(identity)
                refs.append(ImportRef(
                    language="go",
                    workspace=workspace.root,
                    raw_specifier=raw,
                    source_file=source_file,
                    scope=scope,
                    package_name=package_name,
                ))
    return tuple(refs)


def _go_requirements(text: str) -> tuple[tuple[str, str], ...]:
    requirements: list[tuple[str, str]] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if line == "require (":
            in_block = True
            continue
        if in_block and line == ")":
            in_block = False
            continue
        candidate = line
        if line.startswith("require "):
            candidate = line[len("require "):].strip()
        elif not in_block:
            continue
        match = _REQUIRE_LINE.match(candidate)
        if match:
            requirements.append((match.group(1), match.group(2)))
    return tuple(requirements)


class GoProvider:
    ecosystems = (Ecosystem.GO_MODULE,)

    def detect_workspaces(self, repo_path, language_requirements):
        req = language_requirement(language_requirements, "go")
        if req is None:
            return ()
        workspaces = []
        for manifest in find_files(repo_path, {"go.mod"}):
            if is_test_fixture_path(manifest):
                continue
            root = os.path.dirname(manifest) or "."
            text = read_text(os.path.join(repo_path, manifest))
            match = re.search(r"(?m)^\s*go\s+([0-9.]+)", text)
            workspaces.append(Workspace(
                language="go",
                ecosystem=Ecosystem.GO_MODULE,
                root=root,
                manifest_file=manifest,
                package_manager="go",
                version_constraint=match.group(1) if match else req.version_constraint,
                role=req.role,
            ))
        return tuple(workspaces)

    def detect_runtime(self, workspace):
        return workspace.version_constraint

    def resolve(self, repo_path, workspace, target, sandbox=None):
        root = workspace_path(repo_path, workspace)
        path = os.path.join(root, "go.mod")
        text = read_text(path)
        if not text:
            return ResolutionResult(status="failed", error="go.mod unreadable")
        requirements = tuple(RequirementRecord(
            name=name,
            constraint=version,
            source=workspace.manifest_file,
            package_locator=f"{name}@{version}",
        ) for name, version in _go_requirements(text))
        packages = tuple(PackageRecord(
            name=item.name,
            version=item.constraint,
            locator=item.package_locator or f"{item.name}@{item.constraint}",
        ) for item in requirements)
        sum_path = os.path.join(root, "go.sum")
        lock = sum_path if os.path.isfile(sum_path) else path
        base_has_go = target.base_image.startswith(("golang:", "go:"))
        prepare = () if base_has_go else (
            "apt-get update",
            "apt-get install -y --no-install-recommends golang-go ca-certificates",
        )
        native = run_native_resolver_probe(
            sandbox,
            ecosystem=workspace.ecosystem,
            workspace=workspace,
            prepare_commands=prepare,
            command=in_workspace(
                workspace,
                "go list -m -json all > /tmp/graph2env-go-modules.json "
                "&& go mod graph > /tmp/graph2env-go-graph.txt "
                "&& go list -json ./... > /tmp/graph2env-go-packages.json "
                f"&& printf '\\n{_JSON_MARKER}\\n' "
                "&& cat /tmp/graph2env-go-modules.json "
                f"&& printf '\\n{_GRAPH_MARKER}\\n' "
                "&& cat /tmp/graph2env-go-graph.txt "
                f"&& printf '\\n{_PACKAGES_MARKER}\\n' "
                "&& cat /tmp/graph2env-go-packages.json",
            ),
        )
        native_packages = (
            _go_native_packages(native.raw_output)
            if native and native.status == "resolved" else ()
        )
        if native_packages:
            locator_by_name = {
                item.name.lower(): item.locator for item in native_packages
            }
            requirements = tuple(replace(
                item,
                package_locator=locator_by_name.get(item.name.lower())
                or item.package_locator,
            ) for item in requirements)
            packages = native_packages
        scanner_imports = (
            _go_native_imports(
                native.raw_output,
                repo_path=repo_path,
                workspace=workspace,
                container_workdir=str(getattr(sandbox, "workdir", "/app")),
            )
            if native and native.status == "resolved" else ()
        )
        return ResolutionResult(
            requirements=requirements,
            packages=packages,
            status="resolved",
            lock_file=os.path.relpath(lock, repo_path).replace(os.sep, "/"),
            lock_digest=file_digest(lock),
            resolution_source=(
                "resolver_sandbox"
                if native_packages
                else ("go.mod+go.sum" if lock == sum_path else "go.mod")
            ),
            reproducibility_confidence="high" if lock == sum_path else "medium",
            native_evidence=native,
            scanner_imports=scanner_imports,
        )

    def scan_imports(self, repo_path, workspace, target):
        root = workspace_path(repo_path, workspace)
        module_match = re.search(
            r"(?m)^\s*module\s+(\S+)", read_text(os.path.join(root, "go.mod"))
        )
        local_module = module_match.group(1) if module_match else ""
        local_modules = tuple(
            item for item in (local_module, *_go_work_modules(repo_path, workspace))
            if item
        )
        requirements = _go_requirements(read_text(os.path.join(root, "go.mod")))
        module_names = tuple(name for name, _ in requirements)
        goos, goarch, tags = _target_go_tags(
            target,
            workspace.version_constraint,
        )
        refs: list[ImportRef] = []
        seen: set[tuple[str, str]] = set()
        for rel in find_source_files(repo_path, workspace.root, (".go",)):
            text = read_text(os.path.join(repo_path, rel))
            if not _source_matches_build_tags(
                rel,
                text,
                goos=goos,
                goarch=goarch,
                tags=tags,
            ):
                continue
            strings = list(_go_imports_from_ast(text))
            if not strings:
                strings = re.findall(
                    r'(?m)^\s*import\s+(?:[A-Za-z0-9_.]+\s+)?"([^"]+)"',
                    text,
                )
                for block in re.findall(r"(?ms)^\s*import\s*\((.*?)^\s*\)", text):
                    strings.extend(re.findall(
                        r'(?m)^\s*(?:[A-Za-z0-9_.]+\s+)?"([^"]+)"',
                        block,
                    ))
            for raw in strings:
                if "." not in raw.split("/", 1)[0]:
                    continue
                if any(
                    raw == module or raw.startswith(module + "/")
                    for module in local_modules
                ):
                    continue
                package = max(
                    (module for module in module_names
                     if raw == module or raw.startswith(module + "/")),
                    key=len,
                    default=raw,
                )
                key = (rel, raw)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ImportRef(
                    language="go",
                    workspace=workspace.root,
                    raw_specifier=raw,
                    source_file=rel,
                    scope="test" if rel.endswith("_test.go") else "runtime",
                    package_name=package,
                ))
        return tuple(refs)

    def dependency_commands(self, workspace, resolution):
        return (
            (in_workspace(workspace, "go mod download"),),
            in_workspace(workspace, "go mod verify"),
        )

    def project_commands(self, workspace):
        return (
            (in_workspace(workspace, "go build ./..."),),
            in_workspace(workspace, "go list ./..."),
        )

    def test_commands(self, workspace):
        return () if workspace.role == "build_tool" else (
            in_workspace(workspace, "go test ./..."),
        )

    def build_fragment(self, repo_path, workspace, target, sandbox=None):
        resolution = self.resolve(repo_path, workspace, target, sandbox=sandbox)
        version = numeric_version(workspace.version_constraint, "1.22")
        base_has_go = target.base_image.startswith(("golang:", "go:"))
        runtime_setup = ()
        if not base_has_go:
            runtime_setup = (
                "apt-get update && apt-get install -y --no-install-recommends golang-go",
            )
        dep_commands, dep_check = self.dependency_commands(workspace, resolution)
        project_commands, project_check = self.project_commands(workspace)
        return graph_from_resolution(
            workspace=workspace,
            target=target,
            resolution=resolution,
            imports=(
                resolution.scanner_imports
                or self.scan_imports(repo_path, workspace, target)
            ),
            runtime_setup=runtime_setup,
            runtime_check=f"go version | grep -F 'go{version}'",
            tool_name="go",
            tool_setup=(),
            tool_check="go env GOMODCACHE",
            dependency_commands=dep_commands,
            dependency_check=dep_check,
            project_commands=project_commands,
            project_check=project_check,
            test_commands=self.test_commands(workspace),
        )
