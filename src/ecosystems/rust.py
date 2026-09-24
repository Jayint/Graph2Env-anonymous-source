"""Cargo provider."""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

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
    parse_syntax_tree,
    syntax_text,
    walk_syntax_tree,
)


_CARGO_MARKER = "__GRAPH2ENV_CARGO_METADATA_JSON__"


def _cargo_dependencies(data: dict) -> tuple[tuple[str, str, str, str], ...]:
    out: list[tuple[str, str, str, str]] = []
    tables = (
        ("runtime", "dependencies"),
        ("test", "dev-dependencies"),
        ("build", "build-dependencies"),
    )
    for scope, key in tables:
        for alias, value in sorted((data.get(key) or {}).items()):
            if isinstance(value, str):
                name, constraint = alias, value
            elif isinstance(value, dict):
                name = str(value.get("package") or alias)
                constraint = str(value.get("version") or "")
            else:
                continue
            out.append((alias, name, constraint, scope))
    return tuple(out)


def _cargo_native_packages(output: str) -> tuple[PackageRecord, ...]:
    if _CARGO_MARKER not in output:
        return ()
    payload = output.split(_CARGO_MARKER, 1)[1].strip()
    try:
        metadata = json.loads(payload)
    except Exception:
        return ()
    workspace_members = set(metadata.get("workspace_members") or ())
    packages_by_id = {
        str(item.get("id")): item
        for item in metadata.get("packages") or ()
        if item.get("id") and str(item.get("id")) not in workspace_members
    }
    dependency_ids: dict[str, tuple[str, ...]] = {}
    resolve = metadata.get("resolve") or {}
    for node in resolve.get("nodes") or ():
        node_id = str(node.get("id") or "")
        if node_id not in packages_by_id:
            continue
        dependencies = []
        for dependency in node.get("deps") or ():
            package_id = str(dependency.get("pkg") or "")
            if package_id in packages_by_id:
                dependencies.append(package_id)
        if not dependencies:
            dependencies.extend(
                item for item in node.get("dependencies") or ()
                if item in packages_by_id
            )
        dependency_ids[node_id] = tuple(dict.fromkeys(dependencies))
    return tuple(PackageRecord(
        name=str(item.get("name") or ""),
        version=str(item.get("version") or "") or None,
        locator=package_id,
        dependencies=dependency_ids.get(package_id, ()),
    ) for package_id, item in sorted(packages_by_id.items()))


def _rust_imports_from_ast(
    text: str,
) -> tuple[tuple[str, str | None], ...]:
    tree = parse_syntax_tree(text, "rust")
    if tree is None:
        return ()
    source = text.encode("utf-8")
    imports: list[tuple[str, str | None]] = []
    for node in walk_syntax_tree(tree.root_node):
        raw = None
        if node.type == "use_declaration":
            argument = node.child_by_field_name("argument")
            argument_text = syntax_text(source, argument) if argument else ""
            match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", argument_text.lstrip(":"))
            raw = match.group(0) if match else None
        elif node.type == "extern_crate_declaration":
            name = node.child_by_field_name("name")
            raw = syntax_text(source, name) if name else None
        if not raw:
            continue
        prefix = source[max(0, node.start_byte - 300):node.start_byte].decode(
            "utf-8",
            errors="replace",
        )
        condition_match = re.search(
            r"#\s*\[\s*cfg\s*\((.*?)\)\s*\]\s*$",
            prefix,
            re.S,
        )
        imports.append((
            raw,
            condition_match.group(1).strip() if condition_match else None,
        ))
    return tuple(imports)


def _rust_local_modules(repo_path: str, workspace: Workspace) -> set[str]:
    modules: set[str] = set()
    for rel in find_source_files(repo_path, workspace.root, (".rs",)):
        text = read_text(os.path.join(repo_path, rel))
        tree = parse_syntax_tree(text, "rust")
        if tree is None:
            modules.update(re.findall(
                r"(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)",
                text,
            ))
            continue
        source = text.encode("utf-8")
        for node in walk_syntax_tree(tree.root_node):
            if node.type == "mod_item":
                name = node.child_by_field_name("name")
                if name:
                    modules.add(syntax_text(source, name))
    return modules


class RustProvider:
    ecosystems = (Ecosystem.CARGO,)

    def detect_workspaces(self, repo_path, language_requirements):
        req = language_requirement(language_requirements, "rust")
        if req is None:
            return ()
        manifests = find_files(repo_path, {"Cargo.toml"})
        roots: list[str] = []
        workspaces: list[Workspace] = []
        for manifest in sorted(manifests, key=lambda item: (item.count("/"), item)):
            root = os.path.dirname(manifest) or "."
            if any(
                root != parent
                and (parent == "." or root.startswith(parent.rstrip("/") + "/"))
                for parent in roots
            ):
                continue
            roots.append(root)
            try:
                data = tomllib.loads(read_text(os.path.join(repo_path, manifest)))
            except Exception:
                data = {}
            package_version = (data.get("package") or {}).get("rust-version")
            workspace_version = (
                ((data.get("workspace") or {}).get("package") or {}).get(
                    "rust-version"
                )
            )
            if isinstance(package_version, str):
                version = package_version
            elif (
                isinstance(package_version, dict)
                and package_version.get("workspace") is True
                and isinstance(workspace_version, str)
            ):
                version = workspace_version
            elif isinstance(workspace_version, str):
                version = workspace_version
            else:
                version = req.version_constraint
            workspaces.append(Workspace(
                language="rust",
                ecosystem=Ecosystem.CARGO,
                root=root,
                manifest_file=manifest,
                package_manager="cargo",
                version_constraint=str(version) if version else None,
                role=req.role,
            ))
        return tuple(workspaces)

    def detect_runtime(self, workspace):
        return workspace.version_constraint

    def resolve(self, repo_path, workspace, target, sandbox=None):
        root = workspace_path(repo_path, workspace)
        manifest_path = os.path.join(root, "Cargo.toml")
        try:
            manifest = tomllib.loads(read_text(manifest_path))
        except Exception as exc:
            return ResolutionResult(status="failed", error=f"invalid Cargo.toml: {exc}")
        direct = _cargo_dependencies(manifest)
        lock_path = os.path.join(root, "Cargo.lock")
        packages: list[PackageRecord] = []
        name_to_locator: dict[str, str] = {}
        if os.path.isfile(lock_path):
            try:
                lock = tomllib.loads(read_text(lock_path))
                raw_packages = []
                for item in lock.get("package", ()):
                    name = str(item.get("name") or "")
                    version = str(item.get("version") or "") or None
                    if not name:
                        continue
                    source = str(item.get("source") or "")
                    locator = (
                        f"{name}@{version or 'unresolved'}"
                        + (f"#{source}" if source else "")
                    )
                    name_to_locator.setdefault(name.lower(), locator)
                    raw_packages.append((name, version, locator, item))
                locators_by_name: dict[str, list[tuple[str | None, str]]] = {}
                for name, version, locator, _item in raw_packages:
                    locators_by_name.setdefault(name.lower(), []).append(
                        (version, locator)
                    )

                def dependency_locator(raw: str) -> str | None:
                    parts = raw.split()
                    name = parts[0].lower() if parts else ""
                    version = parts[1] if len(parts) > 1 else None
                    candidates = locators_by_name.get(name, ())
                    return next(
                        (locator for candidate_version, locator in candidates
                         if version and candidate_version == version),
                        candidates[0][1] if candidates else None,
                    )

                packages.extend(PackageRecord(
                    name=name,
                    version=version,
                    locator=locator,
                    dependencies=tuple(
                        dependency
                        for raw in item.get("dependencies", ())
                        if (
                            dependency := dependency_locator(str(raw))
                        ) is not None
                    ),
                ) for name, version, locator, item in raw_packages)
            except Exception as exc:
                return ResolutionResult(
                    status="failed",
                    error=f"invalid Cargo.lock: {exc}",
                    lock_file=os.path.relpath(lock_path, repo_path).replace(os.sep, "/"),
                    lock_digest=file_digest(lock_path),
                )
        requirements = tuple(RequirementRecord(
            name=name,
            constraint=constraint,
            scope=scope,
            source=workspace.manifest_file,
            package_locator=name_to_locator.get(name.lower()),
        ) for _alias, name, constraint, scope in direct)
        locked = " --locked" if os.path.isfile(lock_path) else ""
        base_has_rust = target.base_image.startswith("rust:")
        toolchain = workspace.version_constraint or "stable"
        prepare = () if base_has_rust else (
            "apt-get update",
            "apt-get install -y --no-install-recommends "
            "curl ca-certificates build-essential",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            f"| sh -s -- -y --default-toolchain {toolchain}",
            "export PATH=/root/.cargo/bin:$PATH",
        )
        native = run_native_resolver_probe(
            sandbox,
            ecosystem=workspace.ecosystem,
            workspace=workspace,
            prepare_commands=prepare,
            command=(
                "export PATH=/usr/local/cargo/bin:/root/.cargo/bin:$PATH && "
                + in_workspace(
                    workspace,
                    f"cargo metadata --format-version 1{locked} "
                    "> /tmp/graph2env-cargo-metadata.json "
                    f"&& printf '\\n{_CARGO_MARKER}\\n' "
                    "&& cat /tmp/graph2env-cargo-metadata.json",
                )
            ),
        )
        native_packages = (
            _cargo_native_packages(native.raw_output)
            if native and native.status == "resolved" else ()
        )
        if native_packages:
            packages = list(native_packages)
            locators_by_name: dict[str, list[str]] = {}
            for package in native_packages:
                locators_by_name.setdefault(package.name.lower(), []).append(
                    package.locator
                )
            requirements = tuple(replace(
                item,
                package_locator=(
                    locators_by_name.get(item.name.lower(), [None])[0]
                    or item.package_locator
                ),
            ) for item in requirements)
        return ResolutionResult(
            requirements=requirements,
            packages=tuple(packages),
            status="resolved" if packages else "unresolved",
            error=None if packages else "no Cargo.lock; cargo will resolve in the dependency transaction",
            lock_file=(
                os.path.relpath(lock_path, repo_path).replace(os.sep, "/")
                if os.path.isfile(lock_path) else None
            ),
            lock_digest=file_digest(lock_path) if os.path.isfile(lock_path) else None,
            resolution_source="resolver_sandbox" if native_packages else (
                "lockfile" if packages else
                "resolver_sandbox" if native else "manifest"
            ),
            reproducibility_confidence=(
                "high" if os.path.isfile(lock_path)
                else ("medium" if native_packages else "low")
            ),
            native_evidence=native,
        )

    def scan_imports(self, repo_path, workspace, target):
        try:
            manifest = tomllib.loads(read_text(
                os.path.join(workspace_path(repo_path, workspace), "Cargo.toml")
            ))
        except Exception:
            manifest = {}
        alias_to_name = {
            alias: name for alias, name, _constraint, _scope in _cargo_dependencies(manifest)
        }
        local_modules = _rust_local_modules(repo_path, workspace)
        refs: list[ImportRef] = []
        seen: set[tuple[str, str]] = set()
        pattern = re.compile(r"(?m)^\s*(?:use|extern\s+crate)\s+([A-Za-z_][A-Za-z0-9_]*)")
        for rel in find_source_files(repo_path, workspace.root, (".rs",)):
            text = read_text(os.path.join(repo_path, rel))
            ast_imports = _rust_imports_from_ast(text)
            imports = ast_imports or tuple(
                (match.group(1), None) for match in pattern.finditer(text)
            )
            for raw, condition in imports:
                if raw in {"crate", "self", "super", "std", "core", "alloc"}:
                    continue
                if raw in local_modules and raw not in alias_to_name:
                    continue
                key = (rel, raw)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ImportRef(
                    language="rust",
                    workspace=workspace.root,
                    raw_specifier=raw,
                    source_file=rel,
                    scope="test" if "/tests/" in f"/{rel}" else "runtime",
                    conditional=condition,
                    package_name=alias_to_name.get(raw, raw.replace("_", "-")),
                ))
        return tuple(refs)

    def dependency_commands(self, workspace, resolution):
        locked = " --locked" if resolution.lock_file else ""
        return (
            (in_workspace(workspace, f"cargo fetch{locked}"),),
            in_workspace(workspace, f"cargo metadata --format-version 1{locked}"),
        )

    def project_commands(self, workspace):
        return (
            (in_workspace(workspace, "cargo build --all-targets"),),
            in_workspace(workspace, "cargo metadata --format-version 1"),
        )

    def test_commands(self, workspace):
        return () if workspace.role == "build_tool" else (
            in_workspace(workspace, "cargo test --all-targets"),
        )

    def build_fragment(self, repo_path, workspace, target, sandbox=None):
        resolution = self.resolve(repo_path, workspace, target, sandbox=sandbox)
        base_has_rust = target.base_image.startswith("rust:")
        toolchain = workspace.version_constraint or "stable"
        runtime_setup = ()
        if not base_has_rust:
            runtime_setup = (
                "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y "
                f"--default-toolchain {toolchain}",
            )
        dep_commands, dep_check = self.dependency_commands(workspace, resolution)
        project_commands, project_check = self.project_commands(workspace)
        cargo_bin_dir = (
            "/usr/local/cargo/bin" if base_has_rust else "/root/.cargo/bin"
        )
        tool_setup = (
            f"ln -sf {cargo_bin_dir}/cargo /usr/local/bin/cargo && "
            f"ln -sf {cargo_bin_dir}/rustc /usr/local/bin/rustc",
        )
        tests = self.test_commands(workspace)
        return graph_from_resolution(
            workspace=workspace,
            target=target,
            resolution=resolution,
            imports=self.scan_imports(repo_path, workspace, target),
            runtime_setup=runtime_setup,
            runtime_check=f"{cargo_bin_dir}/rustc --version",
            tool_name="cargo",
            tool_setup=tool_setup,
            tool_check="cargo --version",
            dependency_commands=dep_commands,
            dependency_check=dep_check,
            project_commands=project_commands,
            project_check=project_check,
            test_commands=tests,
        )
