"""npm/pnpm/Yarn provider with lock-aware package and JS/TS import graphs."""
from __future__ import annotations

import json
import os
import re
import base64
import shlex
from dataclasses import replace
from fnmatch import fnmatch

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
    quoted_string_value,
    syntax_text,
    walk_syntax_tree,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional lock parser
    yaml = None


_NODE_LOCK_MARKER = "__GRAPH2ENV_NODE_LOCK__"
_TYPESCRIPT_IMPORT_MARKER = "__GRAPH2ENV_TYPESCRIPT_IMPORTS__"


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.split(".")[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def _node_constraint_floor(constraint: str | None) -> tuple[int, int, int] | None:
    """Conservatively extract the lowest viable Node version from npm semver."""
    if not constraint:
        return None
    alternatives = []
    for branch in str(constraint).split("||"):
        versions = re.findall(r"(?<![A-Za-z0-9])v?(\d+(?:\.\d+){0,2})", branch)
        if versions:
            # Conjunctions such as `>=18.18 <19` are governed by their highest
            # lower-looking version only when it is not an explicit upper bound.
            lower = []
            for match in re.finditer(
                r"(?:\^|~|>=?|=)?\s*v?(\d+(?:\.\d+){0,2})", branch
            ):
                prefix = branch[max(0, match.start() - 2):match.start()].strip()
                token = match.group(0).lstrip()
                if token.startswith("<") or prefix.endswith("<"):
                    continue
                lower.append(_version_tuple(match.group(1)))
            alternatives.append(max(lower) if lower else _version_tuple(versions[0]))
    return min(alternatives) if alternatives else None


def _node_runtime_constraints(directory: str, package_data: dict) -> tuple[str, ...]:
    constraints: list[str] = []
    root_constraint = (package_data.get("engines") or {}).get("node")
    if isinstance(root_constraint, str) and root_constraint.strip():
        constraints.append(root_constraint.strip())

    direct_names = {
        str(name)
        for key in ("dependencies", "devDependencies", "optionalDependencies")
        for name in (package_data.get(key) or {})
    }
    lock_path = os.path.join(directory, "package-lock.json")
    try:
        lock_data = json.loads(read_text(lock_path))
    except Exception:
        lock_data = {}
    for path, item in (lock_data.get("packages") or {}).items():
        if not isinstance(item, dict):
            continue
        name = _package_name_from_lock_path(str(path), item)
        if name not in direct_names:
            continue
        constraint = (item.get("engines") or {}).get("node")
        if isinstance(constraint, str) and constraint.strip():
            constraints.append(constraint.strip())
    return tuple(dict.fromkeys(constraints))


def _resolved_node_version(directory: str, package_data: dict) -> tuple[str | None, str | None]:
    constraints = _node_runtime_constraints(directory, package_data)
    floors = [floor for value in constraints if (floor := _node_constraint_floor(value))]
    if not floors:
        return (constraints[0] if constraints else None), None
    selected = max(floors)
    return (", ".join(constraints), ".".join(str(item) for item in selected))


def _node_base_version(base_image: str) -> tuple[int, int, int] | None:
    match = re.search(r"(?:^|/)node:(\d+)(?:\.(\d+))?(?:\.(\d+))?", base_image)
    if not match:
        return None
    return tuple(int(item or 0) for item in match.groups())


def _node_runtime_setup(workspace: Workspace, target: ProviderTarget) -> tuple[str, ...]:
    version = workspace.resolved_runtime_version
    base_version = _node_base_version(target.base_image)
    if base_version is not None:
        # A major-only official tag tracks the current release in that line.  A
        # lower major cannot satisfy a newer direct-dependency engine constraint.
        major_only = re.search(r"(?:^|/)node:\d+(?:[-@]|$)", target.base_image)
        required = _version_tuple(version) if version else None
        if required is None or base_version >= required or (
            major_only and base_version[0] >= required[0]
        ):
            return ()
    if not version:
        return (
            "apt-get update && apt-get install -y --no-install-recommends nodejs npm ca-certificates",
        )
    archive = f"node-v{version}-linux-$node_arch.tar.xz"
    return (
        "apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils",
        "node_arch=$(uname -m); case \"$node_arch\" in "
        "x86_64|amd64) node_arch=x64 ;; aarch64|arm64) node_arch=arm64 ;; "
        "*) echo \"unsupported Node architecture: $node_arch\" >&2; exit 1 ;; esac; "
        f"curl -fsSLO https://nodejs.org/dist/v{version}/{archive} && "
        f"tar -xJf {archive} -C /usr/local --strip-components=1 && rm -f {archive}",
    )


def _package_name_from_lock_path(path: str, item: dict) -> str | None:
    declared = item.get("name")
    if isinstance(declared, str) and declared:
        return declared
    if "node_modules/" not in path:
        return None
    tail = path.rsplit("node_modules/", 1)[-1]
    parts = tail.split("/")
    if parts[0].startswith("@") and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] or None


def _external_package(specifier: str) -> str | None:
    if specifier.startswith((".", "/", "#")):
        return None
    if specifier.startswith("@"):
        parts = specifier.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else specifier
    return specifier.split("/", 1)[0]


def _lock_key_name_version(key: str) -> tuple[str, str | None]:
    key = key.strip().strip("\"'").lstrip("/")
    key = key.split("(", 1)[0]
    if key.startswith("@"):
        separator = key.rfind("@")
        if separator > key.find("/"):
            return key[:separator], key[separator + 1:] or None
        return key, None
    if "@" in key:
        name, version = key.rsplit("@", 1)
        return name, version or None
    return key, None


def _npm_lock_packages(lock: dict) -> tuple[PackageRecord, ...]:
    raw_packages: list[tuple[str, str, str | None, dict]] = []
    for path, item in sorted((lock.get("packages") or {}).items()):
        if not path or not isinstance(item, dict):
            continue
        name = _package_name_from_lock_path(path, item)
        if not name:
            continue
        version = str(item.get("version") or "") or None
        locator = f"{name}@{version or 'unresolved'}#{path}"
        raw_packages.append((name, locator, version, item))
    path_to_locator = {
        path: locator
        for (name, locator, _version, _item), path in zip(
            raw_packages,
            (
                path for path, item in sorted((lock.get("packages") or {}).items())
                if path and isinstance(item, dict)
                and _package_name_from_lock_path(path, item)
            ),
        )
    }

    def dependency_locator(package_path: str, dependency: str) -> str | None:
        base = package_path
        while True:
            candidate = (
                f"{base}/node_modules/{dependency}"
                if base else f"node_modules/{dependency}"
            )
            if candidate in path_to_locator:
                return path_to_locator[candidate]
            if "/node_modules/" not in base:
                break
            base = base.rsplit("/node_modules/", 1)[0]
        return path_to_locator.get(f"node_modules/{dependency}")

    raw_by_locator = {
        locator: (path, name, version, item)
        for path, item in sorted((lock.get("packages") or {}).items())
        if path and isinstance(item, dict)
        and (name := _package_name_from_lock_path(path, item))
        for version in (str(item.get("version") or "") or None,)
        for locator in (f"{name}@{version or 'unresolved'}#{path}",)
    }
    return tuple(PackageRecord(
        name=name,
        version=version,
        locator=locator,
        dependencies=tuple(
            target
            for dep in sorted((item.get("dependencies") or {}).keys())
            if (target := dependency_locator(path, dep)) is not None
        ),
    ) for locator, (path, name, version, item) in raw_by_locator.items())


def _pnpm_lock_packages(text: str) -> tuple[PackageRecord, ...]:
    if yaml is None:
        return ()
    try:
        lock = yaml.safe_load(text) or {}
    except Exception:
        return ()
    package_tables = []
    for key in ("packages", "snapshots"):
        table = lock.get(key)
        if isinstance(table, dict):
            package_tables.append(table)
    records: dict[str, tuple[str, str | None, dict]] = {}
    for table in package_tables:
        for key, item in table.items():
            name, version = _lock_key_name_version(str(key))
            if not name or name.startswith(("file:", "link:", ".")):
                continue
            locator = f"{name}@{version or 'unresolved'}#{key}"
            records.setdefault(locator, (name, version, item or {}))
    first_locator = {}
    for locator, (name, _version, _item) in records.items():
        first_locator.setdefault(name.lower(), locator)
    return tuple(PackageRecord(
        name=name,
        version=version,
        locator=locator,
        dependencies=tuple(
            first_locator[dep.lower()]
            for dep in sorted({
                **(item.get("dependencies") or {}),
                **(item.get("optionalDependencies") or {}),
            })
            if dep.lower() in first_locator
        ),
    ) for locator, (name, version, item) in sorted(records.items()))


def _yarn_lock_packages(text: str) -> tuple[PackageRecord, ...]:
    records: list[PackageRecord] = []
    blocks = re.split(r"(?m)(?=^[^\s#][^:\n]*:\s*$)", text)
    for block in blocks:
        lines = block.splitlines()
        if not lines or not lines[0].rstrip().endswith(":"):
            continue
        first_specifier = lines[0].rstrip()[:-1].split(",", 1)[0].strip()
        name, _constraint = _lock_key_name_version(first_specifier)
        version_match = re.search(r'(?m)^\s+version\s+["\']?([^"\'\s]+)', block)
        if not name or not version_match:
            continue
        version = version_match.group(1)
        records.append(PackageRecord(
            name=name,
            version=version,
            locator=f"{name}@{version}#yarn",
        ))
    return tuple(records)


def _packages_from_lock(manager: str, text: str) -> tuple[PackageRecord, ...]:
    if manager == "npm":
        try:
            return _npm_lock_packages(json.loads(text))
        except Exception:
            return ()
    if manager == "pnpm":
        return _pnpm_lock_packages(text)
    return _yarn_lock_packages(text)


def _json_with_comments(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        cleaned = re.sub(
            r"/\*.*?\*/|^\s*//.*$",
            "",
            text,
            flags=re.S | re.M,
        )
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}


def _module_specifiers_from_ast(
    text: str,
    *,
    grammar: str,
    source_file: str,
) -> tuple[tuple[str, str], ...]:
    tree = parse_syntax_tree(text, grammar)
    if tree is None:
        return ()
    source = text.encode("utf-8")
    found: list[tuple[str, str]] = []
    for node in walk_syntax_tree(tree.root_node):
        candidate = None
        confidence = "high"
        if node.type in {"import_statement", "export_statement"}:
            candidate = node.child_by_field_name("source")
            if candidate is None:
                continue
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            function_text = syntax_text(source, function) if function else ""
            if function_text not in {"require", "import"}:
                continue
            arguments = node.child_by_field_name("arguments")
            named = tuple(arguments.named_children) if arguments else ()
            candidate = named[0] if named else None
            if candidate is None or candidate.type not in {
                "string", "string_literal", "template_string",
            }:
                line = node.start_point[0] + 1
                found.append((f"<dynamic-import:{source_file}:{line}>", "low"))
                continue
        if candidate is None:
            continue
        value = quoted_string_value(syntax_text(source, candidate))
        if value:
            found.append((value, confidence))
    return tuple(found)


def _module_specifiers_fallback(text: str) -> tuple[tuple[str, str], ...]:
    pattern = re.compile(
        r"""(?:\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*|\bimport\s*)"""
        r"""["']([^"']+)["']"""
    )
    return tuple((match.group(1), "medium") for match in pattern.finditer(text))


def _is_local_ts_specifier(
    root_path: str,
    raw: str,
    *,
    path_aliases: tuple[str, ...],
    base_url: str | None,
) -> bool:
    if any(fnmatch(raw, alias) for alias in path_aliases):
        return True
    if not base_url or raw.startswith(("@", "#")):
        return False
    candidate = os.path.join(root_path, base_url, raw)
    variants = (
        candidate,
        *(candidate + suffix for suffix in (".ts", ".tsx", ".js", ".jsx")),
        *(os.path.join(candidate, "index" + suffix)
          for suffix in (".ts", ".tsx", ".js", ".jsx")),
    )
    return any(os.path.exists(path) for path in variants)


def _typescript_scanner_script() -> str:
    return f"""
const fs = require("fs");
const path = require("path");
let ts;
try {{ ts = require("typescript"); }} catch (_) {{ process.exit(0); }}
const suffixes = new Set([".js", ".jsx", ".ts", ".tsx"]);
const skip = new Set(["node_modules", "dist", "build", "target", ".git"]);
const files = [];
function walk(dir) {{
  for (const entry of fs.readdirSync(dir, {{withFileTypes: true}})) {{
    if (skip.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (suffixes.has(path.extname(entry.name))) files.push(full);
  }}
}}
walk(process.cwd());
const found = [];
for (const file of files) {{
  const text = fs.readFileSync(file, "utf8");
  const kind = file.endsWith(".tsx") ? ts.ScriptKind.TSX
    : file.endsWith(".jsx") ? ts.ScriptKind.JSX
    : file.endsWith(".ts") ? ts.ScriptKind.TS : ts.ScriptKind.JS;
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, kind);
  function visit(node) {{
    let value = null;
    let dynamic = false;
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
        && node.moduleSpecifier && ts.isStringLiteralLike(node.moduleSpecifier)) {{
      value = node.moduleSpecifier.text;
    }} else if (ts.isCallExpression(node)) {{
      const isRequire = ts.isIdentifier(node.expression)
        && node.expression.text === "require";
      const isImport = node.expression.kind === ts.SyntaxKind.ImportKeyword;
      if (isRequire || isImport) {{
        const arg = node.arguments[0];
        if (arg && ts.isStringLiteralLike(arg)) value = arg.text;
        else if (isImport) dynamic = true;
      }}
    }}
    if (value !== null || dynamic) {{
      const line = source.getLineAndCharacterOfPosition(node.getStart()).line + 1;
      found.push({{
        file: path.relative(process.cwd(), file).replaceAll(path.sep, "/"),
        specifier: value,
        dynamic,
        line
      }});
    }}
    ts.forEachChild(node, visit);
  }}
  visit(source);
}}
console.log("{_TYPESCRIPT_IMPORT_MARKER}");
console.log(JSON.stringify(found));
"""


def _native_typescript_imports(
    output: str,
    *,
    repo_path: str,
    workspace: Workspace,
) -> tuple[ImportRef, ...]:
    if _TYPESCRIPT_IMPORT_MARKER not in output:
        return ()
    payload = output.split(_TYPESCRIPT_IMPORT_MARKER, 1)[1].strip().splitlines()
    if not payload:
        return ()
    try:
        rows = json.loads(payload[0])
    except Exception:
        return ()
    root_path = workspace_path(repo_path, workspace)
    config = _json_with_comments(read_text(os.path.join(root_path, "tsconfig.json")))
    compiler = config.get("compilerOptions") or {}
    aliases = tuple((compiler.get("paths") or {}).keys())
    base_url = str(compiler.get("baseUrl") or "") or None
    refs = []
    seen = set()
    for row in rows:
        relative = str(row.get("file") or "")
        source_file = (
            relative if workspace.root == "."
            else f"{workspace.root.rstrip('/')}/{relative}"
        )
        if row.get("dynamic"):
            raw = f"<dynamic-import:{source_file}:{int(row.get('line') or 0)}>"
            package = None
            confidence = "low"
        else:
            raw = str(row.get("specifier") or "")
            package = _external_package(raw)
            confidence = "high"
            if not package or _is_local_ts_specifier(
                root_path,
                raw,
                path_aliases=aliases,
                base_url=base_url,
            ):
                continue
        key = (source_file, raw)
        if key in seen:
            continue
        seen.add(key)
        refs.append(ImportRef(
            language=workspace.language,
            workspace=workspace.root,
            raw_specifier=raw,
            source_file=source_file,
            scope=(
                "test" if re.search(
                    r"(?:^|/)(?:test|tests|__tests__)(?:/|$)",
                    source_file,
                ) else "runtime"
            ),
            confidence=confidence,
            package_name=package,
        ))
    return tuple(refs)


class NodeProvider:
    ecosystems = (Ecosystem.NPM,)

    def detect_workspaces(self, repo_path, language_requirements) -> tuple[Workspace, ...]:
        req = language_requirement(language_requirements, "typescript", "javascript")
        if req is None:
            return ()
        manifests = sorted(
            find_files(repo_path, {"package.json"}),
            key=lambda item: (item.count("/"), item),
        )
        selected_roots: list[tuple[str, tuple[str, ...]]] = []
        workspaces: list[Workspace] = []
        for manifest in manifests:
            root = os.path.dirname(manifest) or "."
            covered = False
            for parent, patterns in selected_roots:
                if parent == ".":
                    relative = root
                elif root.startswith(parent.rstrip("/") + "/"):
                    relative = root[len(parent.rstrip("/")) + 1:]
                else:
                    continue
                if any(fnmatch(relative, pattern.rstrip("/")) for pattern in patterns):
                    covered = True
                    break
            if covered:
                continue
            has_ts = os.path.isfile(os.path.join(repo_path, root, "tsconfig.json"))
            language = (
                "typescript"
                if has_ts or req.language == "typescript"
                else "javascript"
            )
            directory = os.path.join(repo_path, root)
            package_manager = "npm"
            if os.path.isfile(os.path.join(directory, "pnpm-lock.yaml")):
                package_manager = "pnpm"
            elif os.path.isfile(os.path.join(directory, "yarn.lock")):
                package_manager = "yarn"
            try:
                package_data = json.loads(read_text(os.path.join(repo_path, manifest)))
                declared_manager = str(package_data.get("packageManager") or "")
                manager_version = None
                if declared_manager:
                    package_manager, _, manager_version = declared_manager.partition("@")
            except Exception:
                package_data = {}
                manager_version = None
            raw_workspaces = package_data.get("workspaces") or ()
            if isinstance(raw_workspaces, dict):
                raw_workspaces = raw_workspaces.get("packages") or ()
            patterns = tuple(
                str(item) for item in raw_workspaces
                if isinstance(item, str)
            )
            selected_roots.append((root.replace(os.sep, "/"), patterns))
            version = (
                (package_data.get("engines") or {}).get("node")
                if isinstance(package_data, dict) else None
            )
            runtime_constraint, resolved_runtime = _resolved_node_version(
                directory, package_data
            )
            workspaces.append(Workspace(
                language=language,
                ecosystem=Ecosystem.NPM,
                root=root.replace(os.sep, "/"),
                manifest_file=manifest,
                package_manager=package_manager,
                version_constraint=str(version or req.version_constraint or "") or None,
                role=req.role,
                package_manager_version=manager_version or None,
                runtime_language="node",
                runtime_version_constraint=runtime_constraint or str(version or "") or None,
                resolved_runtime_version=resolved_runtime,
            ))
        return tuple(workspaces)

    def detect_runtime(self, workspace: Workspace) -> str | None:
        return workspace.version_constraint

    def resolve(self, repo_path, workspace, target, sandbox=None) -> ResolutionResult:
        root = workspace_path(repo_path, workspace)
        package_json_path = os.path.join(root, "package.json")
        try:
            data = json.loads(read_text(package_json_path))
        except Exception as exc:
            return ResolutionResult(status="failed", error=f"invalid package.json: {exc}")

        requirements: list[RequirementRecord] = []
        for scope, key in (
            ("runtime", "dependencies"),
            ("test", "devDependencies"),
            ("runtime", "optionalDependencies"),
            ("runtime", "peerDependencies"),
        ):
            for name, constraint in sorted((data.get(key) or {}).items()):
                requirements.append(RequirementRecord(
                    name=name,
                    constraint=str(constraint),
                    scope=scope,
                    source=workspace.manifest_file,
                ))

        manager = workspace.package_manager
        lock_name = {
            "npm": "package-lock.json",
            "pnpm": "pnpm-lock.yaml",
            "yarn": "yarn.lock",
        }.get(manager, "package-lock.json")
        lock_path = os.path.join(root, lock_name)
        lock_text = read_text(lock_path) if os.path.isfile(lock_path) else ""
        if lock_text and manager == "npm":
            try:
                json.loads(lock_text)
            except Exception as exc:
                return ResolutionResult(
                    requirements=tuple(requirements),
                    status="failed",
                    error=f"invalid package-lock.json: {exc}",
                    lock_file=os.path.relpath(lock_path, repo_path).replace(os.sep, "/"),
                    lock_digest=file_digest(lock_path),
                )
        static_packages = _packages_from_lock(manager, lock_text) if lock_text else ()
        prepare = _node_runtime_setup(workspace, target)
        if manager == "pnpm":
            prepare += ("command -v pnpm >/dev/null || npm install -g pnpm",)
            install = (
                "pnpm install --frozen-lockfile --ignore-scripts"
                if lock_text
                else "pnpm install --no-frozen-lockfile --ignore-scripts"
            )
        elif manager == "yarn":
            prepare += ("command -v yarn >/dev/null || npm install -g yarn",)
            install = (
                "yarn install --frozen-lockfile --ignore-scripts"
                if lock_text else "yarn install --ignore-scripts"
            )
        else:
            install = (
                "npm ci --ignore-scripts --no-audit --no-fund"
                if lock_text
                else "npm install --ignore-scripts "
                "--no-audit --no-fund"
            )
        scanner_script = base64.b64encode(
            _typescript_scanner_script().encode()
        ).decode()
        native_command = (
            f"printf %s {shlex.quote(scanner_script)} | base64 -d "
            "> /tmp/graph2env-typescript-scanner.js && "
            + in_workspace(
                workspace,
                f"{install} >/tmp/graph2env-node-resolver.log 2>&1 "
                f"&& printf '\\n{_NODE_LOCK_MARKER}\\n' && cat {lock_name} "
                "&& node /tmp/graph2env-typescript-scanner.js",
            )
        )
        native = run_native_resolver_probe(
            sandbox,
            ecosystem=workspace.ecosystem,
            workspace=workspace,
            prepare_commands=prepare,
            command=native_command,
        )
        native_packages = ()
        if native and native.status == "resolved" and _NODE_LOCK_MARKER in native.raw_output:
            generated_lock = (
                native.raw_output.split(_NODE_LOCK_MARKER, 1)[1]
                .partition(_TYPESCRIPT_IMPORT_MARKER)[0]
                .strip()
            )
            native_packages = _packages_from_lock(manager, generated_lock)
        scanner_imports = (
            _native_typescript_imports(
                native.raw_output,
                repo_path=repo_path,
                workspace=workspace,
            )
            if native and native.status == "resolved" else ()
        )
        packages = native_packages or static_packages
        if lock_text and not packages:
            return ResolutionResult(
                requirements=tuple(requirements),
                status="failed",
                error=f"unable to parse {lock_name}",
                lock_file=os.path.relpath(lock_path, repo_path).replace(os.sep, "/"),
                lock_digest=file_digest(lock_path),
                native_evidence=native,
            )
        name_to_locator = {}
        for package in packages:
            name_to_locator.setdefault(package.name.lower(), package.locator)
        requirements = tuple(replace(
            item,
            package_locator=name_to_locator.get(item.name.lower()),
        ) for item in requirements)
        return ResolutionResult(
            requirements=requirements,
            packages=tuple(packages),
            status="resolved" if packages else "unresolved",
            error=None if packages else f"no {lock_name}; manifest-driven install",
            lock_file=(
                os.path.relpath(lock_path, repo_path).replace(os.sep, "/")
                if lock_text else None
            ),
            lock_digest=file_digest(lock_path) if lock_text else None,
            resolution_source=(
                "resolver_sandbox" if native_packages
                else ("lockfile" if static_packages else "manifest")
            ),
            reproducibility_confidence=(
                "high" if lock_text and packages
                else ("medium" if native_packages else "low")
            ),
            native_evidence=native,
            scanner_imports=scanner_imports,
        )

    def scan_imports(self, repo_path, workspace, target) -> tuple[ImportRef, ...]:
        root_path = workspace_path(repo_path, workspace)
        aliases: set[str] = set()
        base_url = None
        tsconfig = os.path.join(root_path, "tsconfig.json")
        if os.path.isfile(tsconfig):
            config = _json_with_comments(read_text(tsconfig))
            compiler = config.get("compilerOptions") or {}
            aliases.update((compiler.get("paths") or {}).keys())
            raw_base_url = compiler.get("baseUrl")
            base_url = str(raw_base_url) if raw_base_url else None
        seen: set[tuple[str, str]] = set()
        refs: list[ImportRef] = []
        for rel in find_source_files(repo_path, workspace.root, (".js", ".jsx", ".ts", ".tsx")):
            text = read_text(os.path.join(repo_path, rel))
            extension = os.path.splitext(rel)[1].lower()
            grammar = {
                ".ts": "typescript",
                ".tsx": "tsx",
                ".js": "javascript",
                ".jsx": "javascript",
            }[extension]
            specifiers = _module_specifiers_from_ast(
                text,
                grammar=grammar,
                source_file=rel,
            )
            if not specifiers:
                specifiers = _module_specifiers_fallback(text)
            for raw, confidence in specifiers:
                if raw.startswith("<dynamic-import:"):
                    key = (rel, raw)
                    if key not in seen:
                        seen.add(key)
                        refs.append(ImportRef(
                            language=workspace.language,
                            workspace=workspace.root,
                            raw_specifier=raw,
                            source_file=rel,
                            scope="runtime",
                            confidence="low",
                        ))
                    continue
                package = _external_package(raw)
                if package is None:
                    continue
                if _is_local_ts_specifier(
                    root_path,
                    raw,
                    path_aliases=tuple(aliases),
                    base_url=base_url,
                ):
                    continue
                key = (rel, raw)
                if key in seen:
                    continue
                seen.add(key)
                scope = "test" if re.search(r"(?:^|/)(?:test|tests|__tests__)(?:/|$)", rel) else "runtime"
                refs.append(ImportRef(
                    language=workspace.language,
                    workspace=workspace.root,
                    raw_specifier=raw,
                    source_file=rel,
                    scope=scope,
                    confidence=confidence,
                    package_name=package,
                ))
        return tuple(refs)

    def dependency_commands(self, workspace, resolution):
        manager = workspace.package_manager
        if manager == "pnpm":
            install = "corepack enable && pnpm install --frozen-lockfile"
            check = "pnpm list --depth 0"
        elif manager == "yarn":
            install = "corepack enable && yarn install --immutable"
            check = "yarn list --depth=0"
        else:
            install = "npm ci" if resolution.lock_file else "npm install"
            check = "npm ls --depth=0"
        return (in_workspace(workspace, install),), in_workspace(workspace, check)

    def project_commands(self, workspace):
        manifest = os.path.join(workspace.root, "package.json")
        try:
            data = json.loads(read_text(manifest))
        except Exception:
            return (), None
        if "build" not in (data.get("scripts") or {}):
            return (), None
        command = in_workspace(workspace, f"{workspace.package_manager} run build")
        check = in_workspace(
            workspace,
            "test -d dist || test -d build || test -d .next || test -d out || test -d types",
        )
        return (command,), check

    def test_commands(self, workspace):
        if workspace.role == "build_tool":
            return ()
        try:
            data = json.loads(read_text(os.path.join(workspace.root, "package.json")))
        except Exception:
            return ()
        test_script = str((data.get("scripts") or {}).get("test") or "")
        if not test_script or "no test specified" in test_script.lower():
            return ()
        return (in_workspace(workspace, f"{workspace.package_manager} test"),)

    def build_fragment(self, repo_path, workspace, target, sandbox=None) -> DepGraphFragment:
        resolution = self.resolve(repo_path, workspace, target, sandbox=sandbox)
        imports = (
            resolution.scanner_imports
            or self.scan_imports(repo_path, workspace, target)
        )
        runtime_setup = _node_runtime_setup(workspace, target)
        runtime_check = "node --version"
        if workspace.resolved_runtime_version:
            version = workspace.resolved_runtime_version
            runtime_check = (
                "node -e \"const a=process.versions.node.split('.').map(Number),"
                f"b='{version}'.split('.').map(Number);"
                "process.exit(a[0]>b[0]||a[0]===b[0]&&(a[1]>b[1]||"
                "a[1]===b[1]&&a[2]>=b[2])?0:1)\""
            )
        manager = workspace.package_manager
        tool_setup = ()
        if manager in {"pnpm", "yarn"}:
            manager_version = workspace.package_manager_version or "latest"
            tool_setup = (
                "corepack enable && "
                f"corepack prepare {manager}@{manager_version} --activate",
            )
        dep_commands, dep_check = self.dependency_commands(workspace, resolution)
        try:
            package_data = json.loads(
                read_text(os.path.join(repo_path, workspace.manifest_file))
            )
        except Exception:
            package_data = {}
        scripts = package_data.get("scripts") or {}
        if "build" in scripts:
            project_commands = (
                in_workspace(workspace, f"{workspace.package_manager} run build"),
            )
            project_check = in_workspace(
                workspace,
                "test -d dist || test -d build || test -d .next || test -d out || test -d types",
            )
        else:
            project_commands, project_check = (), None
        if workspace.role != "build_tool" and scripts.get("test") and (
            "no test specified" not in str(scripts.get("test")).lower()
        ):
            tests = (
                in_workspace(workspace, f"{workspace.package_manager} test"),
            )
        else:
            tests = ()
        return graph_from_resolution(
            workspace=workspace,
            target=target,
            resolution=resolution,
            imports=imports,
            runtime_setup=runtime_setup,
            runtime_check=runtime_check,
            tool_name=manager,
            tool_setup=tool_setup,
            tool_check=f"{manager} --version",
            dependency_commands=dep_commands,
            dependency_check=dep_check,
            project_commands=project_commands,
            project_check=project_check,
            test_commands=tests,
        )
