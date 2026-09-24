"""Repository-native Python dependency provider for Full-v3.

The dependency graph remains the source of obligations and host certification,
but a repository lockfile/manifest is the highest-fidelity way to materialize
the initial environment.  This module seeds one deterministic DependencySet
transaction and marks its package/project leaves as managed by that transaction.
Only leaves still missing after the transaction become individual repair work.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path
import shlex

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 deployment
    import tomli as tomllib

from src.python_deps.depgraph.ids import TEST_NODE_ID, dependency_set_id
from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Ecosystem,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
    Strength,
)


def _read_pyproject(root: Path) -> dict:
    try:
        with (root / "pyproject.toml").open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_UV_VERSION_RE = re.compile(
    r"^\s*version\s*:\s*['\"]?([0-9]+\.[0-9]+(?:\.[0-9]+)?)['\"]?\s*(?:#.*)?$"
)
_UV_SYNC_RE = re.compile(r"^\s*run\s*:\s*(uv\s+sync(?:\s+[^#]+?)?)\s*(?:#.*)?$")


def _workflow_paths(root: Path) -> tuple[Path, ...]:
    """Return repository workflows with test/CI workflows first."""
    workflow_root = root / ".github" / "workflows"
    try:
        paths = [
            path for path in workflow_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
        ]
    except OSError:
        return ()
    return tuple(sorted(
        paths,
        key=lambda path: (
            0 if any(token in path.name.lower() for token in ("ci", "test")) else 1,
            path.name,
        ),
    ))


def _workflow_uv_version(root: Path) -> str | None:
    """Read an exact uv version from repository-owned setup-uv CI steps.

    Test/CI workflows take precedence over publishing workflows because this
    provider is constructing the test environment.  Only an exact numeric
    version is accepted; action revisions and unrelated YAML versions are
    ignored.
    """
    for path in _workflow_paths(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            if "astral-sh/setup-uv@" not in line:
                continue
            for candidate in lines[index + 1:index + 10]:
                match = _UV_VERSION_RE.match(candidate)
                if match:
                    return match.group(1)
                if candidate.strip().startswith("- ") or "uses:" in candidate:
                    break
    return None


def _workflow_uv_sync_command(root: Path) -> str | None:
    """Read a directly executable, one-line ``uv sync`` command from CI.

    The repository's test workflow is stronger evidence than host-invented
    flags.  Shell interpolation and compound commands are intentionally
    ignored because GitHub-only environment expressions are not portable to
    setup.sh.
    """
    for path in _workflow_paths(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            match = _UV_SYNC_RE.match(line)
            if not match:
                continue
            command = match.group(1).strip()
            if any(token in command for token in ("$", "`", "&&", "||", ";", "|")):
                continue
            try:
                words = shlex.split(command)
            except ValueError:
                continue
            if words[:2] == ["uv", "sync"]:
                return " ".join(shlex.quote(word) for word in words)
    return None


def _uv_install_spec(root: Path, pyproject: dict) -> str:
    workflow_version = _workflow_uv_version(root)
    if workflow_version:
        return f"uv=={workflow_version}"
    required = str((((pyproject.get("tool") or {}).get("uv") or {}).get("required-version") or "")).strip()
    exact = re.fullmatch(r"(?:==)?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", required)
    return f"uv=={exact.group(1)}" if exact else "uv"


def _selected_groups(data: dict, needed_groups: frozenset[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    project_extras = set(((data.get("project") or {}).get("optional-dependencies") or {}))
    dependency_groups = set(data.get("dependency-groups") or {})
    poetry_groups = set((((data.get("tool") or {}).get("poetry") or {}).get("group") or {}))
    requested = set(needed_groups) | ({"test"} if "test" in project_extras | dependency_groups | poetry_groups else set())
    extras = tuple(sorted(requested & project_extras))
    groups = tuple(sorted(requested & (dependency_groups | poetry_groups)))
    return extras, groups


def _venv_pth_command() -> str:
    """Keep the project root importable from the venv that ``uv`` created.

    Do not ask the currently selected ``python3`` for ``site.getsitepackages``:
    during Docker replay PATH can resolve either the system interpreter or a
    partially-created project venv.  That previously produced a hard-coded
    `/testbed/.venv/...` target that did not exist in clean images.  The venv's
    discovered site-packages directory is the only stable target.
    """
    code = (
        "import pathlib; "
        "roots=sorted(pathlib.Path('.venv').glob('lib/python*/site-packages')); "
        "assert roots, 'uv environment has no site-packages'; "
        "target=roots[-1]/'rat-project-venv.pth'; "
        "target.parent.mkdir(parents=True, exist_ok=True); "
        "target.write_text(str(pathlib.Path.cwd().resolve())+'\\n')"
    )
    return "python3 -c " + shlex.quote(code)


def _requirement_lines(root: Path, relative_path: str) -> tuple[str, ...]:
    """Return installable top-level requirement lines for a bounded fallback.

    A bulk pip transaction is preferable, but it is all-or-nothing when one
    declared/private distribution cannot be resolved.  These lines let the
    manifest provider salvage the other top-level requirements without asking
    the LLM to repair an entire resolved closure one package at a time.
    """
    path = root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    requirements: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("-r", "--requirement")):
            continue
        # Preserve URL fragments while removing ordinary trailing comments.
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        if line and not line.startswith("-"):
            requirements.append(line)
    return tuple(requirements)


def _requirements_command(root: Path, relative_path: str) -> str:
    pip = "python3 -m pip install --break-system-packages"
    bulk = f"{pip} -r {shlex.quote(relative_path)}"
    requirements = _requirement_lines(root, relative_path)
    if not requirements:
        return bulk
    salvage = "; ".join(
        f"{pip} {shlex.quote(requirement)} || true"
        for requirement in requirements
    )
    return f"{bulk} || ( {salvage} )"


def _requirement_files(
    root: Path, needed_groups: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    files: list[str] = []
    names = ["requirements.txt"]
    if needed_groups & {"test", "tests", "testing", "ci"}:
        names += [
            "requirements-test.txt", "test-requirements.txt",
            "requirements/test.txt",
        ]
    if needed_groups & {"dev", "development"}:
        names += [
            "requirements-dev.txt", "requirements_dev.txt",
            "requirements/dev.txt",
        ]
    for name in names:
        if (root / name).is_file() and name not in files:
            files.append(name)
    return tuple(files)


def _provider_commands(root: Path, needed_groups: frozenset[str]) -> tuple[str, tuple[str, ...], bool] | None:
    data = _read_pyproject(root)
    extras, groups = _selected_groups(data, needed_groups)

    if (root / "uv.lock").is_file():
        sync_command = _workflow_uv_sync_command(root)
        if sync_command is None:
            args = ["uv", "sync", "--locked"]
            for extra in extras:
                args += ["--extra", extra]
            for group in groups:
                args += ["--group", group]
            sync_command = " ".join(shlex.quote(item) for item in args)
        uv_spec = _uv_install_spec(root, data)
        commands = (
            "python3 -m pip install --break-system-packages " + shlex.quote(uv_spec),
            sync_command,
            _venv_pth_command(),
        )
        # Some repositories keep runtime/test dependencies in requirements
        # files while their uv.lock only covers pyproject dependency groups.
        # Treat those files as authoritative supplements instead of silently
        # omitting packages absent from the lock (for example fastmcp).
        commands += tuple(
            _requirements_command(root, name)
            for name in _requirement_files(root, needed_groups)
        )
        return "uv", commands, True

    if (root / "poetry.lock").is_file():
        install = "POETRY_VIRTUALENVS_CREATE=false poetry install"
        if groups:
            install += " --with " + shlex.quote(",".join(groups))
        return "poetry", (
            "python3 -m pip install --break-system-packages poetry",
            install,
        ), True

    requirement_files = _requirement_files(root, needed_groups)
    if requirement_files:
        return "requirements", tuple(
            _requirements_command(root, name)
            for name in requirement_files
        ), False

    if (root / "pyproject.toml").is_file() or (root / "setup.py").is_file() or (root / "setup.cfg").is_file():
        target = ".[" + ",".join(extras) + "]" if extras else "."
        return "editable", (
            "python3 -m pip install --break-system-packages -e " + shlex.quote(target),
        ), True
    return None


def _seed_manifest_provider_single(
    graph: DepGraph,
    repo_path: str | os.PathLike[str],
    *,
    needed_groups: frozenset[str] = frozenset(),
    workspace: str = ".",
    managed_node_ids: frozenset[str] | None = None,
) -> DepGraph:
    """Add one native DependencySet and bind resolved Python leaves to it."""
    root = Path(repo_path)
    detected = _provider_commands(root, needed_groups)
    if detected is None:
        return graph
    manager, commands, manages_project = detected
    if workspace != ".":
        prefix = f"cd {shlex.quote(workspace)} && "
        commands = tuple(prefix + command for command in commands)
    node_id = dependency_set_id(Ecosystem.PYPI, workspace)
    digest = hashlib.sha256((workspace + "\n" + "\n".join(commands)).encode()).hexdigest()[:12]
    marker = f"/tmp/rat-manifest-provider-{digest}.ok"
    commands = commands + (
        f"printf '%s\\n' {shlex.quote(digest)} > {shlex.quote(marker)}",
    )
    provider = Node(
        id=node_id,
        type=NodeType.DEPENDENCY_SET,
        name=f"{manager} project dependencies ({workspace})",
        layer=Layer.DEPENDENCIES,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=State.MISSING,
        check_command=f"test -f {shlex.quote(marker)}",
        evidence=f"repository-native provider selected from {workspace}/{manager} manifest/lockfile",
        chosen_fix=f"manifest:{manager}",
        ecosystem=Ecosystem.PYPI,
        workspace=workspace,
        package_manager=manager,
        setup_commands=commands,
        strength=Strength.HARD,
        data={
            "provider_backed": True,
            "provider_kind": "manifest",
            "manifest_manager": manager,
            "marker": marker,
        },
    )
    new = graph.with_node(provider)

    for node in tuple(new.nodes):
        should_manage = node.type is NodeType.PACKAGE and node.ecosystem in (None, Ecosystem.PYPI)
        should_manage = should_manage or (manages_project and node.type is NodeType.PROJECT)
        if not should_manage or (managed_node_ids is not None and node.id not in managed_node_ids):
            continue
        data = {**dict(node.data), "managed_by_dependency_set": node_id}
        # A native manager already installs the local project.  Keep a
        # read-only Project certification block in the plan so cross-language
        # ordering remains visible (Rust/maturin -> manifest -> Python project)
        # without repeating `pip install -e .`.
        managed_commands: tuple[str, ...] = ()
        managed_check = marker if node.type is NodeType.PROJECT else node.check_command
        if node.type is NodeType.PROJECT:
            # Tests execute from the checked-out source tree and Import nodes
            # separately certify importability.  Once the native transaction
            # has completed, do not require PEP 660 editable metadata: flat
            # layout/script repositories may be runnable but intentionally not
            # editable-installable.
            managed_check = f"test -f {shlex.quote(marker)}"
            # Keep a no-op certification block in the compiled plan so
            # cross-language ordering (for example Rust extension -> Python
            # project) remains explicit after the native transaction.
            managed_commands = (managed_check,)
        new = new.with_node(replace(
            node,
            check_command=managed_check,
            setup_commands=managed_commands,
            data=data,
        ))
        if node.type is NodeType.PACKAGE:
            new = new.with_edge(Edge(
                src=node_id,
                dst=node.id,
                relation=EdgeType.DESCRIBES,
                origin="manifest-provider",
                data={"hard": False},
            ))
        elif node.type is NodeType.PROJECT:
            new = new.with_edge(Edge(
                src=node.id,
                dst=node_id,
                relation=EdgeType.REQUIRES,
                origin="manifest-provider",
            ))
            # Preserve build-tool/cross-language prerequisites on the actual
            # transaction which installs the Python project.  Package leaves
            # are deliberately excluded because the transaction provides them.
            for requirement in graph.requires_of(node.id):
                if requirement.type in {NodeType.PACKAGE, NodeType.DEPENDENCY_SET}:
                    continue
                new = new.with_edge(Edge(
                    src=node_id,
                    dst=requirement.id,
                    relation=EdgeType.REQUIRES,
                    origin="manifest-provider",
                ))

    if new.get(TEST_NODE_ID) is not None:
        new = new.with_edge(Edge(
            src=TEST_NODE_ID,
            dst=node_id,
            relation=EdgeType.REQUIRES,
            origin="manifest-provider",
        ))
    return new


def seed_manifest_provider(
    graph: DepGraph,
    repo_path: str | os.PathLike[str],
    *,
    needed_groups: frozenset[str] = frozenset(),
) -> DepGraph:
    """Seed one native transaction per test-bearing Python workspace."""
    from src.python_deps.evidence import (
        discover_test_project_roots,
        discover_test_requirement_files,
    )

    root = Path(repo_path)
    roots = set(discover_test_project_roots(root))
    roots.update(path.parent for path in discover_test_requirement_files(root))
    roots.add(root)
    available = [
        path for path in sorted(roots, key=lambda item: (len(item.parts), str(item)))
        if _provider_commands(path, needed_groups) is not None
    ]
    if not available:
        return graph

    relatives = {path: (path.relative_to(root).as_posix() or ".") for path in available}
    nested = sorted((rel for rel in relatives.values() if rel != "."), key=len, reverse=True)
    assignments: dict[str, set[str]] = {relative: set() for relative in relatives.values()}
    for node in graph.nodes:
        if node.type is NodeType.PROJECT:
            project_path = str(node.data.get("project_path") or ".").strip("/") or "."
            if project_path in assignments:
                assignments[project_path].add(node.id)
            continue
        if node.type is not NodeType.PACKAGE or node.ecosystem not in (None, Ecosystem.PYPI):
            continue
        source = str(node.manifest_source or node.provenance or "").replace("\\", "/").lstrip("./")
        assigned = next(
            (relative for relative in nested if source == relative or source.startswith(relative + "/")),
            "." if "." in assignments else None,
        )
        if assigned is not None:
            assignments[assigned].add(node.id)

    updated = graph
    for path in available:
        relative = relatives[path]
        updated = _seed_manifest_provider_single(
            updated, path, needed_groups=needed_groups, workspace=relative,
            managed_node_ids=frozenset(assignments[relative]),
        )
    return updated
