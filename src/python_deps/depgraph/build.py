"""Stage orchestrator — repo path in, host-certified ``DepGraph`` out.

Wires the pipeline of
``docs/DESIGN-static-probe-certified-dependency-graph.md`` /
``docs/superpowers/specs/2026-06-23-uv-enriched-depgraph.md`` in order:

    1. scan      static import scan          -> Import + Test nodes   (cycle 1)
    2. map       roots.select_roots          -> resolver roots
    3. resolve   uv.lock closure (HOST)      -> Package nodes/edges   (cycle 2)
    3b. seed     predicted native nodes      -> Tool/SystemLib        (cycle 2)
    4. probe     install + import (CONTAINER)-> SystemLib/Tool nodes  (cycle 3)
    4.5 ldd      ldd ext .so (CONTAINER)     -> run-time SystemLib     (cycle 3)
    5. certify   host check_commands (CONTAINER) -> node ``state``    (cycle 4)

**Executor split (spec "Architecture change"):** resolution is HOST-side — ``uv``
cross-platform resolves the container target without a container interpreter — so
it runs through ``host_executor``.  Install/probe/certify must observe the real
target environment, so they run through ``container_executor``.  Both default-safe
for unit tests (a single ``FakeExecutor`` can be injected for both).

Discovery order and execution order differ (design 3.3 / 10.10): probing
discovers a SystemLib *after* installing the pip package that needs it, but
certification then runs in execution layer order (system before pip).  Every
stage returns a NEW immutable graph; this function only ever rebinds ``graph``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import replace

try:  # tomllib is stdlib on 3.11+; fall back to the tomli backport on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

from src.python_deps.depgraph.apt_verify import reconcile_apt_names
from src.python_deps.depgraph.certify import EXECUTION_LAYER_ORDER, certify_all
from src.python_deps.depgraph.executor import Executor, LocalSubprocessExecutor
from src.python_deps.depgraph.ids import TEST_NODE_ID, package_id, project_id, tool_id
from src.python_deps.depgraph.ldd_probe import ldd_probe
from src.python_deps.depgraph.pins import compute_exclude_newer
from src.python_deps.depgraph.probe import import_probe, install_closure
from src.python_deps.depgraph.relink import certified_import_links
from src.python_deps.depgraph.resolve import (
    link_imports_to_packages,
    resolve_closure,
)
from src.python_deps.depgraph.roots import ManifestRoot, select_manifest_roots, select_roots
from src.python_deps.depgraph.scan import scan_to_nodes
from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
)
from src.python_deps.depgraph.seed import seed_wheel_oracle_prior
from src.python_deps.depgraph.tables import TOOL_TO_APT, apt_for_tool
from src.python_deps.depgraph.config_scan import _is_dockerfile_name
from src.python_deps.depgraph.target_env import (
    detect_target_env,
    detect_target_stdlib_modules,
)
from src.python_deps.evidence import (
    collect_python_dependency_evidence,
    discover_test_project_roots,
)
from src.python_deps.import_mapping import normalize_package_name

logger = logging.getLogger(__name__)

# discovered_cycle stamps, one per discovery stage (design 5.2 example uses 3 for
# probe-discovered SystemLibs).
_SCAN_CYCLE = 1
_RESOLVER_CYCLE = 2
_PROBE_CYCLE = 3
_CERTIFY_CYCLE = 4

# ``DockerExecutor`` is an intentionally repository-free dependency probe.  It
# can certify installed packages/imports/native tools, but it cannot run the
# repository's Test obligation: there is no source tree or repository workdir
# in that container.  Full test execution belongs to the seeded live Sandbox's
# anti-hollow gate, which has the real checkout and later repeats from a fresh
# base before success.
_SCRATCH_CERTIFY_LAYER_ORDER: tuple[Layer, ...] = tuple(
    layer for layer in EXECUTION_LAYER_ORDER if layer is not Layer.TESTS
)


def _restamp(graph: DepGraph, node_ids: set[str], cycle: int) -> DepGraph:
    """Return a new graph with ``discovered_cycle = cycle`` on the named nodes."""
    new = graph
    for node_id in node_ids:
        node = new.get(node_id)
        if node is not None:
            new = new.with_node(replace(node, discovered_cycle=cycle))
    return new


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _project_name(repo_path: str) -> str:
    """Project name from ``[project].name`` in pyproject.toml, else dir basename."""
    pyproject = os.path.join(repo_path, "pyproject.toml")
    try:
        with open(pyproject, "rb") as fh:
            data = tomllib.load(fh)
        name = (data.get("project") or {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return os.path.basename(repo_path.rstrip("/\\")) or "project"


def _project_manifest(project_root: str) -> str:
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        path = os.path.join(project_root, name)
        if os.path.isfile(path):
            return path
    return project_root


def _editable_project_check(name: str, relative_path: str) -> str:
    expected = "/app" if relative_path == "." else f"/app/{relative_path}"
    # Distribution names declared by setup.py/setup.cfg are not always the
    # directory name.  Certify the PEP 610 editable source path instead of
    # guessing a distribution name.
    program = (
        "import importlib.metadata as m, json\n"
        "from pathlib import Path\n"
        "from urllib.parse import unquote, urlparse\n"
        f"expected = Path({expected!r}).resolve()\n"
        "for dist in m.distributions():\n"
        "    try:\n"
        "        direct = json.loads(dist.read_text('direct_url.json') or '{}')\n"
        "        editable = bool(direct.get('dir_info', {}).get('editable'))\n"
        "        actual = Path(unquote(urlparse(direct.get('url', '')).path)).resolve()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if editable and actual == expected:\n"
        "        raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    # Keep the shell command on one physical line.  The final setup artifact
    # carries checks in line-oriented ``#@check`` annotations; embedding the
    # raw multi-line program there would comment only its first line and let
    # the remaining Python source escape into executable shell.  ``exec`` of
    # the repr preserves the exact PEP 610 check while making the command safe
    # for both live execution and artifact rendering.
    payload = f"exec({program!r})"
    return f"python3 -c {shlex.quote(payload)}"


def _project_uses_vcs_version(project_root: str) -> bool:
    path = os.path.join(project_root, "pyproject.toml")
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = data.get("tool") or {}
    hatch_version = ((tool.get("hatch") or {}).get("version") or {})
    if isinstance(hatch_version, dict) and hatch_version.get("source") == "vcs":
        return True
    if isinstance(tool.get("setuptools_scm"), dict):
        return True
    build_requires = (data.get("build-system") or {}).get("requires") or []
    return any(
        isinstance(item, str)
        and re.split(r"[<>=!~\[]", item, maxsplit=1)[0].lower().replace("_", "-")
        in {"hatch-vcs", "setuptools-scm"}
        for item in build_requires
    )


def _add_project_node(graph: DepGraph, repo_path: str) -> DepGraph:
    """Add a Project hub node and connect declared direct deps to it.

    The repo under test is otherwise only reachable through the Test->Import
    chain, so its declared direct dependencies have no shared parent (e.g.
    ``certifi`` had no incoming Package->Package edge).  This node makes "what
    does the project directly require" a single explorable subtree:

    * ``Test --requires--> Project``
    * ``Project --requires--> <runtime declared dep Package>``  (kind=dependency)
    * ``Test --requires--> <test/optional declared dep Package>`` (kind=optional)

    Runtime vs test classification reuses ``evidence`` (kind ``dependency`` vs
    ``optional_dependency``); no new parsing.  Transitive deps still hang off
    their parents, and Import->Package reconciliation is unchanged.
    """
    canon_to_pkg = {
        _canon(n.name): n.id for n in graph.nodes if n.type is NodeType.PACKAGE
    }
    evidence = collect_python_dependency_evidence(repo_path)
    root = os.path.abspath(repo_path)
    discovered = [os.path.abspath(str(path)) for path in discover_test_project_roots(root)]
    # Preserve the historical structural hub for un-packaged repositories.
    project_roots = discovered or [root]
    nested_prefixes = {
        os.path.relpath(path, root).replace(os.sep, "/") + "/"
        for path in discovered if path != root
    }

    for project_root in project_roots:
        relative_path = os.path.relpath(project_root, root).replace(os.sep, "/")
        name = _project_name(project_root)
        proj_id = project_id(name)
        # ``discover_test_project_roots`` uses the shared content-aware
        # installability predicate.  A root that only has tool configuration
        # still gets the historical structural Project hub via the fallback
        # above, but must not acquire an impossible editable-install recipe.
        has_metadata = project_root in discovered
        chosen_fix = None
        check_command = None
        if has_metadata:
            target = "." if relative_path == "." else relative_path
            chosen_fix = (
                "python3 -m pip install --break-system-packages -e "
                + shlex.quote(target)
            )
            check_command = _editable_project_check(name, target)
        graph = graph.with_node(
            Node(
                id=proj_id,
                type=NodeType.PROJECT,
                name=name,
                layer=Layer.PIP,
                discovered_by=DiscoveredBy.STATIC_SCAN,
                state=State.UNKNOWN,
                check_command=check_command,
                chosen_fix=chosen_fix,
                provenance=_project_manifest(project_root),
                data={"project_path": relative_path},
            )
        )
        graph = graph.with_edge(
            Edge(
                src=TEST_NODE_ID,
                dst=proj_id,
                relation=EdgeType.REQUIRES,
                origin="project",
            )
        )

        if has_metadata and _project_uses_vcs_version(project_root):
            git_id = tool_id("git")
            if graph.get(git_id) is None:
                graph = graph.with_node(
                    Node(
                        id=git_id,
                        type=NodeType.TOOL,
                        name="git",
                        layer=Layer.TOOLCHAIN,
                        discovered_by=DiscoveredBy.STATIC_SCAN,
                        state=State.UNKNOWN,
                        check_command="command -v git",
                        evidence=f"{_project_manifest(project_root)}: VCS-derived version",
                        fix_candidates=("apt:git",),
                        chosen_fix="apt:git",
                        provenance=_project_manifest(project_root),
                    )
                )
            graph = graph.with_edge(
                Edge(
                    src=proj_id,
                    dst=git_id,
                    relation=EdgeType.REQUIRES,
                    origin="project-vcs",
                )
            )

        own_prefix = "" if relative_path == "." else relative_path + "/"
        for req in evidence.declared_dependencies:
            source = getattr(req, "source", "") or ""
            if own_prefix:
                if not source.startswith(own_prefix):
                    continue
            elif any(source.startswith(prefix) for prefix in nested_prefixes):
                continue
            if getattr(req, "kind", "dependency") == "constraint":
                continue
            pkg_id = canon_to_pkg.get(_canon(normalize_package_name(req.name)))
            if pkg_id is None:
                continue
            # runtime deps hang off their Project; test/optional deps off Test.
            src = (
                TEST_NODE_ID
                if getattr(req, "kind", "dependency") == "optional_dependency"
                else proj_id
            )
            graph = graph.with_edge(
                Edge(src=src, dst=pkg_id, relation=EdgeType.REQUIRES, origin="project")
            )
    return graph


def _dockerfile_install_tools(repo_path: str) -> dict[str, str]:
    """Return curated tool hints from repository Dockerfile apt installs.

    This is intentionally narrow: parse only Dockerfile ``RUN`` instructions that
    contain ``apt-get install``/``apt install`` and only accept tokens already in
    the curated tool-provider table (or their exact curated apt package).  It
    does not execute shell and does not infer arbitrary package names.
    """
    tools: dict[str, str] = {}
    apt_to_tool = {apt: tool for tool, apt in TOOL_TO_APT.items()}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in {".git", "__pycache__"}]
        for fname in filenames:
            if not _is_dockerfile_name(fname):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_path).replace(os.sep, "/")
            try:
                text = open(full, encoding="utf-8").read()
            except OSError:
                continue
            logical = re.sub(r"\\\s*\n", " ", text)
            for line in logical.splitlines():
                stripped = line.strip()
                if not stripped.upper().startswith("RUN "):
                    continue
                body = stripped[4:]
                if not re.search(r"\bapt(?:-get)?\s+install\b", body):
                    continue
                try:
                    toks = shlex.split(body, comments=True)
                except ValueError:
                    continue
                for raw in toks:
                    token = raw.strip()
                    if not token or token.startswith("-"):
                        continue
                    token = token.split("=", 1)[0]
                    if token in TOOL_TO_APT:
                        tools.setdefault(token, rel)
                    elif token in apt_to_tool:
                        tools.setdefault(apt_to_tool[token], rel)
    return tools


def _seed_dockerfile_hints(graph: DepGraph, repo_path: str) -> DepGraph:
    tools = _dockerfile_install_tools(repo_path)
    if not tools:
        return graph
    project_nodes = [n for n in graph.nodes if n.type is NodeType.PROJECT]
    new = graph
    for tool, rel in sorted(tools.items()):
        apt = apt_for_tool(tool)
        if not apt:
            continue
        tid = tool_id(tool)
        if new.get(tid) is None:
            new = new.with_node(
                Node(
                    id=tid,
                    type=NodeType.TOOL,
                    name=tool,
                    layer=Layer.TOOLCHAIN,
                    discovered_by=DiscoveredBy.STATIC_SCAN,
                    state=State.UNKNOWN,
                    check_command=(
                        "python -c \"import sysconfig, pathlib, sys; "
                        f"sys.exit(0 if pathlib.Path(sysconfig.get_paths()['include'], {tool!r}).exists() else 1)\""
                        if tool.endswith(".h")
                        else f"command -v {tool}"
                    ),
                    evidence=f"{rel}: apt install includes {tool}",
                    fix_candidates=(f"apt:{apt}",),
                    chosen_fix=f"apt:{apt}",
                    provenance=rel,
                )
            )
        for project in project_nodes:
            new = new.with_edge(
                Edge(
                    src=project.id,
                    dst=tid,
                    relation=EdgeType.REQUIRES,
                    origin="dockerfile",
                )
            )
    return new


def _seed_manifest_packages(
    graph: DepGraph, manifest_roots: list[ManifestRoot], *, target_env
) -> DepGraph:
    """Materialize declarations before resolution using stable package ids."""
    new = graph
    for root in manifest_roots:
        normalized = normalize_package_name(root.name)
        new = new.with_node(
            Node(
                id=package_id(normalized, None),
                type=NodeType.PACKAGE,
                name=normalized,
                layer=Layer.PIP,
                discovered_by=DiscoveredBy.STATIC_SCAN,
                state=State.UNKNOWN,
                provenance=root.source,
                discovered_cycle=_SCAN_CYCLE,
                resolved_python=target_env.python_version,
                resolved_platform=target_env.python_platform_tag,
                declared_specifier=root.specifier or None,
                declared_marker=root.marker or None,
                manifest_source=root.source or None,
                resolution_status="unresolved",
                data={"declared_extras": list(root.extras)},
            )
        )
    return new


_PYTEST_ROOT = "pytest"
_REQ_NAME_PREFIX_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _root_names(roots: list[tuple[str | None, str]]) -> set[str]:
    names: set[str] = set()
    for _import_id, token in roots:
        match = _REQ_NAME_PREFIX_RE.match(token)
        if match:
            names.add(normalize_package_name(match.group(1)))
    return names


def _ensure_pytest_resolver_root(
    roots: list[tuple[str | None, str]],
) -> list[tuple[str | None, str]]:
    """Ensure the fixed pytest test goal can run even without manifest test deps.

    The graph's test obligation is always ``python -m pytest -q``.  Some repos
    have tests or benchmark-provided tests but do not declare pytest in their
    own manifests, so a fresh environment can fail before collection with
    ``No module named pytest``.  Add only the test runner package, and only when
    it is not already present in manifest/scan roots.
    """
    if normalize_package_name(_PYTEST_ROOT) in _root_names(roots):
        return roots
    return [*roots, (None, _PYTEST_ROOT)]


def _link_test_runner_package(graph: DepGraph) -> DepGraph:
    pytest_pkg = package_id(_PYTEST_ROOT, None)
    if graph.get(pytest_pkg) is None:
        return graph
    return graph.with_edge(
        Edge(
            src=TEST_NODE_ID,
            dst=pytest_pkg,
            relation=EdgeType.REQUIRES,
            origin="test-runner",
        )
    )


def _merge_resolved_packages(
    graph: DepGraph,
    resolver_nodes: list[Node],
    resolver_edges: list[Edge],
) -> DepGraph:
    """Update stable package identities with resolver output and rewire edges."""
    id_map: dict[str, str] = {}
    new = graph

    for node in resolver_nodes:
        if node.type is not NodeType.PACKAGE:
            id_map[node.id] = node.id
            new = new.with_node(node)
            continue

        stable_id = package_id(normalize_package_name(node.name), None)
        id_map[node.id] = stable_id
        declared = new.get(stable_id)
        data = dict(declared.data) if declared is not None else {}
        data.update(dict(node.data))
        failed = node.state is State.MISSING
        merged = replace(
            node,
            id=stable_id,
            discovered_cycle=_RESOLVER_CYCLE,
            declared_specifier=(declared.declared_specifier if declared else None),
            declared_marker=(declared.declared_marker if declared else None),
            manifest_source=(declared.manifest_source if declared else None),
            resolution_status="failed" if failed else "resolved",
            resolution_error=node.evidence if failed else None,
            data=data,
        )
        new = new.with_node(merged)

    for edge in resolver_edges:
        src = id_map.get(edge.src, edge.src)
        dst = id_map.get(edge.dst, edge.dst)
        if src == dst:
            continue
        new = new.with_edge(replace(edge, src=src, dst=dst))

    # A resolver may fail without producing a structured diagnostic node. Keep
    # the declaration in the graph and make that failure explicit rather than
    # silently deleting the package obligation.
    for node in tuple(new.nodes):
        if node.type is NodeType.PACKAGE and node.resolution_status == "unresolved":
            new = new.with_node(
                replace(
                    node,
                    state=State.MISSING,
                    resolution_status="failed",
                    resolution_error="resolver returned no concrete package",
                )
            )
    return new


def _pad_python_full(target_python: str) -> str:
    """``"3.13"`` -> ``"3.13.0"`` (padding for a caller-supplied override).

    Mirrors the padding ``resolve_lock._target_env_for`` applies so an
    overridden ``target_python`` still produces a valid ``python_full_version``
    for marker evaluation (``python_full_version < '3.12'`` style forks).
    """
    parts = [p for p in target_python.split(".") if p]
    return ".".join((parts + ["0", "0"])[:3]) if parts else target_python


# Minor-version token in a ``Python 3.13.14`` banner.
_PY_VER_RE = re.compile(r"(\d+\.\d+)")
# Last-resort interpreter version when the container probe yields nothing.
_DEFAULT_TARGET_PYTHON = "3.11"


def _detect_target_python(
    container_executor: Executor, default: str = _DEFAULT_TARGET_PYTHON
) -> str:
    """Probe the container's interpreter minor version (e.g. ``"3.13"``).

    The resolve MUST target the python the container actually runs, or it pins
    versions that have no wheel for that interpreter (observed: a
    3.11-resolved ``pyarrow==2.0.0`` cannot build on a 3.13 container). Tries
    ``python3`` then ``python``, reading both streams (``--version``
    historically printed to stderr). Falls back to ``default`` when nothing
    parses, so a fake/empty executor preserves the legacy 3.11 target.

    Superseded in :func:`build_dep_graph` by :func:`target_env.detect_target_env`
    (Task 7, one combined probe covering python + platform); kept standalone
    (directly unit-tested) as it captures a slightly different signal (a
    ``--version`` banner rather than ``sys.version``) that some callers may
    still want in isolation.
    """
    for cmd in ("python3 --version", "python --version"):
        result = container_executor.run(cmd)
        if not result.ok:
            continue
        m = _PY_VER_RE.search((result.stdout or "") + " " + (result.stderr or ""))
        if m:
            return m.group(1)
    return default


def build_dep_graph(
    repo_path: str,
    container_executor: Executor,
    *,
    host_executor: Executor | None = None,
    target_python: str | None = None,
    target_platform: str | None = None,
    exclude_newer: str | None = None,
    needed_extras: frozenset[str] = frozenset(),
) -> DepGraph:
    """Build a host-certified dependency graph for ``repo_path``.

    ``container_executor`` runs install/probe/certify inside the target container;
    ``host_executor`` (default :class:`LocalSubprocessExecutor`) runs the
    host-side ``uv`` resolve.  A single :class:`TargetEnv` (Task 7) is detected
    from the container (``detect_target_env`` — one probe covering interpreter
    version, ``sys_platform``/``os_name``/``platform_machine``, and a glibc/musl
    guess for the ``uv lock --python-platform`` tag) so the resolve — and every
    PEP 508 marker it evaluates — targets the CONTAINER, never the host running
    this function.  ``target_python`` / ``target_platform`` remain accepted as
    caller overrides that patch the detected env (a hardcoded python would pin
    wheels for the wrong interpreter; an unset default would leak the dev host's
    own platform into the resolve).  The detected/patched ``TargetEnv`` OBJECT is
    passed straight into :func:`resolve_closure` (never decomposed into two
    strings first) so its RAW ``platform_machine`` — not a normalized wheel-tag
    stand-in — is what every marker evaluation downstream actually sees.  See
    the module docstring for the staged pipeline.  Returns the final immutable
    ``DepGraph``; certificates produced here are provisional (scratch-container
    scope) per design section 4.6.

    ``needed_extras`` (Task 8, targeted extras) is the set of
    ``[project.optional-dependencies]`` / ``extras_require`` group names this
    build actually needs (e.g. ``{"test"}`` when the goal is running the test
    suite). It is threaded, unchanged, into both :func:`select_roots` (which
    gates which optional groups become roots at all — fixing the prior
    "union every group" bug) and :func:`resolve_closure` (which records the
    chosen groups' scope in the resolver's temp pyproject). The default is
    deliberately runtime-only (``frozenset()``), NOT a union of every declared
    group. **Seam, not policy**: this function does not itself discover which
    extras a repo's CI/tox/Makefile actually invokes (e.g. `pip install -e
    .[test]`) — that discovery is separate future enrichment (cluster-1); a
    caller that already knows the needed groups passes them here.
    """
    host_executor = host_executor or LocalSubprocessExecutor()

    # Stage 1 — static import scan -> Import + Test nodes.
    graph = scan_to_nodes(repo_path)
    graph = _restamp(graph, {n.id for n in graph.nodes}, _SCAN_CYCLE)

    # Stage 1.5 — detect the TARGET container's env (Task 7) BEFORE root
    # selection (moved ahead of Stage 2, review fix: Stage 2's environment-
    # marker filter needs a real TargetEnv to evaluate against). ONE detected
    # TargetEnv replaces the previous two independent probes; explicit
    # target_python/target_platform (if given) patch the detected env rather
    # than skipping detection, so every other target-honest field (used by
    # marker evaluation in resolve_lock.py and roots.py) still reflects the
    # real container. The resulting `target_env` OBJECT (never decomposed into
    # separate strings) is what gets passed to select_roots below and
    # resolve_closure further down, so its RAW `platform_machine` (e.g. a
    # container reporting "arm64") reaches PEP 508 marker evaluation instead
    # of being lost to a normalized wheel-tag split.
    target_env = detect_target_env(container_executor)
    if target_python:
        target_env = replace(
            target_env,
            python_version=target_python,
            python_full=_pad_python_full(target_python),
        )
    if target_platform:
        if "/" in target_platform:
            # Docker uses ``linux/arm64`` while uv expects a normalized wheel
            # tag such as ``aarch64-manylinux_2_28``. The scratch-container
            # probe above already produced that tag for the requested Docker
            # platform, so only preserve the raw machine for PEP 508 markers.
            target_env = replace(
                target_env,
                platform_machine=(
                    target_platform.rsplit("/", 1)[-1]
                    or target_env.platform_machine
                ),
            )
        else:
            target_env = replace(
                target_env,
                platform_machine=(
                    target_platform.split("-", 1)[0]
                    or target_env.platform_machine
                ),
                python_platform_tag=target_platform,
            )
    target_python = target_env.python_version
    target_stdlib_modules = detect_target_stdlib_modules(container_executor)

    # Stage 2 — manifest-first, scan-gap-filled, filtered resolver roots.
    # needed_extras gates which optional-dependency groups become roots at all
    # (Task 8) -- logged here since it silently determines closure membership.
    # target_env (Task 8 review fix) additionally drops a manifest dep whose
    # PEP 508 environment marker evaluates False for the TARGET (e.g. `foo ;
    # sys_platform == 'win32'` on a Linux target); see
    # roots._env_marker_excludes for the conservative keep-unless-certain rule
    # (extra-gated markers are left untouched -- that's needed_extras' job).
    logger.info("build_dep_graph: needed_extras=%s", sorted(needed_extras))
    manifest_roots = select_manifest_roots(
        repo_path,
        needed_extras=needed_extras,
        target_env=target_env,
        target_stdlib_modules=target_stdlib_modules,
    )
    roots = select_roots(
        repo_path,
        graph,
        needed_extras=needed_extras,
        target_env=target_env,
        target_stdlib_modules=target_stdlib_modules,
    )
    roots = _ensure_pytest_resolver_root(roots)
    graph = _seed_manifest_packages(graph, manifest_roots, target_env=target_env)

    # Stage 2a — anchor the resolve cutoff to the project's pinned era (HOST,
    # PyPI). A pinned old root (opencv-python==4.9.0.80) otherwise lets uv pull an
    # ABI-incompatible latest transitive dep (numpy 2.x); resolving as-of the pin
    # era keeps the closure compatible. Unset/unpinned -> None -> resolve latest.
    if exclude_newer is None:
        exclude_newer = compute_exclude_newer(roots)

    # Runtime-tier obligation: the container must run the targeted python minor.
    # Certified later by a host check (rc 0 iff sys.version_info matches); discovery
    # here never implies SATISFIED.
    from src.python_deps.depgraph.ids import runtime_id as _runtime_id
    _maj, _min = target_python.split(".")[:2]
    _rt_check = f'python3 -c "import sys; sys.exit(0 if sys.version_info[:2]==({_maj},{_min}) else 1)"'
    graph = graph.with_node(
        Node(
            id=_runtime_id(target_python),
            type=NodeType.RUNTIME,
            name=f"python {target_python}",
            layer=Layer.RUNTIME,
            discovered_by=DiscoveredBy.STATIC_SCAN,
            state=State.UNKNOWN,
            version=target_python,
            check_command=_rt_check,
            resolved_python=target_python,
        )
    )
    pkg_nodes, pkg_edges = resolve_closure(
        roots,
        host_executor,
        target_env=target_env,
        exclude_newer=exclude_newer,
        extras=needed_extras,
    )
    pre_resolve_ids = {n.id for n in graph.nodes}
    graph = _merge_resolved_packages(graph, pkg_nodes, pkg_edges)
    graph = _link_test_runner_package(graph)

    # Stage 3a — reconcile: link EVERY Import to its resolved Package (covers
    # manifest-declared deps whose root carried import_id=None, which would
    # otherwise leave the scanned Import node orphaned from its Package).
    graph = link_imports_to_packages(graph)

    # Stage 3a' — Project hub: connect declared direct deps to a Project node so
    # the package layer is fully connected (runtime deps off Project, test deps
    # off the Test goal).
    graph = _add_project_node(graph, repo_path)
    graph = _seed_dockerfile_hints(graph, repo_path)
    # Python projects with native Rust extensions still need Cargo's declared
    # MSRV represented in the Python DepGraph.  A distro ``cargo`` package can
    # be older than Cargo.toml/rust-toolchain.toml (Qiskit requires 1.87 while
    # Debian currently supplies 1.85), so compile a versioned Rustup provider.
    from src.python_deps.depgraph.rust_toolchain import enrich_rust_toolchain
    graph = enrich_rust_toolchain(graph, repo_path)

    # Stage 3b — predicted native Tool/SystemLib nodes (resolver-origin).
    # PACKAGE_TO_SYSTEM_DEPS here is a PROACTIVE FALLBACK (pre-install / install-fail
    # hint); Stage 4.5 ldd_probe is the authoritative run-time native-lib source.
    graph = seed_wheel_oracle_prior(graph)
    resolver_ids = {n.id for n in graph.nodes} - pre_resolve_ids
    graph = _restamp(graph, resolver_ids, _RESOLVER_CYCLE)

    # Stage 4 — CONTAINER probe: install once (build-time gaps -> Tool) then
    # import-probe (run-time gaps -> SystemLib); predictions reconcile in place.
    pre_probe_ids = {n.id for n in graph.nodes}
    graph = install_closure(graph, container_executor)
    # Stage 4.5 — AUTHORITATIVE run-time native-lib discovery: ldd each installed
    # package's extension .so files and surface ``=> not found`` sonames as
    # SystemLib nodes (DT_NEEDED ground truth). Runs after install (needs the
    # built .so) and before relink/import-probe. The curated table (Stage 3b) is
    # demoted to a proactive fallback; ldd is the source of truth here.
    graph = ldd_probe(graph, container_executor)
    # Stage 4a — certified Import->Package relink (packages_distributions, CONTAINER).
    graph = certified_import_links(graph, container_executor)
    # import_probe is now the dlopen BACKSTOP only: DT_NEEDED gaps are covered by
    # Stage 4.5 (ldd_probe); this catches libs loaded at run time via dlopen that
    # never appear in the binary's NEEDED list.
    graph = import_probe(graph, container_executor)
    probe_ids = {n.id for n in graph.nodes} - pre_probe_ids
    graph = _restamp(graph, probe_ids, _PROBE_CYCLE)

    # Stage 4b — release-aware apt-name reconciliation against the TARGET image:
    # remap stale predicted/table names (e.g. libglib2.0-0 -> libglib2.0-0t64)
    # so the fix-candidate is correct for the actual base image.
    graph = reconcile_apt_names(graph, container_executor)

    # Stage 5 — dependency certification in the repository-free scratch
    # container.  Test is deliberately deferred to the live anti-hollow gate;
    # running pytest here would scan ``/`` rather than the checkout and record
    # a non-evidentiary failure (or spend the whole command timeout doing so).
    graph = certify_all(
        graph,
        container_executor,
        cycle=_CERTIFY_CYCLE,
        layer_order=_SCRATCH_CERTIFY_LAYER_ORDER,
    )

    # Stage 6 — repository-native materialization.  The scratch graph above is
    # still used for discovery and checks, while the live/fresh replay starts
    # with the highest-fidelity lockfile/manifest transaction.  Individual
    # package repair is deferred until that transaction has run and host checks
    # identify the genuinely missing leaves.
    from src.python_deps.depgraph.manifest_provider import seed_manifest_provider
    graph = seed_manifest_provider(
        graph,
        repo_path,
        needed_groups=needed_extras,
    )
    from src.python_deps.depgraph.test_config_provider import enrich_test_environment
    graph = enrich_test_environment(graph, repo_path)

    return graph
