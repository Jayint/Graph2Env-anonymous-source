"""Pure uv.lock parsers: parse_uv_lock, native_risk_from_lock, and supporting helpers.

These functions are side-effect-free (no network, no subprocess, no executor) and
are unit-tested in isolation.  The orchestration layer (resolve_closure) lives in
resolve.py, which imports from here.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

try:  # tomllib is stdlib on 3.11+; fall back to the tomli backport on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

from src.python_deps.depgraph.ids import package_id
from src.python_deps.depgraph.schema import (
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
)
from src.python_deps.depgraph.target_env import TargetEnv

# Locked decision 1: the 'uv' binary, invoked (never imported) via the Executor.
def _find_uv_bin() -> str:
    """Prefer uv installed beside the active virtualenv Python.

    Workers may be launched with an absolute virtualenv Python while the shell
    itself is not activated. In that case PATH does not contain the matching uv.
    """
    configured = os.environ.get("DEPGRAPH_UV_BIN")
    if configured:
        return configured
    sibling = os.path.join(os.path.dirname(sys.executable), "uv")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return shutil.which("uv") or "uv"


UV_BIN = _find_uv_bin()

# Default container target when the caller does not detect/inject one. NEVER
# manylinux2014 (silently downgrades e.g. numpy) — use the modern 2_28 baseline.
DEFAULT_TARGET_PLATFORM = "x86_64-manylinux_2_28"

# A plausible PEP 508 requirement — a distribution name with optional extras and
# an optional version specifier (e.g. ``urllib3<1.21`` / ``numpy>=2,<3``).  The
# allowed alphabet excludes quotes, spaces, newlines, slashes and shell
# metacharacters so nothing untrusted can break out of a TOML string or a heredoc
# body.  Version constraints are kept so the resolver can detect conflicts.
_DIST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQUIREMENT_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"  # distribution name
    r"(?:\[[A-Za-z0-9._,-]+\])?"  # optional extras
    r"[<>=!~,.*+A-Za-z0-9_!-]*$"  # optional version specifier(s)
)
# The bare distribution name at the head of a requirement token (drops the
# extras/specifier) — used for case/separator-insensitive node matching.
_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
# A simple ``X.Y`` / ``X.Y.Z`` python version, for the resolve target.
_PY_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")

# uv.lock source kinds that denote the *local* project (the synthetic resolve
# root or a path/editable dependency) rather than an installable distribution.
_LOCAL_SOURCE_KEYS = frozenset({"virtual", "editable", "directory"})


def _safe_dist_names(dist_names: list[str]) -> list[str]:
    """Keep only injection-safe requirement tokens (name + optional specifier)."""
    return [d for d in dist_names if _REQUIREMENT_RE.match(d)]


def _req_name(token: str) -> str:
    """Bare distribution name of a requirement token (``urllib3<1.21`` -> ``urllib3``)."""
    match = _REQ_NAME_RE.match(token.strip())
    return match.group(1) if match else token.strip()


def _validate_target_python(target_python: str) -> str:
    """Return ``target_python`` if it is a plain version, else raise."""
    if not _PY_VERSION_RE.match(target_python):
        raise ValueError(f"invalid target python version: {target_python!r}")
    return target_python


def _canon(name: str) -> str:
    """PEP 503-style normalization for case/separator-insensitive matching."""
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------- #
# Forked-version resolution: pick the ONE package version applicable to the
# target python when a uv.lock forks a dependency across python markers.
# --------------------------------------------------------------------------- #
def _marker_env(target: TargetEnv) -> dict[str, str]:
    """Marker-evaluation environment for ``target`` — ALL PEP 508 fields.

    Delegates to :meth:`TargetEnv.marker_env`. Passing every field
    ``packaging.markers`` may reference is what keeps ``Marker.evaluate()``
    from falling back to its HOST-derived ``default_environment()`` for
    ``sys_platform`` / ``platform_machine`` / ``os_name`` — the leak that let a
    non-x86_64-linux dev host silently mis-evaluate platform-gated deps.
    """
    return target.marker_env()


def _target_env_for(
    target_python: str, target_platform: str | None = None
) -> TargetEnv:
    """Build a :class:`TargetEnv` from BARE ``target_python`` / ``target_platform``
    strings, for callers with no real :class:`TargetEnv` to pass (this module's
    pure parsers are also unit-tested directly, string-only, in isolation).

    ``python_version`` keeps two components (``3.11``); ``python_full_version``
    is padded to three (``3.11.0``) so ``python_full_version < '3.12'`` style
    fork markers evaluate correctly. The container this codebase resolves for
    is always linux (see :data:`DEFAULT_TARGET_PLATFORM`), so ``sys_platform`` /
    ``os_name`` / ``platform_system`` are fixed; ``platform_machine`` is taken
    from ``target_platform``'s arch token (default ``x86_64``).

    CAUTION: ``target_platform`` here is the NORMALIZED wheel/uv tag (e.g.
    ``"aarch64-manylinux_2_28"``), so the ``platform_machine`` this recovers is
    ALWAYS the canonical arch — it can never reproduce a raw, non-canonical
    ``platform.machine()`` value (e.g. ``"arm64"``) a real container might
    report. ``resolve_closure`` therefore never uses this reconstruction: it
    always threads the actual detected/overridden :class:`TargetEnv` (see
    :func:`_resolved_target_env`) so a marker like ``platform_machine ==
    'arm64'`` is evaluated against the container's own RAW fact, not a
    normalized stand-in.
    """
    parts = [p for p in target_python.split(".") if p]
    version = ".".join(parts[:2]) if len(parts) >= 2 else target_python
    full = ".".join((parts + ["0", "0"])[:3]) if parts else target_python
    platform = target_platform or DEFAULT_TARGET_PLATFORM
    machine = platform.split("-", 1)[0] if platform else "x86_64"
    return TargetEnv(
        python_full=full,
        python_version=version,
        platform_machine=machine,
        sys_platform="linux",
        os_name="posix",
        platform_system="Linux",
        python_platform_tag=platform,
    )


def _resolved_target_env(
    target_python: str,
    target_platform: str | None,
    target_env: TargetEnv | None,
) -> TargetEnv:
    """The :class:`TargetEnv` to marker-evaluate a fork/prune decision against.

    Prefers an explicitly supplied ``target_env`` — the actual detected (or
    caller-overridden) container facts ``resolve_closure`` always threads down
    now, carrying the RAW ``platform_machine`` reported by the container.
    Falls back to reconstructing one from the legacy ``target_python`` /
    ``target_platform`` strings ONLY when no real ``target_env`` is available
    (:func:`parse_uv_lock` / :func:`native_risk_from_lock` are also exercised
    directly, string-only, by unit tests) — that reconstruction can only ever
    recover the NORMALIZED arch token (see :func:`_target_env_for`'s caution),
    so it must never be reached once a real ``target_env`` exists.
    """
    if target_env is not None:
        return target_env
    return _target_env_for(target_python, target_platform)


def _marker_applies(marker: str, env: dict[str, str]) -> bool | None:
    """Evaluate a resolution ``marker`` for ``env``; ``None`` if not evaluable.

    ``packaging`` is a near-universal transitive dependency, but its absence (or
    a malformed marker) must not crash the resolver — the caller treats ``None``
    as "unknown" and falls back to a deterministic version pick.
    """
    try:
        from packaging.markers import Marker
    except ImportError:
        return None
    try:
        return bool(Marker(marker).evaluate(env))
    except Exception:
        return None


def _version_key(version: str | None):
    """Sort key for picking the highest version (packaging-aware, str fallback)."""
    if not version:
        return (0,)
    try:
        from packaging.version import Version

        return (1, Version(version))
    except Exception:
        return (0, version)


def _entry_applies(pkg: dict, env: dict[str, str]) -> bool | None:
    """Whether a forked ``[[package]]`` applies under ``env``.

    A package with no ``resolution-markers`` always applies.  Otherwise it
    applies when ANY of its markers evaluates true; ``None`` (unknown) is
    returned only when no marker could be evaluated at all.
    """
    markers = pkg.get("resolution-markers") or []
    if not markers:
        return True
    saw_unknown = False
    for marker in markers:
        verdict = _marker_applies(marker, env)
        if verdict is True:
            return True
        if verdict is None:
            saw_unknown = True
    return None if saw_unknown else False


def _select_applicable_packages(
    raw_packages: list[dict],
    target_python: str | None,
    target_platform: str | None = None,
    target_env: TargetEnv | None = None,
) -> list[dict]:
    """Drop fork duplicates: keep one version per name for the TARGET.

    When a name appears once, it is kept as-is.  When it forks into multiple
    versions, the version whose ``resolution-markers`` match the target
    (python version AND platform — e.g. ``platform_machine == 'aarch64'``) is
    kept; ties / unknown markers fall back to the highest version so two
    versions of one distribution are never emitted (which would break ``pip
    install``).  With no ``target_python`` the list is returned unchanged
    (legacy behavior).

    ``target_env``, when given, is the real container :class:`TargetEnv` and
    is used AS-IS (RAW ``platform_machine``) for marker evaluation instead of
    reconstructing one from ``target_platform`` — see
    :func:`_resolved_target_env`.
    """
    if target_python is None:
        return raw_packages

    env = _marker_env(_resolved_target_env(target_python, target_platform, target_env))
    by_canon: dict[str, list[dict]] = {}
    order: list[str] = []
    for pkg in raw_packages:
        name = pkg.get("name")
        if not name:
            continue
        key = _canon(name)
        if key not in by_canon:
            by_canon[key] = []
            order.append(key)
        by_canon[key].append(pkg)

    selected: list[dict] = []
    for key in order:
        group = by_canon[key]
        if len(group) == 1:
            selected.append(group[0])
            continue
        applicable = [p for p in group if _entry_applies(p, env) is True]
        pool = applicable or group
        selected.append(max(pool, key=lambda p: _version_key(p.get("version"))))
    return selected


def _prune_to_applicable(
    nodes: list[Node],
    edges: list[Edge],
    seed_specs: list[tuple[str, str | None]],
    env: dict[str, str],
) -> tuple[list[Node], list[Edge]]:
    """Keep only packages reachable from the project's direct deps via edges whose
    markers apply under ``env``.

    A ``uv.lock`` is a UNIVERSAL lock: it lists packages needed across the whole
    ``requires-python`` range, including marker-gated ones (e.g. ``audioop-lts``
    pulled only under ``python_full_version >= '3.13'``). On a 3.11 target such a
    package is not part of the environment and has no installable distribution
    there, so installing it collapses the whole closure. Prune any node not
    reachable through marker-applicable edges from the direct deps.

    Conservative: an unknown/unevaluable marker (``_marker_applies`` -> ``None``)
    is treated as APPLICABLE (kept), so only definitely-false markers prune. If
    the roots cannot be determined, all nodes are kept (no-op).
    """
    if not nodes:
        return nodes, edges

    canon_to_id: dict[str, str] = {}
    for n in nodes:
        canon_to_id.setdefault(_canon(n.name), n.id)

    adj: dict[str, list[str]] = {}
    has_incoming: set[str] = set()
    for e in edges:
        has_incoming.add(e.dst)
        if e.marker is None or _marker_applies(e.marker, env) is not False:
            adj.setdefault(e.src, []).append(e.dst)

    # Seeds: applicable project direct deps, plus any node with no incoming edge
    # at all (a genuine root uv listed). A node whose only incoming edges are
    # marker-false is NOT a seed (it has incoming edges) -> correctly prunable.
    seeds: set[str] = set()
    for dep_name, marker in seed_specs:
        if marker is not None and _marker_applies(marker, env) is False:
            continue
        nid = canon_to_id.get(_canon(dep_name))
        if nid is not None:
            seeds.add(nid)
    for n in nodes:
        if n.id not in has_incoming:
            seeds.add(n.id)
    if not seeds:
        return nodes, edges

    keep: set[str] = set()
    stack = list(seeds)
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        keep.add(nid)
        for nxt in adj.get(nid, ()):
            if nxt not in keep:
                stack.append(nxt)

    kept_nodes = [n for n in nodes if n.id in keep]
    kept_edges = [e for e in edges if e.src in keep and e.dst in keep]
    return kept_nodes, kept_edges


def _package_node(
    name: str,
    version: str | None,
    *,
    provenance: str = "uv.lock",
    resolvable: bool = True,
) -> Node:
    """A resolver-discovered ``Package`` node (unknown until host-certified).

    ``resolvable=False`` marks an UNRESOLVED placeholder (a root ``uv`` could not
    resolve, or a conflict): it carries NO ``pip:<name>`` fix, because that name
    is exactly what failed to resolve — prescribing ``pip install <name>`` would
    fail (or install a squatter). "No known fix" is the honest signal.
    """
    fix = f"pip:{name}" if resolvable else None
    return Node(
        id=package_id(name, version),
        type=NodeType.PACKAGE,
        name=name,
        layer=Layer.PIP,
        discovered_by=DiscoveredBy.RESOLVER,
        version=version,
        check_command=f"python -m pip show {name}",
        fix_candidates=(fix,) if resolvable else (),
        chosen_fix=fix,
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Pure parser 1: uv.lock -> Package nodes + Package->Package requires edges.
# --------------------------------------------------------------------------- #
def _is_local_source(source: dict) -> bool:
    """True when a ``[[package]].source`` denotes the local project, not a dist."""
    return any(key in source for key in _LOCAL_SOURCE_KEYS)


def parse_uv_lock(
    text: str,
    target_python: str | None = None,
    target_platform: str | None = None,
    target_env: TargetEnv | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Parse a ``uv.lock`` into Package nodes + Package->Package requires edges.

    Each ``[[package]]`` with a registry/url/git source becomes a ``Package``
    node; local sources (the synthetic resolve root, path/editable deps) are
    skipped.  Each ``dependencies = [{name, marker?}]`` entry becomes a
    parent->child ``requires`` edge carrying the optional dependency ``marker``.

    When ``target_python`` is given and the lock forks a package across
    ``resolution-markers`` (python version — e.g. ``numpy`` 2.4.6 for ``<3.12``
    and 2.5.0 for ``>=3.12`` — OR platform, e.g. ``platform_machine ==
    'aarch64'``), only the version applicable to the target is emitted, so the
    closure stays container-accurate and never hands two versions of one
    distribution to ``pip install``.  ``target_env`` (the real container
    facts — RAW ``platform_machine``) is used for that evaluation when given;
    ``target_platform`` (e.g. ``"aarch64-manylinux_2_28"``) is only a
    fallback used to reconstruct a NORMALIZED-arch stand-in when no
    ``target_env`` is available (see :func:`_resolved_target_env`). Without
    either, every marker is evaluated against the TARGET container's assumed
    default facts, never the HOST running this parse.
    """
    data = tomllib.loads(text)
    raw_packages = _select_applicable_packages(
        data.get("package", []), target_python, target_platform, target_env
    )

    nodes: list[Node] = []
    entries: list[tuple[dict, Node]] = []
    canon_to_id: dict[str, str] = {}
    # The project's direct deps (from the local root entry) seed marker-reachability.
    seed_specs: list[tuple[str, str | None]] = []
    for pkg in raw_packages:
        name = pkg.get("name")
        if not name:
            continue
        if _is_local_source(pkg.get("source", {})):
            for dep in pkg.get("dependencies", []):
                dep_name = dep.get("name")
                if dep_name:
                    seed_specs.append((dep_name, dep.get("marker")))
            continue
        node = _package_node(name, pkg.get("version"))
        nodes.append(node)
        entries.append((pkg, node))
        canon_to_id[_canon(name)] = node.id

    # Dedup edges by (src, dst), MERGING markers: a parent may list the same child
    # under several markers (e.g. version-fork markers ``numpy<3.12`` / ``>=3.12``
    # that collapse to one node after fork-dedup). The child is then needed under
    # the UNION of those markers, so differing markers collapse to unconditional
    # (None) — otherwise keeping just the first marker would wrongly prune a child
    # that IS applicable via a later edge.
    edge_marker: dict[tuple[str, str], str | None] = {}
    edge_order: list[tuple[str, str]] = []
    for pkg, node in entries:
        for dep in pkg.get("dependencies", []):
            dep_name = dep.get("name")
            if not dep_name:
                continue
            child_id = canon_to_id.get(_canon(dep_name))
            if child_id is None or child_id == node.id:
                continue
            key = (node.id, child_id)
            marker = dep.get("marker")
            if key not in edge_marker:
                edge_marker[key] = marker
                edge_order.append(key)
            elif edge_marker[key] != marker:
                edge_marker[key] = None  # union of distinct markers -> unconditional
    edges: list[Edge] = [
        Edge(
            src=src,
            dst=dst,
            relation=EdgeType.REQUIRES,
            origin="resolver",
            marker=edge_marker[(src, dst)],
        )
        for (src, dst) in edge_order
    ]

    # Prune packages reachable only through markers that don't hold for the target
    # (a universal lock lists deps for the whole requires-python range). Only when a
    # target is given, mirroring the fork-dedup above (legacy no-op without one).
    if target_python is not None:
        nodes, edges = _prune_to_applicable(
            nodes,
            edges,
            seed_specs,
            _marker_env(_resolved_target_env(target_python, target_platform, target_env)),
        )
    return nodes, edges


# --------------------------------------------------------------------------- #
# Pure parser 2: per-package native-build risk — delegates to wheel_oracle.py.
# --------------------------------------------------------------------------- #
from src.python_deps.depgraph.wheel_oracle import (  # noqa: E402
    _artifact_filename,
    _wheel_matches_platform,
    risk_from_packages,
)


def native_risk_from_lock(
    text: str,
    target_platform: str,
    target_python: str | None = None,
    target_env: TargetEnv | None = None,
) -> dict[str, dict]:
    """Map ``package name -> {build_from_source, artifact, hash}`` from a lock.

    Thin orchestrator: parses the TOML, resolves fork duplicates against the
    real target environment the same way :func:`parse_uv_lock` does (so a
    forked package's risk reflects the version actually applicable on the
    target — correctness Task 7), filters out local-source entries with this
    module's OWN ``_is_local_source``, then delegates the per-package
    wheel-vs-sdist decision to :func:`wheel_oracle.risk_from_packages`.
    """
    data = tomllib.loads(text)
    raw_packages = _select_applicable_packages(
        data.get("package", []), target_python, target_platform, target_env
    )
    raw_packages = [
        p for p in raw_packages
        if p.get("name") and not _is_local_source(p.get("source", {}))
    ]
    return risk_from_packages(raw_packages, target_platform)
