"""Stage 3 — resolver v2: ``uv.lock``-driven Package closure.

Primary resolve source is **``uv.lock``** (the richest single uv artifact: nodes
+ versions + transitive edges + markers + sdist/wheel artifacts).  The orchestrator
(:func:`resolve_closure`) creates a throwaway uv project in a temp dir, runs
``uv lock`` (targeted at the container's python/platform) *on the host* through the
injected ``Executor`` (locked decision 1: the ``uv`` binary is invoked, never
imported), reads the produced ``uv.lock`` and feeds it to the PURE parsers below.

Pure, unit-testable parsers (no executor / no network / no uv):

* :func:`parse_uv_lock` — ``tomllib``-parse a lock into Package nodes +
  Package->Package ``requires`` edges (carrying optional dependency markers).
* :func:`native_risk_from_lock` — per-package native-build risk: a package with
  an ``sdist`` and **no wheel matching the target platform** ⇒
  ``build_from_source=True`` (also captures the chosen artifact + hash).
* :func:`parse_resolver_error` — structure a failed ``uv lock`` stderr into
  missing packages, version constraints/conflicts, and python-version incompat.

Resilience (no all-or-nothing): when the combined lock fails, the offending
roots are dropped and the lock is retried so the remaining good roots still yield
a graph; the dropped roots are surfaced as ``missing`` Package nodes (+ conflict
edges) with evidence.  A degraded ``uv pip compile`` ``# via`` parse is kept only
as a last-resort fallback when ``uv.lock`` cannot be produced at all.

Implementation note: the three parsing concerns are split into focused modules
(resolve_lock, resolve_errors, resolve_link); this module re-exports all their
public symbols and contains the orchestration layer (resolve_closure,
_pip_compile_fallback) that ties them together.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace

try:  # tomllib is stdlib on 3.11+; fall back to the tomli backport on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

from src.python_deps.depgraph.executor import Executor
from src.python_deps.depgraph.pins import (
    PythonIncompatiblePin,
    _default_fetch as _fetch_pypi_release,
    incompatible_python_pins,
)
from src.python_deps.depgraph.target_env import TargetEnv

# --------------------------------------------------------------------------- #
# Re-exports from sub-modules (keep every previously-public name importable
# from src.python_deps.depgraph.resolve without changes at any call site).
# --------------------------------------------------------------------------- #
from src.python_deps.depgraph.resolve_lock import (
    UV_BIN,
    DEFAULT_TARGET_PLATFORM,
    _DIST_NAME_RE,
    _REQUIREMENT_RE,
    _REQ_NAME_RE,
    _PY_VERSION_RE,
    _LOCAL_SOURCE_KEYS,
    _safe_dist_names,
    _req_name,
    _validate_target_python,
    _canon,
    _marker_env,
    _target_env_for,
    _marker_applies,
    _version_key,
    _entry_applies,
    _select_applicable_packages,
    _prune_to_applicable,
    _package_node,
    _is_local_source,
    parse_uv_lock,
    _artifact_filename,
    _wheel_matches_platform,
    native_risk_from_lock,
)
from src.python_deps.depgraph.resolve_errors import (
    MissingPackage,
    VersionConstraint,
    VersionConflict,
    PythonIncompat,
    ResolverDiagnosis,
    _REGISTRY_MISS_RE,
    _NO_VERSION_RE,
    _NO_VERSION_PLAIN_RE,
    _YOU_REQUIRE_RE,
    _DEPENDS_ON_RE,
    _PY_INCOMPAT_RE,
    _PY_IMPOSER_BEFORE_RE,
    _PY_IMPOSER_AFTER_RE,
    _BUILD_FAILURE_RE,
    _UNUSABLE_RE,
    _excerpt,
    parse_resolver_error,
    _missing_package_node,
    _conflict_package_node,
    _ROOT_IMPOSER_NAMES,
    _real_imposer,
    _conflict_endpoint_node,
    _conflict_endpoints,
    _diagnosis_to_graph,
    _offending_root_names,
)
from src.python_deps.depgraph.resolve_link import (
    _stamp,
    _import_edges,
    link_imports_to_packages,
    _merge,
)
from src.python_deps.depgraph.schema import Edge, EdgeType, Node, State

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants local to the orchestration / fallback layer.
# --------------------------------------------------------------------------- #
# Heredoc delimiter for feeding the root requirements on stdin (fallback path).
_HEREDOC = "DEPGRAPH_REQS"

# A pinned line, e.g. ``opencv-python==4.9.0.80`` (fallback path only).
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")

# Tokens that mark an annotation source rather than a parent distribution.
_SOURCE_FLAGS = {"-r", "-c", "--requirement", "--constraint"}

# uv 0.8.x can target a newer Python for solving while silently using the host's
# older interpreter to build sdist metadata. In that mode it has been observed
# to emit transitive pins whose Requires-Python excludes the declared target.
_BUILD_METADATA_PYTHON_FALLBACK_RE = re.compile(
    r"requested Python version\s+\S+\s+is not available;.*?"
    r"will be used to build dependencies instead",
    re.IGNORECASE | re.DOTALL,
)
_MAX_PYTHON_COMPAT_BACKTRACKS = 4
_PYTHON_COMPAT_CONSTRAINTS_FILE = ".depgraph-python-compat.constraints.txt"
# Static resolution is context, not the final authority: the ReAct builder can
# diagnose unresolved manifest obligations in the construction container.  A
# malformed or very broad monorepo manifest must therefore not consume the
# entire Agent wall-clock budget before the first turn.
_RESOLVE_COMMAND_TIMEOUT_SECONDS = 90
_MAX_UV_LOCK_ATTEMPTS = 6
_MAX_PIP_COMPILE_ATTEMPTS = 4


# --------------------------------------------------------------------------- #
# Orchestrator: throwaway uv project -> uv lock -> parse, with resilience.
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _project_dir(project_dir: str | None):
    if project_dir is not None:
        yield project_dir
    else:
        with tempfile.TemporaryDirectory(prefix="depgraph-uv-") as tmp:
            yield tmp


# Injection-safe optional-dependency group name (a bare TOML table key: no
# quotes, brackets, newlines or whitespace).
_SAFE_GROUP_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_group_names(names) -> list[str]:
    """Sorted, injection-safe extras-group names (bare TOML table keys only)."""
    return sorted(n for n in names if isinstance(n, str) and _SAFE_GROUP_RE.match(n))


def _write_pyproject(
    workdir: str,
    dist_names: list[str],
    target_python: str,
    *,
    extras: frozenset[str] = frozenset(),
) -> None:
    _validate_target_python(target_python)
    deps = ",\n    ".join(f'"{d}"' for d in _safe_dist_names(dist_names))
    content = (
        "[project]\n"
        'name = "depgraph-resolve-root"\n'
        'version = "0.0.0"\n'
        f'requires-python = ">={target_python}"\n'
        "dependencies = [\n"
        f"    {deps}\n"
        "]\n"
    )
    if extras:
        # The chosen groups' own requirement bodies already flow into
        # `dist_names` above -- roots.select_roots (Task 8) gates a group by
        # `needed_extras` and, when selected, flattens its members straight
        # into the plain root list, so they are already being resolved via
        # `dependencies`. This table is a provenance record of which
        # optional-dependency groups were IN SCOPE for this resolve (visible
        # in the produced uv.lock's root-package metadata for debugging/
        # traceability), not a second resolution path -- declaring the same
        # requirement under two different TOML keys would be redundant.
        content += "\n[project.optional-dependencies]\n"
        for group in _safe_group_names(extras):
            content += f"{group} = []\n"
    with open(os.path.join(workdir, "pyproject.toml"), "w", encoding="utf-8") as fh:
        fh.write(content)


def _uv_python_platform(platform_tag: str) -> str:
    """Translate musllinux wheel tags to platform triples accepted by uv.

    TargetEnv retains the wheel tag for compatibility checks; only the uv CLI
    spelling changes at this boundary.
    """
    match = re.fullmatch(r"(aarch64|x86_64)-musllinux(?:_\d+_\d+)?", platform_tag)
    if match:
        return f"{match.group(1)}-unknown-linux-musl"
    return platform_tag


def _lock_command(
    workdir: str,
    target_python: str,
    exclude_newer: str | None,
    python_platform_tag: str,
) -> str:
    """Build the ``uv lock`` shell command, ALWAYS targeted at the container.

    ``--python-platform`` (Task 7) is what stops ``uv`` from resolving for the
    HOST's platform: without it a dev host's own OS/arch tags leak into the
    lock's wheel selection and marker environment, silently diverging from the
    container being built.
    """
    parts = [shlex.quote(UV_BIN), "lock", "--python", shlex.quote(target_python)]
    if exclude_newer:
        parts += ["--exclude-newer", shlex.quote(exclude_newer)]
    parts += [
        "--python-platform",
        shlex.quote(_uv_python_platform(python_platform_tag)),
    ]
    return f"cd {shlex.quote(workdir)} && {' '.join(parts)}"


def _lock_supports_python_platform(host_executor: Executor) -> bool:
    """Whether this uv build can target a platform from ``uv lock``.

    uv versions differ here: some expose ``--python-platform`` on ``uv lock``;
    others expose it only on ``uv pip compile``.  An unsuccessful capability
    probe is treated as "supported" so injected/fake executors and transient
    help failures retain the historical lock-first behavior.
    """
    result = host_executor.run(
        f"{shlex.quote(UV_BIN)} help lock",
        timeout=min(30, _RESOLVE_COMMAND_TIMEOUT_SECONDS),
    )
    if not result.ok:
        return True
    help_text = f"{result.stdout}\n{result.stderr}".strip()
    if not help_text:
        return True
    return "--python-platform" in help_text


def _read_lock(workdir: str) -> str | None:
    try:
        with open(os.path.join(workdir, "uv.lock"), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _stamp_cutoff_relaxation(
    node: Node, requested_cutoff: str | None, effective_cutoff: str | None
) -> Node:
    """Record that explicit requirements overrode the era cutoff heuristic."""
    if not requested_cutoff or effective_cutoff == requested_cutoff:
        return node
    data = dict(node.data)
    data.update(
        {
            "requested_exclude_newer": requested_cutoff,
            "exclude_newer_relaxed": True,
        }
    )
    return replace(node, data=data)


def _prune_stale_diagnostics(
    resolved_nodes: list[Node],
    diag_nodes: list[Node],
    diag_edges: list[Edge],
) -> tuple[list[Node], list[Edge]]:
    """Drop failed-attempt diagnostics superseded by a successful closure.

    Resolver diagnostics use stable, unversioned ids (``pkg:name``), while a
    successful closure uses versioned ids (``pkg:name==version``).  Keeping both
    lets the later build-stage id normalization overwrite the concrete package
    with the stale diagnostic; an old ``conflicts_with`` edge can also suppress
    emission of the now-resolved package.  Diagnostics for names that remain
    unresolved are preserved.
    """
    resolved_names = {_canon(node.name) for node in resolved_nodes}
    stale_ids = {
        node.id for node in diag_nodes if _canon(node.name) in resolved_names
    }
    if not stale_ids:
        return diag_nodes, diag_edges
    return (
        [node for node in diag_nodes if node.id not in stale_ids],
        [
            edge
            for edge in diag_edges
            if edge.src not in stale_ids and edge.dst not in stale_ids
        ],
    )


def resolve_closure(
    roots: list[tuple[str | None, str]],
    host_executor: Executor,
    *,
    target_env: TargetEnv,
    exclude_newer: str | None = None,
    project_dir: str | None = None,
    extras: frozenset[str] = frozenset(),
    release_metadata_fetch: Callable[[str, str], dict] = _fetch_pypi_release,
) -> tuple[list[Node], list[Edge]]:
    """Resolve ``roots`` to a Package closure (nodes + edges) via ``uv.lock``.

    ``roots`` is a list of ``(import_id | None, dist_name)`` pairs (the shape
    produced by ``roots.select_roots`` / ``naming.package_roots``); a ``None``
    import id is a manifest-declared root with no Import node to attach.

    ``target_env`` is the single :class:`TargetEnv` (Task 7) the whole resolve
    honors: ``target_env.python_version`` / ``target_env.python_platform_tag``
    (the NORMALIZED wheel/uv tag) drive ``uv lock --python``/``--python-platform``
    and the wheel-artifact match, while ``target_env`` ITSELF (carrying the RAW
    ``platform.machine()`` the container reported) is threaded into
    ``parse_uv_lock``/``native_risk_from_lock`` so every PEP 508 marker
    evaluated against a forked/conditional lock entry sees the container's own
    facts — never a normalized stand-in, and never the host running this
    resolve.

    ``extras`` is the set of ``[project.optional-dependencies]`` / extras_require
    group names IN SCOPE for this resolve (Task 8's targeted-extras fix — the
    caller, typically ``roots.select_roots(..., needed_extras=...)`` upstream
    of here, has already gated which groups' members are present in ``roots``
    at all). It is written into the temp pyproject's own
    ``[project.optional-dependencies]`` table as a provenance record of which
    groups were considered (see :func:`_write_pyproject`); the groups'
    requirement bodies themselves reach the resolver via ``roots`` /
    ``dist_names``, not through this table.

    A throwaway uv project is created in a temp dir (or ``project_dir`` when
    injected, for tests), ``uv lock`` is run on the host through
    ``host_executor``, and the resulting ``uv.lock`` is parsed.  On lock failure
    the offending roots are dropped and the lock is retried (per-root
    resilience); the dropped roots are emitted as ``missing``/conflict Package
    nodes with evidence.  If no lock can be produced at all, a degraded
    ``uv pip compile`` ``# via`` parse is used as a last resort.
    """
    target_python = target_env.python_version
    platform = target_env.python_platform_tag
    if not roots:
        return [], []

    diag_nodes: list[Node] = []
    diag_edges: list[Edge] = []

    with _project_dir(project_dir) as workdir:
        current = list(roots)
        requested_cutoff = exclude_newer
        effective_cutoff = exclude_newer
        cutoff_relaxed = False

        # A target-honest lock is impossible when this uv version does not
        # support `uv lock --python-platform`.  In that case use the supported
        # target-aware compile interface directly instead of issuing lock
        # commands that are guaranteed to fail at argument parsing time.
        if not _lock_supports_python_platform(host_executor):
            fb_nodes, fb_edges = _pip_compile_fallback(
                roots,
                host_executor,
                target_python,
                target_platform=platform,
                exclude_newer=effective_cutoff,
                drop_bad_roots=effective_cutoff is None,
                compatibility_constraints_path=os.path.join(
                    workdir, _PYTHON_COMPAT_CONSTRAINTS_FILE
                ),
                release_metadata_fetch=release_metadata_fetch,
            )
            if not fb_nodes and effective_cutoff is not None:
                effective_cutoff = None
                cutoff_relaxed = True
                fb_nodes, fb_edges = _pip_compile_fallback(
                    roots,
                    host_executor,
                    target_python,
                    target_platform=platform,
                    exclude_newer=None,
                    drop_bad_roots=True,
                    compatibility_constraints_path=os.path.join(
                        workdir, _PYTHON_COMPAT_CONSTRAINTS_FILE
                    ),
                    release_metadata_fetch=release_metadata_fetch,
                )
            if fb_nodes:
                fb_nodes = [
                    _stamp_cutoff_relaxation(
                        _stamp(node, {}, target_python, platform, effective_cutoff),
                        requested_cutoff,
                        effective_cutoff,
                    )
                    for node in fb_nodes
                ]
            return fb_nodes, fb_edges

        # Root dropping remains bounded. A same-roots retry without the cutoff
        # is extra because the manifest constraint outranks the era heuristic.
        drop_attempts = 0
        lock_attempts = 0
        while (
            drop_attempts <= len(roots)
            and lock_attempts < _MAX_UV_LOCK_ATTEMPTS
        ):
            names = [dist for _import_id, dist in current]
            if not names:
                break
            _write_pyproject(workdir, names, target_python, extras=extras)
            lock_attempts += 1
            result = host_executor.run(
                _lock_command(workdir, target_python, effective_cutoff, platform),
                timeout=_RESOLVE_COMMAND_TIMEOUT_SECONDS,
            )
            lock_text = _read_lock(workdir) if result.ok else None

            if lock_text:
                try:
                    nodes, edges = parse_uv_lock(
                        lock_text,
                        target_python,
                        target_platform=platform,
                        target_env=target_env,
                    )
                    risk = native_risk_from_lock(
                        lock_text, platform, target_python, target_env=target_env
                    )
                    nodes = [
                        _stamp_cutoff_relaxation(
                            _stamp(n, risk, target_python, platform, effective_cutoff),
                            requested_cutoff,
                            effective_cutoff,
                        )
                        for n in nodes
                    ]
                    edges = list(edges) + _import_edges(current, nodes)
                    diag_nodes, diag_edges = _prune_stale_diagnostics(
                        nodes, diag_nodes, diag_edges
                    )
                    return _merge(nodes, edges, diag_nodes, diag_edges)
                except tomllib.TOMLDecodeError:
                    # A truncated/corrupt lock (uv exit 0 but partial write on a
                    # disk-full/NFS hiccup) is treated like a missing lock: fall
                    # through to diagnosis/retry instead of crashing the pipeline.
                    pass

            if effective_cutoff is not None and not cutoff_relaxed:
                effective_cutoff = None
                cutoff_relaxed = True
                continue

            # Lock failed: diagnose and try dropping the offending roots.
            drop_attempts += 1
            diag = parse_resolver_error(result.stderr)
            dn, de = _diagnosis_to_graph(diag)
            diag_nodes, diag_edges = _merge(diag_nodes, diag_edges, dn, de)

            current_root_names = {_canon(_req_name(r[1])) for r in current}
            offending = _offending_root_names(diag, current_root_names)
            remaining = [
                r for r in current if _canon(_req_name(r[1])) not in offending
            ]
            if not offending or remaining == current:
                break  # cannot make progress by dropping roots.
            current = remaining

        # Last resort: degraded `uv pip compile` `# via` fallback.
        fb_nodes, fb_edges = _pip_compile_fallback(
            roots,
            host_executor,
            target_python,
            target_platform=platform,
            exclude_newer=effective_cutoff,
            drop_bad_roots=True,
            compatibility_constraints_path=os.path.join(
                workdir, _PYTHON_COMPAT_CONSTRAINTS_FILE
            ),
            release_metadata_fetch=release_metadata_fetch,
        )
        concrete_fb_nodes = [
            node for node in fb_nodes if node.chosen_fix is not None
        ]
        diag_nodes, diag_edges = _prune_stale_diagnostics(
            concrete_fb_nodes, diag_nodes, diag_edges
        )
        return _merge(fb_nodes, fb_edges, diag_nodes, diag_edges)


# --------------------------------------------------------------------------- #
# Degraded fallback: `uv pip compile` `# via` annotation parse (last resort).
# --------------------------------------------------------------------------- #
def _used_non_target_python_for_metadata(stderr: str) -> bool:
    """Whether uv admitted that dependency metadata used another Python."""
    return bool(_BUILD_METADATA_PYTHON_FALLBACK_RE.search(stderr or ""))


def _write_python_compat_constraints(
    path: str,
    exclusions: dict[str, tuple[str, set[str]]],
) -> None:
    """Persist safe ``name!=bad,...`` constraints for the next uv solve."""
    lines: list[str] = []
    for key in sorted(exclusions):
        name, versions = exclusions[key]
        token = name + ",".join(f"!={version}" for version in sorted(versions))
        if _REQUIREMENT_RE.match(token):
            lines.append(token)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


def _explicit_exact_root_pins(
    roots: list[tuple[str | None, str]],
) -> dict[str, str]:
    """Canonical root name -> immutable exact version declared by the repo.

    Compatibility backtracking may narrow a range or an unbounded transitive
    dependency, but it must never silently override ``project==1.2.3``. Such a
    root is instead surfaced as an incompatible declared obligation.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return {}

    exact: dict[str, str] = {}
    for _import_id, token in roots:
        try:
            requirement = Requirement(token)
        except Exception:
            continue
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) == 1
            and specifiers[0].operator == "=="
            and "*" not in specifiers[0].version
        ):
            exact[_canon(requirement.name)] = specifiers[0].version
    return exact


def _compile_command(
    dist_names: list[str],
    target_python: str,
    *,
    target_platform: str | None = None,
    exclude_newer: str | None = None,
    constraint_file: str | None = None,
) -> str:
    """Build the ``uv pip compile`` shell command (roots fed via heredoc stdin).

    Names are validated (``_safe_dist_names``) so a value containing a newline +
    the heredoc delimiter cannot prematurely terminate the body; ``UV_BIN`` and
    ``target_python`` are quoted so neither can inject a second shell statement.
    """
    _validate_target_python(target_python)
    reqs = "\n".join(_safe_dist_names(dist_names))
    parts = [
        shlex.quote(UV_BIN),
        "pip",
        "compile",
        "--python-version",
        shlex.quote(target_python),
    ]
    if target_platform:
        parts += [
            "--python-platform",
            shlex.quote(_uv_python_platform(target_platform)),
        ]
    if exclude_newer:
        parts += ["--exclude-newer", shlex.quote(exclude_newer)]
    if constraint_file:
        parts += ["--constraint", shlex.quote(constraint_file)]
    parts.append("-")
    return f"{' '.join(parts)} <<'{_HEREDOC}'\n{reqs}\n{_HEREDOC}"


def _parse_parents(text: str) -> list[str]:
    """Parent distribution names from a ``# via ...`` annotation body."""
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    parents: list[str] = []
    for tok in tokens:
        if tok in _SOURCE_FLAGS:
            return []  # remainder is a filename/source, not a dependency.
        if tok == "-" or tok.startswith("-"):
            continue
        parents.append(tok)
    return parents


def _parse_closure(stdout: str) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """Parse pinned lines + ``# via`` annotations (fallback path)."""
    packages: list[tuple[str, str]] = []
    via: dict[str, set[str]] = {}
    current_canon: str | None = None
    in_via = False

    for raw in stdout.splitlines():
        stripped = raw.strip()
        if not stripped:
            in_via = False
            continue
        if stripped.startswith("#"):
            body = stripped[1:].strip()
            if body.startswith("via"):
                in_via = True
                for parent in _parse_parents(body[3:]):
                    if current_canon is not None:
                        via.setdefault(current_canon, set()).add(_canon(parent))
            elif in_via:
                for parent in _parse_parents(body):
                    if current_canon is not None:
                        via.setdefault(current_canon, set()).add(_canon(parent))
            else:
                in_via = False
            continue

        in_via = False
        match = _PIN_RE.match(stripped)
        if match:
            name, version = match.group(1), match.group(2)
            packages.append((name, version))
            current_canon = _canon(name)
    return packages, via


def _pip_compile_fallback(
    roots: list[tuple[str | None, str]],
    host_executor: Executor,
    target_python: str,
    *,
    target_platform: str | None = None,
    exclude_newer: str | None = None,
    drop_bad_roots: bool = False,
    compatibility_constraints_path: str | None = None,
    release_metadata_fetch: Callable[[str, str], dict] = _fetch_pypi_release,
) -> tuple[list[Node], list[Edge]]:
    """Degraded resolve via ``uv pip compile`` when ``uv.lock`` is unavailable.

    When ``drop_bad_roots`` is enabled, the fallback mirrors the lock path's
    bounded resilience: a resolver-diagnosed bad root is represented as a
    diagnostic node, removed, and the surviving roots are retried.  This is
    important on uv builds that only support target-platform selection on
    ``uv pip compile``; one low-confidence import mapping must not erase every
    manifest-backed package from the graph.
    """
    current = list(roots)
    diag_nodes: list[Node] = []
    diag_edges: list[Edge] = []
    result = None
    compatibility_guard = True
    compatibility_rounds = 0
    exclusions: dict[str, tuple[str, set[str]]] = {}
    rejected_pins: dict[str, PythonIncompatiblePin] = {}
    rejected_exact_roots: set[str] = set()
    exact_root_pins = _explicit_exact_root_pins(roots)
    compatibility_rejections: dict[tuple[str, str], PythonIncompatiblePin] = {}
    release_cache: dict[tuple[str, str], dict] = {}
    release_cache_lock = threading.Lock()

    if compatibility_constraints_path:
        try:
            os.unlink(compatibility_constraints_path)
        except FileNotFoundError:
            pass

    def cached_release_fetch(name: str, version: str) -> dict:
        key = (_canon(name), version)
        with release_cache_lock:
            cached = release_cache.get(key)
        if cached is not None:
            return cached
        try:
            fetched = release_metadata_fetch(name, version)
        except Exception:  # metadata is best-effort; cache the unknown result
            fetched = {}
        with release_cache_lock:
            release_cache[key] = fetched
        return fetched

    compile_attempts = 0
    while current and compile_attempts < _MAX_PIP_COMPILE_ATTEMPTS:
        dist_names = [dist for _import_id, dist in current]
        constraint_file = (
            compatibility_constraints_path if exclusions else None
        )
        compile_attempts += 1
        result = host_executor.run(
            _compile_command(
                dist_names,
                target_python,
                target_platform=target_platform,
                exclude_newer=exclude_newer,
                constraint_file=constraint_file,
            ),
            timeout=_RESOLVE_COMMAND_TIMEOUT_SECONDS,
        )
        if result.ok:
            compatibility_guard = compatibility_guard or (
                _used_non_target_python_for_metadata(result.stderr)
            )
            if compatibility_guard:
                packages, _via = _parse_closure(result.stdout)
                incompatible = incompatible_python_pins(
                    packages,
                    target_python,
                    fetch=cached_release_fetch,
                )
                if incompatible:
                    for pin in incompatible:
                        compatibility_rejections[
                            (_canon(pin.name), pin.version)
                        ] = pin
                    immutable = [
                        pin
                        for pin in incompatible
                        if exact_root_pins.get(_canon(pin.name)) == pin.version
                    ]
                    if immutable:
                        # The repository deliberately fixed this exact release.
                        # Do not disguise the incompatibility by overriding the
                        # root; reject every known bad pin from this closure and
                        # let the graph explain the unsatisfied declaration.
                        rejected_pins = {
                            _canon(pin.name): pin for pin in incompatible
                        }
                        rejected_exact_roots = {
                            _canon(pin.name) for pin in immutable
                        }
                        logger.warning(
                            "target Python %s excludes explicit exact root pin(s): %s",
                            target_python,
                            ", ".join(
                                f"{pin.name}=={pin.version} "
                                f"(Requires-Python {pin.requires_python})"
                                for pin in immutable
                            ),
                        )
                        break

                    added = False
                    for pin in incompatible:
                        key = _canon(pin.name)
                        if key not in exclusions:
                            exclusions[key] = (pin.name, set())
                        versions = exclusions[key][1]
                        if pin.version not in versions:
                            versions.add(pin.version)
                            added = True

                    if (
                        added
                        and compatibility_constraints_path
                        and compatibility_rounds < _MAX_PYTHON_COMPAT_BACKTRACKS
                    ):
                        _write_python_compat_constraints(
                            compatibility_constraints_path, exclusions
                        )
                        compatibility_rounds += 1
                        logger.warning(
                            "uv emitted target-incompatible pins for Python %s; "
                            "bounded compatibility retry %s/%s excludes: %s",
                            target_python,
                            compatibility_rounds,
                            _MAX_PYTHON_COMPAT_BACKTRACKS,
                            ", ".join(
                                f"{pin.name}=={pin.version} "
                                f"(Requires-Python {pin.requires_python})"
                                for pin in incompatible
                            ),
                        )
                        continue

                    # A constraint was ignored, no constraint path is available,
                    # or the bounded search is exhausted. Preserve the graph
                    # shape below, but replace each bad concrete pin with a
                    # non-installable diagnostic node: setup must never emit it.
                    rejected_pins = {
                        _canon(pin.name): pin for pin in incompatible
                    }
            break

        error_text = "\n".join(
            part for part in (result.stderr, result.stdout) if part
        )
        logger.warning(
            "uv pip compile fallback failed (rc=%s): %s",
            result.returncode,
            error_text.strip()[-1200:],
        )
        if not drop_bad_roots:
            return [], []

        diagnosis = parse_resolver_error(error_text)
        dn, de = _diagnosis_to_graph(diagnosis)
        diag_nodes, diag_edges = _merge(diag_nodes, diag_edges, dn, de)
        current_names = {_canon(_req_name(dist)) for _iid, dist in current}
        offending = _offending_root_names(diagnosis, current_names)
        remaining = [
            root for root in current
            if _canon(_req_name(root[1])) not in offending
        ]
        if not offending or remaining == current:
            return diag_nodes, diag_edges
        current = remaining

    if result is None or not result.ok:
        if current and compile_attempts >= _MAX_PIP_COMPILE_ATTEMPTS:
            logger.warning(
                "uv pip compile fallback stopped after bounded attempt budget %s",
                _MAX_PIP_COMPILE_ATTEMPTS,
            )
        return diag_nodes, diag_edges

    packages, via = _parse_closure(result.stdout)

    nodes: list[Node] = []
    canon_to_id: dict[str, str] = {}
    for name, version in packages:
        canonical_name = _canon(name)
        rejected = rejected_pins.get(canonical_name)
        if rejected is None:
            node = _package_node(name, version, provenance="uv pip compile")
            prior_rejections = sorted(
                (
                    pin
                    for (key, _version), pin in compatibility_rejections.items()
                    if key == canonical_name
                ),
                key=lambda pin: pin.version,
            )
            if prior_rejections:
                rejected_summary = ", ".join(
                    f"{pin.name}=={pin.version} "
                    f"(Requires-Python {pin.requires_python})"
                    for pin in prior_rejections
                )
                node = replace(
                    node,
                    evidence=(
                        f"selected {name}=={version} for Python {target_python} "
                        f"after rejecting incompatible pin(s): {rejected_summary}"
                    ),
                    data={
                        "python_compatibility_target": target_python,
                        "python_compatibility_backtracks": compatibility_rounds,
                        "python_compatibility_rejected": [
                            {
                                "version": pin.version,
                                "requires_python": pin.requires_python,
                            }
                            for pin in prior_rejections
                        ],
                    },
                )
        else:
            is_exact_root = canonical_name in rejected_exact_roots
            rejection_kind = (
                "declared exact root pin" if is_exact_root else "resolved pin"
            )
            evidence = (
                f"rejected {rejection_kind} {name}=={rejected.version}: "
                f"Requires-Python "
                f"{rejected.requires_python} excludes target Python {target_python}"
            )
            node = replace(
                _package_node(
                    name,
                    None,
                    provenance="uv pip compile compatibility guard",
                    resolvable=False,
                ),
                state=State.MISSING,
                evidence=evidence,
                resolution_status="failed",
                resolution_error=evidence,
                data={
                    "rejected_pin": rejected.version,
                    "requires_python": rejected.requires_python,
                    "target_python": target_python,
                    "compatibility_backtracks": compatibility_rounds,
                    "explicit_exact_root": is_exact_root,
                },
            )
        nodes.append(node)
        canon_to_id[canonical_name] = node.id

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()

    def _add_edge(src: str, dst: str) -> None:
        if src == dst or (src, dst) in seen:
            return
        seen.add((src, dst))
        edges.append(
            Edge(src=src, dst=dst, relation=EdgeType.REQUIRES, origin="resolver")
        )

    edges.extend(_import_edges(current, nodes))
    for e in edges:
        seen.add((e.src, e.dst))

    for child_canon, parents in via.items():
        child_id = canon_to_id.get(child_canon)
        if child_id is None:
            continue
        for parent_canon in parents:
            parent_id = canon_to_id.get(parent_canon)
            if parent_id is not None:
                _add_edge(parent_id, child_id)

    diag_nodes, diag_edges = _prune_stale_diagnostics(
        nodes, diag_nodes, diag_edges
    )
    return _merge(nodes, edges, diag_nodes, diag_edges)
