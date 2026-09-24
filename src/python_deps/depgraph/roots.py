"""Root selection for the resolver — manifest-first, scan-gap-filled, filtered.

The resolver (stage 3) needs a set of *distribution* roots to lock.  Picking the
wrong roots is what caused the "Run-A collapse": feeding non-distributions
(stdlib modules, Python-2 shims, typing-only stubs) to ``uv`` made the whole
resolve fail and produced an empty Package layer.

This module realizes the spec's "Root selection" ladder
(``docs/superpowers/specs/2026-06-23-uv-enriched-depgraph.md``):

1. **Manifest first** — declared dependencies (parsed via
   :func:`python_deps.evidence.collect_python_dependency_evidence`) are the
   highest-trust roots.  They carry no import node, so their ``import_id`` is
   ``None``.
2. **Scan gap-fill** — mapped scanned imports (:func:`naming.package_roots`) are
   added only for distributions not already covered by a manifest declaration.
3. **Filter non-distributions** — stdlib modules, known Py2 shims, typing-only
   stubs, and obvious junk are dropped before anything reaches ``uv``: a name
   that isn't plausibly a PyPI distribution never becomes a root.

Pure (no Executor / no network): it reads the repo on disk and an already-scanned
graph, and returns the root list.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.python_deps.depgraph.naming import package_roots
from src.python_deps.depgraph.resolve_lock import _marker_applies
from src.python_deps.depgraph.schema import DepGraph
from src.python_deps.evidence import collect_python_dependency_evidence
from src.python_deps.import_mapping import (
    normalize_package_name,
    top_level_import_name,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids any import-order risk
    from src.python_deps.depgraph.target_env import TargetEnv

# Python-2 modules that survive a py3 static scan as "external" (they are neither
# py3 stdlib nor project-local) but are NOT installable PyPI distributions.
PY2_SHIM_DENYLIST: frozenset[str] = frozenset(
    {
        "StringIO",
        "cStringIO",
        "BaseHTTPServer",
        "SimpleHTTPServer",
        "CGIHTTPServer",
        "urllib2",
        "urlparse",
        "Queue",
        "cPickle",
        "cProfile",
        "Tkinter",
        "httplib",
        "cookielib",
        "Cookie",
        "thread",
        "copy_reg",
        "__builtin__",
        "ConfigParser",
        "HTMLParser",
        "SocketServer",
        "xmlrpclib",
        "robotparser",
        "repr",
    }
)

# Typing-only / stub-only modules that exist for static analysis, not runtime.
TYPING_ONLY_DENYLIST: frozenset[str] = frozenset({"_typeshed"})

# Obvious junk that is never a distribution root.
JUNK_DENYLIST: frozenset[str] = frozenset({
    "", "__future__", "__main__",
    # Conventional source/documentation roots are repository layout, not PyPI
    # distributions.  A low-trust direct-name guess for these is catastrophic:
    # one bogus package can fail the fail-fast replay before real deps install.
    "src", "docs_src", "doc_src", "tests", "test",
})

# Every PEP 508 marker variable name `packaging` can parse into a Marker (the
# grammar restricts marker variables to exactly this set -- see
# https://peps.python.org/pep-0508/#environment-markers). Used by
# ``_env_marker_excludes`` to detect a marker referencing a field the TARGET
# cannot supply (see that function's docstring for why this matters).
_PEP508_MARKER_FIELDS: frozenset[str] = frozenset(
    {
        "os_name",
        "sys_platform",
        "platform_machine",
        "platform_python_implementation",
        "platform_release",
        "platform_system",
        "platform_version",
        "python_version",
        "python_full_version",
        "implementation_name",
        "implementation_version",
        "extra",
    }
)

# A version specifier safe to embed verbatim in the resolver's temp pyproject
# (a TOML double-quoted string) and the fallback heredoc body: PEP 440 operators
# plus version characters only — no quotes, spaces, newlines, shell metacharacters
# or environment markers.  Anything else falls back to the bare name.
_SAFE_SPECIFIER_RE = re.compile(
    r"^[<>=!~]=?[A-Za-z0-9.][A-Za-z0-9.*+!-]*"
    r"(?:,[<>=!~]=?[A-Za-z0-9.][A-Za-z0-9.*+!-]*)*$"
)


def _manifest_root_token(req) -> str:
    """Distribution token for a manifest dep, carrying a safe version specifier.

    Carrying the declared constraint (e.g. ``numpy<2``) is what lets the resolver
    SEE a conflict — the spec's "project pinning numpy<2 plus a dep requiring
    numpy>=2" example.  The specifier is dropped (bare name only) when it is empty,
    carries a marker, or fails the injection-safety check.

    Per-dep extras (``uvicorn[standard]``) are carried too — NOT stripped — so
    the extra's own transitive deps reach the resolver instead of silently
    vanishing under a ``--no-deps`` install (Task 8).
    """
    spec = (getattr(req, "specifier", "") or "").replace(" ", "")
    extras = getattr(req, "extras", ()) or ()
    name = f"{req.name}[{','.join(extras)}]" if extras else req.name
    if spec and _SAFE_SPECIFIER_RE.match(spec):
        return f"{name}{spec}"
    return name


# Optional-dependency group name embedded at the tail of a requirement's
# ``source`` string by evidence.py, e.g.
# ``pyproject.toml:project.optional-dependencies.test`` -> ``test``, or
# ``setup.cfg:options.extras_require.docs`` -> ``docs``.
_OPTIONAL_GROUP_RE = re.compile(
    r"(?:optional-dependencies|extras_require|dependency-groups|"
    r"test-requirements|tox-deps|hatch-test-deps)\.(.+)$"
)


@dataclass(frozen=True)
class ManifestRoot:
    """A selected manifest dependency before a concrete version is resolved."""

    name: str
    resolver_token: str
    specifier: str
    marker: str
    source: str
    extras: tuple[str, ...] = ()


def _requirement_group(source: str) -> str:
    """Extras-group name a ``kind=="optional_dependency"`` requirement belongs to."""
    match = _OPTIONAL_GROUP_RE.search(source or "")
    return match.group(1) if match else ""


def _is_non_distribution(
    import_name: str,
    target_stdlib_modules: frozenset[str] | None = None,
) -> bool:
    """True when ``import_name`` cannot be a real PyPI distribution root."""
    top = top_level_import_name(import_name).strip()
    if top in JUNK_DENYLIST:
        return True
    if top in PY2_SHIM_DENYLIST:
        return True
    if top in TYPING_ONLY_DENYLIST:
        return True
    # Prefer the target container's stdlib set. Falling back to the host keeps
    # historical behavior when the target probe is unavailable.
    stdlib_modules = (
        target_stdlib_modules
        if target_stdlib_modules is not None
        else frozenset(sys.stdlib_module_names)
    )
    if top in stdlib_modules:
        return True
    # Leading-underscore private/dunder modules are not distributions.
    if top.startswith("_"):
        return True
    return False


def _env_marker_excludes(req, target_env: "TargetEnv | None") -> bool:
    """True only when we are CONFIDENT ``req`` must not be a root for the target.

    Conservative-skip rule (review "no silent shrink" constraint), generalized
    past the original ``extra``-only special case: a dep is excluded ONLY when
    ALL of the following hold —

    * a real ``target_env`` was given (no target -> nothing is filterable),
    * the requirement HAS a marker at all,
    * EVERY PEP 508 field the marker references is one ``target_env.marker_env()``
      actually supplies (see ``uncovered`` below), and
    * the marker EVALUATES to False against ``target_env.marker_env()``.

    Why generalize: ``TargetEnv.marker_env()`` supplies only 6 of the 12 PEP 508
    marker fields (``python_version``, ``python_full_version``,
    ``platform_machine``, ``sys_platform``, ``os_name``, ``platform_system``).
    ``packaging.markers.Marker.evaluate()`` silently fills any field missing from
    the given environment from the HOST's own ``default_environment()`` -- so a
    marker referencing e.g. ``platform_python_implementation`` or
    ``implementation_name`` would be evaluated against the HOST's value, not the
    TARGET's, making a drop decision confidently WRONG. ``extra`` is simply one
    more field ``marker_env()`` never supplies (it is inherently a per-requirement
    grouping flag, not an environment fact) -- so the same "field not covered by
    the target" rule subsumes the old ``"extra" in marker`` special-case without
    naming it specially: ``uncovered`` always contains ``extra`` alongside the
    5 host-fallback fields, and gating on ANY uncovered field being referenced
    keeps ``extra``-gated deps out of this filter exactly as before (they are
    already handled by the ``needed_extras`` mechanism above; this function must
    never re-judge them, or it would silently drop an extra's dep for a reason
    unrelated to environment).
    ``uncovered`` is DERIVED from ``target_env.marker_env()`` each call, so
    widening that method (a tracked follow-up) automatically shrinks
    ``uncovered`` down to just ``{"extra"}`` with no change needed here.

    Every other case — no target_env, no marker, a marker referencing an
    uncovered field, a True evaluation, or any evaluation error
    (``_marker_applies`` returns ``None`` on ``packaging`` absence or a
    malformed marker) — KEEPS the dep. Keeping on uncertainty is the point: an
    over-eager filter here would silently shrink the closure fed to ``uv``,
    which is worse than resolving one extra unneeded root.

    Substring matching (``field in marker``) below errs toward KEEP: a marker
    string that merely happens to contain a field name as a substring (of e.g.
    a version literal) only causes over-inclusion into ``uncovered``, which
    means MORE deps get kept, never fewer — the safe direction. Do not tighten
    this into exact tokenization; that could only ever make the filter drop
    more, which is the wrong direction for this "no silent shrink" invariant.
    """
    if target_env is None:
        return False
    marker = getattr(req, "marker", "") or ""
    if not marker:
        return False
    covered = set(target_env.marker_env())
    uncovered = _PEP508_MARKER_FIELDS - covered
    if any(field in marker for field in uncovered):
        return False
    try:
        verdict = _marker_applies(marker, target_env.marker_env())
    except Exception:
        return False
    return verdict is False


def select_manifest_roots(
    repo_path: str,
    needed_extras: frozenset[str] = frozenset(),
    *,
    target_env: "TargetEnv | None" = None,
    target_stdlib_modules: frozenset[str] | None = None,
) -> list[ManifestRoot]:
    """Return applicable, deduplicated manifest dependencies with provenance."""
    evidence = collect_python_dependency_evidence(repo_path)
    selected: list[ManifestRoot] = []
    seen: set[str] = set()
    for req in evidence.declared_dependencies:
        if getattr(req, "kind", "dependency") == "optional_dependency":
            if _requirement_group(req.source) not in needed_extras:
                continue
        if _env_marker_excludes(req, target_env) or _is_non_distribution(
            req.name, target_stdlib_modules
        ):
            continue
        normalized = normalize_package_name(req.name)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            ManifestRoot(
                name=req.name,
                resolver_token=_manifest_root_token(req),
                specifier=(getattr(req, "specifier", "") or "").replace(" ", ""),
                marker=getattr(req, "marker", "") or "",
                source=getattr(req, "source", "") or "",
                extras=tuple(getattr(req, "extras", ()) or ()),
            )
        )
    return selected


def select_roots(
    repo_path: str,
    graph: DepGraph,
    needed_extras: frozenset[str] = frozenset(),
    *,
    target_env: "TargetEnv | None" = None,
    target_stdlib_modules: frozenset[str] | None = None,
) -> list[tuple[str | None, str]]:
    """Return ``(import_id | None, distribution_name)`` resolver roots.

    Declared manifest dependencies come first (``import_id=None``); mapped
    scanned imports fill gaps only for distributions not already declared.
    Non-distributions are filtered out, and each distribution appears once
    (deduped by normalized name).

    ``needed_extras`` TARGETS which ``[project.optional-dependencies]`` /
    ``extras_require`` groups are in scope: a ``kind=="optional_dependency"``
    requirement is only added as a root when its group is a member of
    ``needed_extras``; runtime (non-optional) deps are always included. This
    is the fix for the "uv unions all extras groups" bug — previously every
    optional group was appended as a root with no filter, so mutually
    exclusive groups (e.g. ``cpu``/``gpu``) collided into one unsatisfiable
    resolve. The default (``frozenset()``) is deliberately runtime-only; see
    ``build.py`` for the seam that would eventually source this set from
    discovered CI/tox/Makefile ``pip install -e .[...]`` invocations
    (cluster-1 enrichment, not this task).

    ``target_env`` (Task 8 review fix), when given, additionally drops a
    manifest dependency whose PEP 508 environment marker evaluates False for
    the TARGET (e.g. ``foo ; sys_platform == 'win32'`` on a Linux target) —
    see :func:`_env_marker_excludes` for the conservative "keep unless
    certain" rule. Default ``None`` preserves the pre-Task-8-review behavior
    for every existing caller (``advise.py`` and current tests).
    """
    evidence = collect_python_dependency_evidence(repo_path)
    declared_names = {req.name for req in evidence.declared_dependencies}

    roots: list[tuple[str | None, str]] = []
    seen: set[str] = set()

    # 1. Manifest-declared dependencies (highest trust).
    for manifest_root in select_manifest_roots(
        repo_path,
        needed_extras=needed_extras,
        target_env=target_env,
        target_stdlib_modules=target_stdlib_modules,
    ):
        seen.add(normalize_package_name(manifest_root.name))
        roots.append((None, manifest_root.resolver_token))

    # 2. Scan gap-fill: mapped imports not already covered by a declaration.
    for import_node_id, dist_name in package_roots(graph, declared_names):
        node = graph.get(import_node_id)
        module_name = node.name if node is not None else import_node_id
        if _is_non_distribution(module_name, target_stdlib_modules):
            continue
        normalized = normalize_package_name(dist_name)
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append((import_node_id, dist_name))

    return roots
