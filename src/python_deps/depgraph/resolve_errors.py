"""Resolver-error parsing: error dataclasses, parse_resolver_error, diagnosis -> graph.

Imports ``_canon`` and ``_package_node`` from resolve_lock (lowest-level module).
No imports from resolve_link or resolve.py — dependency is one-way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.python_deps.depgraph.schema import (
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
)
from src.python_deps.depgraph.resolve_lock import _canon, _package_node

# --------------------------------------------------------------------------- #
# Pure parser 3: failed ``uv lock`` stderr -> structured diagnosis.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MissingPackage:
    name: str
    version: str | None
    evidence: str


@dataclass(frozen=True)
class VersionConstraint:
    package: str  # the constrained package
    specifier: str  # e.g. "<2.0" or ">=2.0"
    imposed_by: str | None  # the package imposing it, or None for the root


@dataclass(frozen=True)
class VersionConflict:
    package: str  # the package under conflict
    left: VersionConstraint
    right: VersionConstraint
    evidence: str


@dataclass(frozen=True)
class PythonIncompat:
    floor: str  # e.g. ">=3.11"
    evidence: str
    imposer: str | None = None  # the package that requires the floor, if known


@dataclass(frozen=True)
class ResolverDiagnosis:
    missing: tuple[MissingPackage, ...] = ()
    constraints: tuple[VersionConstraint, ...] = ()
    conflicts: tuple[VersionConflict, ...] = ()
    python_incompat: PythonIncompat | None = None
    raw: str = ""


# uv wraps long diagnostic lines, so inter-token whitespace may be a newline +
# indent rather than a single space — match runs of whitespace, not literal spaces.
# (Distribution names are never split: uv only wraps at whitespace boundaries.)
_REGISTRY_MISS_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)\s+(?:was|were)\s+not\s+found\s+in\s+the\s+"
    r"(?:package\s+)?registry"
)
_NO_VERSION_RE = re.compile(
    r"there\s+(?:is|are)\s+no\s+versions?\s+of\s+([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s,)]+)"
)
_NO_VERSION_PLAIN_RE = re.compile(
    r"there\s+(?:is|are)\s+no\s+versions?\s+of\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?![=\w.-])"
)
_YOU_REQUIRE_RE = re.compile(
    r"you require ([A-Za-z0-9][A-Za-z0-9._-]*)\s*([<>=!~]=?[0-9][^\s,)]*)"
)
_DEPENDS_ON_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:==[^\s]+)? depends on "
    r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*([<>=!~]=?[0-9][^\s,)]*)"
)
_PY_INCOMPAT_RE = re.compile(
    r"(?:does not satisfy Python|requires Python)\s*(>=?\s*[0-9][0-9.]*)"
)
# The package imposing a python floor: ``X==1.0 requires Python>=3.12`` (form B,
# imposer before the clause) or ``... and you require X`` (form A, after it).
_PY_IMPOSER_BEFORE_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:==[^\s]+)?\s+requires Python"
)
_PY_IMPOSER_AFTER_RE = re.compile(
    r"and you require ([A-Za-z0-9][A-Za-z0-9._-]*)"
)
# An sdist whose metadata BUILD fails (old/broken setup.py, missing build deps).
# uv prints ``× Failed to build `factory==1.2```; this is NOT a version conflict
# or a registry miss, so without recognizing it the diagnosis is empty and the
# drop-retry loop can never attribute/drop the offending root -> whole-closure
# collapse (P0). The distribution name is never line-wrapped (uv wraps only at
# whitespace, and there is none inside ``name==version``).
_BUILD_FAILURE_RE = re.compile(
    r"Failed to build\s+`?([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s`]+)"
)
# A root with NO usable version: every release yanked ("all versions of X cannot
# be used"), or a requires-python the target lacks ("X==V cannot be used"). uv's
# universal conclusion for a fundamentally-unsatisfiable package is "<dist>[==V]
# cannot be used"; it is neither a registry miss, a build failure, nor a version
# tug-of-war (which says "unsatisfiable", never "cannot be used"), so it was
# previously unattributable -> whole-closure collapse (RATBench mcp-atlassian +
# docling). uv WRAPS "cannot\n      be used", so match whitespace runs inside the
# phrase too (same reason the other diagnostic regexes use ``\s+``).
_UNUSABLE_RE = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9._-]*)(?:==[^\s]+)?\s+cannot\s+be\s+used"
)


def _excerpt(text: str, needle_start: int, width: int = 200) -> str:
    """A short evidence window around ``needle_start`` (single-lined)."""
    lo = max(0, needle_start - 20)
    snippet = text[lo : lo + width].strip()
    return " ".join(snippet.split())


def parse_resolver_error(stderr: str) -> ResolverDiagnosis:
    """Structure a failed ``uv lock`` stderr into a :class:`ResolverDiagnosis`.

    Recognizes registry misses, no-such-version, version conflicts (root and
    transitive bounds on a shared package), and python-version incompatibility.
    """
    text = stderr or ""

    missing: list[MissingPackage] = []
    seen_missing: set[str] = set()

    def _add_missing(name: str, version: str | None, start: int) -> None:
        key = _canon(name)
        if key in seen_missing:
            return
        seen_missing.add(key)
        missing.append(MissingPackage(name, version, _excerpt(text, start)))

    for m in _REGISTRY_MISS_RE.finditer(text):
        _add_missing(m.group(1), None, m.start())
    for m in _NO_VERSION_RE.finditer(text):
        _add_missing(m.group(1), m.group(2), m.start())
    for m in _NO_VERSION_PLAIN_RE.finditer(text):
        _add_missing(m.group(1), None, m.start())
    # A build failure makes the root unresolvable in this environment; surface it
    # as a missing package (with the build error as evidence) so the drop-retry
    # loop attributes and drops it instead of collapsing the entire closure.
    for m in _BUILD_FAILURE_RE.finditer(text):
        _add_missing(m.group(1), m.group(2), m.start())
    # A package uv concludes "cannot be used" (all-yanked, or requires a Python the
    # target lacks) is likewise unresolvable; attribute it so its root is dropped.
    for m in _UNUSABLE_RE.finditer(text):
        _add_missing(m.group(1), None, m.start())

    constraints: list[VersionConstraint] = []
    for m in _YOU_REQUIRE_RE.finditer(text):
        constraints.append(VersionConstraint(m.group(1), m.group(2), None))
    for m in _DEPENDS_ON_RE.finditer(text):
        constraints.append(VersionConstraint(m.group(2), m.group(3), m.group(1)))

    # A conflict = a single package carrying >=2 distinct specifiers.
    by_pkg: dict[str, list[VersionConstraint]] = {}
    for c in constraints:
        by_pkg.setdefault(_canon(c.package), []).append(c)
    conflicts: list[VersionConflict] = []
    for cons in by_pkg.values():
        specs = {c.specifier for c in cons}
        if len(cons) >= 2 and len(specs) >= 2:
            # Pick two constraints with DISTINCT specifiers (uv may restate the
            # same bound several times across wrapped lines, so cons[0]/cons[1]
            # are not reliably the two conflicting bounds).  Prefer a pair whose
            # imposers also differ so the edge joins two real distributions.
            left = cons[0]
            right = next(
                (c for c in cons[1:] if c.specifier != left.specifier
                 and _real_imposer(c.imposed_by) != _real_imposer(left.imposed_by)),
                None,
            )
            if right is None:
                right = next(c for c in cons[1:] if c.specifier != left.specifier)
            conflicts.append(
                VersionConflict(
                    package=left.package,
                    left=left,
                    right=right,
                    evidence=_excerpt(text, text.find(left.package)),
                )
            )

    python_incompat: PythonIncompat | None = None
    py = _PY_INCOMPAT_RE.search(text)
    if py:
        floor = re.sub(r"\s+", "", py.group(1))
        before = _PY_IMPOSER_BEFORE_RE.search(text)
        after = _PY_IMPOSER_AFTER_RE.search(text)
        imposer = before.group(1) if before else (after.group(1) if after else None)
        python_incompat = PythonIncompat(floor, _excerpt(text, py.start()), imposer)

    return ResolverDiagnosis(
        missing=tuple(missing),
        constraints=tuple(constraints),
        conflicts=tuple(conflicts),
        python_incompat=python_incompat,
        raw=text,
    )


# --------------------------------------------------------------------------- #
# Diagnosis -> graph (missing/conflict nodes + conflicts_with edges).
# --------------------------------------------------------------------------- #
def _missing_package_node(name: str, version: str | None, evidence: str) -> Node:
    return replace(
        _package_node(name, version, provenance="uv lock (unresolved)", resolvable=False),
        state=State.MISSING,
        evidence=evidence,
    )


def _conflict_package_node(name: str, evidence: str) -> Node:
    return replace(
        _package_node(name, None, provenance="uv lock (conflict)", resolvable=False),
        state=State.UNKNOWN,
        evidence=evidence,
    )


# Synthetic resolve-root labels uv uses for the root requirements ("your project
# depends on ...", "the root project ...").  These are NOT real distributions and
# must never leak in as Package nodes; treat them like the unnamed root (None).
_ROOT_IMPOSER_NAMES: frozenset[str] = frozenset({"project", "root"})


def _real_imposer(imposer: str | None) -> str | None:
    """An imposing distribution name, or ``None`` for the synthetic resolve root."""
    if imposer is None or _canon(imposer) in _ROOT_IMPOSER_NAMES:
        return None
    return imposer


def _conflict_endpoint_node(name: str, c: VersionConflict) -> Node:
    """A Package node for a conflict endpoint.

    The shared package (``c.package``) has no version satisfying every bound, so it
    is ``MISSING``; an imposing distribution resolves fine on its own and stays
    ``UNKNOWN`` (only the *combination* is unsatisfiable).
    """
    if _canon(name) == _canon(c.package):
        return _missing_package_node(name, None, c.evidence)
    return _conflict_package_node(name, c.evidence)


def _conflict_endpoints(c: VersionConflict) -> tuple[str | None, str | None]:
    """Two distinct Package names to join with a conflicts_with edge.

    A synthetic-root imposer (``None``/"your project") falls back to the shared
    package, so a "package P needs D>=x but the root pins D<x" conflict joins the
    real P and the shared D (never the synthetic root).
    """
    a = _real_imposer(c.left.imposed_by) or c.package
    b = _real_imposer(c.right.imposed_by) or c.package
    if a == b:
        imposers = [
            x
            for x in (_real_imposer(c.left.imposed_by), _real_imposer(c.right.imposed_by))
            if x
        ]
        if not imposers:
            return None, None
        a, b = imposers[0], c.package
        if a == b:
            return None, None
    return a, b


def _diagnosis_to_graph(diag: ResolverDiagnosis) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    seen_ids: set[str] = set()

    def _add(node: Node) -> Node:
        if node.id not in seen_ids:
            seen_ids.add(node.id)
            nodes.append(node)
        return node

    edges: list[Edge] = []

    for m in diag.missing:
        _add(_missing_package_node(m.name, m.version, m.evidence))

    for c in diag.conflicts:
        a, b = _conflict_endpoints(c)
        if a is None or b is None:
            continue
        na = _add(_conflict_endpoint_node(a, c))
        nb = _add(_conflict_endpoint_node(b, c))
        edges.append(
            Edge(
                src=na.id,
                dst=nb.id,
                relation=EdgeType.CONFLICTS_WITH,
                origin="resolver",
                data={
                    "package": c.package,
                    "src_bound": c.left.specifier,
                    "dst_bound": c.right.specifier,
                    "evidence": c.evidence,
                },
            )
        )

    if diag.python_incompat is not None:
        incompat = diag.python_incompat
        floor = incompat.floor
        py_node = _add(
            Node(
                id="pkg:python",
                type=NodeType.PACKAGE,
                name="python",
                layer=Layer.INTERPRETER,
                discovered_by=DiscoveredBy.RESOLVER,
                version=floor,
                state=State.MISSING,
                evidence=incompat.evidence,
                provenance="uv lock (python incompat)",
            )
        )
        # Conflict edge to the interpreter need with the floor (spec §"Conflict/
        # failure → graph"): the imposing package conflicts_with the interpreter.
        if incompat.imposer and _canon(incompat.imposer) != _canon(py_node.name):
            imposer_node = _add(
                _conflict_package_node(incompat.imposer, incompat.evidence)
            )
            edges.append(
                Edge(
                    src=imposer_node.id,
                    dst=py_node.id,
                    relation=EdgeType.CONFLICTS_WITH,
                    origin="resolver",
                    data={
                        "package": "python",
                        "floor": floor,
                        "dst_bound": floor,
                        "evidence": incompat.evidence,
                    },
                )
            )

    return nodes, edges


def _offending_root_names(diag: ResolverDiagnosis, current_root_names: set[str]) -> set[str]:
    """Canonical names of ROOTS to drop for a retry. Missing packages are always
    dropped. For a version conflict: if the conflicted package is itself a root
    (a direct pin), drop it and KEEP the imposers (avoids collapsing both
    subtrees). If it is purely transitive (not a current root), dropping its
    name would not shrink the root set, so drop one imposer that is a current
    root instead so the retry can still make progress and the other subtree
    survives. The conflict is recorded as an advisory edge either way."""
    names: set[str] = {_canon(m.name) for m in diag.missing}
    for c in diag.conflicts:
        pkg = _canon(c.package)
        if pkg in current_root_names:
            names.add(pkg)
        else:
            # Prefer an imposer that is itself a current root — dropping a
            # non-root (transitive) imposer cannot shrink the root set, so the
            # retry would make no progress. If neither immediate imposer is a
            # current root, the diagnosis (immediate imposers only) can't name a
            # droppable root; add nothing and let the retry fall through.
            root_imposers = sorted(
                _canon(i)
                for i in (_real_imposer(c.left.imposed_by), _real_imposer(c.right.imposed_by))
                if i and _canon(i) in current_root_names
            )
            if root_imposers:
                names.add(root_imposers[0])
    return names
