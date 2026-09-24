"""Import-to-package linking: link_imports_to_packages and supporting helpers.

Imports ``_canon`` and ``_req_name`` from resolve_lock (lowest-level module).
No imports from resolve_errors or resolve.py — dependency is one-way.
"""

from __future__ import annotations

from dataclasses import replace

from src.python_deps.depgraph.schema import (
    DepGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
)
from src.python_deps.import_mapping import map_import_to_package
from src.python_deps.depgraph.resolve_lock import _canon, _req_name


def _stamp(
    node: Node,
    risk: dict[str, dict],
    target_python: str,
    target_platform: str,
    exclude_newer: str | None = None,
) -> Node:
    """Stamp targeting provenance + native-build risk onto a Package node."""
    changes: dict = {
        "resolved_python": target_python,
        "resolved_platform": target_platform,
        "exclude_newer": exclude_newer,
    }
    info = risk.get(node.name) or risk.get(_canon(node.name))
    if info is None:
        # Case/separator-insensitive fallback.
        for key, val in risk.items():
            if _canon(key) == _canon(node.name):
                info = val
                break
    if info is not None:
        changes["build_from_source"] = info.get("build_from_source")
        changes["artifact"] = info.get("artifact")
        changes["hash"] = info.get("hash")
    return replace(node, **changes)


def _import_edges(
    roots: list[tuple[str | None, str]],
    nodes: list[Node],
) -> list[Edge]:
    """Import->Package edges for each root with a resolved Package node."""
    canon_to_id = {_canon(n.name): n.id for n in nodes}
    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for import_id_, dist in roots:
        if import_id_ is None:
            continue  # manifest-declared root: no Import node to attach.
        pkg_id = canon_to_id.get(_canon(_req_name(dist)))
        if pkg_id is None:
            continue
        key = (import_id_, pkg_id)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            Edge(
                src=import_id_,
                dst=pkg_id,
                relation=EdgeType.REQUIRES,
                origin="resolver",
            )
        )
    return edges


def link_imports_to_packages(graph: DepGraph) -> DepGraph:
    """Connect every Import node to its resolved Package by canonical dist name.

    Complements :func:`_import_edges`, which only links imports that were
    themselves resolver roots.  A manifest-declared dependency seeds a Package via
    a root with ``import_id=None`` (see ``roots.select_roots``), so its scanned
    Import node would otherwise be orphaned from the Package — breaking the
    symptom->owner walk.  This pass links any Import whose mapped distribution
    matches a Package node, regardless of how the root was sourced.  ``_canon``
    collapses ``_``/``-``/``.`` so e.g. ``charset_normalizer`` matches
    ``charset-normalizer`` even via the identity fallback.
    """
    canon_to_pkg = {
        _canon(n.name): n.id for n in graph.nodes if n.type is NodeType.PACKAGE
    }
    existing = {
        (e.src, e.dst) for e in graph.edges if e.relation is EdgeType.REQUIRES
    }
    new = graph
    for node in graph.nodes:
        if node.type is not NodeType.IMPORT:
            continue
        dist = map_import_to_package(node.name).package_name
        pkg_id = canon_to_pkg.get(_canon(dist))
        if pkg_id is None or (node.id, pkg_id) in existing:
            continue
        new = new.with_edge(
            Edge(
                src=node.id,
                dst=pkg_id,
                relation=EdgeType.REQUIRES,
                origin="reconcile",
            )
        )
    return new


def _merge(
    primary_nodes: list[Node],
    primary_edges: list[Edge],
    extra_nodes: list[Node],
    extra_edges: list[Edge],
) -> tuple[list[Node], list[Edge]]:
    """Merge node/edge lists; primary entries win on id/edge-key collisions."""
    nodes: list[Node] = list(primary_nodes)
    have_ids = {n.id for n in nodes}
    for n in extra_nodes:
        if n.id not in have_ids:
            have_ids.add(n.id)
            nodes.append(n)

    edges: list[Edge] = list(primary_edges)
    have_keys = {e.key() for e in edges}
    for e in extra_edges:
        if e.key() not in have_keys:
            have_keys.add(e.key())
            edges.append(e)
    return nodes, edges
