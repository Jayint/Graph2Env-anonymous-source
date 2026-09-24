"""Stage 4a — certified Import->Package relink from packages_distributions().

After ``install_closure`` has installed the resolved closure, the container can
report the ground-truth import-name -> distribution map via
``importlib.metadata.packages_distributions()`` (Python 3.10+). This stage uses it
to add CERTIFIED ``Import->Package`` edges that the pre-install heuristic
(``resolve.link_imports_to_packages``) missed — e.g. ``import dateutil`` provided
by dist ``python-dateutil``. Discovery only: it adds edges, never node state.

Pure parser + pure edge builder + thin executor orchestrator (repo immutability:
every "mutation" returns a NEW ``DepGraph``).
"""

from __future__ import annotations

import json

from src.python_deps.depgraph.executor import Executor
from src.python_deps.depgraph.schema import DepGraph, Edge, EdgeType, NodeType, State
from src.python_deps.import_mapping import (
    normalize_package_name,
    top_level_import_name,
)

PACKAGES_DIST_CMD = (
    'python -c "import importlib.metadata, json; '
    'print(json.dumps(importlib.metadata.packages_distributions()))"'
)


def parse_packages_distributions(stdout: str) -> dict[str, list[str]]:
    """Parse the JSON ``{import_name: [dist, ...]}`` map; ``{}`` if malformed."""
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, val in data.items():
        if isinstance(key, str) and isinstance(val, list):
            out[key] = [v for v in val if isinstance(v, str)]
    return out


def import_to_package_edges(
    graph: DepGraph, dist_map: dict[str, list[str]]
) -> list[Edge]:
    """Certified Import->Package edges from a packages_distributions() map.

    Module keys are real names (``PIL``, ``MySQLdb``) so match case-insensitively;
    distribution names match a Package node by canonical (PEP 503) name. A
    namespace import (multiple dists) links to every dist that is present as a
    Package node. Edges already in the graph are skipped (no duplicates).
    """
    pkg_by_canon = {
        normalize_package_name(n.name): n.id
        for n in graph.nodes
        if n.type is NodeType.PACKAGE
    }
    dist_by_module = {module.lower(): dists for module, dists in dist_map.items()}
    existing = {
        (e.src, e.dst) for e in graph.edges if e.relation is EdgeType.REQUIRES
    }

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()
    for node in graph.nodes:
        if node.type is not NodeType.IMPORT:
            continue
        module = top_level_import_name(node.name).lower()
        for dist in dist_by_module.get(module, ()):
            pkg_id = pkg_by_canon.get(normalize_package_name(dist))
            if pkg_id is None:
                continue
            key = (node.id, pkg_id)
            if key in existing or key in seen:
                continue
            seen.add(key)
            edges.append(
                Edge(
                    src=node.id,
                    dst=pkg_id,
                    relation=EdgeType.REQUIRES,
                    origin="certified",
                )
            )
    return edges


def _drop_superseded_ghosts(graph: DepGraph, certified_edges: list[Edge]) -> DepGraph:
    """Remove identity-fallback ghost packages superseded by a certified link.

    When the relink certifies ``import:X -> pkg:realname``, the unresolved
    placeholder ``pkg:X`` (MISSING — the identity-fallback root ``uv`` could not
    resolve) is a phantom duplicate of an already-resolved need, and (pre-fix) a
    trap carrying a ``pip:X`` fix that cannot work. Drop it (and its dangling
    edges) so the graph holds only the real, certified provider. Returns a NEW
    graph.

    The placeholder is identified by ``state is MISSING`` (+ name match + not a
    certified target), NOT by ``version is None``: an unresolvable root can carry
    a version — a ``no version of X==Y`` registry miss, or a sdist that ``Failed
    to build X==Y`` (P0). Those versioned ghosts are equally superseded once the
    real provider is certified, so the drop must not hinge on the version field.
    """
    new = graph
    # Never drop ANY certified target (not just the current edge's) -- a missing
    # placeholder could, pathologically, be another import's certified provider;
    # dropping it would orphan that link.
    protected = {edge.dst for edge in certified_edges}
    for edge in certified_edges:
        imp = new.get(edge.src)
        if imp is None or imp.type is not NodeType.IMPORT:
            continue
        imp_canon = normalize_package_name(top_level_import_name(imp.name))
        for node in list(new.nodes):
            if (
                node.type is NodeType.PACKAGE
                and node.id not in protected
                and node.state is State.MISSING
                and normalize_package_name(node.name) == imp_canon
            ):
                new = new.without_node(node.id)
    return new


def certified_import_links(graph: DepGraph, executor: Executor) -> DepGraph:
    """Stage 4a: add certified Import->Package edges from the container.

    Runs ``packages_distributions()`` in the (post-install) container and links
    every Import to its certified provider Package, then drops any unresolved
    identity-fallback ghost package the certified link supersedes. On command
    failure the graph is returned unchanged — never worse than the pre-install
    heuristic alone.
    """
    result = executor.run(PACKAGES_DIST_CMD)
    if not result.ok:
        return graph
    dist_map = parse_packages_distributions(result.stdout)
    edges = import_to_package_edges(graph, dist_map)
    new = graph
    for edge in edges:
        new = new.with_edge(edge)
    return _drop_superseded_ghosts(new, edges)
