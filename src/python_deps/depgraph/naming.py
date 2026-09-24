"""Stage 2: map ``Import`` nodes to PyPI distribution names.

Realizes design sections 4.2 / 10.2 / 10.3: Python code imports *modules* while
pip installs *distributions*, so each ``Import`` node must be resolved to a
distribution root before the resolver (stage 3) can run.

This module is pure (no Executor, no I/O) and reuses the curated table in
``python_deps.import_mapping``.  Precedence follows the design's ladder (10.3):
declared manifest names outrank the curated table, which outranks the
direct-name identity fallback.
"""

from __future__ import annotations

from src.python_deps.depgraph.schema import DepGraph, NodeType
from src.python_deps.import_mapping import (
    map_import_to_package,
    normalize_package_name,
)


def package_roots(
    graph: DepGraph,
    declared_names: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(import_id, distribution_name)`` for every Import node.

    Resolution per Import node, highest precedence first:

    1. **declared manifest name** — if a declared distribution's normalized
       name equals the import's normalized name, the declared name wins
       (original manifest form preserved). Project declarations are the
       highest-trust evidence (design 4.2 / 10.3).
    2. **curated table** — ``python_deps.import_mapping.map_import_to_package``
       (handles ``cv2 -> opencv-python`` and friends).
    3. **identity fallback** — also via ``map_import_to_package`` for unknowns.

    Non-Import nodes are ignored. Output order follows graph node order, one
    pair per Import node.
    """
    declared = declared_names or set()
    declared_by_normalized = {normalize_package_name(name): name for name in declared}

    roots: list[tuple[str, str]] = []
    for node in graph.nodes:
        if node.type is not NodeType.IMPORT:
            continue
        normalized = normalize_package_name(node.name)
        if normalized in declared_by_normalized:
            roots.append((node.id, declared_by_normalized[normalized]))
            continue
        result = map_import_to_package(node.name, declared_package_names=declared)
        roots.append((node.id, result.package_name))
    return roots
