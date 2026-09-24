"""Wheel-oracle prior — a build-essential Tool for every from-source package.

Realizes construction-enrichment cluster 1a (design 2026-07-01): the
resolver's own wheel-vs-sdist signal (``Node.build_from_source``, computed by
``wheel_oracle.risk_from_packages``) is the ONLY basis for this prediction —
no curated package->syslib table. A package with no compatible wheel needs a
compiler to build its sdist; that is the one thing this stage predicts.

This REPLACES the deleted ``seed_predicted_native`` / ``PACKAGE_TO_SYSTEM_DEPS``
path. Specific ``-dev`` headers (psycopg2->libpq-dev, Pillow->libjpeg-dev) are
no longer predicted from a table. They are recovered by: ``install_closure``
parsing the real build error (stage 4), or ``ldd_probe`` for runtime libs
(stage 4.5) — an expected coverage tradeoff, see the design doc's "What this
loses, honestly" and Risk #2. (Declaration mining, cluster 1b, is deferred —
not part of this plan.)

Pure: every "mutation" returns a NEW ``DepGraph`` (repo immutability rule).
"""

from __future__ import annotations

from src.python_deps.depgraph.ids import tool_id
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

_BUILD_ESSENTIAL_APT = "build-essential"
_BUILD_ESSENTIAL_ID = tool_id(_BUILD_ESSENTIAL_APT)


def _build_essential_node() -> Node:
    fix = f"apt:{_BUILD_ESSENTIAL_APT}"
    return Node(
        id=_BUILD_ESSENTIAL_ID,
        type=NodeType.TOOL,
        name=_BUILD_ESSENTIAL_APT,
        layer=Layer.TOOLCHAIN,
        discovered_by=DiscoveredBy.RESOLVER,
        state=State.UNKNOWN,
        check_command=f"dpkg -s {_BUILD_ESSENTIAL_APT}",
        fix_candidates=(fix,),
        chosen_fix=fix,
        provenance="wheel-oracle (build_from_source)",
    )


def seed_wheel_oracle_prior(graph: DepGraph) -> DepGraph:
    """Emit ONE ``tool:build-essential`` node for every from-source or unknown Package.

    For each ``Package`` with ``build_from_source`` not False (i.e., True or None —
    known from-source builds or unknown build mode from degraded resolution), add a
    ``requires`` edge to the single, deduped ``build-essential`` Tool node (created
    once, on first need). Only packages with a confirmed matching wheel
    (``build_from_source=False``) skip the prediction. Unresolved diagnostic packages
    (``version=None``, representing resolver failures) are always excluded. Returns
    a NEW graph; a no-op when no package needs a source build.
    """
    new = graph
    packages = [
        n for n in graph.nodes
        if n.type is NodeType.PACKAGE and n.version and n.build_from_source is not False
    ]
    if not packages:
        return new
    if new.get(_BUILD_ESSENTIAL_ID) is None:
        new = new.with_node(_build_essential_node())
    for pkg in packages:
        new = new.with_edge(
            Edge(src=pkg.id, dst=_BUILD_ESSENTIAL_ID, relation=EdgeType.REQUIRES, origin="resolver")
        )
    return new
