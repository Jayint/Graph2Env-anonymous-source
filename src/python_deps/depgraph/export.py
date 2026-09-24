"""GraphML export — render a ``DepGraph`` for the existing HTML viewer.

Supersets the key schema of ``docs/sample-dependency-graph.graphml``: the
original keys (``d0..d7`` + ``e0``) are preserved byte-for-byte so the output
still drops straight into ``docs/sample-dependency-graph-visualization.html``,
and the uv-enrichment adds backward-compatible keys the viewer ignores when
unknown:

    node keys: label, type, layer, state, discovered_by, check, fix, evidence,
               build_from_source (d8)
    edge keys: relation (e0), marker (e1), constraint (e2)

``build_from_source`` surfaces native-build risk; ``marker`` carries a
conditional-dependency marker on a ``requires`` edge; ``constraint`` carries the
version bounds on a ``conflicts_with`` edge (so conflicts are diagnosable).
Predicted-vs-observed is already visible via ``discovered_by`` (resolver = a
prediction, probe = an observation).

The ``fix`` value prefers ``chosen_fix`` and falls back to the joined
``fix_candidates`` (the viewer shows a single provider string).  Empty fields are
omitted (matching the sample).  All values are XML-escaped.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from src.python_deps.depgraph.schema import DepGraph, Edge, Node, NodeType

# (key-id, attr.name) — d0..d7 ids/order mirror docs/sample-dependency-graph.graphml;
# d8 is the backward-compatible native-risk addition.
_NODE_KEYS: tuple[tuple[str, str], ...] = (
    ("d0", "label"),
    ("d1", "type"),
    ("d2", "layer"),
    ("d3", "state"),
    ("d4", "discovered_by"),
    ("d5", "check"),
    ("d6", "fix"),
    ("d7", "evidence"),
    ("d8", "build_from_source"),
)
# e0 mirrors the sample; e1/e2 are the backward-compatible enrichment additions.
_EDGE_KEYS: tuple[tuple[str, str], ...] = (
    ("e0", "relation"),
    ("e1", "marker"),
    ("e2", "constraint"),
)


def _label(node: Node) -> str:
    """Viewer label.  Package nodes pin the version (``name==version``) to match
    the design-doc sample render (e.g. ``opencv-python==4.9.0.80``); every other
    node type uses its bare name."""
    if node.type is NodeType.PACKAGE and node.version:
        return f"{node.name}=={node.version}"
    return node.name


def _fix_value(node: Node) -> str:
    if node.chosen_fix:
        return node.chosen_fix
    if node.fix_candidates:
        return "; ".join(node.fix_candidates)
    return ""


def _build_from_source_value(node: Node) -> str:
    """``"true"``/``"false"`` when known, else ``""`` (omitted)."""
    if node.build_from_source is None:
        return ""
    return "true" if node.build_from_source else "false"


def _node_data(node: Node) -> dict[str, str]:
    return {
        "d0": _label(node),
        "d1": node.type.value,
        "d2": node.layer.value,
        "d3": node.state.value,
        "d4": node.discovered_by.value,
        "d5": node.check_command or "",
        "d6": _fix_value(node),
        "d7": node.evidence or "",
        "d8": _build_from_source_value(node),
    }


def _constraint_value(edge: Edge) -> str:
    """Version-bound summary for a conflict edge (empty when absent)."""
    data = edge.data or {}
    src_bound = data.get("src_bound")
    dst_bound = data.get("dst_bound")
    if not src_bound and not dst_bound:
        return ""
    pkg = data.get("package")
    bounds = " vs ".join(b for b in (src_bound, dst_bound) if b)
    return f"{pkg}: {bounds}" if pkg else bounds


def _edge_data(edge: Edge) -> dict[str, str]:
    return {
        "e0": edge.relation.value,
        "e1": edge.marker or "",
        "e2": _constraint_value(edge),
    }


def to_graphml(graph: DepGraph) -> str:
    """Serialize ``graph`` to a GraphML document string (viewer-compatible)."""
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
    ]
    for key_id, attr_name in _NODE_KEYS:
        lines.append(
            f'  <key id="{key_id}" for="node" '
            f'attr.name="{attr_name}" attr.type="string"/>'
        )
    for key_id, attr_name in _EDGE_KEYS:
        lines.append(
            f'  <key id="{key_id}" for="edge" '
            f'attr.name="{attr_name}" attr.type="string"/>'
        )
    lines.append('  <graph edgedefault="directed">')

    for node in graph.nodes:
        lines.append(f"    <node id={quoteattr(node.id)}>")
        data = _node_data(node)
        for key_id, _ in _NODE_KEYS:
            value = data[key_id]
            if value:
                lines.append(f'      <data key="{key_id}">{escape(value)}</data>')
        lines.append("    </node>")

    for edge in graph.edges:
        data = _edge_data(edge)
        parts = [
            f'<data key="{key_id}">{escape(data[key_id])}</data>'
            for key_id, _ in _EDGE_KEYS
            if data[key_id]
        ]
        lines.append(
            f"    <edge source={quoteattr(edge.src)} target={quoteattr(edge.dst)}>"
            f"{''.join(parts)}</edge>"
        )

    lines.append("  </graph>")
    lines.append("</graphml>")
    return "\n".join(lines) + "\n"
