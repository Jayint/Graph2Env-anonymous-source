"""Project a certified DepGraph into one whole, install-only setup.sh artifact
(design 2026-06-29). Pure: no Docker, no network, no LLM, no src.envstate.

Distinct from script.render_setup_sh (the live block-stepped, round-trippable
format): this renderer hoists shared setup and adds tier section headers, so it
is intentionally NOT parseable back to one-block-per-node.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.python_deps.depgraph.block import Block

from src.python_deps.depgraph.certify import EXECUTION_LAYER_ORDER
from src.python_deps.depgraph.emit import _is_reciped, _apt_name, topo_order
from src.python_deps.depgraph.populate import populate_setup_commands
from src.python_deps.depgraph.schema import DepGraph, Layer, Node, NodeType

_BANNER = (
    "#!/usr/bin/env bash",
    "#",
    "# setup.sh — COMPILED from the certified dependency graph. DO NOT EDIT.",
    "# Edit the graph and re-render; this file is an artifact, not a source.",
    "#",
)
# Shared with certify's execution-layer walk so the rendered artifact's section
# order never contradicts host certification order; any Layer not part of the
# certified walk (e.g. SERVICES) is appended so no section is silently dropped.
_LAYER_ORDER: tuple[Layer, ...] = EXECUTION_LAYER_ORDER + tuple(
    L for L in Layer if L not in EXECUTION_LAYER_ORDER
)


def _section_header(layer: Layer) -> str:
    label = layer.value.upper()
    return f"# ==================== {label} ===================="


def _check_annotation(command: str) -> str:
    """Render check metadata as exactly one shell-comment line.

    Host checks are executed separately and may legitimately contain a
    multi-line quoted payload.  The compiled setup script must never let a
    continuation line escape the comment and become an install command.
    """
    one_line = (command or "").replace("\r", r"\r").replace("\n", r"\n")
    return f"#@check {one_line}"



def _annotation(graph: DepGraph, node: Node) -> list[str]:
    from src.python_deps.depgraph.advise import _best_evidence_line  # lazy: avoid load-order coupling
    toks = [f"#@node {node.id}"]
    if node.ecosystem is not None:
        toks.append(f"ecosystem={node.ecosystem.value}")
    if node.workspace:
        toks.append(f"workspace={node.workspace}")
    if node.package_manager:
        toks.append(f"manager={node.package_manager}")
    if node.version:
        toks.append(f"version={node.version}")
    if _apt_name(node) is not None:
        toks.append(f"provider={node.chosen_fix}")
    reqs = [d.id for d in graph.requires_of(node.id) if _is_reciped(d)]
    toks.append("requires=" + (",".join(sorted(reqs)) if reqs else "-"))
    unblocks = sorted(n.id for n in graph.required_by(node.id) if _is_reciped(n))
    if unblocks:
        toks.append("unblocks=" + ",".join(unblocks))
    if node.build_from_source:
        toks.append("build-from-source")
    if node.layer is Layer.TOOLCHAIN:
        toks.append("toolchain")
    ev = _best_evidence_line(node.evidence)
    if ev:
        toks.append(f"evidence={ev}")
    out = ["  ".join(toks)]
    if node.check_command:
        out.append(_check_annotation(node.check_command))
    return out


def _node_block(graph: DepGraph, node: Node, apt_done: list[bool]) -> list[str]:
    out: list[str] = []
    if _apt_name(node) is not None and not apt_done[0]:
        out += ["export DEBIAN_FRONTEND=noninteractive", "apt-get update"]
        apt_done[0] = True
    out += _annotation(graph, node)
    if node.setup_commands:
        # Fresh replay executes the complete artifact in one shell.  Isolate
        # each graph block so a workspace ``cd`` cannot leak into later nodes.
        out += ["("]
        out += list(node.setup_commands)
        out += [")"]
    return out


def _reciped_topologically(graph: DepGraph) -> tuple[Node, ...]:
    """Return every executable graph node in one dependency-first order.

    A layer is a useful presentation and certification abstraction, but it is
    not a substitute for a graph edge.  A local ``PROJECT`` may live in the
    ``pip`` layer while its repository-native ``DEPENDENCY_SET`` provider lives
    in ``dependencies``.  Rendering one complete section per layer would then
    run the project's marker check before the provider block despite the
    explicit ``project -> dependency-set`` requires edge.

    The graph-wide topological order keeps replay faithful to cross-layer
    dependencies while preserving :func:`topo_order`'s deterministic ties.
    """
    nodes = tuple(n for n in graph.nodes if _is_reciped(n))
    return topo_order(graph, nodes)


_NEED_TYPES: tuple[NodeType, ...] = (NodeType.CONFIG, NodeType.SERVICE, NodeType.DATA_ASSET)


def _need_block(graph: DepGraph, node: Node) -> list[str]:
    from src.python_deps.depgraph.advise import _best_evidence_line  # lazy: avoid load-order coupling
    reqs = [d.id for d in graph.requires_of(node.id) if _is_reciped(d)]
    head = f"#@need {node.id}  state={node.state.value}"
    if reqs:
        head += "  requires=" + ",".join(sorted(reqs))
    out = ["#", head]
    if node.check_command:
        out.append(_check_annotation(node.check_command))
    ev = _best_evidence_line(node.evidence)
    if ev:
        out.append(f"#@evidence {ev}")
    out.append("#     (no command — propose a governed block to satisfy this)")
    return out


def _need_in_layer(graph: DepGraph, layer: Layer, covered: set[str]) -> list[Node]:
    nodes = [n for n in graph.nodes
             if n.layer is layer and n.type in _NEED_TYPES
             and not _is_reciped(n) and n.id not in covered]
    return sorted(nodes, key=lambda n: n.id)


def _block_block(block: Block) -> list[str]:
    head = f"#@block {block.block_id}  source=llm-patch"
    if block.target_node_ids:
        head += "  targets=" + ",".join(block.target_node_ids)
    if block.evidence_refs:
        head += "  evidence=" + ",".join(block.evidence_refs)
    out = [head]
    for chk in block.check_commands:
        out.append(_check_annotation(chk))
    if block.commands:
        out.append("(")
        out.extend(block.commands)
        out.append(")")
    return out


def _graph_hash(graph: DepGraph) -> str:
    reciped_ids = {n.id for n in graph.nodes if _is_reciped(n)}
    nodes_payload = sorted(
        (n.id, n.version or "", n.chosen_fix or "", tuple(n.setup_commands))
        for n in graph.nodes if _is_reciped(n)
    )
    edges_payload = sorted(
        (e.src, e.dst, e.relation.value)
        for e in graph.edges
        if e.src in reciped_ids and e.dst in reciped_ids
    )
    blob = json.dumps({"nodes": nodes_payload, "edges": edges_payload},
                      separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def _closure_meta(graph: DepGraph) -> dict[str, str]:
    meta: dict[str, str] = {}
    for n in sorted((n for n in graph.nodes if n.type is NodeType.PACKAGE),
                    key=lambda n: n.id):
        for key, attr in (("python", "resolved_python"),
                          ("platform", "resolved_platform"),
                          ("exclude-newer", "exclude_newer")):
            val = getattr(n, attr, None)
            if val and key not in meta:
                meta[key] = val
    return meta


_TYPE_WORD = {
    NodeType.SYSTEM_LIB: "system",
    NodeType.TOOL: "toolchain",
    NodeType.RUNTIME: "runtime",
    NodeType.PACKAGE: "pip",
    NodeType.DEPENDENCY_SET: "dependencies",
    NodeType.PROJECT: "project",
}
_NEED_WORD = {NodeType.SERVICE: "service", NodeType.CONFIG: "config",
              NodeType.DATA_ASSET: "data_asset"}


def _manifest(graph: DepGraph, manual_blocks) -> list[str]:
    reciped = [n for n in graph.nodes if _is_reciped(n)]
    covered = {nid for b in manual_blocks for nid in b.target_node_ids}
    needs = [n for n in graph.nodes
             if n.type in _NEED_TYPES and not _is_reciped(n) and n.id not in covered]
    counts = Counter(_TYPE_WORD.get(n.type, n.type.value) for n in reciped)
    count_str = ", ".join(
        f"{counts[w]} {w}"
        for w in ("system", "runtime", "toolchain", "pip", "dependencies", "project")
                          if counts.get(w))
    need_counts = Counter(_NEED_WORD.get(n.type, n.type.value) for n in needs)
    need_str = ", ".join(f"{need_counts[w]} {w}"
                         for w in ("service", "config", "data_asset")
                         if need_counts.get(w))
    needs_suffix = f" ({need_str})" if need_str else ""
    meta = _closure_meta(graph)
    meta_str = "   ".join(f"{k}: {v}" for k, v in meta.items())
    lines = list(_BANNER)  # full banner; _BANNER[-1] is the "#" separator (keep it)
    lines.append(f"#   nodes: {len(reciped)} reciped ({count_str or 'none'}) "
                 f"+ {len(needs)} needs{needs_suffix}")
    hash_line = f"#   graph-hash: {_graph_hash(graph)}"
    if meta_str:
        hash_line += "   " + meta_str
    lines.append(hash_line)
    lines.append("#")
    return lines


def render_build_script(graph: DepGraph | None, manual_blocks: tuple[Block, ...] = ()) -> str:
    if graph is None:
        graph = DepGraph()
    graph = populate_setup_commands(graph)  # single call site: derive commands, then emit
    parts: list[str] = _manifest(graph, manual_blocks) + [
        "set -Eeuo pipefail",
        # Make the replay artifact itself resilient to short-lived index/network
        # failures.  These are pip's native bounded retries; real dependency or
        # build failures still return non-zero and remain binding.
        "export PIP_RETRIES=${PIP_RETRIES:-10}",
        "export PIP_TIMEOUT=${PIP_TIMEOUT:-120}",
        "export PIP_DISABLE_PIP_VERSION_CHECK=${PIP_DISABLE_PIP_VERSION_CHECK:-1}",
    ]
    covered = {nid for b in manual_blocks for nid in b.target_node_ids}
    blocks_by_wave: dict[str, list] = {}
    for b in manual_blocks:
        blocks_by_wave.setdefault(b.wave, []).append(b)
    apt_done = [False]

    # Executable graph blocks must obey one global topology.  Headers are only
    # presentation, so a layer can appear more than once for a valid
    # cross-layer dependency chain.
    current_layer: Layer | None = None
    for node in _reciped_topologically(graph):
        if node.layer is not current_layer:
            parts.append("")
            parts.append(_section_header(node.layer))
            current_layer = node.layer
        parts.extend(_node_block(graph, node, apt_done))

    # Manual blocks and advisory needs are not executable graph nodes. Keep
    # their declared waves after the compiled graph blocks, as before.
    for layer in _LAYER_ORDER:
        section: list[str] = []
        for b in blocks_by_wave.get(layer.value, ()):
            section += _block_block(b)
        for node in _need_in_layer(graph, layer, covered):
            section += _need_block(graph, node)
        if section:
            parts.append("")
            parts.append(_section_header(layer))
            parts.extend(section)
    # Fail-fast: PatchGate (Phase 1) rejects illegal waves, so any manual block whose
    # wave is not a Layer value is a programming error, not user input — never silently
    # render it into an UNSCHEDULED section.
    known_waves = {layer.value for layer in _LAYER_ORDER}
    illegal = [b.block_id for b in manual_blocks if b.wave not in known_waves]
    if illegal:
        raise ValueError(f"render_build_script: manual blocks have illegal waves "
                         f"(not a Layer value): {illegal}")
    return "\n".join(parts) + "\n"
