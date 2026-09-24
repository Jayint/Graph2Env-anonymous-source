"""Compile a certified DepGraph's emittable wave into annotated, one-action-per-block
script blocks (design §6). Pure: no Docker, no network, no LLM."""
from __future__ import annotations

from dataclasses import dataclass

from src.python_deps.depgraph.emit import partition, topo_order, _apt_name, _pip_spec, _is_reciped
from src.python_deps.depgraph.schema import DepGraph, Node, NodeType


@dataclass(frozen=True)
class Block:
    block_id: str
    wave: str                              # node.layer.value
    commands: tuple[str, ...]
    target_node_ids: tuple[str, ...]
    provider_ids: tuple[str, ...] = ()
    check_commands: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    mutates_env: bool = True
    can_batch: bool = False                # v1: one action per block (defer batching to v2)


def _command_for(node: Node) -> str:
    apt = _apt_name(node)
    if apt is not None:
        # Self-sufficient block: a fresh base image ships empty apt lists, so the
        # block MUST refresh metadata before install or it fails (exit 100 "Unable
        # to locate package"). Keeping update+install in one block preserves the
        # one-action-per-block replay contract and matches emit.build_recipe.
        return f"apt-get update && apt-get install -y --no-install-recommends {apt}"
    if node.type is NodeType.PACKAGE:
        return f"python3 -m pip install --break-system-packages {_pip_spec(node)}"
    if node.type is NodeType.PROJECT:
        return node.chosen_fix or ""
    # Fallback: a node with an explicit chosen_fix that is not apt: (e.g. a shell recipe).
    return node.chosen_fix or ""


def _block_id_for(node: Node) -> str:
    short = node.id.split(":", 1)[-1]
    return f"{node.layer.value}.{short}"


def _block_for(node: Node) -> Block | None:
    """Build the one-action block for an installable node, or None if it has no command."""
    if (
        node.setup_commands
        and node.data.get("provider_backed") is True
        and node.data.get("asset_kind") in {"browser_binary", "config_template", "runtime_env"}
    ):
        return Block(
            block_id=_block_id_for(node),
            wave=node.layer.value,
            commands=node.setup_commands,
            target_node_ids=(node.id,),
            provider_ids=(node.chosen_fix,) if node.chosen_fix else (),
            check_commands=(node.check_command,) if node.check_command else (),
            evidence_refs=(node.evidence,) if node.evidence else (),
        )
    cmd = _command_for(node)
    if not cmd:
        return None
    apt = _apt_name(node)
    return Block(
        block_id=_block_id_for(node),
        wave=node.layer.value,
        commands=(cmd,),
        target_node_ids=(node.id,),
        provider_ids=(node.chosen_fix,) if apt is not None else (),
        check_commands=(node.check_command,) if node.check_command else (),
    )


def compile_blocks(graph: DepGraph) -> tuple[Block, ...]:
    """Emit-phase compile: ONLY the emittable wave (partition().emittable = MISSING nodes
    whose deps are satisfied). Used by the live block-emit phase."""
    if graph is None:
        return ()
    ready = topo_order(graph, partition(graph).emittable)
    return tuple(b for n in ready if (b := _block_for(n)) is not None)


def compile_replay_blocks(graph: DepGraph) -> tuple[Block, ...]:
    """Artifact/replay compile: one block per installable (_is_reciped) node, in topo
    order, REGARDLESS of state. Reproduces the certified environment on a fresh
    container — so SATISFIED nodes ARE included (unlike compile_blocks). Pure."""
    if graph is None:
        return ()
    installable = tuple(n for n in graph.nodes if _is_reciped(n))
    ready = topo_order(graph, installable)
    return tuple(b for n in ready if (b := _block_for(n)) is not None)
