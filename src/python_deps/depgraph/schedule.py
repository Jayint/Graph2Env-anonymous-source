"""Pure scheduling layer: select and frame the next actionable obligation.

The DECIDE role's "what & when". Given a host-certified DepGraph, pick the MISSING
obligations whose every dependency is already SATISFIED and that carry a host
check_command (the agent's stop condition), ordered deps-before-dependents.

PURE: must not import from src.envstate. Depends only on the depgraph schema and
emit helpers.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.python_deps.depgraph.schema import (
    DepGraph, DiscoveredBy, EdgeType, Node, NodeType, State,
)
from src.python_deps.depgraph.emit import topo_order
from src.python_deps.depgraph.req_slice import RequirementSlice, build_requirement_slice


def _dependencies_satisfied(graph: DepGraph, node: Node) -> bool:
    """True when every HARD node this one REQUIRES is SATISFIED.

    Soft edges (``Edge.data["hard"] is False``) never block scheduling (invariant #10);
    they are hints/candidates promoted to hard only on runtime/gate failure.
    """
    for edge in graph.edges:
        if (edge.src == node.id and edge.relation is EdgeType.REQUIRES
                and edge.data.get("hard", True)):
            dep = graph.get(edge.dst)
            if dep is None or dep.state is not State.SATISFIED:
                return False
    return True


def _is_actionable(graph: DepGraph, node: Node, *, allow_services: bool = False) -> bool:
    # Lazy import to avoid any circular dependency: schedule.py must stay PURE
    # (no src.envstate imports), and emit.py is a pure sibling in this package.
    from src.python_deps.depgraph.emit import _is_emittable, _conflicted_ids
    service_ok = (
        node.type is not NodeType.SERVICE
        or (allow_services and node.data.get("service_confidence") == "confirmed")
    )
    manager_id = node.data.get("managed_by_dependency_set")
    managed_static_leaf = bool(
        manager_id
        and not node.setup_commands
        and node.discovered_by is not DiscoveredBy.RUNTIME
        and not node.data.get("runtime_confidence")
    )
    return (
        node.state is State.MISSING
        # Let the repository-native provider run first.  A managed package is
        # promoted back into the actionable frontier only when pytest/runtime
        # evidence proves that this exact leaf is required.
        and not managed_static_leaf
        and service_ok
        # In the service-enabled v3 arm, full Test execution belongs to
        # next_decision's anti-hollow callback.  A stale MISSING certificate
        # from the earlier scratch graph must not route the same suite through
        # the generic obligation/repair path before that governed gate runs.
        and not (allow_services and node.type is NodeType.TEST)
        and node.type is not NodeType.DATA_ASSET   # DataAsset (tier 6) is a construction-time
                                                   # advisory Hint/Candidate from the LLM env
                                                   # classifier (Slice C) — never a scheduled
                                                   # obligation (like CONFIG/SERVICE). Its
                                                   # `test -f ...`-style check would otherwise make
                                                   # it actionable; keep classifier soft nodes out
                                                   # of the frontier.
        and (node.type is not NodeType.CONFIG      # config is advisory-only (tier 6); its `printenv X`
                                                   # check is unsatisfiable in a fresh-shell exec, and a
                                                   # genuinely-required var is set reactively via the
                                                   # discover-task (tests certify SUFFICIENT). See
                                                   # docs/.../2026-06-26-unified-executor-loop-delta.md §9.
             or (allow_services and bool(node.data.get("binding"))))  # ...EXCEPT a service-binding
                                                   # CONFIG node (Option B): it carries a REAL psql
                                                   # check_command and `data["binding"]`, so when
                                                   # services are armed it is a genuine, host-certifiable
                                                   # obligation (its REQUIRES service gate still applies
                                                   # via _dependencies_satisfied below).
        and bool(node.check_command)              # the agent needs a host stop condition
        and _dependencies_satisfied(graph, node)
        and not _is_emittable(graph, node, _conflicted_ids(graph))  # deterministic prefix handles these
    )


def scheduler_frontier(graph: DepGraph, *, allow_services: bool = False) -> tuple[Node, ...]:
    """Actionable MISSING obligations, topologically ordered (deps first)."""
    actionable = [n for n in graph.nodes if _is_actionable(graph, n, allow_services=allow_services)]
    if not actionable:
        return ()
    return tuple(topo_order(graph, tuple(actionable)))   # topo_order returns tuple[Node, ...]


@dataclass(frozen=True)
class ObligationPacket:
    """The agent's problem statement for one obligation — assembled from the graph."""
    node_id: str
    node_type: str
    tier: int
    layer: str
    goal: str
    evidence: str
    check_command: str
    depends_on: tuple[str, ...] = ()
    blocks: tuple[str, ...] = ()
    certified_context: tuple[str, ...] = ()
    start_recipe: dict | None = None
    bind_recipe: dict | None = None
    requirement_slice: RequirementSlice | None = None


def frame_obligation(graph: DepGraph, node: Node) -> ObligationPacket:
    depends_on = tuple(
        e.dst for e in graph.edges
        if e.src == node.id and e.relation is EdgeType.REQUIRES
    )
    blocks = tuple(
        e.src for e in graph.edges
        if e.dst == node.id and e.relation is EdgeType.REQUIRES
    )
    certified_context = tuple(n.id for n in graph.nodes if n.state is State.SATISFIED)
    goal = (
        f"Satisfy obligation '{node.name}' ({node.type.value}, tier {node.tier}): "
        f"make the host check `{node.check_command}` succeed."
    )
    return ObligationPacket(
        node_id=node.id,
        node_type=node.type.value,
        tier=node.tier,
        layer=node.layer.value,
        goal=goal,
        evidence=node.evidence or "",
        check_command=node.check_command or "",
        depends_on=depends_on,
        blocks=blocks,
        certified_context=certified_context,
        start_recipe=node.data.get("start_recipe"),
        bind_recipe=node.data.get("bind_recipe"),
        requirement_slice=build_requirement_slice(graph, node),
    )
