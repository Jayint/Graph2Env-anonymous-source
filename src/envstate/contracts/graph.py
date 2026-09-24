"""Immutable ContractGraph container + status projection + traversal."""
from __future__ import annotations

import dataclasses
from typing import Optional

from .nodes import Edge, Node, edge_from_dict, edge_to_dict, node_from_dict, node_to_dict


@dataclasses.dataclass(frozen=True)
class ContractGraph:
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    diagnostic_notes: tuple[str, ...] = ()

    @staticmethod
    def empty() -> "ContractGraph":
        return ContractGraph()

    def node(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def has_node(self, node_id: str) -> bool:
        return self.node(node_id) is not None

    def active_nodes(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if not n.invalidated)

    def nodes_by_type(self, t: str) -> tuple[Node, ...]:
        return tuple(n for n in self.active_nodes() if n.type == t)

    def contracts(self) -> tuple[Node, ...]:
        return self.nodes_by_type("Contract")

    def blockers(self) -> tuple[Node, ...]:
        return self.nodes_by_type("Blocker")

    def attempts(self) -> tuple[Node, ...]:
        return self.nodes_by_type("Attempt")

    def out_edges(self, source: str, edge_type: Optional[str] = None) -> tuple[Edge, ...]:
        return tuple(
            e for e in self.edges
            if not e.invalidated and e.source == source
            and (edge_type is None or e.type == edge_type)
        )

    def in_edges(self, target: str, edge_type: Optional[str] = None) -> tuple[Edge, ...]:
        return tuple(
            e for e in self.edges
            if not e.invalidated and e.target == target
            and (edge_type is None or e.type == edge_type)
        )

    def goal_contracts(self) -> tuple[Node, ...]:
        return tuple(n for n in self.contracts() if n.data.get("level") == "goal")

    def required_goal_contracts(self) -> tuple[Node, ...]:
        return tuple(n for n in self.goal_contracts() if bool(n.data.get("required", False)))

    def to_dict(self) -> dict:
        return {
            "nodes": [node_to_dict(n) for n in self.nodes],
            "edges": [edge_to_dict(e) for e in self.edges],
            "diagnostic_notes": list(self.diagnostic_notes),
        }

    @staticmethod
    def from_dict(d: dict) -> "ContractGraph":
        d = d or {}
        return ContractGraph(
            nodes=tuple(node_from_dict(x) for x in d.get("nodes", [])),
            edges=tuple(edge_from_dict(x) for x in d.get("edges", [])),
            diagnostic_notes=tuple(d.get("diagnostic_notes", [])),
        )


def _active_blocker_violates(graph: ContractGraph, contract_id: str) -> bool:
    for e in graph.in_edges(contract_id, "violates"):
        b = graph.node(e.source)
        if b is not None and not b.invalidated and bool(b.data.get("active", True)):
            return True
    return False


def project_status(graph: ContractGraph, contract_id: str, host_satisfied: frozenset[str]) -> str:
    if contract_id in host_satisfied:
        return "satisfied"
    if _active_blocker_violates(graph, contract_id):
        return "violated"
    return "unknown"


def depends_on_closure(graph: ContractGraph, goal_id: str) -> tuple[str, ...]:
    seen: set[str] = {goal_id}
    stack = [goal_id]
    out: list[str] = []
    while stack:
        cur = stack.pop()
        for e in graph.out_edges(cur, "depends_on"):
            if e.target not in seen:
                seen.add(e.target)
                out.append(e.target)
                stack.append(e.target)
    return tuple(out)


def root_blockers(graph: ContractGraph) -> tuple[Node, ...]:
    active = [b for b in graph.blockers() if bool(b.data.get("active", True))]
    return tuple(
        sorted(active, key=lambda b: 0 if b.data.get("root_or_downstream") == "root" else 1)
    )


def frontier_by_layer(graph: ContractGraph, host_satisfied: frozenset[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for c in graph.contracts():
        if project_status(graph, c.id, host_satisfied) != "satisfied":
            out.setdefault(c.data.get("layer", "deps"), []).append(c.id)
    return {k: tuple(v) for k, v in out.items()}


def goal_ready(graph: ContractGraph, host_satisfied: frozenset[str]) -> bool:
    required = graph.required_goal_contracts()
    if not required:
        return False
    for goal in required:
        if project_status(graph, goal.id, host_satisfied) != "satisfied":
            return False
        for dep in depends_on_closure(graph, goal.id):
            if project_status(graph, dep, host_satisfied) != "satisfied":
                return False
    return True


def _prereqs_satisfied(graph: ContractGraph, contract_id: str, host_satisfied: frozenset[str]) -> bool:
    """True iff every contract this one depends_on is already satisfied."""
    for e in graph.out_edges(contract_id, "depends_on"):
        if project_status(graph, e.target, host_satisfied) != "satisfied":
            return False
    return True


def _is_actionable_contract(node: Node) -> bool:
    """Directly repairable obligations: atomic contracts, or goals carrying a
    concrete `check` command (e.g. repo_tests_pass).  Pure aggregator goals
    (`check=""`) are satisfied derivatively and are not repair targets."""
    return node.data.get("level") == "atomic" or bool(node.data.get("check"))


def find_next_target_contracts(
    graph: ContractGraph, goal_id: str, host_satisfied: frozenset[str]
) -> tuple[str, ...]:
    """The actionable repair frontier for a goal.

    Returns contract ids on ``goal_id``'s depends_on path that are NOT satisfied
    and whose own prerequisites are ALL satisfied — i.e. the lowest obligations
    the planner can act on right now.  "Unsatisfied" means ``violated`` OR
    ``unknown`` (blockers are sparse, so many genuinely-unmet contracts project
    ``unknown`` rather than ``violated`` — both belong on the frontier).

    Ordered ``violated`` first (diagnosed, with a blocker), then ``unknown``,
    then by id for determinism.
    """
    candidate_ids = (goal_id,) + depends_on_closure(graph, goal_id)
    status_rank = {"violated": 0, "unknown": 1}
    ranked: list[tuple[int, str]] = []
    for cid in candidate_ids:
        node = graph.node(cid)
        if node is None or node.invalidated or node.type != "Contract":
            continue
        status = project_status(graph, cid, host_satisfied)
        if status == "satisfied":
            continue
        if not _is_actionable_contract(node):
            continue
        if not _prereqs_satisfied(graph, cid, host_satisfied):
            continue
        ranked.append((status_rank.get(status, 2), cid))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return tuple(cid for _, cid in ranked)


def attempts_for_contract(graph: ContractGraph, contract_id: str) -> tuple[Node, ...]:
    """Active Attempt nodes that `addresses` this contract, in graph order.

    Empty ⇒ the contract is untried.  Their outcomes (``ok`` / ``failed`` /
    ``ok_but_still_blocked``) tell the planner whether a prior repair was
    effective — the signal that prevents repeating an ineffective fix.
    """
    out: list[Node] = []
    for e in graph.in_edges(contract_id, "addresses"):
        a = graph.node(e.source)
        if a is not None and not a.invalidated and a.type == "Attempt":
            out.append(a)
    return tuple(out)
