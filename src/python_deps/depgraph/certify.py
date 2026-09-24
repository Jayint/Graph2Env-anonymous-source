"""Stage 5 — host certification.

This realizes the certification invariant of design section 3.1: a node's
``state`` is flipped **only** by running its ``check_command`` on the host and
observing the exit code.  Install/import *actions* never imply ``satisfied`` —
only a passing check does.  State is therefore:

* **revocable** — re-certifying after a mutation can flip ``SATISFIED`` back to
  ``MISSING`` (a later install can break an earlier import; design 10.9);
* **host-issued** — nothing here is inferred from an action outcome.

``certify`` certifies one node; ``certify_all`` walks the graph in execution
layer order (design section 6): interpreter -> system -> toolchain -> pip ->
naming -> tests.  Every "mutation" returns a NEW ``DepGraph`` (repo immutability
rule).
"""

from __future__ import annotations

from src.python_deps.depgraph.executor import Executor
from src.python_deps.depgraph.schema import DepGraph, Layer, NodeType, State

# Execution layer priority (design section 6). Runtime joins the walk first: the
# interpreter minor is the platform floor every later layer assumes.
#
# Public: also consumed by build_script.py so the rendered artifact's section
# order never contradicts the order certify actually walks the graph in.
EXECUTION_LAYER_ORDER: tuple[Layer, ...] = (
    Layer.RUNTIME,
    Layer.INTERPRETER,
    Layer.SYSTEM,
    Layer.TOOLCHAIN,
    Layer.PIP,
    Layer.DEPENDENCIES,
    Layer.BUILD,
    Layer.NAMING,
    Layer.CONFIG,
    Layer.TESTS,
)
_LAYER_ORDER = EXECUTION_LAYER_ORDER  # backwards-compat alias

# Confirmed in-image services are checked after their system/package providers
# but before the Test layer that requires them.  Off-arm never uses this order.
_SERVICE_LAYER_ORDER: tuple[Layer, ...] = (
    tuple(layer for layer in _LAYER_ORDER if layer is not Layer.TESTS)
    + (Layer.SERVICES, Layer.TESTS)
)


def _test_waits_for_confirmed_service(graph: DepGraph, node_id: str) -> bool:
    """Whether a Test check is structurally blocked on a known live service."""
    node = graph.get(node_id)
    if node is None or node.type is not NodeType.TEST:
        return False
    return any(
        requirement.type is NodeType.SERVICE
        and requirement.data.get("service_confidence") == "confirmed"
        and requirement.state is not State.SATISFIED
        for requirement in graph.requires_of(node_id)
    )


def certify(
    graph: DepGraph,
    node_id: str,
    executor: Executor,
    cycle: int = 0,
    *,
    allow_service_certify: bool = False,
) -> DepGraph:
    """Run one node's ``check_command`` and write its host-certified ``state``.

    * rc 0          -> ``SATISFIED`` with ``certified_cycle = cycle``;
    * rc != 0       -> ``MISSING`` with the check's stderr as ``evidence``;
    * no check_command -> left ``UNKNOWN`` (the host ran nothing).

    ``certified_cycle`` always records the cycle in which the host check was last
    actually run — including on the revocation path (SATISFIED -> MISSING) — so a
    consumer can distinguish "never certified" (``None``) from "certified then
    revoked" (the cycle of the failing re-check).

    Unknown ``node_id`` returns the graph unchanged.  Returns a NEW graph.

    SERVICE nodes are certified (loopback probe) only when ``allow_service_certify``
    is True AND the node has ``data["service_confidence"] == "confirmed"`` (design
    §4.3).  Off-arm / inferred services stay UNKNOWN — the scratch container cannot
    host the daemon.
    """
    node = graph.get(node_id)
    if node is None or not node.check_command:
        return graph
    # Services are reachability-certified only on the live in-image path (arm
    # v3) and only when CONFIRMED. Off-arm / inferred: stay UNKNOWN (design
    # §4.3). The scratch container cannot host the daemon, so the scratch
    # certify_all call leaves allow_service_certify=False.
    if node.type is NodeType.SERVICE:
        if not (allow_service_certify
                and node.data.get("service_confidence") == "confirmed"):
            return graph

    result = executor.run(node.check_command)
    if result.ok:
        updated = node.with_state(State.SATISFIED, cycle=cycle)
    else:
        # Preserve the node's DISCOVERY evidence (the real build/import failure the
        # probe captured) — only fall back to the check's stderr when there is no
        # prior evidence. A presence check like ``ldconfig -p | grep`` or
        # ``command -v`` prints nothing on failure, so writing its empty stderr
        # would otherwise clobber the diagnostic line that explains WHY the need
        # exists. (design 3.1: certify owns ``state``, not the evidence of need.)
        evidence = node.evidence or result.stderr or None
        updated = node.with_state(State.MISSING, evidence=evidence, cycle=cycle)
    return graph.with_node(updated)


def certify_all(
    graph: DepGraph,
    executor: Executor,
    cycle: int = 0,
    *,
    allow_service_certify: bool = False,
    layer_order: tuple[Layer, ...] = _LAYER_ORDER,
) -> DepGraph:
    """Certify every node in execution layer order (design section 6).

    Re-reads the evolving graph after each certification so revocation/ordering
    side effects compose.  Returns a NEW graph.

    Pass ``allow_service_certify=True`` and ``layer_order=_SERVICE_LAYER_ORDER``
    (via ``certify_refresh``) to also certify confirmed SERVICE nodes on the live
    in-image path (arm v3).
    """
    new = graph
    for layer in layer_order:
        node_ids = [n.id for n in new.nodes if n.layer is layer]
        for node_id in node_ids:
            # Do not launch a potentially long/hanging test suite while a
            # confirmed runtime service it directly requires is still down.
            # The service's quick loopback check has already run earlier in the
            # live layer order; scheduler repair can now act on that obligation.
            if _test_waits_for_confirmed_service(new, node_id):
                continue
            new = certify(new, node_id, executor, cycle=cycle,
                          allow_service_certify=allow_service_certify)
    return new
