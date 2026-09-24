"""Typed, immutable concrete dependency graph.

This realizes the model in ``docs/DESIGN-static-probe-certified-dependency-graph.md``
section 5: one concrete node type per layer joined by ``requires`` edges, where
each node carries a host-certified ``state`` axis (the certification invariant of
section 3.1) that is separate from the ``attempts`` action axis.

Everything here is a frozen dataclass.  Every "mutation" returns a NEW object;
the originals are never changed (repo immutability rule).
"""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass, field, replace


class NodeType(enum.Enum):
    TEST = "Test"
    PROJECT = "Project"  # the repo under test; hub for its declared direct deps
    IMPORT = "Import"
    REQUIREMENT = "Requirement"
    PACKAGE = "Package"
    DEPENDENCY_SET = "DependencySet"
    SYSTEM_LIB = "SystemLib"
    TOOL = "Tool"
    RUNTIME = "Runtime"
    PLATFORM = "Platform"      # tier 1
    SERVICE = "Service"        # tier 5
    CONFIG = "Config"          # tier 6
    DATA_ASSET = "DataAsset"   # tier 6


# Provider-stack tier per node type; goal types (Test/Project/Import) map to 0
# (they are the demand side, not a tier). See design §3.1/§3.2.
TYPE_TO_TIER: dict["NodeType", int] = {
    NodeType.PLATFORM: 1,
    NodeType.SYSTEM_LIB: 2,
    NodeType.TOOL: 2,
    NodeType.RUNTIME: 3,
    NodeType.DEPENDENCY_SET: 4,
    NodeType.PACKAGE: 4,
    NodeType.SERVICE: 5,
    NodeType.CONFIG: 6,
    NodeType.DATA_ASSET: 6,
}


def tier_for_type(node_type: "NodeType") -> int:
    """Provider tier for ``node_type``; 0 for goal nodes (Test/Project/Import)."""
    return TYPE_TO_TIER.get(node_type, 0)


class EdgeType(enum.Enum):
    REQUIRES = "requires"
    MAPS_TO = "maps_to"
    RESOLVES_TO = "resolves_to"
    DESCRIBES = "describes"
    ALTERNATIVE_TO = "alternative_to"  # reserved; not emitted in this plan
    CONFLICTS_WITH = "conflicts_with"  # emitted by the resolver (uv unsat core)


class State(enum.Enum):
    """Certification axis. Only a host-run check_command flips this (3.1)."""

    UNKNOWN = "unknown"
    MISSING = "missing"
    SATISFIED = "satisfied"


class DiscoveredBy(enum.Enum):
    GOAL = "goal"
    STATIC_SCAN = "static_scan"
    RESOLVER = "resolver"
    PROBE = "probe"
    RUNTIME = "runtime"


class Layer(enum.Enum):
    INTERPRETER = "interpreter"
    SYSTEM = "system"
    TOOLCHAIN = "toolchain"
    PIP = "pip"
    DEPENDENCIES = "dependencies"
    BUILD = "build"
    NAMING = "naming"
    RUNTIME = "runtime"
    TESTS = "tests"
    CONFIG = "config"
    SERVICES = "services"


class Strength(enum.Enum):
    """Blocking semantics. SOFT = hint/candidate (does not block dependents or
    gates); HARD = required obligation. The populator sets HARD on reciped tiers;
    static/LLM discovery stays SOFT."""

    SOFT = "soft"
    HARD = "hard"


class Phase(enum.Enum):
    """Where a node's commands belong in the final artifact (distinct from Layer,
    which drives topological order)."""

    SETUP = "setup"
    RUNTIME = "runtime"
    TEST = "test"
    GATE = "gate"


class Ecosystem(enum.Enum):
    PYPI = "pypi"
    NPM = "npm"
    CARGO = "cargo"
    GO_MODULE = "gomod"
    MAVEN = "maven"
    GRADLE = "gradle"


# relation -> (allowed src node-type values, allowed dst node-type values)
EDGE_RULES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "requires": (
        frozenset({"Test", "Project", "Import", "Package", "DependencySet",
                   "Tool", "Service", "Config", "DataAsset"}),
        frozenset({"Project", "Import", "Requirement", "Package", "DependencySet",
                   "SystemLib", "Tool", "Runtime", "Platform", "Service",
                   "Config", "DataAsset"}),
    ),
    "maps_to": (
        frozenset({"Import"}),
        frozenset({"Package"}),
    ),
    "resolves_to": (
        frozenset({"Requirement"}),
        frozenset({"Package"}),
    ),
    "describes": (
        frozenset({"DependencySet"}),
        frozenset({"Package"}),
    ),
    # Resolver-discovered version conflicts join two Packages (uv unsat core).
    "conflicts_with": (
        frozenset({"Package"}),
        frozenset({"Package"}),
    ),
}


@dataclass(frozen=True)
class Attempt:
    command: str
    outcome: str  # "succeeded" | "failed" | "unknown"
    check: str = ""
    cycle: int = 0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "outcome": self.outcome,
            "check": self.check,
            "cycle": self.cycle,
        }


@dataclass(frozen=True)
class Node:
    id: str
    type: NodeType
    name: str
    layer: Layer
    discovered_by: DiscoveredBy
    tier: int = 0  # 0 = derive from type in __post_init__ (goal nodes stay 0)
    state: State = State.UNKNOWN
    version: str | None = None
    check_command: str | None = None
    evidence: str | None = None
    fix_candidates: tuple[str, ...] = ()
    chosen_fix: str | None = None
    attempts: tuple[Attempt, ...] = ()
    provenance: str | None = None
    discovered_cycle: int = 0
    certified_cycle: int | None = None
    # uv-resolution enrichment (native-build risk + targeting provenance).
    # All default-safe so existing construction is unaffected.
    build_from_source: bool | None = None
    artifact: str | None = None  # chosen wheel/sdist filename
    hash: str | None = None
    resolved_python: str | None = None
    resolved_platform: str | None = None
    exclude_newer: str | None = None  # uv resolve cutoff (reproducibility)
    # Original manifest intent is retained independently of the resolver's
    # selected concrete version.  This lets later repair reason from what the
    # repository declared even when resolution failed or was degraded.
    declared_specifier: str | None = None
    declared_marker: str | None = None
    manifest_source: str | None = None
    resolution_status: str | None = None  # "unresolved" | "resolved" | "failed"
    resolution_error: str | None = None
    # Cross-ecosystem identity/provenance. Defaults preserve the existing
    # Python graph byte-for-byte while non-Python providers use explicit
    # ecosystem/workspace-qualified identities.
    ecosystem: Ecosystem | None = None
    workspace: str | None = None
    package_manager: str | None = None
    declared_constraint: str | None = None
    resolved_locator: str | None = None
    lock_digest: str | None = None
    language: str | None = None
    language_role: str | None = None
    version_constraint: str | None = None
    data: dict = field(default_factory=dict)  # general per-node metadata bag
    # --- install-command generation (uniform-graph) ---
    # setup_commands is the node's canonical "how" (the only command source the
    # renderer reads); strength is blocking semantics; phase is artifact placement.
    setup_commands: tuple[str, ...] = ()
    strength: Strength = Strength.SOFT
    phase: Phase = Phase.SETUP

    def __post_init__(self) -> None:
        # Derive tier from type when left at the 0 sentinel. Idempotent for goal
        # nodes (tier_for_type returns 0 for them). frozen=True blocks rebinding,
        # so set through object.__setattr__.
        if self.tier == 0:
            object.__setattr__(self, "tier", tier_for_type(self.type))
        if not isinstance(self.data, types.MappingProxyType):
            object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))

    def with_state(
        self,
        state: State,
        *,
        evidence: str | None = None,
        cycle: int | None = None,
    ) -> "Node":
        """Return a NEW node with an updated certification state.

        ``evidence`` overrides only when provided; ``cycle`` (when provided)
        records the certification cycle.
        """
        changes: dict = {"state": state}
        if evidence is not None:
            changes["evidence"] = evidence
        if cycle is not None:
            changes["certified_cycle"] = cycle
        return replace(self, **changes)

    def with_attempt(self, attempt: Attempt) -> "Node":
        """Return a NEW node with ``attempt`` appended to the history."""
        return replace(self, attempts=self.attempts + (attempt,))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "layer": self.layer.value,
            "tier": self.tier,
            "discovered_by": self.discovered_by.value,
            "state": self.state.value,
            "version": self.version,
            "check_command": self.check_command,
            "evidence": self.evidence,
            "fix_candidates": list(self.fix_candidates),
            "chosen_fix": self.chosen_fix,
            "attempts": [a.to_dict() for a in self.attempts],
            "provenance": self.provenance,
            "discovered_cycle": self.discovered_cycle,
            "certified_cycle": self.certified_cycle,
            "build_from_source": self.build_from_source,
            "artifact": self.artifact,
            "hash": self.hash,
            "resolved_python": self.resolved_python,
            "resolved_platform": self.resolved_platform,
            "exclude_newer": self.exclude_newer,
            "declared_specifier": self.declared_specifier,
            "declared_marker": self.declared_marker,
            "manifest_source": self.manifest_source,
            "resolution_status": self.resolution_status,
            "resolution_error": self.resolution_error,
            "ecosystem": self.ecosystem.value if self.ecosystem is not None else None,
            "workspace": self.workspace,
            "package_manager": self.package_manager,
            "declared_constraint": self.declared_constraint,
            "resolved_locator": self.resolved_locator,
            "lock_digest": self.lock_digest,
            "language": self.language,
            "language_role": self.language_role,
            "version_constraint": self.version_constraint,
            "setup_commands": list(self.setup_commands),
            "strength": self.strength.value,
            "phase": self.phase.value,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: EdgeType = EdgeType.REQUIRES
    origin: str | None = None  # "scan"|"resolver"|"probe"|"runtime"|"project"|"certified"|"reconcile"
    marker: str | None = None  # conditional-dep marker on a `requires` edge
    # Auxiliary relation data (e.g. conflict version bounds on conflicts_with).
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # frozen=True only blocks attribute rebinding, not in-place dict mutation.
        # Wrap data in a read-only view so `frozen` actually means immutable; a
        # plain dict passed in is snapshotted (copy then freeze) so later edits to
        # the caller's dict cannot leak into a stored edge.
        if not isinstance(self.data, types.MappingProxyType):
            object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.relation.value)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation.value,
            "origin": self.origin,
            "marker": self.marker,
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class DepGraph:
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()

    def get(self, node_id: str) -> Node | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def with_node(self, node: Node) -> "DepGraph":
        """Add ``node``, or replace any existing node with the same id."""
        kept = tuple(n for n in self.nodes if n.id != node.id)
        return replace(self, nodes=kept + (node,))

    def with_edge(self, edge: Edge) -> "DepGraph":
        """Add ``edge`` deduped by (src, dst, relation), validating EDGE_RULES."""
        self._validate_edge(edge)
        if any(e.key() == edge.key() for e in self.edges):
            return self
        return replace(self, edges=self.edges + (edge,))

    def _validate_edge(self, edge: Edge) -> None:
        src_node = self.get(edge.src)
        dst_node = self.get(edge.dst)
        if src_node is None or dst_node is None:
            raise ValueError(
                f"edge {edge.relation.value} references unknown node(s): "
                f"{edge.src!r} -> {edge.dst!r}"
            )
        rule = EDGE_RULES.get(edge.relation.value)
        if rule is None:
            # Reserved relations (e.g. alternative_to) carry no type rule, but
            # endpoints must still exist.
            return
        allowed_src, allowed_dst = rule
        if src_node.type.value not in allowed_src:
            raise ValueError(
                f"illegal {edge.relation.value} source type "
                f"{src_node.type.value!r} ({edge.src!r})"
            )
        if dst_node.type.value not in allowed_dst:
            raise ValueError(
                f"illegal {edge.relation.value} destination type "
                f"{dst_node.type.value!r} ({edge.dst!r})"
            )

    def without_node(self, node_id: str) -> "DepGraph":
        """Remove ``node_id`` and every edge that touches it.  Returns a NEW graph.

        Used to drop a node that has been superseded (e.g. an unresolved
        identity-fallback placeholder once the relink certifies the real
        provider).  A no-op for an unknown id.
        """
        nodes = tuple(n for n in self.nodes if n.id != node_id)
        edges = tuple(e for e in self.edges if e.src != node_id and e.dst != node_id)
        return replace(self, nodes=nodes, edges=edges)

    def requires_of(self, node_id: str) -> tuple[Node, ...]:
        """Successor nodes reachable from ``node_id`` via a requires edge."""
        out = []
        for edge in self.edges:
            if edge.relation is EdgeType.REQUIRES and edge.src == node_id:
                dst = self.get(edge.dst)
                if dst is not None:
                    out.append(dst)
        return tuple(out)

    def required_by(self, node_id: str) -> tuple[Node, ...]:
        """Predecessor nodes that require ``node_id`` via a requires edge."""
        out = []
        for edge in self.edges:
            if edge.relation is EdgeType.REQUIRES and edge.dst == node_id:
                src = self.get(edge.src)
                if src is not None:
                    out.append(src)
        return tuple(out)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
