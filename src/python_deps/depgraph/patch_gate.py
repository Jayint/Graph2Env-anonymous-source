"""Deterministic PatchGate (design §10): validate -> apply -> recompose.

The v3 replacement for the LLM Maintainer. validate_proposal returns an error list
(empty = accept); apply_proposal is a pure immutable reducer that NEVER writes
SATISFIED; compose_script re-derives the artifact from the graph plus governed
manual blocks. Pure: no Docker/network/LLM."""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, replace

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from src.python_deps.depgraph.action_class import matches_action_class
from src.python_deps.depgraph.block import Block, compile_blocks
from src.python_deps.depgraph.certify import EXECUTION_LAYER_ORDER
from src.python_deps.depgraph.patch import (
    PatchProposal, ProviderSpec, ScriptPatch,
)
from src.python_deps.depgraph.schema import (
    DepGraph, Ecosystem, Node, Edge, NodeType, Layer, EdgeType, State,
    DiscoveredBy, EDGE_RULES,
)

# Node-type -> canonical id prefix (ids.py).  Types not listed accept any "<kind>:<rest>".
_KIND_PREFIX: dict[NodeType, str] = {
    NodeType.PACKAGE: "pkg:", NodeType.SYSTEM_LIB: "syslib:", NodeType.TOOL: "tool:",
    NodeType.CONFIG: "config:", NodeType.SERVICE: "service:", NodeType.RUNTIME: "runtime:",
    NodeType.IMPORT: "import:", NodeType.PROJECT: "project:", NodeType.DATA_ASSET: "data:",
    NodeType.REQUIREMENT: "req:", NodeType.DEPENDENCY_SET: "deps:",
}
_ALLOWED_PROMOTION = frozenset({"hint", "candidate"})
_BENIGN_REDIR = re.compile(r"\s*(?:\d?>>?\s*/dev/null|\d?>&\d)")
_MUTATING = re.compile(
    r"(\bapt-get\s+install\b|\bapt\s+install\b|\bpip\d?\s+install\b|\bnpm\s+(?:install|ci)\b"
    r"|\brm\b|\bmkdir\b|\bmv\b|\bcp\b|\btee\b|\bdd\b|\btruncate\b|\bln\s+-s\b"
    r"|\bcurl\b|\bwget\b|>>|>)")
_COMMAND_V_ONLY = re.compile(r"^\s*command\s+-v\s+[A-Za-z0-9_.+-]+\s*$")
_SHELL_CONTROL = frozenset({"&&", "||", ";", "|", ">", ">>", "<", "<<"})

# pip install options whose following token is an option value, not a package
# requirement.  Keeping this list conservative means an unfamiliar/ambiguous
# command is rejected for unresolved Package promotion instead of guessed at.
_PIP_OPTIONS_WITH_VALUE = frozenset({
    "-c", "--constraint", "-r", "--requirement", "-e", "--editable",
    "-t", "--target", "--platform", "--python-version", "--implementation",
    "--abi", "--root", "--prefix", "--src", "-i", "--index-url",
    "--extra-index-url", "-f", "--find-links",
    "--trusted-host", "--proxy", "--retries", "--timeout", "--cache-dir",
    "--config-settings", "--hash", "--report", "--progress-bar",
    "--root-user-action",
})


def is_read_only(cmd: str) -> bool:
    """True when *cmd* performs no env mutation. Benign /dev/null and fd-dup (2>&1)
    redirects are stripped first. Load-bearing: admitted check_commands are host-executed."""
    scrubbed = _BENIGN_REDIR.sub("", cmd or "")
    if _COMMAND_V_ONLY.fullmatch(scrubbed):
        return True
    return not _MUTATING.search(scrubbed)


def _node_type(value: str) -> NodeType | None:
    try:
        return NodeType(value)
    except ValueError:
        return None


def _safe_manifest_provider(command: str) -> bool:
    """Whether a manifest provider is a single repository-local transaction.

    PatchGate remains pure and cannot inspect the checkout, so deterministic
    discovery owns existence checks.  This guard constrains Agent overrides to
    known package-manager forms and relative manifest/project paths; candidate
    replay plus host checks remain the binding proof.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or any(token in _SHELL_CONTROL for token in tokens):
        return False
    if tokens[:2] == ["uv", "sync"] or tokens[:2] == ["poetry", "install"]:
        return True
    install_at = None
    if tokens[:2] == ["pip", "install"]:
        install_at = 2
    elif len(tokens) >= 4 and tokens[0] in {"python", "python3"} and tokens[1:4] == ["-m", "pip", "install"]:
        install_at = 4
    if install_at is None:
        return False
    args = tokens[install_at:]
    for index, token in enumerate(args):
        if token in {"-r", "--requirement", "-e", "--editable"}:
            if index + 1 >= len(args):
                return False
            target = args[index + 1].split("[", 1)[0]
            if os.path.isabs(target) or ".." in target.replace("\\", "/").split("/"):
                return False
    return any(token in {"-r", "--requirement", "-e", "--editable"} for token in args)


def exact_pypi_provider_version(
    provider: ProviderSpec, package_name: str
) -> str | None:
    """Return the exact matching PyPI version installed by *provider*.

    This deliberately recognizes only one unambiguous positional requirement,
    e.g. ``python3 -m pip install --no-deps pytest==8.3.3``.  Requirements
    files, URLs, ranges, wildcards, multiple packages, shell compounds, and a
    package name different from the target node all return ``None``.
    """
    if provider.kind != "pip":
        return None
    try:
        tokens = shlex.split(provider.command)
        install_index = tokens.index("install")
    except (ValueError, TypeError):
        return None

    requirements: list[Requirement] = []
    skip_value = False
    positional_only = False
    for token in tokens[install_index + 1:]:
        if skip_value:
            skip_value = False
            continue
        if not positional_only and token == "--":
            positional_only = True
            continue
        if not positional_only and token.startswith("-"):
            option = token.split("=", 1)[0]
            if "=" not in token and option in _PIP_OPTIONS_WITH_VALUE:
                skip_value = True
            continue
        if token in {"&&", "||", ";", "|"}:
            return None
        try:
            requirements.append(Requirement(token))
        except InvalidRequirement:
            return None

    if skip_value or len(requirements) != 1:
        return None
    requirement = requirements[0]
    if (
        requirement.url is not None
        or requirement.marker is not None
        or canonicalize_name(requirement.name) != canonicalize_name(package_name)
    ):
        return None
    specifiers = tuple(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        return None
    version = specifiers[0].version
    if "*" in version:
        return None
    try:
        Version(version)
    except InvalidVersion:
        return None
    return version


def _exact_pypi_pin(provider: ProviderSpec, node: Node) -> str | None:
    if (
        node.type is not NodeType.PACKAGE
        or node.ecosystem not in (None, Ecosystem.PYPI)
    ):
        return None
    return exact_pypi_provider_version(provider, node.name)


def validate_proposal(graph: DepGraph, proposal: PatchProposal, *,
                      known_evidence_ids: frozenset[str],
                      manual_blocks: tuple[Block, ...] = ()) -> list[str]:
    errs: list[str] = []
    existing_ids = {n.id for n in graph.nodes}
    proposed_node_ids = {r.id for r in proposal.add_requirements}
    # Lazy import (not module-level) keeps python_deps.depgraph envstate-free; used by both the
    # requirement-check and script-patch-check anti-weakening guards below.
    from src.python_deps.depgraph.check_quality import check_can_detect_absence

    # within-proposal duplicate ids (nodes / providers / script blocks)
    for label, ids in (("add_requirements", [r.id for r in proposal.add_requirements]),
                       ("add_providers", [p.id for p in proposal.add_providers]),
                       ("script_patches", [s.block_id for s in proposal.script_patches])):
        if len(ids) != len(set(ids)):
            errs.append(f"duplicate id within {label}")

    for r in proposal.add_requirements:
        nt = _node_type(r.type)
        if nt is None:
            errs.append(f"unknown node type {r.type!r} for {r.id}")
            continue
        try:
            Layer(r.layer)
        except ValueError:
            errs.append(f"unknown layer {r.layer!r} for {r.id}")
        prefix = _KIND_PREFIX.get(nt)
        if prefix is not None and not r.id.startswith(prefix):
            errs.append(f"non-canonical id {r.id!r}: {nt.value} requires prefix {prefix!r}")
        elif ":" not in r.id:
            errs.append(f"non-canonical id {r.id!r}: missing '<kind>:' prefix")
        if r.promotion is not None and r.promotion not in _ALLOWED_PROMOTION:
            errs.append(f"illegal promotion {r.promotion!r} for {r.id} "
                        f"(only {sorted(_ALLOWED_PROMOTION)} or none; SATISFIED is host-only)")
        if not r.evidence_ref or r.evidence_ref not in known_evidence_ids:
            errs.append(f"requirement {r.id} cites unknown/absent evidence {r.evidence_ref!r}")
        if nt is NodeType.PACKAGE:
            if not r.version:
                errs.append(f"package requirement {r.id} must include a pinned version")
            elif r.ecosystem in (None, "", Ecosystem.PYPI.value):
                try:
                    Version(r.version)
                except InvalidVersion:
                    errs.append(
                        f"package requirement {r.id} must use one exact PEP-440 version, "
                        f"not {r.version!r}"
                    )
        if r.ecosystem is not None:
            try:
                Ecosystem(r.ecosystem)
            except ValueError:
                errs.append(f"unknown ecosystem {r.ecosystem!r} for {r.id}")
        if r.check_command and not is_read_only(r.check_command):
            errs.append(f"check command for {r.id} is not read-only: {r.check_command!r}")
        if r.check_command and not check_can_detect_absence(r.check_command):
            errs.append(f"check command for {r.id} cannot detect absence "
                        f"(structurally trivial): {r.check_command!r}")
        # conflicting redefinition vs graph
        cur = graph.get(r.id)
        if cur is not None and (
            cur.type.value != r.type
            or cur.layer.value != r.layer
            or (cur.check_command or None) != (r.check_command or None)
            or (r.version is not None and cur.version != r.version)
        ):
            errs.append(
                f"conflicting redefinition of existing node {r.id}; remove "
                f"{r.id} from add_requirements and, when replacing its install "
                "action, use add_providers with override=true"
            )

    known_after = existing_ids | proposed_node_ids
    manual_block_ids = {block.block_id for block in manual_blocks}
    proposed_provider_targets = {
        node_id for provider in proposal.add_providers for node_id in provider.provides
    }
    proposed_types = {requirement.id: _node_type(requirement.type)
                      for requirement in proposal.add_requirements}
    for p in proposal.add_providers:
        if not matches_action_class(p.kind, p.command):
            errs.append(f"provider {p.id} command does not match action class "
                        f"{p.kind!r}: {p.command!r}")
        if p.kind == "manifest" and not _safe_manifest_provider(p.command):
            errs.append(
                f"manifest provider {p.id} must be one repository-local uv sync, "
                f"poetry install, or pip install -r/-e transaction: {p.command!r}"
            )
        for nid in p.provides:
            if nid not in known_after:
                errs.append(f"provider {p.id} provides unknown node {nid!r}")
                continue
            current = graph.get(nid)
            target_type = current.type if current is not None else proposed_types.get(nid)
            if p.kind == "manifest" and target_type is not NodeType.DEPENDENCY_SET:
                errs.append(
                    f"manifest provider {p.id} may provide only a DependencySet, not {nid}"
                )
            if (
                current is not None
                and current.type is NodeType.PACKAGE
                and current.ecosystem in (None, Ecosystem.PYPI)
                and not current.version
                and _exact_pypi_pin(p, current) is None
            ):
                errs.append(
                    f"provider {p.id} for unresolved package {nid} must install "
                    f"exactly one matching pinned requirement "
                    f"({current.name}==<PEP-440-version>)"
                )
            if current is not None and current.chosen_fix is not None and not p.override:
                # Applying this proposal would be a silent no-op because the
                # reducer is first-writer-wins.  Reject it with the exact
                # repair instruction so the bounded re-prompt can correct an
                # existing package/provider instead of cycling between a
                # conflicting NodeSpec and an ineffective ProviderSpec.
                errs.append(
                    f"provider {p.id} cannot replace existing provider "
                    f"{current.chosen_fix!r} for {nid}; set override=true and do "
                    f"not repeat {nid} in add_requirements"
                )

    for s in proposal.script_patches:
        if s.op not in {"add_block", "replace_block"}:
            errs.append(
                f"script block {s.block_id} has illegal op {s.op!r} "
                "(must be 'add_block' or 'replace_block')"
            )
        elif s.op == "add_block" and s.block_id in manual_block_ids:
            errs.append(
                f"script block {s.block_id} already exists; use op='replace_block'"
            )
        elif s.op == "replace_block" and s.block_id not in manual_block_ids:
            errs.append(
                f"script block {s.block_id} cannot be replaced because it does not exist"
            )
        if s.op == "add_block":
            for nid in s.target_node_ids:
                node = graph.get(nid)
                node_type = node.type if node is not None else proposed_types.get(nid)
                has_graph_recipe = (
                    node_type is NodeType.PACKAGE
                    or (
                        node_type in {NodeType.SYSTEM_LIB, NodeType.TOOL}
                        and nid in proposed_provider_targets
                    )
                )
                if has_graph_recipe:
                    errs.append(
                        f"script block {s.block_id} duplicates graph-native target {nid}; "
                        "repair its provider instead"
                    )
        if not s.evidence_ref or s.evidence_ref not in known_evidence_ids:
            errs.append(f"script block {s.block_id} cites unknown/absent evidence {s.evidence_ref!r}")
        if not s.target_node_ids:
            errs.append(f"script block {s.block_id} has empty target_node_ids")
        for nid in s.target_node_ids:
            if nid not in known_after:
                errs.append(f"script block {s.block_id} targets unknown node {nid!r}")
        for chk in s.checks:
            if not is_read_only(chk):
                errs.append(f"script block {s.block_id} check is not read-only: {chk!r}")
            if not check_can_detect_absence(chk):
                errs.append(f"script block {s.block_id} check cannot detect absence "
                            f"(structurally trivial): {chk!r}")
        if not s.commands:
            errs.append(f"script block {s.block_id} has empty commands")
        if any(not c.strip() for c in s.commands):
            errs.append(f"script block {s.block_id} has a blank/whitespace-only command")
        try:
            Layer(s.wave)
        except ValueError:
            errs.append(f"script block {s.block_id} has illegal wave {s.wave!r} "
                        f"(must be a Layer value)")
        for nid in s.provides:
            if nid not in known_after:
                errs.append(f"script block {s.block_id} provides unknown node {nid!r}")

    # edges: replicate EDGE_RULES against the post-add_requirements view (with_edge would RAISE).
    type_of = {n.id: n.type.value for n in graph.nodes}
    type_of.update({r.id: r.type for r in proposal.add_requirements})
    for e in proposal.add_edges:
        try:
            EdgeType(e.relation)
        except ValueError:
            errs.append(f"unknown edge relation {e.relation!r}")
            continue
        rule = EDGE_RULES.get(e.relation)
        if e.source not in type_of or e.target not in type_of:
            errs.append(f"edge {e.relation} references unknown node(s): {e.source!r} -> {e.target!r}")
            continue
        if rule is not None:
            allowed_src, allowed_dst = rule
            if type_of[e.source] not in allowed_src:
                errs.append(f"illegal {e.relation} source type {type_of[e.source]!r} ({e.source!r})")
            if type_of[e.target] not in allowed_dst:
                errs.append(f"illegal {e.relation} destination type {type_of[e.target]!r} ({e.target!r})")

    return errs


@dataclass(frozen=True)
class ApplyResult:
    graph: DepGraph
    blocks: tuple[Block, ...]


def _provider_fix(p: ProviderSpec) -> str:
    # apt providers store the "apt:NAME" form (emit._apt_name strips the prefix);
    # everything else stores the literal command (compile_blocks' fallback emits it,
    # and for PACKAGE nodes compile_blocks derives the pip command from name/version).
    return p.id if p.id.startswith("apt:") else p.command


def _script_patch_to_block(s: ScriptPatch) -> Block:
    return Block(
        block_id=s.block_id, wave=s.wave, commands=s.commands,
        target_node_ids=s.target_node_ids, provider_ids=s.provides,
        check_commands=s.checks,
        evidence_refs=(s.evidence_ref,) if s.evidence_ref else (),
    )


def apply_proposal(graph: DepGraph, proposal: PatchProposal) -> ApplyResult:
    g = graph
    # 1. requirement nodes — always MISSING (never SATISFIED), promotion tag if present.
    for r in proposal.add_requirements:
        if g.get(r.id) is not None:
            continue                                    # dedup no-op (validate ensured non-conflicting)
        data = {"promotion": r.promotion} if r.promotion else {}
        g = g.with_node(Node(
            id=r.id, type=NodeType(r.type), name=r.name or r.id.split(":", 1)[-1],
            layer=Layer(r.layer), discovered_by=DiscoveredBy.PROBE, state=State.MISSING,
            version=r.version, check_command=r.check_command,
            evidence=r.evidence_ref, data=data,
            ecosystem=Ecosystem(r.ecosystem) if r.ecosystem else None,
            workspace=r.workspace,
            package_manager=r.package_manager,
            declared_constraint=r.declared_constraint,
            resolved_locator=r.resolved_locator,
        ))
    # 2. providers -> chosen_fix on each provided node (first writer wins).
    for p in proposal.add_providers:
        fix = _provider_fix(p)
        for nid in p.provides:
            node = g.get(nid)
            if node is not None and (node.chosen_fix is None or p.override):
                resolved_version = (
                    _exact_pypi_pin(p, node)
                    if node.type is NodeType.PACKAGE and not node.version
                    else None
                )
                # ``chosen_fix`` keeps the stable provider identity (notably
                # ``apt:NAME``), while setup_commands is the canonical, gated
                # action.  Preserve the exact Agent-supplied command for both
                # first writers and overrides; otherwise populate would
                # reconstruct a generic apt command and silently discard
                # approved retry flags, mirror handling, or other semantics.
                g = g.with_node(replace(
                    node,
                    version=resolved_version or node.version,
                    chosen_fix=fix,
                    setup_commands=(p.command,),
                ))
    # 3. edges — endpoints now exist (validate guaranteed legality; with_edge dedupes).
    for e in proposal.add_edges:
        g = g.with_edge(Edge(src=e.source, dst=e.target, relation=EdgeType(e.relation),
                             data={"hard": e.hard}))
    # 4. script_patches -> governed blocks; they NEVER mutate node state.
    blocks = tuple(_script_patch_to_block(s) for s in proposal.script_patches)
    return ApplyResult(graph=g, blocks=blocks)


# Wave rank for slotting manual blocks; shares the certified EXECUTION_LAYER_ORDER
# (design section 6, also used by certify.certify_all and build_script.render_build_script)
# so the live compose order matches render_build_script's artifact order (replay
# fidelity) and RUNTIME/CONFIG/TESTS waves sort correctly, rather than the raw Layer
# enum declaration order. Layers not in EXECUTION_LAYER_ORDER (e.g. SERVICES) sort
# last, in enum order. compile_blocks already emits compiled blocks in topo
# (wave-rank-nondecreasing) order, and Python's sort is STABLE, so sorting the
# merged list by wave rank leaves compiled blocks in place and slots each manual
# block after the compiled blocks of its wave.
_WAVE_ORDER: tuple[Layer, ...] = EXECUTION_LAYER_ORDER + tuple(
    L for L in Layer if L not in EXECUTION_LAYER_ORDER
)
_WAVE_RANK: dict[str, int] = {layer.value: i for i, layer in enumerate(_WAVE_ORDER)}


def compose_script(graph: DepGraph, manual_blocks: tuple[Block, ...] = ()) -> tuple[Block, ...]:
    compiled = compile_blocks(graph)
    seen = {b.block_id for b in compiled}
    fresh = []
    for b in manual_blocks:
        if b.block_id in seen:                 # graph-compiled block wins on id collision
            continue
        seen.add(b.block_id)
        fresh.append(b)
    if not fresh:
        return compiled
    merged = list(compiled) + fresh            # compiled first -> stable sort keeps them first per wave
    merged.sort(key=lambda b: _WAVE_RANK.get(b.wave, len(_WAVE_RANK)))
    return tuple(merged)


@dataclass(frozen=True)
class AdmitResult:
    """PatchGate result; accepted state is a candidate, never an official commit."""

    accepted: bool
    errors: tuple[str, ...]
    graph: DepGraph
    blocks: tuple[Block, ...]
    manual_blocks: tuple[Block, ...]

    @property
    def candidate_graph(self) -> DepGraph:
        return self.graph

    @property
    def candidate_manual_blocks(self) -> tuple[Block, ...]:
        return self.manual_blocks


def admit_proposal(graph: DepGraph, proposal: PatchProposal, *,
                   manual_blocks: tuple[Block, ...] = (),
                   known_evidence_ids: frozenset[str]) -> AdmitResult:
    """Purely build a validated candidate graph and plan.

    On reject, graph/manual_blocks are unchanged and errors are non-empty. On
    accept, ``candidate_graph``/``candidate_manual_blocks`` still require a Host
    CandidateTransaction before they may replace official state. NEVER writes
    SATISFIED, runs commands, or commits official state.
    """
    errs = validate_proposal(
        graph,
        proposal,
        known_evidence_ids=known_evidence_ids,
        manual_blocks=manual_blocks,
    )
    if errs:
        return AdmitResult(False, tuple(errs), graph,
                           compose_script(graph, manual_blocks), manual_blocks)
    applied = apply_proposal(graph, proposal)
    new_manual_list = list(manual_blocks)
    for spec, block in zip(proposal.script_patches, applied.blocks):
        if spec.op == "replace_block":
            index = next(
                i for i, current in enumerate(new_manual_list)
                if current.block_id == spec.block_id
            )
            new_manual_list[index] = block
        else:
            new_manual_list.append(block)
    new_manual = tuple(new_manual_list)
    return AdmitResult(True, (), applied.graph,
                       compose_script(applied.graph, new_manual), new_manual)
