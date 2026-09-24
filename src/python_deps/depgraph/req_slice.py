# src/python_deps/depgraph/req_slice.py
"""Read-time derivation of the structured RequirementSlice the v3 agent sees (design
2026-06-29). Pure: no Docker/LLM/subprocess and NO dependency on src.envstate."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.python_deps.depgraph.action_class import ACTION_CLASSES   # pure leaf (only imports re); no cycle


@dataclass(frozen=True)
class ProviderCand:
    id: str
    action_class: str            # "apt" | "pip" | "npm" | "shell" (action_class.py) | "" (undeterminable)


@dataclass(frozen=True)
class TriedProvider:
    command: str
    outcome: str
    provider_id: str | None      # best-effort reverse-parse of `command`; None for batch/unparseable


@dataclass(frozen=True)
class ProviderView:
    candidates: tuple[ProviderCand, ...]
    chosen: str | None
    tried_failed: tuple[TriedProvider, ...]


def _action_class_for(provider_id: str) -> str:
    # Reuse the canonical provider taxonomy (apt/pip/npm/shell) — do NOT reimplement the map.
    head = provider_id.split(":", 1)[0] if ":" in provider_id else ""
    return head if head in ACTION_CLASSES else ""


def _provider_from_command(command: str) -> str | None:
    """Map an install command back to a provider id when EXACTLY one package is named
    (single-token); batch installs are not cleanly attributable -> None."""
    m = re.search(r"\bapt(?:-get)?\s+install\b(.*)", command)
    if m:
        toks = [t for t in m.group(1).split() if not t.startswith("-")]
        return f"apt:{toks[0]}" if len(toks) == 1 else None
    m = re.search(r"\bpip3?\s+install\b(.*)", command)
    if m:
        args = m.group(1)
        # requirements/constraints-file or editable installs are not a single named package
        if re.search(r"(?:^|\s)(?:-r|--requirement|-c|--constraint|-e|--editable)(?:\s|=|$)", args):
            return None
        toks = [t for t in args.split() if not t.startswith("-")]
        return f"pip:{toks[0].split('==')[0]}" if len(toks) == 1 else None
    return None


def providers_view(node) -> ProviderView:
    cand_ids = list(node.fix_candidates)
    if node.chosen_fix and node.chosen_fix not in cand_ids:
        cand_ids.append(node.chosen_fix)
    candidates = tuple(ProviderCand(id=c, action_class=_action_class_for(c)) for c in cand_ids)
    tried = tuple(
        TriedProvider(command=a.command, outcome=a.outcome,
                      provider_id=_provider_from_command(a.command))
        for a in node.attempts if a.outcome == "failed"
    )
    return ProviderView(candidates=candidates, chosen=node.chosen_fix, tried_failed=tried)


@dataclass(frozen=True)
class DepView:
    id: str
    state: str


@dataclass(frozen=True)
class RequirementSlice:
    node_id: str
    kind: str
    layer: str
    state: str
    check: str
    evidence: str
    deps: tuple[DepView, ...]
    chain_to_goal: str
    unblocks: tuple[str, ...]
    layer_cohort_satisfied: tuple[str, ...]
    layer_cohort_missing: tuple[str, ...]
    conflict: str | None
    providers: ProviderView
    active_gate: str
    platform: str | None


def render_requirement_slice(s: RequirementSlice) -> tuple[str, ...]:
    """Compact, agent-readable fact lines. Empty sections are omitted."""
    lines = [f"target: {s.node_id}  ({s.kind}, {s.layer}, {s.state})"]
    gate = f"   [active gate: {s.active_gate}]" if s.active_gate else ""
    lines.append(f"why: {s.chain_to_goal or '(no chain to goal)'}{gate}")
    if s.check:
        lines.append(f"check: {s.check}")
    if s.deps:
        lines.append("deps: " + ", ".join(f"{d.id}={d.state}" for d in s.deps))
    if s.unblocks:
        # the reverse-REQUIRES the old packet computed then discarded — what this node frees up
        lines.append("unblocks: " + ", ".join(s.unblocks))
    pv = s.providers
    if pv.candidates:
        chosen = f"  chosen={pv.chosen}" if pv.chosen else ""
        lines.append("providers: candidates=[" + ", ".join(c.id for c in pv.candidates) + "]" + chosen)
    for t in pv.tried_failed:
        avoid = f"  (=> avoid {t.provider_id})" if t.provider_id else ""
        lines.append(f"tried & FAILED: {t.command}{avoid}")
    if s.layer_cohort_satisfied or s.layer_cohort_missing:
        lines.append(
            f"layer ({s.layer}): satisfied=[{', '.join(s.layer_cohort_satisfied)}]  "
            f"missing=[{', '.join(s.layer_cohort_missing)}]"
        )
    if s.conflict:
        lines.append(s.conflict)
    if s.platform:
        lines.append(s.platform)
    if s.evidence:
        lines.append(f"evidence: {s.evidence}")
    return tuple(lines)


def build_requirement_slice(graph, node) -> RequirementSlice:
    """Pure read-time projection of `node` for the agent. Reuses advise's render helpers
    (imported lazily to avoid module load-order coupling)."""
    from src.python_deps.depgraph.advise import (
        _chain_to_goal, _conflict_note, _best_evidence_line, _platform_note,
    )
    from src.python_deps.depgraph.schema import NodeType, State

    deps = tuple(DepView(id=d.id, state=d.state.value) for d in graph.requires_of(node.id))
    unblocks = tuple(n.id for n in graph.required_by(node.id))
    cohort = [n for n in graph.nodes if n.layer == node.layer and n.id != node.id]
    sat = tuple(n.id for n in cohort if n.state is State.SATISFIED)
    miss = tuple(n.id for n in cohort if n.state is State.MISSING)
    goal = next((n for n in graph.nodes if n.type is NodeType.TEST), None)
    active_gate = goal.check_command if (goal and goal.check_command) else ""
    return RequirementSlice(
        node_id=node.id, kind=node.type.value, layer=node.layer.value, state=node.state.value,
        check=node.check_command or "", evidence=_best_evidence_line(node.evidence) or "",
        deps=deps, chain_to_goal=_chain_to_goal(graph, node) or "", unblocks=unblocks,
        layer_cohort_satisfied=sat, layer_cohort_missing=miss,
        conflict=_conflict_note(graph, node), providers=providers_view(node),
        active_gate=active_gate, platform=_platform_note(node),
    )
