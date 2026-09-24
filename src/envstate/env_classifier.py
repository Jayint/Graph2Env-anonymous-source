# src/envstate/env_classifier.py
"""Construction-time LLM environment classifier (design 2026-06-29, Slice C). The allowed
LLM bridge: python_deps/depgraph stays LLM-free; this envstate module calls the model and
feeds the result through the pure patch_gate. Best-effort: never raises into the build."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

logger = logging.getLogger(__name__)

_GOAL = ("Infer LOCAL install/test/run environment requirements (not deployment). For each "
         "need cite >=1 evidence_id from the bundle. Deployment-only / release / secret-store / "
         "cache / optional-matrix signals -> promotion 'hint' only. Every requirement needs a real "
         "check_command or null (a hint).")
_SYSTEM_PROMPT = (
    "You classify a compact evidence bundle into environment obligations for running a repo's "
    "tests locally. Output ONLY a JSON object: {\"add_requirements\":[{id,type,name,layer,"
    "check_command,promotion,evidence_ref}], \"add_edges\":[{source,target,relation,hard}]}.\n"
    "type in {Service,Config,DataAsset}; id is 'service:<name>' / 'config:<VAR>' / 'data:<name>'; "
    "layer in {services,config}; promotion in {hint,candidate} (NEVER active); evidence_ref MUST be "
    "an evidence_id from the bundle. Edges connect an existing node (e.g. a pkg: or project: id from "
    "the bundle) to your new node. "
    "Some bundle hits include a \"node_id\" (e.g. \"pkg:psycopg\", \"project:foo\"). To link a "
    "new node to an existing one, add an edge whose source/target are those exact node_id values. "
    "Valid edge relations are ONLY: requires, alternative_to, conflicts_with (default requires). "
    "Do NOT invent other relations, and do NOT create a node per package. " + _GOAL
)


def _normalize(d: dict) -> dict:
    """Map the recalled output shape onto what parse_patch_proposal expects:
    requirements->add_requirements, state->promotion (lowercased), evidence_refs->evidence_ref."""
    if not isinstance(d, dict):
        return {}
    patch = dict(d.get("patch", d))
    if "requirements" in patch and "add_requirements" not in patch:
        patch["add_requirements"] = patch.get("requirements")
    norm_reqs = []
    for r in (patch.get("add_requirements") or []):
        if not isinstance(r, dict):
            continue
        r = dict(r)
        prom = r.get("promotion") or r.get("state")
        if isinstance(prom, str):
            r["promotion"] = prom.strip().lower()
        if "evidence_ref" not in r:
            refs = r.get("evidence_refs")
            if isinstance(refs, (list, tuple)) and refs:
                r["evidence_ref"] = refs[0]
        norm_reqs.append(r)
    patch["add_requirements"] = norm_reqs
    return {"patch": patch}


def _sanitize(proposal, bundle_ids, graph):
    """Drop ungrounded/illegal requirements; force ALL edges soft; keep only edges whose
    endpoints exist (after the kept new nodes are accounted for) AND whose relation is a
    valid EdgeType value. An invalid-relation edge is dropped (not batch-voiding)."""
    from src.python_deps.depgraph.patch import PatchProposal
    from src.python_deps.depgraph.check_quality import check_can_detect_absence
    from src.python_deps.depgraph.patch_gate import _ALLOWED_PROMOTION, _KIND_PREFIX, is_read_only
    from src.python_deps.depgraph.service_tables import KNOWN_SERVICE_KINDS
    from src.python_deps.depgraph.schema import NodeType, EdgeType

    _valid_relations = {e.value for e in EdgeType}

    def _ok(r):
        # The parser is deliberately tolerant and therefore may preserve malformed
        # JSON value types from the model.  Treat those values as a bad entry instead
        # of letting an unhashable list/dict abort the whole classifier batch.
        if not isinstance(r.evidence_ref, str) or r.evidence_ref not in bundle_ids:
            return False
        try:
            nt = NodeType(r.type)
        except (TypeError, ValueError):
            return False
        prefix = _KIND_PREFIX.get(nt)
        if not (bool(prefix) and isinstance(r.id, str) and r.id.startswith(prefix)):
            return False
        if nt is NodeType.SERVICE:
            if r.name and not isinstance(r.name, str):
                return False
            service_name = (r.name or r.id.split(":", 1)[-1]).strip().lower()
            if service_name not in KNOWN_SERVICE_KINDS:
                return False
        # Drop (don't void the batch on) entries the gate would reject all-or-nothing:
        # an illegal promotion or a non-read-only check_command. One bad LLM field then
        # only loses that requirement, not every valid sibling in the proposal.
        if (r.promotion is not None
                and (not isinstance(r.promotion, str)
                     or r.promotion not in _ALLOWED_PROMOTION)):
            return False
        if r.check_command is not None and not isinstance(r.check_command, str):
            return False
        if r.check_command and not is_read_only(r.check_command):
            return False
        if r.check_command and not check_can_detect_absence(r.check_command):
            return False
        return True

    good_reqs = tuple(r for r in proposal.add_requirements if _ok(r))
    known = {r.id for r in good_reqs} | {n.id for n in graph.nodes}
    good_edges = tuple(replace(e, hard=False) for e in proposal.add_edges
                       if isinstance(e.source, str) and isinstance(e.target, str)
                       and isinstance(e.relation, str)
                       and e.source in known and e.target in known
                       and e.relation in _valid_relations)
    return PatchProposal(add_requirements=good_reqs, add_edges=good_edges)


def make_construction_classifier(complete_fn: Callable[[list[dict]], str]):
    """Return classify(graph, repo_path) -> graph. complete_fn(messages)->text (temp-0, JSON)."""
    def classify(graph, repo_path: str):
        try:
            from src.python_deps.depgraph.static_collect import (
                collect_static_evidence, compact_bundle_json)
            from src.python_deps.depgraph.patch import parse_patch_proposal
            from src.python_deps.depgraph.patch_gate import admit_proposal
            from src.envstate.jsonutil import extract_json_object

            hits = collect_static_evidence(repo_path, graph)
            if not hits:
                return graph
            bundle_ids = frozenset(h.evidence_id for h in hits)
            messages = [{"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": compact_bundle_json(hits, _GOAL)}]
            obj = extract_json_object(complete_fn(messages))
            if obj is None:
                return graph
            proposal = _sanitize(parse_patch_proposal(_normalize(obj)), bundle_ids, graph)
            if proposal.is_empty():
                return graph
            result = admit_proposal(graph, proposal, known_evidence_ids=bundle_ids)
            if not result.accepted:
                logger.warning("env classifier proposal rejected: %s", result.errors)
                return graph
            return result.graph
        except Exception as exc:                       # best-effort: never crash the build
            logger.warning("env classifier skipped: %s", exc)
            return graph
    return classify
