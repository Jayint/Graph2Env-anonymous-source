"""Deterministic static evidence collectors → compact LLM bundle (design §5.2). Pure.

Thin adapter that RESHAPES the existing config/service scanners' output into the
§5.2 bundle rows. It does NOT scan the repo blindly and does NOT create graph truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.python_deps.depgraph.config_scan import (
    scan_env_reads, parse_env_example, scan_framework_config_reads, configured_vars,
    scan_env_defaults,
)
from src.python_deps.depgraph.service_scan import scan_ci_services, scan_compose_services

_GOAL = ("Infer local install/test/run environment requirements, not deployment "
         "requirements.")


@dataclass(frozen=True)
class DeterministicHit:
    evidence_id: str
    file: str
    kind: str                     # ci_service | compose_service | env_var | env_read | package | project
    snippet: str = ""
    name: str | None = None
    node_id: str | None = None


def collect_static_evidence(repo_path: str, graph=None) -> tuple[DeterministicHit, ...]:
    hits: list[DeterministicHit] = []
    n = 0

    def _add(file, kind, *, name=None, snippet="", node_id=None):
        nonlocal n
        prefix = {"ci_service": "ci", "compose_service": "svc",
                  "env_var": "env", "env_read": "code", "package": "pkg",
                  "project": "proj"}.get(kind, "ev")
        hits.append(DeterministicHit(f"{prefix}.{n:02d}", file, kind,
                                     snippet=snippet, name=name, node_id=node_id))
        n += 1

    ci_services, _has_ci = scan_ci_services(repo_path)
    for svc, meta in sorted(ci_services.items()):
        # VERIFIED: service meta carries image/host/port — NOT a file path; use a generic label.
        _add(".github/workflows", "ci_service", name=svc, snippet=str(meta.get("image", svc)))
    for svc, meta in sorted(scan_compose_services(repo_path).items()):
        _snip = str(meta.get("image", svc))
        if meta.get("port"):
            _snip += f" port={meta['port']}"
        if meta.get("healthcheck"):
            _snip += f" healthcheck={meta['healthcheck']}"
        _add("docker-compose.yml", "compose_service", name=svc, snippet=_snip)
    provided_vars = configured_vars(repo_path)
    defaulted_vars = set(scan_env_defaults(repo_path))
    for var, default in sorted(parse_env_example(repo_path).items()):
        if var in provided_vars:
            continue
        _add(".env.example", "env_var", name=var, snippet=str(default))
    for var, file in sorted(scan_env_reads(repo_path).items()):
        if var in provided_vars:
            continue
        if var in defaulted_vars:
            continue
        _add(file, "env_read", name=var)
    seen_vars = {h.name for h in hits if h.kind in ("env_var", "env_read")}
    for var, src in sorted(scan_framework_config_reads(repo_path).items()):
        if var in provided_vars:
            continue
        if var in seen_vars:
            continue
        seen_vars.add(var)
        _add(str(src), "env_var", name=var, snippet="settings/framework config field")
    if graph is not None:
        from src.python_deps.depgraph.schema import NodeType
        for node in sorted((n_ for n_ in graph.nodes if n_.type is NodeType.PACKAGE),
                           key=lambda x: x.name):
            _add("manifest", "package", name=node.name,
                 snippet=node.version or "", node_id=node.id)
        for node in sorted((n_ for n_ in graph.nodes if n_.type is NodeType.PROJECT),
                           key=lambda x: x.id):
            _add("manifest", "project", name=node.name, node_id=node.id)
    return tuple(hits)


def compact_bundle_json(hits: tuple[DeterministicHit, ...], goal: str = _GOAL) -> str:
    rows = []
    for h in hits:
        row = {"evidence_id": h.evidence_id, "file": h.file, "kind": h.kind}
        if h.name is not None:
            row["name"] = h.name
        if h.snippet:
            row["snippet"] = h.snippet
        if h.node_id is not None:
            row["node_id"] = h.node_id
        rows.append(row)
    return json.dumps({"goal": goal, "deterministic_hits": rows}, indent=2)
