"""Python dependency utilities — v3-only branch.

The z3-era modules (models bulk classes, graph, z3_adapter, resolver,
constraints, report, pypi_metadata, external_graph) are excluded from the
v3-only branch.  Only the depgraph sub-package (uv.lock-based closure) and
the shared scan utilities (import_mapping, import_graph, import_graph.models
stub) are kept.

evidence.py and failure_classifier.py are also present but their eager
re-exports have been removed from this __init__ so that importing any
python_deps.depgraph.* submodule does not trigger z3-era class imports.
"""
from __future__ import annotations

__all__: list[str] = []
