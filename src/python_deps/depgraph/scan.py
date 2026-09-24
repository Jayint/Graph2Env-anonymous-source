"""Stage 1: static import scan -> Import + Test nodes.

Wraps :func:`python_deps.import_graph.scan_imports` (which classifies each
top-level import as ``stdlib`` / ``project_local`` / ``external``) and lifts the
``external`` findings into the concrete dependency graph of
``docs/DESIGN-static-probe-certified-dependency-graph.md`` section 5:

  * one ``Import`` node per external import
    (``type=IMPORT``, ``layer=NAMING``, ``discovered_by=STATIC_SCAN``,
    ``state=UNKNOWN``, ``provenance`` = the source file(s),
    ``check_command = python -c "import <name>"``);
  * one ``Test`` goal node (``TEST_NODE_ID``, ``layer=TESTS``,
    ``discovered_by=GOAL``, ``check_command = "python -m pytest -q"``) with a
    ``requires`` edge to every Import.

Static scanning is evidence, not completeness (design 4.1 / 10.1): dynamic and
plugin imports are not visible here.  This stage never sets ``state`` beyond
``UNKNOWN`` — only the host certifier (Task 8) flips it.
"""

from __future__ import annotations

import os
import re

from src.python_deps.import_graph import scan_imports

from .ids import TEST_NODE_ID, import_id
from .schema import (
    DepGraph,
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
)

TEST_NODE_NAME = "repo_tests_pass"
TEST_CHECK_COMMAND = "python -m pytest -q"

# Directory segments whose imports are NOT part of "running the repo properly":
# examples/docs/build artifacts pull in non-project deps (and sometimes non-PyPI
# example names) that wrongly inflate — and can collapse — the resolver closure.
# Scope the scan to project source + tests; an import survives if it appears in at
# least one non-excluded file.
_EXCLUDED_SEGMENTS: frozenset[str] = frozenset(
    {
        "examples", "example", "docs", "doc", "build", "dist", "samples",
        "sample", "benchmarks", "benchmark", "bench", "scripts", "script",
        ".github", ".tox", "node_modules", "site-packages", ".venv", "venv",
    }
)


def _is_excluded_path(path: str) -> bool:
    """True when every meaningful segment routes through an excluded directory."""
    segments = {seg.lower() for seg in re.split(r"[\\/]+", path) if seg}
    return bool(segments & _EXCLUDED_SEGMENTS)


def _in_scope_files(source_files: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only source files that are not under an excluded directory."""
    return tuple(f for f in source_files if not _is_excluded_path(f))


# Dirs never worth walking for local-name detection (vcs/build/venv noise).
_SKIP_WALK_DIRS: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache"}
) | _EXCLUDED_SEGMENTS


def _local_module_names(repo_path: str) -> frozenset[str]:
    """Names defined *inside* the repo — packages (dirs with ``__init__.py``) and
    top-level module files (``*.py`` stems), found anywhere (incl. nested under
    ``tests/``).

    ``scan_imports`` only treats root/``src`` modules as project-local, so test
    fixture sub-packages (e.g. flask's ``blueprintapp``/``site_package`` under
    ``tests/``) leak through as "external" and become bogus PyPI roots.  A name
    that resolves to a local file/dir is never a distribution the environment must
    install.
    """
    names: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_WALK_DIRS]
        if "__init__.py" in filenames:
            names.add(os.path.basename(dirpath))
        for fname in filenames:
            if fname.endswith(".py") and fname != "__init__.py":
                names.add(fname[:-3])
    return frozenset(names)


def local_module_names(repo_path: str) -> frozenset[str]:
    """Public alias for :func:`_local_module_names` (used by the diagnosis router)."""
    return _local_module_names(repo_path)


def _import_check_command(name: str) -> str:
    return f'python -c "import {name}"'


def _build_test_node() -> Node:
    return Node(
        id=TEST_NODE_ID,
        type=NodeType.TEST,
        name=TEST_NODE_NAME,
        layer=Layer.TESTS,
        discovered_by=DiscoveredBy.GOAL,
        state=State.UNKNOWN,
        check_command=TEST_CHECK_COMMAND,
    )


def _build_import_node(name: str, source_files: tuple[str, ...]) -> Node:
    provenance = ", ".join(source_files) if source_files else None
    return Node(
        id=import_id(name),
        type=NodeType.IMPORT,
        name=name,
        layer=Layer.NAMING,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=State.UNKNOWN,
        check_command=_import_check_command(name),
        provenance=provenance,
    )


def scan_to_nodes(repo_path: str) -> DepGraph:
    """Scan ``repo_path`` and return a graph of external Import nodes plus the
    Test goal node, joined by ``requires`` edges (Test -> each Import).

    Only ``external`` imports become nodes; stdlib and project-local imports are
    dropped (their classification is reused from ``scan_imports``).
    """
    findings, _project_local, _errors = scan_imports(repo_path)
    local_names = _local_module_names(repo_path)

    graph = DepGraph().with_node(_build_test_node())

    for finding in findings:
        if finding.classification != "external":
            continue
        name = finding.import_name
        # Typing-only / private modules (e.g. ``_typeshed``) are not installable.
        if name.startswith("_"):
            continue
        # In-repo packages/modules (incl. nested test fixtures like flask's
        # ``blueprintapp``) are local, not PyPI distributions — drop them.
        if name in local_names:
            continue
        # Scope to project source + tests: drop imports seen ONLY in
        # examples/docs/build (they pull non-project / non-PyPI names).
        in_scope = _in_scope_files(finding.source_files)
        if finding.source_files and not in_scope:
            continue
        provenance_files = in_scope or finding.source_files
        graph = graph.with_node(
            _build_import_node(finding.import_name, provenance_files)
        )
        graph = graph.with_edge(
            Edge(
                src=TEST_NODE_ID,
                dst=import_id(finding.import_name),
                relation=EdgeType.REQUIRES,
                origin="scan",
            )
        )

    return graph
