"""Release-aware apt-name verification (fixes stale predicted/table names).

The curated tables (``PACKAGE_TO_SYSTEM_DEPS``, ``apt_for_soname``) encode ONE
Debian release's apt names (bookworm-era). On a newer target image they go stale
— most visibly the **t64 ABI transition** (``libglib2.0-0`` ->
``libglib2.0-0t64``). The probe/seed stages emit those names verbatim, so the
advisory's fix-candidate can be wrong for the actual image.

This stage closes that gap WITHOUT abandoning the curated knowledge (which is the
only proactive source of "opencv needs a GL lib"): it keeps the table's *fact*
but verifies the *name* against the target image, remapping when the predicted
name is not installable. Pure parser + thin executor orchestrator (mirrors
``apt_resolve.py``); Debian/Ubuntu only.
"""

from __future__ import annotations

import shlex
from dataclasses import replace

from src.python_deps.depgraph.executor import Executor
from src.python_deps.depgraph.schema import DepGraph, Node


def apt_name_installable(stdout: str) -> bool:
    """True when ``apt-cache show <name>`` returned a real package record.

    A missing package yields empty stdout (+ an ``N: Unable to locate`` notice on
    stderr), so the presence of a ``Package:`` field is the reliable signal.
    """
    return any(line.startswith("Package:") for line in (stdout or "").splitlines())


def t64_variant(name: str) -> str | None:
    """The Debian time_t-64 transition appends ``t64`` to the runtime-lib package
    (``libglib2.0-0`` -> ``libglib2.0-0t64``). ``None`` when already a t64 name."""
    if name.endswith("t64"):
        return None
    return name + "t64"


_SHOWPKG_REVERSE_PROVIDES = "Reverse Provides:"


def parse_showpkg_reverse_provides(stdout: str) -> list[str]:
    """Extract provider package names from ``apt-cache showpkg`` stdout.

    Locates the ``Reverse Provides:`` section and returns the list of provider
    package names found there (one per ``<name> <version>`` entry, order
    preserved, duplicates included — callers must dedup). Returns an empty list
    when the section is absent or empty.
    """
    lines = (stdout or "").splitlines()
    in_section = False
    providers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped.startswith(_SHOWPKG_REVERSE_PROVIDES):
                in_section = True
            continue
        # A line ending with ":" is the next section header — stop. (An empty
        # string can never satisfy ``endswith(":")``, so no extra guard is needed
        # — review LOW-1.)
        if stripped.endswith(":"):
            break
        if stripped:
            parts = stripped.split()
            if parts:
                providers.append(parts[0])
    return providers


def resolve_virtual_provider(name: str, executor: Executor) -> str | None:
    """Real provider of a virtual/renamed package via ``apt-cache showpkg``
    'Reverse Provides' (e.g. ``libglib2.0-0`` -> ``libglib2.0-0t64``), else None.

    ``showpkg`` always exits 0; the decision is based on output content, not
    return code. Multiple 'Reverse Provides' lines for different versions of the
    *same* provider are deduplicated. Returns the unique provider name when
    exactly one distinct provider exists, else None (zero or more-than-one).
    """
    result = executor.run(f"apt-cache showpkg {shlex.quote(name)}")
    providers = parse_showpkg_reverse_provides(result.stdout)
    unique = list(dict.fromkeys(providers))  # dedup, preserve insertion order
    return unique[0] if len(unique) == 1 else None


def resolve_installable_apt_name(candidate: str, executor: Executor) -> str:
    """Return an apt package name that actually exists on the target image.

    Resolution order:

    1. Candidate as-is (``apt-cache show``).
    2. Virtual-provider lookup via ``apt-cache showpkg`` Reverse Provides —
       authoritative remap for virtual/renamed packages (e.g. the t64 ABI
       transition) without relying on a suffix heuristic.
    3. t64 suffix heuristic (``<name>t64``) as a last-resort fallback.

    Best-effort: returns the original candidate unchanged when none resolves
    (never worse than the table name).
    """
    # 1. Candidate installable as-is.
    result = executor.run(f"apt-cache show {shlex.quote(candidate)}")
    if apt_name_installable(result.stdout):
        return candidate

    # 2. Virtual-provider resolution via showpkg (authoritative remap).
    provider = resolve_virtual_provider(candidate, executor)
    if provider is not None:
        result = executor.run(f"apt-cache show {shlex.quote(provider)}")
        if apt_name_installable(result.stdout):
            return provider

    # 3. t64 suffix heuristic (last resort; kept for robustness).
    t64 = t64_variant(candidate)
    if t64 is not None:
        result = executor.run(f"apt-cache show {shlex.quote(t64)}")
        if apt_name_installable(result.stdout):
            return t64

    return candidate


def _apt_package(node: Node) -> str | None:
    """The apt package in a node's first ``apt:`` fix-candidate, else None."""
    for fix in node.fix_candidates:
        if fix.startswith("apt:"):
            return fix[len("apt:") :]
    return None


def reconcile_apt_names(graph: DepGraph, executor: Executor) -> DepGraph:
    """Verify every node's ``apt:`` fix-candidate against the target image and
    remap stale names (e.g. the t64 transition). Returns a NEW graph.

    Runs ``apt-get update`` once so apt metadata is available on slim images
    (which ship empty lists). No-op — and no ``apt-get update`` — when the graph
    has no apt fix-candidates, so pure-Python repos pay nothing.

    Only the ``fix_candidates`` (the actionable part) are rewritten; a node named
    by its apt package (a seed prediction) also has its ``name``/``chosen_fix``
    and its ``dpkg -s <name>`` ``check_command`` updated (else certification would
    forever check the stale name and report MISSING after the correct package is
    installed), while a soname-named node (e.g. ``libGL.so.1``) keeps its identity
    and its ``ldconfig``-based check (which is release-independent).
    """
    apt_nodes = [(n, pkg) for n in graph.nodes if (pkg := _apt_package(n))]
    if not apt_nodes:
        return graph

    executor.run("apt-get update")  # best-effort; slim images ship empty lists
    resolved: dict[str, str] = {}
    new = graph
    for node, pkg in apt_nodes:
        if pkg not in resolved:
            resolved[pkg] = resolve_installable_apt_name(pkg, executor)
        fixed = resolved[pkg]
        if fixed == pkg:
            continue
        changes: dict = {"fix_candidates": (f"apt:{fixed}",)}
        if node.name == pkg:  # named by its apt package -> a seed prediction
            changes["name"] = fixed
        if node.chosen_fix == f"apt:{pkg}":
            changes["chosen_fix"] = f"apt:{fixed}"
        # Remap a seed node's apt-name check too; leave soname/ldconfig checks
        # (and any non-dpkg check) untouched.
        if node.check_command == f"dpkg -s {pkg}":
            changes["check_command"] = f"dpkg -s {fixed}"
        new = new.with_node(replace(node, **changes))
    return new
