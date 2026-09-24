"""Stage 4.5 — ldd-based run-time native library discovery.

After ``install_closure`` installs the resolved closure, this stage runs ``ldd``
on each package's compiled extension ``.so`` files and collects shared libraries
the dynamic linker reports as ``=> not found``.  Each missing soname becomes a
``SystemLib`` node (``discovered_by=PROBE``, ``state=MISSING``) with a
``requires`` edge from the owning ``Package``.

This is the *primary authoritative* source for run-time native-lib nodes
(option A of the plan).  The curated ``PACKAGE_TO_SYSTEM_DEPS`` table stays as a
proactive/install-fail fallback (seeded before install; ldd supersedes it for
successfully installed packages).

``resolve_soname_apt`` is table-first with an apt-file fallback that is ABSENT on
slim images.  An unknown soname yields a node with EMPTY ``fix_candidates``
(option A: *need* surfaced, apt *name* not known).  Option B (lazy apt-file)
closes that gap — see plan Future TODOs.

Pure parser + thin executor orchestrator (repo immutability: every "mutation"
returns a NEW ``DepGraph``).
"""

from __future__ import annotations

import json
import shlex

from src.python_deps.depgraph.apt_resolve import resolve_soname_apt
from src.python_deps.depgraph.executor import Executor
from src.python_deps.depgraph.ids import syslib_id
from src.python_deps.depgraph.probe import reconcile_predicted
from src.python_deps.depgraph.schema import (
    Attempt,
    DepGraph,
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
)
from src.python_deps.import_mapping import normalize_package_name

# One container round-trip: emit JSON {canonical_dist_name: [absolute ext-.so paths]}
# for all installed distributions via importlib.metadata.
#
# MUST-FIX guards applied inside the command (plan Task 1):
#   * d.files is None   -> skip (not crashed; ~9% of dists have no RECORD)
#   * absolute paths    -> via d.locate_file(f) (relative files() cause silent ldd misses)
#   * ext modules only  -> basename matches .cpython-NN[N]*.so or .abi3.so
#       (NN = 2-digit tag for Python 3.0-3.9 e.g. cpython-39; NNN = 3-digit for
#        3.10+ e.g. cpython-311 — the count MUST allow both or 3.9 ext modules
#        are silently skipped and their native libs never get ldd-probed)
#   * bundled helpers excluded:
#       - path containing .libs/  (manylinux auditwheel directory)
#       - basename matching ^lib<name>-<8hex>.so (auditwheel-renamed soname)
#
# Shell-quoting notes:
#   * Outer python -c uses double quotes -> inner Python strings use single quotes.
#   * $ in regex end-of-string anchors becomes \Z (shell double-quotes expand $
#     to empty; \Z is not a shell escape and passes through unchanged, then the
#     regex engine interprets \Z as end-of-string).
#   * DockerExecutor wraps in sh -c '...'; its single-quote escaping preserves
#     the inner Python single-quoted strings correctly.
EXT_SO_MAP_CMD = (
    "python -c \""
    "import importlib.metadata as M, json, re, os; "
    "dists = {}; "
    "EXT = re.compile('[.]cpython-[0-9]{2,3}.*[.]so\\Z|[.]abi3[.]so\\Z'); "
    "BND = re.compile('^lib[a-z0-9._+-]+-[0-9a-f]{8}[.]so\\Z'); "
    "_ = [dists.setdefault(c, []).append(p) "
    "for d in M.distributions() if d.files is not None "
    "for c in [re.sub('[-_.]+', '-', (d.metadata.get('Name') or '').strip()).lower()] "
    "if c "
    "for f in d.files "
    "if EXT.search(bn := os.path.basename(p := str(d.locate_file(f)))) "
    "and not BND.match(bn) "
    "and '.libs/' not in p]; "
    "print(json.dumps(dists))\""
)


def parse_ext_so_map(stdout: str) -> dict[str, list[str]]:
    """Parse the JSON ``{canonical_dist_name: [abs ext-.so paths]}`` output.

    Returns an empty dict on any parse failure so the caller no-ops gracefully.
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, val in data.items():
        if isinstance(key, str) and isinstance(val, list):
            out[key] = [v for v in val if isinstance(v, str)]
    return out


def parse_ldd_not_found(stdout: str) -> list[str]:
    """Sonames from ``=> not found`` lines; deduped, preserving first occurrence.

    Input may be multi-file ldd output (per-file ``path:`` headers).  Only lines
    containing ``=> not found`` are kept; the token before ``=>`` is extracted
    and stripped.  A soname that appears multiple times is returned once.
    """
    seen: set[str] = set()
    result: list[str] = []
    for line in stdout.splitlines():
        if "=> not found" not in line:
            continue
        arrow = line.find("=>")
        if arrow < 0:
            continue
        token = line[:arrow].strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def ldd_probe(graph: DepGraph, executor: Executor) -> DepGraph:
    """Stage 4.5: discover run-time native-lib gaps via ldd on extension modules.

    For each Package node: batch-ldd its extension ``.so`` files; collect
    ``=> not found`` sonames; resolve soname → apt via ``resolve_soname_apt``
    (fills ``chosen_fix`` only — never the id); then either reconcile with a
    seed RESOLVER prediction of the same CANONICAL SONAME id (keeping
    ``discovered_by=RESOLVER``) or create a fresh ``discovered_by=PROBE`` node.
    Adds a ``requires`` edge Package→SystemLib.  Returns a NEW graph; no-op for
    packages with no extension modules.

    Canonical identity (Task 9): the soname IS the SystemLib node's id (see
    ``seed.py`` module docstring "canonical rule").  Reconciliation is keyed by
    the soname, so the seed prediction and this observation always collapse
    onto ONE node — independent of whether ``resolve_soname_apt`` succeeds this
    round (the prior apt-keyed reconciliation split into two nodes whenever
    resolution failed; this cannot happen anymore).

    Option A: ``resolve_soname_apt`` is table-first with an apt-file fallback
    ABSENT on slim images.  An unknown soname (not in ``NATIVE_LIB_TO_APT``)
    yields a node with EMPTY ``fix_candidates`` — the *need* is surfaced but
    the apt *name* is not.  Option B (lazy apt-file) closes this gap — see
    plan Future TODOs.
    """
    so_result = executor.run(EXT_SO_MAP_CMD)
    if not so_result.ok:
        return graph
    so_map = parse_ext_so_map(so_result.stdout)
    if not so_map:
        return graph

    new = graph
    for pkg in [n for n in graph.nodes if n.type is NodeType.PACKAGE]:
        canon = normalize_package_name(pkg.name)
        so_paths = so_map.get(canon, [])
        if not so_paths:
            continue

        # Batch-ldd all extension .so files for this package in one round-trip.
        ldd_cmd = "ldd " + " ".join(shlex.quote(p) for p in so_paths)
        ldd_result = executor.run(ldd_cmd)
        if not ldd_result.ok:
            continue

        sonames = parse_ldd_not_found(ldd_result.stdout or "")
        for soname in sonames:
            apt, _source = resolve_soname_apt(soname, executor)
            check = f"ldconfig -p | grep {soname}"
            evidence = _first_line_with(ldd_result.stdout or "", soname)

            # Reconcile with a RESOLVER seed prediction of the same CANONICAL
            # SONAME id (keeps discovered_by=RESOLVER per the spec); fall back
            # to a fresh PROBE node using the soname as the id.
            predicted_id = syslib_id(soname)
            reconciled = reconcile_predicted(
                new, predicted_id, check=check, evidence=evidence, command=ldd_cmd
            )
            if reconciled is not None:
                node_id = reconciled.id
                new = new.with_node(reconciled)
            else:
                # A prior package in this same pass may already have produced a
                # PROBE node for this soname (e.g. two dists both report
                # ``libGL.so.1 => not found`` and no RESOLVER seed exists). Append
                # this package's attempt to the existing node instead of replacing
                # it wholesale, which would silently drop the earlier package's
                # attempt history (review MEDIUM-1).
                existing = new.get(predicted_id)
                if existing is not None:
                    node = existing.with_attempt(
                        Attempt(command=ldd_cmd, outcome="failed", check=check)
                    )
                else:
                    node = _make_syslib_node(
                        soname, ldd_result.stdout or "", ldd_cmd, apt=apt
                    )
                node_id = node.id
                new = new.with_node(node)

            new = new.with_edge(
                Edge(src=pkg.id, dst=node_id, relation=EdgeType.REQUIRES, origin="probe")
            )

    return new


# --------------------------------------------------------------------------- #
# Node builders and helpers                                                    #
# --------------------------------------------------------------------------- #

def _make_syslib_node(
    soname: str, ldd_output: str, command: str, *, apt: str | None = None
) -> Node:
    """Fresh probe-discovered SystemLib for a soname reported by ldd."""
    check = f"ldconfig -p | grep {soname}"
    node = Node(
        id=syslib_id(soname),
        type=NodeType.SYSTEM_LIB,
        name=soname,
        layer=Layer.SYSTEM,
        discovered_by=DiscoveredBy.PROBE,
        state=State.MISSING,
        check_command=check,
        evidence=_first_line_with(ldd_output, soname),
        fix_candidates=(f"apt:{apt}",) if apt else (),
        chosen_fix=f"apt:{apt}" if apt else None,
        provenance="ldd (observed)",
    )
    return node.with_attempt(Attempt(command=command, outcome="failed", check=check))


def _first_line_with(text: str, needle: str, max_chars: int = 500) -> str:
    """First line of ``text`` containing ``needle``, truncated; else head of text.

    Guards ``text is None`` (``(text or "")``) to mirror ``probe._first_line_with``
    so the two copies cannot drift into different crash behavior (review LOW).
    """
    for line in (text or "").splitlines():
        if needle in line:
            return line.strip()[:max_chars]
    return (text or "").strip()[:max_chars]
