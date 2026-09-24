# src/envstate/world_model.py
"""EnvState v1 — shared world-model types and pure map helpers.

All types are frozen dataclasses (immutable).  The only mutable container is
WorldModelMap.progress (a plain dict), but merge_map() always copies it so
callers never receive an alias to a live dict.

Serialization helpers (map_to_dict / map_from_dict) let the maintainer and
planner embed the map as JSON in LLM messages.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Any

from src.envstate.contracts.graph import ContractGraph

if TYPE_CHECKING:
    # Imported only for type-checking to avoid a runtime import cycle
    # (python_deps depends on nothing here; this stays a string annotation).
    from src.python_deps.depgraph.block import Block
    from src.python_deps.depgraph.schema import DepGraph


# ---------------------------------------------------------------------------
# Primitive facts
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Fact:
    name: str               # e.g. "flask", "pytest", "libpq-dev"
    detail: str = ""        # version / note, taken from real command output


@dataclasses.dataclass(frozen=True)
class OpenProblem:
    signature: str          # short id, e.g. "ModuleNotFoundError: psycopg2"
    interpretation: str     # maintainer's interpretation of the failure
    layer: str              # "base" | "system" | "runtime" | "deps" | "build" | "tests"
    out_of_scope: bool = False  # set by the planner when routing around it


# ---------------------------------------------------------------------------
# Dependency state + recipe types (keystone)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DependencyState:
    declared: tuple[Fact, ...] = ()
    resolved: tuple[Fact, ...] = ()          # from `python -m pip inspect`
    package_manager: str = "pip"
    test_framework: str = "pytest"


@dataclasses.dataclass(frozen=True)
class RecipeStep:
    id: str
    kind: str                 # AttemptKind value
    command: str
    target_node_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class RecipePatch:
    steps: tuple[RecipeStep, ...] = ()


# ---------------------------------------------------------------------------
# The shared map (single writer: Maintainer)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WorldModelMap:
    base_image: str
    workdir: str
    language: str                        # e.g. "python 3.12"
    build_system: str                    # "poetry" | "pip" | "hatchling" | "unknown"
    repo_layout: tuple[str, ...]         # key dirs/files: ("tests/", "src/", "pyproject.toml")
    required: tuple[Fact, ...]           # declared by manifests (not yet verified)
    installed: tuple[Fact, ...]          # confirmed present from real command results
    open_problems: tuple[OpenProblem, ...]
    progress: dict[str, bool]            # keys: base, system, runtime, deps, build, tests
    done_flag: bool = False              # True once pytest --collect-only exited 0
    notes: tuple[str, ...] = ()          # durable cautions the maintainer wants kept
    env: dict[str, str] = dataclasses.field(default_factory=dict)   # scalar probe facts
    system_installed: tuple[Fact, ...] = ()   # apt names + pkg-config modules + tools present
    contract_graph: ContractGraph = dataclasses.field(default_factory=ContractGraph.empty)
    dependency_state: DependencyState | None = None
    import_results: tuple[tuple[str, bool], ...] = ()   # (import_name, ok) from the sweep
    host_satisfied: frozenset[str] = dataclasses.field(default_factory=frozenset)
    # Phase-0 dep-graph advisory: a static, host-certified rendering built once in
    # a scratch container and spliced into the planner prompt. "" when the feature
    # is off or the build failed (graceful degradation). Advisory only — never
    # consulted by control/done/synthesis; the depgraph writes nowhere else.
    dep_advisory: str = ""
    # The built DepGraph object itself, carried in-process so a later seed adapter
    # can project contract nodes from it. None when the feature is off or the build
    # failed. Not serialized (map_to_dict/from_dict) — it is consumed in-process only.
    dep_graph: "DepGraph | None" = None
    # Governed manual blocks (LLM-admitted ScriptPatches) carried in-process so the
    # FINAL rendered setup.sh includes them. Not serialized (map_to_dict/from_dict) —
    # consumed in-process only, same as dep_graph above.
    manual_blocks: tuple["Block", ...] = ()


# ---------------------------------------------------------------------------
# Planner → BuildAgent message
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TransitionProposal:
    kind: str
    target: str
    intent: str
    command_templates: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Task:
    goal: str               # one concrete sub-goal: "install project deps from pyproject"
    done_when: str          # checkable criterion: "pip install exits 0 and import works"
    layer: str              # which stack layer this targets
    facts: tuple[str, ...]  # relevant map facts handed down (so the agent does not re-discover)
    target_node_ids: tuple[str, ...] = ()
    transition_proposal: "TransitionProposal | None" = None


@dataclasses.dataclass(frozen=True)
class PlannerDecision:
    action: str                       # "task" | "done" | "giveup" | "apply_recipe_patch"
    task: Task | None = None
    reason: str = ""                  # explanation for done/giveup
    satisfied_goal_contract_ids: tuple[str, ...] = ()
    recipe_patch: RecipePatch | None = None


# ---------------------------------------------------------------------------
# BuildAgent → Maintainer message
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CommandRecord:
    cmd: str
    rc: int                 # 0 = success, non-zero = failure (proxy, same as ActionLedger)
    output: str             # truncated salient output


@dataclasses.dataclass(frozen=True)
class TaskReport:
    task_goal: str
    status: str             # "done" | "blocked"
    commands: tuple[CommandRecord, ...]
    learning: str           # one line: what was learned / why blocked
    # Number of leading recipe steps whose command completed successfully before
    # the run stopped.  Set by build_agent.run_recipe so the orchestrator can
    # attribute per-step Attempt outcomes (BUG-10).  None on reports that do not
    # come from a recipe run (back-compat: derive_attempt_outcome falls back to
    # the recipe-level status flag).
    completed_steps: int | None = None


# ---------------------------------------------------------------------------
# Factories and pure helpers
# ---------------------------------------------------------------------------

_PROGRESS_LAYERS: tuple[str, ...] = (
    "base", "system", "runtime", "deps", "build", "tests"
)


def initial_map(
    base_image: str,
    workdir: str,
    language: str,
    build_system: str,
    repo_layout: tuple[str, ...],
    required: tuple[Fact, ...] = (),
    dep_advisory: str = "",
    dep_graph: "DepGraph | None" = None,
) -> WorldModelMap:
    """Return a fresh WorldModelMap at cycle 0.

    Sets done_flag=False, empty installed/open_problems/notes, and
    progress={layer: False} for all six known layers.
    Called once in run_v1() before the loop.
    base_image and workdir come from Synthesizer;
    language/build_system/repo_layout come from ImageSelector + repo tree scan.
    """
    return WorldModelMap(
        base_image=base_image,
        workdir=workdir,
        language=language,
        build_system=build_system,
        repo_layout=repo_layout,
        required=required,
        installed=(),
        open_problems=(),
        progress={layer: False for layer in _PROGRESS_LAYERS},
        done_flag=False,
        notes=(),
        contract_graph=ContractGraph.empty(),
        dependency_state=None,
        import_results=(),
        host_satisfied=frozenset(),
        dep_advisory=dep_advisory,
        dep_graph=dep_graph,
    )


def merge_map(
    current: WorldModelMap,
    *,
    installed: tuple[Fact, ...] | None = None,
    open_problems: tuple[OpenProblem, ...] | None = None,
    progress: dict[str, bool] | None = None,
    done_flag: bool | None = None,
    notes: tuple[str, ...] | None = None,
    required: tuple[Fact, ...] | None = None,
    env: dict[str, str] | None = None,
    build_system: str | None = None,
    language: str | None = None,
    system_installed: tuple[Fact, ...] | None = None,
    contract_graph: ContractGraph | None = None,
    dependency_state: DependencyState | None = None,
    import_results: tuple[tuple[str, bool], ...] | None = None,
    host_satisfied: frozenset[str] | None = None,
    dep_advisory: str | None = None,
    dep_graph: "DepGraph | None" = None,
    manual_blocks: "tuple[Block, ...] | None" = None,
) -> WorldModelMap:
    """Return a new WorldModelMap with only the supplied keyword fields replaced.

    progress and env are always copied so callers never share a live dict.
    Never raises.
    """
    return dataclasses.replace(
        current,
        installed=installed if installed is not None else current.installed,
        open_problems=open_problems if open_problems is not None else current.open_problems,
        progress=dict(progress) if progress is not None else dict(current.progress),
        done_flag=done_flag if done_flag is not None else current.done_flag,
        notes=notes if notes is not None else current.notes,
        required=required if required is not None else current.required,
        env=dict(env) if env is not None else dict(current.env),
        build_system=build_system if build_system is not None else current.build_system,
        language=language if language is not None else current.language,
        system_installed=system_installed if system_installed is not None else current.system_installed,
        contract_graph=contract_graph if contract_graph is not None else current.contract_graph,
        dependency_state=dependency_state if dependency_state is not None else current.dependency_state,
        import_results=import_results if import_results is not None else current.import_results,
        host_satisfied=host_satisfied if host_satisfied is not None else current.host_satisfied,
        dep_advisory=dep_advisory if dep_advisory is not None else current.dep_advisory,
        dep_graph=dep_graph if dep_graph is not None else current.dep_graph,
        manual_blocks=manual_blocks if manual_blocks is not None else current.manual_blocks,
    )




def _build_required(repo_layout: tuple[str, ...]) -> bool:
    """Return True when the repo needs a compile step before tests can run.

    Detects (case-insensitive, by basename):
    - Build-system files: Makefile, makefile, GNUmakefile, CMakeLists.txt,
      configure, configure.ac, meson.build, Cargo.toml, go.mod
    - Compiled source extensions: .c, .cc, .cpp, .cxx, .go, .rs

    Pure; never mutates its argument. Returns False for empty layouts and
    for repos whose layout contains only Python/scripted files.
    """
    _BUILD_FILENAMES: frozenset[str] = frozenset({
        "makefile", "gnumakefile", "cmakelists.txt",
        "configure", "configure.ac", "meson.build",
        "cargo.toml", "go.mod",
    })
    _BUILD_EXTENSIONS: frozenset[str] = frozenset({
        ".c", ".cc", ".cpp", ".cxx", ".go", ".rs",
    })
    for entry in repo_layout:
        basename = entry.rstrip("/").rsplit("/", 1)[-1].lower()
        if basename in _BUILD_FILENAMES:
            return True
        _, _, ext = basename.rpartition(".")
        if ext and "." + ext in _BUILD_EXTENSIONS:
            return True
    return False


def _derive_progress(prev: dict[str, bool], m: WorldModelMap) -> dict[str, bool]:
    """Deterministically compute layer progress from facts; monotonic vs prev.

    Clean-signal layers: base/runtime/deps/tests. Signal-less layers
    (system/build) are complete unless an unresolved open_problem targets
    them. OR-merged with prev so a layer never flips True->False mid-run.

    Build-layer special case (Phase 3):
      For repos where _build_required() is True (Makefile, C/Go/Rust sources, …),
      the build layer is NOT auto-completed merely because no open_problem targets
      it. Keeping build=False as a standing planning concern nudges the Planner to
      emit a compile step (e.g. `make`) rather than skipping it.
      For pure-Python repos (_build_required() is False) the signal-less logic is
      unchanged: build is True unless a build open_problem is present.
      Finalization is the execution gate (done_flag), not progress; this never
      produces a false success, only a more conservative planning signal.
    """
    installed_lower = {f.name.lower() for f in m.installed}
    deps_ok = bool(m.required) and all(
        r.name.lower() in installed_lower for r in m.required
    )
    open_layers = {p.layer for p in m.open_problems if not p.out_of_scope}
    build_required = _build_required(m.repo_layout)
    # For build-required repos, suppress the signal-less True so the Planner
    # is forced to propose a compile step; for pure-Python repos, keep old behaviour.
    computed_build = ("build" not in open_layers) and not build_required
    computed = {
        "base": bool(m.base_image),
        "system": "system" not in open_layers,
        "runtime": bool(m.env.get("python_version")),
        "deps": deps_ok,
        "build": computed_build,
        "tests": bool(m.done_flag),
    }
    return {
        layer: bool(prev.get(layer, False)) or bool(computed.get(layer, False))
        for layer in _PROGRESS_LAYERS
    }


def _auto_resolve_problems(
    open_problems: tuple[OpenProblem, ...],
    installed: tuple[Fact, ...],
) -> tuple[OpenProblem, ...]:
    """Drop a problem whose package name appears (case-insensitive) in its
    signature once that package is in `installed`.

    Conservative exact-name match: covers pip ModuleNotFoundError cases.
    Package-vs-import-name mismatches (psycopg2 vs psycopg2-binary) are left
    for the Maintainer's `resolved` list. Never raises.
    """
    if not installed:
        return open_problems
    names = [f.name.lower() for f in installed if f.name]
    kept: list[OpenProblem] = []
    for p in open_problems:
        sig = p.signature.lower()
        if any(name in sig for name in names):
            continue  # resolved
        kept.append(p)
    return tuple(kept)


# Failure-signature shapes → the missing artifact token.
_SYS_ARTIFACT_PATTERNS = (
    re.compile(r"([A-Za-z0-9_.+-]+)\s*(?::\s*)?(?:command not found|executable not found|: not found)", re.I),
    re.compile(r"no package '([^']+)' found", re.I),
    re.compile(r"cannot find -l(\S+)", re.I),
    re.compile(r"fatal error:\s*([A-Za-z0-9_./+-]+)\.h\b", re.I),
)


def _system_artifact(signature: str) -> str | None:
    """Best-effort extraction of the missing tool/lib/module from a signature."""
    for pat in _SYS_ARTIFACT_PATTERNS:
        m = pat.search(signature)
        if m:
            return m.group(1).strip().lower()
    return None


def _auto_resolve_system_problems(
    open_problems: tuple[OpenProblem, ...],
    system_installed: tuple[Fact, ...],
) -> tuple[OpenProblem, ...]:
    """Drop a layer=='system' problem once the probe confirms its missing artifact
    is present in system_installed (apt names / pkg-config modules / tools on PATH).

    Matches against WHAT THE ARTIFACT IS, not the apt package name (the signature
    names pg_config, not libpq-dev). Conservative: unrecognized shapes are kept.
    Never raises; leaves non-system problems untouched.
    """
    if not system_installed:
        return open_problems
    present = {f.name.lower() for f in system_installed if f.name}
    # also index without a leading 'lib' so 'libpq' matches 'pq' artifacts and vice-versa
    present |= {n[3:] for n in present if n.startswith("lib")}
    present |= {"lib" + n for n in list(present)}

    kept: list[OpenProblem] = []
    for p in open_problems:
        if p.layer != "system":
            kept.append(p)
            continue
        art = _system_artifact(p.signature)
        if art is not None and (art in present or art.replace("lib", "") in present):
            continue  # resolved deterministically
        kept.append(p)
    return tuple(kept)


def apply_deterministic(
    current: WorldModelMap,
    snap: Any,   # EnvSnapshot (duck-typed: .installed, .env)
    man: Any,    # ManifestResult (duck-typed: .build_system, .required)
) -> WorldModelMap:
    """Fold deterministic facts into the map. Pure; never raises.

    REPLACES installed/env/build_system/required/language from snap+man.
    Empty snap.env => probe failed => keep prior installed/env (degrade).
    AUTO-RESOLVES pip problems; RECOMPUTES progress deterministically.
    Leaves notes/base_image/workdir/repo_layout/done_flag untouched.
    """
    if snap.env:  # non-empty env == probe succeeded (arch always reads on success)
        installed = tuple(snap.installed)
        env = dict(snap.env)
        language = snap.env.get("python_version") or current.language
        system_installed = tuple(getattr(snap, "system_installed", ()))
        # Preserve current import_results when the snapshot has none: probe_env()
        # always returns import_results=() because the sweep is driven separately
        # (validators.build_import_sweep_command / exec_readonly), not by the extractor.
        # Unconditionally overwriting with () would erase results written by the sweep.
        _snap_ir: tuple[tuple[str, bool], ...] = tuple(getattr(snap, "import_results", ()))
        import_results: tuple[tuple[str, bool], ...] = _snap_ir if _snap_ir else current.import_results
    else:
        installed = current.installed
        env = dict(current.env)
        language = current.language
        system_installed = current.system_installed
        import_results = current.import_results

    build_system = (
        man.build_system
        if man.build_system and man.build_system != "unknown"
        else current.build_system
    )
    resolved = _auto_resolve_problems(current.open_problems, installed)
    resolved = _auto_resolve_system_problems(resolved, system_installed)   # NEW

    interim = merge_map(
        current,
        installed=installed,
        env=env,
        required=tuple(man.required),
        build_system=build_system,
        language=language,
        open_problems=resolved,
        system_installed=system_installed,
        import_results=import_results,
    )
    progress = _derive_progress(current.progress, interim)
    return merge_map(interim, progress=progress)

# ---------------------------------------------------------------------------
# JSON serialization helpers (for LLM message embedding)
# ---------------------------------------------------------------------------

def _fact_to_dict(f: Fact) -> dict[str, Any]:
    return {"name": f.name, "detail": f.detail}


def _fact_from_dict(d: dict[str, Any]) -> Fact:
    return Fact(name=d["name"], detail=d.get("detail", ""))


def _open_problem_to_dict(op: OpenProblem) -> dict[str, Any]:
    return {
        "signature": op.signature,
        "interpretation": op.interpretation,
        "layer": op.layer,
        "out_of_scope": op.out_of_scope,
    }


def _open_problem_from_dict(d: dict[str, Any]) -> OpenProblem:
    return OpenProblem(
        signature=d["signature"],
        interpretation=d["interpretation"],
        layer=d["layer"],
        out_of_scope=bool(d.get("out_of_scope", False)),
    )


def _dependency_state_to_dict(ds: DependencyState) -> dict[str, Any]:
    return {
        "declared": [_fact_to_dict(f) for f in ds.declared],
        "resolved": [_fact_to_dict(f) for f in ds.resolved],
        "package_manager": ds.package_manager,
        "test_framework": ds.test_framework,
    }


def _dependency_state_from_dict(d: dict[str, Any]) -> DependencyState:
    return DependencyState(
        declared=tuple(_fact_from_dict(f) for f in d.get("declared", [])),
        resolved=tuple(_fact_from_dict(f) for f in d.get("resolved", [])),
        package_manager=d.get("package_manager", "pip"),
        test_framework=d.get("test_framework", "pytest"),
    )


def map_to_dict(m: WorldModelMap) -> dict[str, Any]:
    """Serialize a WorldModelMap to a plain JSON-safe dict.

    Suitable for embedding in LLM messages via json.dumps().
    Tuples are serialized as lists (JSON has no tuple type).
    """
    return {
        "base_image": m.base_image,
        "workdir": m.workdir,
        "language": m.language,
        "build_system": m.build_system,
        "repo_layout": list(m.repo_layout),
        "required": [_fact_to_dict(f) for f in m.required],
        "installed": [_fact_to_dict(f) for f in m.installed],
        "open_problems": [_open_problem_to_dict(op) for op in m.open_problems],
        "progress": dict(m.progress),
        "done_flag": m.done_flag,
        "notes": list(m.notes),
        "env": dict(m.env),
        "system_installed": [_fact_to_dict(f) for f in m.system_installed],
        "contract_graph": m.contract_graph.to_dict(),
        "dependency_state": _dependency_state_to_dict(m.dependency_state) if m.dependency_state is not None else None,
        "import_results": [list(pair) for pair in m.import_results],
        "host_satisfied": sorted(m.host_satisfied),
        "dep_advisory": m.dep_advisory,
    }


def map_from_dict(d: dict[str, Any]) -> WorldModelMap:
    """Deserialize a WorldModelMap from a plain dict (inverse of map_to_dict).

    Never mutates *d*.
    """
    raw_ds = d.get("dependency_state")
    raw_ir = d.get("import_results", [])
    raw_hs = d.get("host_satisfied", [])
    return WorldModelMap(
        base_image=d["base_image"],
        workdir=d["workdir"],
        language=d["language"],
        build_system=d["build_system"],
        repo_layout=tuple(d.get("repo_layout", [])),
        required=tuple(_fact_from_dict(f) for f in d.get("required", [])),
        installed=tuple(_fact_from_dict(f) for f in d.get("installed", [])),
        open_problems=tuple(
            _open_problem_from_dict(op) for op in d.get("open_problems", [])
        ),
        progress=dict(d.get("progress", {layer: False for layer in _PROGRESS_LAYERS})),
        done_flag=bool(d.get("done_flag", False)),
        notes=tuple(d.get("notes", [])),
        env=dict(d.get("env", {})),
        system_installed=tuple(_fact_from_dict(f) for f in d.get("system_installed", [])),
        contract_graph=ContractGraph.from_dict(d.get("contract_graph", {})),
        dependency_state=_dependency_state_from_dict(raw_ds) if raw_ds is not None else None,
        import_results=tuple((str(pair[0]), bool(pair[1])) for pair in raw_ir),
        host_satisfied=frozenset(raw_hs),
        dep_advisory=str(d.get("dep_advisory", "")),
    )


# ---------------------------------------------------------------------------
# Derived views over the contract graph
# ---------------------------------------------------------------------------

def derive_open_problems(graph: ContractGraph) -> tuple[OpenProblem, ...]:
    """Return OpenProblem entries derived from active Blockers in the graph.

    Only Blockers with ``active=True`` (or missing ``active``, defaulting True)
    are included.  ``layer`` defaults to ``"deps"`` when absent.
    """
    out: list[OpenProblem] = []
    for b in graph.blockers():
        if bool(b.data.get("active", True)):
            out.append(OpenProblem(
                signature=b.data.get("signature", ""),
                interpretation=b.data.get("summary", ""),
                layer=b.data.get("layer", "deps"),
            ))
    return tuple(out)
