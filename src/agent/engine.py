"""Free-action ReAct environment builder with an ordinary chat trajectory.

The Agent owns every construction decision.  The host executes requested
operations, performs representation conversion, and records evidence; it does
not validate the semantics of Agent-authored graph, script, or runtime changes.
"""
from __future__ import annotations

import copy
import difflib
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.python_deps.depgraph.build_script import render_build_script
from src.python_deps.depgraph.populate import populate_setup_commands
from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Ecosystem,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    Phase,
    State,
    Strength,
)
from src.agent.artifacts import append_jsonl, write_json, write_llm_exchange
from src.agent.contracts import Event, SessionState
from src.agent.context_compressor import CompressionConfig, compress_observation
from src.agent.conversation import (
    fit_messages_to_token_budget,
    graph_summary,
    replace_setup_block,
    resolve_context_budget,
    setup_block,
    truncate_messages,
    truncate_observation,
)
from src.agent.observation_fallback import (
    DEFAULT_FALLBACK_CHARS,
    DEFAULT_SUMMARY_CHARS,
    apply_rule_fallback,
    observation_sha256,
    observation_slice,
)
from src.agent.trajectory import ReActStep, local_window, message_chars, render_messages
from src.envstate.llm_response import complete_with_retry, json_response_format_kwargs


UPDATE_GRAPH_CHANGE_FIELDS = frozenset({
    "type", "name", "layer", "discovered_by",
    "version", "check_command", "evidence", "fix_candidates", "chosen_fix",
    "provenance", "certified_cycle", "build_from_source", "artifact", "hash",
    "resolved_python", "resolved_platform", "exclude_newer", "declared_specifier",
    "declared_marker", "manifest_source", "resolution_status", "resolution_error",
    "workspace", "package_manager", "declared_constraint", "resolved_locator",
    "lock_digest", "language", "language_role", "version_constraint",
    "setup_commands", "data",
})
_UPDATE_GRAPH_FIELDS_TEXT = ", ".join(sorted(UPDATE_GRAPH_CHANGE_FIELDS))
_LAYER_VALUES_TEXT = ", ".join(layer.value for layer in Layer)

_FAILURE_SIGNAL_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError|Traceback|command not found|"
    r"No such file or directory|connection (?:refused|failed)|"
    r"could not connect|cannot connect|collect(?:ion)? error|"
    r"^E\s{2,}|^ERROR\b|FAILED\b|Exception\b)",
    re.IGNORECASE,
)


def _failure_signal_lines(output: str, *, limit: int = 10) -> list[str]:
    """Extract a compact Agent-facing diagnostic from a large command output."""
    selected: list[str] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        line = " ".join(raw.strip().split())
        if not line or not _FAILURE_SIGNAL_RE.search(line):
            continue
        clipped = line[:600]
        if clipped not in seen:
            selected.append(clipped)
            seen.add(clipped)
        if len(selected) >= limit:
            break
    if selected:
        return selected
    fallback = [" ".join(line.strip().split()) for line in output.splitlines()]
    return [line[:600] for line in fallback if line][-min(limit, 6):]


def _failure_signature(output: str) -> str:
    """Return a stable-enough signature for repeated identical diagnoses."""
    signals = _failure_signal_lines(output, limit=3)
    return "\n".join(signals) or "(empty failure output)"


def _failure_focus(output: str, graph: DepGraph, repeat_count: int) -> str:
    """Render compact failure evidence plus unresolved service/config context."""
    unresolved = [
        {
            "id": node.id,
            "state": node.state.value,
            "check_command": node.check_command,
        }
        for node in graph.nodes
        if node.type in {NodeType.SERVICE, NodeType.CONFIG}
        and node.state is not State.SATISFIED
    ]
    return (
        "FAILURE_FOCUS\n"
        f"same_signature_count={repeat_count}\n"
        "first_diagnostic_signals="
        + json.dumps(_failure_signal_lines(output), ensure_ascii=False)
        + "\n"
        "unresolved_service_config_nodes="
        + json.dumps(unresolved[:30], ensure_ascii=False)
        + "\n"
        "Use this compact evidence to choose one repair. If the same signature "
        "repeats without a graph/setup/runtime change, persist a targeted repair "
        "before running setup or collect again."
    )


def _agent_node(spec: dict[str, Any]) -> Node:
    """Convert an Agent-authored JSON node into the typed DepGraph form.

    This is representation conversion only.  The Agent remains responsible for
    deciding whether the node, its commands, and its edges are correct.
    """
    required = ("id", "type", "name", "layer")
    missing = [field for field in required if not str(spec.get(field, "")).strip()]
    if missing:
        raise ValueError("node is missing required field(s): " + ", ".join(missing))
    values = dict(spec)
    values["id"] = str(values["id"]).strip()
    values["name"] = str(values["name"]).strip()
    values["type"] = NodeType(str(values["type"]))
    values["layer"] = Layer(str(values["layer"]))
    discovered_by = str(values.get("discovered_by", "runtime"))
    if discovered_by in {"agent", "react", "repair"}:
        discovered_by = "runtime"
    values["discovered_by"] = DiscoveredBy(discovered_by)
    values["state"] = State(str(values.get("state", "unknown")))
    for field_name, enum_type in (("ecosystem", Ecosystem), ("strength", Strength), ("phase", Phase)):
        if values.get(field_name) is not None:
            values[field_name] = enum_type(str(values[field_name]))
    for field_name in ("setup_commands", "fix_candidates"):
        if field_name in values:
            if not isinstance(values[field_name], list) or not all(
                isinstance(value, str) for value in values[field_name]
            ):
                raise ValueError(f"{field_name} must be a list of strings")
            values[field_name] = tuple(values[field_name])
    if "data" in values and not isinstance(values["data"], dict):
        raise ValueError("data must be an object")
    allowed = set(Node.__dataclass_fields__)
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError("unsupported node field(s): " + ", ".join(unknown))
    return Node(**values)


def _agent_edge(spec: dict[str, Any]) -> Edge:
    required = ("src", "dst")
    missing = [field for field in required if not str(spec.get(field, "")).strip()]
    if missing:
        raise ValueError("edge is missing required field(s): " + ", ".join(missing))
    data = spec.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("edge.data must be an object")
    return Edge(
        src=str(spec["src"]).strip(),
        dst=str(spec["dst"]).strip(),
        relation=EdgeType(str(spec.get("relation", "requires"))),
        origin=str(spec["origin"]) if spec.get("origin") is not None else None,
        marker=str(spec["marker"]) if spec.get("marker") is not None else None,
        data=data,
    )


def _apply_graph_action(graph: DepGraph, action: dict) -> tuple[DepGraph, str, bool]:
    """Apply Agent-authored graph edits without semantic gating.

    In addition to field updates, the Agent can upsert newly discovered nodes
    and add their edges.  This is necessary for repairs such as a newly found
    Rust toolchain prerequisite; refusing those edits made the supposedly
    free-action builder unable to express its own diagnosis.
    """
    updates: list[dict] = []
    if "updates" in action:
        if not isinstance(action["updates"], list):
            return graph, "update_graph representation error: updates must be a list.", False
        updates = action["updates"]
    elif "node_id" in action:
        updates = [action]
    additions = action.get("add_nodes", [])
    edges = action.get("add_edges", [])
    if not isinstance(additions, list) or not isinstance(edges, list):
        return graph, "update_graph representation error: add_nodes and add_edges must be lists.", False
    if not updates and not additions and not edges:
        return graph, "NO_GRAPH_CHANGES", False

    updated_graph = graph
    changed: list[str] = []
    for index, update in enumerate(updates, start=1):
        if not isinstance(update, dict):
            return graph, f"update_graph representation error: update {index} must be a JSON object.", False
        # Accept both the documented compact representation
        # ``{"id": ..., "setup_commands": [...]}`` and the original nested
        # representation ``{"node_id": ..., "changes": {...}}``.  The prior
        # mismatch caused valid Agent repairs to be rejected and subsequently
        # disappear when setup.sh was recompiled from the unchanged graph.
        node_id = str(update.get("node_id") or update.get("id") or "").strip()
        node = updated_graph.get(node_id)
        if node is None:
            # A complete typed update is also an unambiguous upsert
            # representation. This lets the Agent discover a node and repair it
            # in one action without the host making a semantic decision.
            if all(str(update.get(field, "")).strip() for field in ("type", "name", "layer")):
                spec = {
                    key: field_value
                    for key, field_value in update.items()
                    if key not in {"node_id", "changes"}
                }
                spec["id"] = node_id
                nested_changes = update.get("changes", {})
                if not isinstance(nested_changes, dict):
                    return graph, (
                        f"update_graph representation error for {node_id}: "
                        "changes must be an object."
                    ), False
                spec.update(nested_changes)
                try:
                    added = _agent_node(spec)
                except (TypeError, ValueError) as exc:
                    return graph, (
                        f"update_graph representation error for new node {node_id}: {exc}"
                    ), False
                updated_graph = updated_graph.with_node(added)
                changed.append(f"added {node_id}")
                continue
            return graph, (
                f"update_graph representation error: unknown node: {node_id}. "
                "Use add_nodes, or include type, name, and layer in this update "
                "to create the node."
            ), False
        try:
            value = State(str(update.get("state", node.state.value)))
        except ValueError:
            return graph, (
                f"update_graph representation error for {node_id}: state must be satisfied, "
                "missing, or unknown."
            ), False
        changes = update.get("changes", {})
        if not isinstance(changes, dict):
            return graph, f"update_graph representation error for {node_id}: changes must be an object.", False
        direct_changes = {
            key: field_value
            for key, field_value in update.items()
            if key in UPDATE_GRAPH_CHANGE_FIELDS
        }
        changes = {**changes, **direct_changes}
        control_fields = {"id", "node_id", "state", "changes"}
        unsupported_top_level = sorted(
            set(update).difference(control_fields | UPDATE_GRAPH_CHANGE_FIELDS)
        )
        if unsupported_top_level:
            return graph, (
                "update_graph representation error: unsupported update field(s): "
                f"{', '.join(unsupported_top_level)}. Allowed fields: "
                f"{_UPDATE_GRAPH_FIELDS_TEXT}."
            ), False
        unsupported = sorted(set(changes).difference(UPDATE_GRAPH_CHANGE_FIELDS))
        if unsupported:
            return graph, (
                "update_graph representation error: unsupported changes field(s): "
                f"{', '.join(unsupported)}. Allowed fields: {_UPDATE_GRAPH_FIELDS_TEXT}. "
                "Put free-form metadata under changes.data."
            ), False
        if "data" in changes and not isinstance(changes["data"], dict):
            return graph, f"update_graph representation error for {node_id}: changes.data must be an object.", False
        try:
            converted = dict(changes)
            if "type" in converted:
                converted["type"] = NodeType(str(converted["type"]))
            if "name" in converted:
                converted["name"] = str(converted["name"]).strip()
            if "layer" in converted:
                converted["layer"] = Layer(str(converted["layer"]))
            if "discovered_by" in converted:
                discovered_by = str(converted["discovered_by"])
                if discovered_by in {"agent", "react", "repair"}:
                    discovered_by = "runtime"
                converted["discovered_by"] = DiscoveredBy(discovered_by)
            for field_name in ("setup_commands", "fix_candidates"):
                if field_name in converted and isinstance(converted[field_name], list):
                    converted[field_name] = tuple(str(item) for item in converted[field_name])
            updated_graph = updated_graph.with_node(replace(node, state=value, **converted))
        except (TypeError, ValueError) as exc:
            return graph, (
                f"update_graph representation error for {node_id}: {exc}. "
                f"Valid layer values: {_LAYER_VALUES_TEXT}."
            ), False
        changed.append(f"{node_id} -> {value.value}")
    for index, spec in enumerate(additions, start=1):
        if not isinstance(spec, dict):
            return graph, f"update_graph representation error: add_nodes[{index}] must be a JSON object.", False
        try:
            node = _agent_node(spec)
        except (TypeError, ValueError) as exc:
            return graph, f"update_graph representation error for add_nodes[{index}]: {exc}", False
        existed = updated_graph.get(node.id) is not None
        updated_graph = updated_graph.with_node(node)
        changed.append(("replaced " if existed else "added ") + node.id)
    for index, spec in enumerate(edges, start=1):
        if not isinstance(spec, dict):
            return graph, f"update_graph representation error: add_edges[{index}] must be a JSON object.", False
        try:
            edge = _agent_edge(spec)
            before = len(updated_graph.edges)
            updated_graph = updated_graph.with_edge(edge)
        except (TypeError, ValueError) as exc:
            return graph, f"update_graph representation error for add_edges[{index}]: {exc}", False
        if len(updated_graph.edges) != before:
            changed.append(f"edge {edge.src} -> {edge.dst}")
    return updated_graph, f"graph updated ({len(changed)} node(s)): " + "; ".join(changed), True


SYSTEM_PROMPT = r"""ROLE
You are the sole ReAct environment-construction Agent for a source repository mounted at /app.
You control the construction container, DepGraph, setup.sh, candidate containers, and runtime handoff.
The host executes exactly the operations you request and records their results. It does not judge the
semantic correctness of your changes.

MANDATORY GOAL
Make `pytest --collect-only -q --disable-warnings` return exit code 0 and leave a DepGraph, setup.sh,
and runtime handoff that reproduce the working environment from the clean selected base image.
The final external evaluator rebuilds a clean Docker image, starts declared runtime services, and runs
that same collection command from `/testbed`; Agent-authored absolute runtime paths rooted at `/app`
are projected to the corresponding `/testbed` paths. A stateful-container success is insufficient unless
every effective repair is represented in DepGraph, setup.sh, and runtime handoff as applicable.

CONTAINER MECHANISM
- Active is the official construction container.
- create_candidate snapshots Active and creates an isolated repair branch.
- A candidate inherits Active state; all execution actions target it until promote_candidate or abort_candidate.
- promote_candidate atomically replaces Active with the candidate and retains its graph/script/runtime state.
- abort_candidate discards the candidate.
- validate_candidate can run setup.sh and collect in the current candidate and promote it when you explicitly
  request promote_on_success=true. If the candidate contains mutations, successful promotion deliberately
  returns FINALIZATION_PENDING instead of ending the session so you can persist runtime state and clean-replay it.
- run_clean_replay creates a temporary container from the clean seeded base checkpoint, runs the current
  setup.sh, starts runtime services, and runs collect. It never replaces Active; use its Observation to repair
  replay defects. A successful clean replay is terminal; do not issue another collect afterward.
- If you explicitly send finish with reason success while a candidate is active, that is your request to
  promote the candidate before ending. The host does not add a semantic judgment; the independent clean-image
  evaluator remains authoritative. Use this only when you intend the candidate graph/setup/runtime to persist.
- `SETUP_FAILURE`, `COLLECT_FAILURE`, `CLEAN_SETUP_FAILURE`, and `CLEAN_COLLECT_FAILURE` are unresolved
  construction failures. After any of these observations, do not promote a candidate and do not finish with
  `reason:"success"`: inspect the first failing block, repair the matching DepGraph node(s) and setup.sh,
  then validate again. A candidate may be promoted only after its latest validation reports both
  `SETUP_SUCCESS` and `COLLECT_SUCCESS`. Clean replay must report both `CLEAN_SETUP_SUCCESS` and
  `CLEAN_COLLECT_SUCCESS` before success is declared.

ARTIFACT RESPONSIBILITY
- Any effective container mutation must be reflected in setup.sh and the corresponding DepGraph node(s).
- A successful `run_shell` with `effect:"mutating"` changes only the selected container. It does not update
  DepGraph, setup.sh, or runtime handoff. Persist the repair immediately with commit_repair (or the equivalent
  update_graph/update_setup/update_runtime actions) before validation. For example, after proving that pytest
  is missing, persist both artifacts:
  `{"action":"commit_repair","command":"python3 -m pip install pytest",
  "setup":{"append":"python3 -m pip install pytest"},
  "graph_updates":[{"id":"pkg:pytest","type":"Package","name":"pytest","layer":"dependencies",
  "discovered_by":"runtime","state":"satisfied","check_command":"python3 -m pytest --version",
  "setup_commands":["python3 -m pip install pytest"]}]}`
- Services and persistent environment variables must be reflected with update_runtime.
- If you start a daemon with run_shell (for example `redis-server --daemonize yes`), immediately persist it:
  `{"action":"update_runtime","services":[{"kind":"redis","start":"redis-server --daemonize yes",
  "check":"redis-cli ping"}]}`. A daemon process in a candidate does not survive clean replay or evaluation.
- Mark nodes SATISFIED yourself after performing the checks you consider appropriate.
- setup.sh is an in-memory artifact. Use run_setup to execute it; do not assume /app/setup.sh exists.
- Do not delete, skip, or rewrite tests to make collection easier. The task is environment construction.
  In particular, never create or alter `conftest.py`, test files, or test configuration to mock missing services.

CONVERSATION
You receive the prior assistant/action/observation trajectory. Exceptionally large outputs from run_collect,
run_setup, run_block, validate_candidate, and run_clean_replay may be replaced in the model-visible trajectory
by a deterministic OBSERVATION_SAFETY_SUMMARY. The exact immutable output remains available through
inspect_observation. Exact graph/setup data remains available through inspect_graph and inspect_setup.
After a failed setup/collect action, FAILURE_FOCUS supplies the first diagnostic signals, the repeat count for
that signature, and unresolved Service/Config nodes. If the same signature repeats without an artifact change,
repair the focused cause before invoking setup or collect again.

RESPONSE FORMAT
Return one JSON object. Never include Markdown fences, an invented Observation, or a second JSON object.
You may use action_batch or commit_repair to perform several related operations in one reasoning turn.

OPERATIONS
1. run_shell: {"thought":"...","action":"run_shell","effect":"read_only|mutating","command":"..."}
2. run_block: {"thought":"...","action":"run_block","node_id":"..."}
3. run_setup: {"thought":"...","action":"run_setup"}
4. run_collect: {"thought":"...","action":"run_collect"}
5. inspect_graph: workspace-level graph by default; one node_id/node_ids is exact; filters state/type/workspace;
   include_managed=true returns the exact full graph including manifest-managed static package leaves.
6. inspect_setup: full script or one stable block: {"action":"inspect_setup","node_id":"pkg:x"}
7. update_graph: update existing nodes and/or add discovered nodes and edges. To add Rust after a build failure:
   {"action":"update_graph","add_nodes":[{"id":"tool:rust","type":"Tool","name":"rust/cargo",
   "layer":"toolchain","discovered_by":"runtime","state":"missing",
   "check_command":"/root/.cargo/bin/rustc --version",
   "setup_commands":["apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential",
   "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.87",
   "ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo",
   "ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc"],"strength":"hard"}],
   "add_edges":[{"src":"deps:pypi:.","dst":"tool:rust","relation":"requires","origin":"runtime"}]}
   Read Cargo.toml `rust-version` or rust-toolchain.toml first. Do not use distro `apt install cargo`
   as the provider for a versioned Rust requirement: it can install an older compiler while returning 0.
   Existing nodes may use compact updates such as
   `{"updates":[{"id":"deps:pypi:.","state":"missing","setup_commands":["uv lock","uv sync"]}]}`;
   the legacy `node_id` plus nested `changes` form is also accepted. Complete typed updates that
   repeat `type`, `name`, `layer`, and `discovered_by` are accepted for existing nodes too, so the
   same repair representation can safely be used for either an update or an upsert. `updates:[]` is optional.
   `layer` must be one of: interpreter, system, toolchain, pip, dependencies, build, naming, runtime,
   tests, config, services. There is no `project` layer; Project nodes use one of the listed graph layers.
   A node absent from inspect_graph must be created, not sent as a field-only update. Use add_nodes, or include
   its complete `type`, `name`, and `layer` in the update to upsert it. For example:
   `{"updates":[{"id":"config:DJANGO_SETTINGS_MODULE","type":"Config",
   "name":"DJANGO_SETTINGS_MODULE","layer":"config","discovered_by":"runtime","state":"missing",
   "check_command":"python3 -c 'import os; assert os.environ.get(\"DJANGO_SETTINGS_MODULE\")'"}]}`
8. recompile_setup: deterministically regenerate setup.sh from the current DepGraph after graph edits.
9. update_setup supports append, exact replace, or stable block replacement. A new node can use a structured
   block and will be appended when no existing block is present:
   {"action":"update_setup","block":{"node_id":"tool:rust","commands":["rustup toolchain install 1.87",
   "rustup default 1.87"],"check":"rustc --version"}}
   Whole-script replacement accepts `content:"..."` or `replace:"..."`. A nested
   `block:{"append":"..."}` is treated as a script append for compatibility.
10. update_runtime: {"action":"update_runtime","services":[{"kind":"redis","start":"redis-server --daemonize yes","check":"redis-cli ping"}],
   "environment":{"PATH":"/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"},
   "capabilities":["docker_socket"]}. Use capabilities for evaluator resources that cannot be started by setup.sh.
    A daemon started by setup.sh during Docker build does not survive final `docker run`; declare it here too.
11. create_candidate / promote_candidate / abort_candidate.
12. validate_candidate: {"action":"validate_candidate","run_setup":true,"run_collect":true,
    "promote_on_success":true}
13. run_clean_replay: {"action":"run_clean_replay"}
14. commit_repair combines an optional shell mutation, setup edit, graph updates, and runtime update:
    {"action":"commit_repair","command":"...","setup":{"append":"..."},"graph_updates":[...],
     "runtime":{"services":[],"environment":{}}}
    It also accepts `graph_add_nodes` and `graph_add_edges`. Omit graph_updates entirely when no graph
    change is needed; never send an empty list.
15. action_batch executes Agent-authored actions in order and stops after the first failed sub-action:
    {"action":"action_batch","actions":[{...},{...}]}
16. finish: {"thought":"...","action":"finish","reason":"success|blocked: ..."}
17. inspect_observation retrieves an exact bounded slice of a prior raw Observation:
    {"action":"inspect_observation","turn":3,"match":"ModuleNotFoundError","limit":6000}
    `match` is a case-insensitive literal. Alternatively provide `offset`; `limit` is capped by the host.

WORKFLOW
Inspect manifests and the initial plan, execute setup, diagnose the first concrete failure, repair in a
candidate, immediately synchronize graph/setup/runtime, validate and promote, then use run_clean_replay.
FINALIZATION_PENDING means the successful candidate is now Active but clean replay is still required; do not
repeat ordinary collect or create a new candidate. If a shell-started daemon, environment variable, or external
capability is not represented in runtime handoff, update_runtime once; otherwise call run_clean_replay as your
next action. A successful clean replay ends the session automatically. Prefer targeted graph/setup inspection and
batch operations over repeatedly returning entire artifacts. Treat every failed validation or clean replay as
the next diagnosis task, not as a reason to promote or finish. Do not finish success inside an unpromoted candidate.
Manage the finite turn budget: after identifying the first failure, use a batched repair/validation sequence
instead of repeatedly inspecting unchanged files. When fewer than five turns remain, prioritize committing the
best evidence-backed graph/setup/runtime repair and one validation over further exploration.
"""


def _raw_exec(sandbox: Sandbox, container, command: str) -> tuple[bool, str]:
    # Login shells may replace the image/runtime PATH and hide a project venv.
    # Agent execution and the external evaluator must observe the same PATH.
    exit_code, output = sandbox.exec_with_timeout(
        container,
        command,
        workdir=sandbox.workdir,
    )
    return exit_code == 0, output


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object without greedily consuming trailing text."""
    value = str(text or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _parse_single_json_object(text: str) -> dict | None:
    """Parse a protocol response only when its entire content is one JSON object.

    ``_extract_json`` remains deliberately permissive for recovery-oriented
    utilities, but it is unsafe as the ReAct action acceptance criterion: a
    response containing many actions would otherwise execute only the first
    one while the entire invalid response is retained in the trajectory.
    """
    try:
        value = json.loads(str(text or "").strip())
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _replacement_miss_observation(script: str, old: str) -> str:
    lines = script.splitlines()
    requested = [line for line in old.splitlines() if line.strip()]
    anchor = next((i for wanted in requested for i, actual in enumerate(lines) if actual == wanted), None)
    excerpt = script[:3000] if anchor is None else "\n".join(
        lines[max(0, anchor - 4):min(len(lines), anchor + max(20, len(old.splitlines()) + 6))]
    )
    return (
        "REPLACE_MISS\nFAILURE: replace.old was not found exactly; setup.sh is unchanged. "
        "Use stable block replacement when possible.\n<<<EXACT_SETUP_EXCERPT\n"
        f"{excerpt}\nEXACT_SETUP_EXCERPT"
    )


def _apply_setup_action(script: str, action: dict, turn: int) -> tuple[str, str, bool]:
    content = action.get("content")
    if isinstance(content, str):
        if not content.strip():
            return script, "UPDATE_SETUP_ERROR\nFAILURE: content is empty.", False
        return (content, "SCRIPT_REPLACED_FULL", True) if content != script else (
            script, "REPLACE_NOOP\nFAILURE: setup.sh is unchanged.", False
        )
    if isinstance(action.get("append"), str):
        appended = action["append"].rstrip()
        if not appended:
            return script, "UPDATE_SETUP_ERROR\nFAILURE: append is empty.", False
        return script + f"\n#@react-edit turn={turn}\n{appended}\n", "SCRIPT_APPENDED", True
    block = action.get("block")
    if isinstance(block, dict):
        block_append = block.get("append")
        if isinstance(block_append, str):
            appended = block_append.rstrip()
            if not appended:
                return script, "UPDATE_SETUP_ERROR\nFAILURE: block.append is empty.", False
            return (
                script + f"\n#@react-edit turn={turn}\n{appended}\n",
                "SCRIPT_APPENDED",
                True,
            )
        node_id = str(block.get("node_id") or "").strip()
        new = block.get("new", block.get("content"))
        commands = block.get("commands")
        structured_block = False
        if not isinstance(new, str) and isinstance(commands, list) and all(
            isinstance(command, str) and command.strip() for command in commands
        ):
            structured_block = True
            lines = [f"#@node {node_id}"]
            check = block.get("check")
            if isinstance(check, str) and check.strip():
                lines.append(f"#@check {check.strip()}")
            lines.extend(["(", *commands, ")"])
            new = "\n".join(lines)
        if not node_id or not isinstance(new, str) or not new.strip():
            return script, (
                "UPDATE_SETUP_ERROR\nFAILURE: block needs node_id plus non-empty new text, "
                "or commands:[...]."
            ), False
        updated, changed = replace_setup_block(script, node_id, new)
        if changed:
            return updated, f"SCRIPT_BLOCK_REPLACED\nnode_id={node_id}", True
        # A newly added DepGraph node has no existing stable block yet.  Append
        # its explicit block instead of rejecting the repair representation.
        if not structured_block and not new.lstrip().startswith(f"#@node {node_id}"):
            return script, f"BLOCK_REPLACE_MISS\nFAILURE: no changed setup block found for {node_id}.", False
        return (
            script + f"\n#@react-edit turn={turn}\n{new.rstrip()}\n",
            f"SCRIPT_BLOCK_APPENDED\nnode_id={node_id}",
            True,
        )
    replacement = action.get("replace")
    if isinstance(replacement, str):
        if not replacement.strip():
            return script, "UPDATE_SETUP_ERROR\nFAILURE: replace is empty.", False
        return (replacement, "SCRIPT_REPLACED_FULL", True) if replacement != script else (
            script, "REPLACE_NOOP\nFAILURE: setup.sh is unchanged.", False
        )
    if isinstance(replacement, dict):
        old, new = replacement.get("old"), replacement.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return script, "UPDATE_SETUP_ERROR\nFAILURE: replace needs non-empty old and string new.", False
        if old not in script:
            return script, _replacement_miss_observation(script, old), False
        updated = script.replace(old, new, 1)
        return (updated, "SCRIPT_REPLACED", True) if updated != script else (
            script, "REPLACE_NOOP\nFAILURE: setup.sh is unchanged.", False
        )
    return script, "UPDATE_SETUP_ERROR\nFAILURE: provide append, block, or replace.", False


def _runtime_update(current: dict[str, Any], action: dict) -> tuple[dict[str, Any], str, bool]:
    services = action.get("services", current.get("services", []))
    environment = action.get("environment", current.get("environment", {}))
    capabilities = action.get("capabilities", current.get("capabilities", []))
    if not isinstance(services, list) or not all(isinstance(item, dict) for item in services):
        return current, "UPDATE_RUNTIME_ERROR\nFAILURE: services must be a list of objects.", False
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        return current, "UPDATE_RUNTIME_ERROR\nFAILURE: environment must map strings to strings.", False
    if not isinstance(capabilities, list) or not all(
        isinstance(value, str) and value.strip() for value in capabilities
    ):
        return current, "UPDATE_RUNTIME_ERROR\nFAILURE: capabilities must be a list of strings.", False
    updated = {
        "version": 1,
        "services": copy.deepcopy(services),
        "environment": dict(environment),
        "capabilities": list(dict.fromkeys(value.strip() for value in capabilities)),
    }
    return updated, (
        f"RUNTIME_UPDATED\nservices={len(services)}; environment={len(environment)}; "
        f"capabilities={len(capabilities)}"
    ), updated != current


def _runtime_commands(runtime: dict[str, Any]) -> str:
    commands: list[str] = []
    for key, value in sorted((runtime.get("environment") or {}).items()):
        commands.append(f"export {key}={shlex.quote(str(value))}")
    for service in runtime.get("services") or []:
        start = str(service.get("start") or "").strip()
        check = str(service.get("check") or "").strip()
        if start:
            commands.append(start)
        if check:
            commands.append(
                f"for _react_ready in $(seq 1 30); do ({check}) && break; "
                "sleep 1; done; (" + check + ")"
            )
    return "\n".join(commands)


def _initial_user_context(graph, script: str, runtime: dict[str, Any], max_turns: int) -> str:
    return (
        "INITIAL CONSTRUCTION CONTEXT\n"
        f"Turn budget: {max_turns}\n"
        f"Graph summary:\n{json.dumps(graph_summary(graph), ensure_ascii=False, indent=2)}\n\n"
        "INITIAL AGENT DEPGRAPH VIEW (manifest-managed static package leaves are represented by "
        "their DependencySet; use inspect_graph include_managed=true or a node_id for exact leaves):\n"
        f"{json.dumps(_agent_graph_view(graph), ensure_ascii=False)}\n\n"
        "FULL INITIAL setup.sh:\n"
        f"{script}\n"
        "INITIAL RUNTIME HANDOFF:\n"
        f"{json.dumps(runtime, ensure_ascii=False, indent=2)}\n\n"
        "Begin by judging this plan and then constructing the environment."
    )


def _diet_initial_user_context(graph, script: str, runtime: dict[str, Any], max_turns: int) -> str:
    """A compact, inspectable bootstrap for Diet mode.

    Exact graph and setup content remains available through existing inspection
    actions.  Repeating a complete generated setup script in every prompt made
    the immutable prefix alone exceed the context budget on large repositories.
    """
    setup_blocks = [
        line.split(maxsplit=2)[1]
        for line in script.splitlines()
        if line.startswith("#@node ") and len(line.split(maxsplit=2)) >= 2
    ]
    nodes = [
        {"id": node.id, "type": node.type.value, "state": node.state.value}
        for node in graph.nodes[:120]
    ]
    return (
        "INITIAL CONSTRUCTION CONTEXT (COMPACT VIEW)\n"
        f"Turn budget: {max_turns}\n"
        f"Graph summary:\n{json.dumps(graph_summary(graph), ensure_ascii=False, indent=2)}\n\n"
        "Graph index (exact node detail is available through inspect_graph):\n"
        f"{json.dumps(nodes, ensure_ascii=False)}\n\n"
        "Setup block index (exact blocks are available through inspect_setup):\n"
        f"{json.dumps(setup_blocks[:240], ensure_ascii=False)}\n\n"
        "INITIAL RUNTIME HANDOFF:\n"
        f"{json.dumps(runtime, ensure_ascii=False, indent=2)}\n\n"
        "Use inspect_graph and inspect_setup whenever exact artifacts matter. "
        "Begin by judging this plan and then constructing the environment."
    )


def _initial_runtime(graph, supplied: dict[str, Any] | None) -> dict[str, Any]:
    services: list[dict[str, str]] = []
    environment = {"PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"}
    for node in graph.nodes:
        recipe = node.data.get("start_recipe")
        if node.type.value == "Service" and isinstance(recipe, dict):
            start, check = recipe.get("start"), node.check_command
            if isinstance(start, str) and start.strip() and isinstance(check, str) and check.strip():
                services.append({"kind": node.name, "start": start.strip(), "check": check.strip()})
        if node.type.value == "Config" and node.data.get("asset_kind") == "runtime_env":
            name, value = node.data.get("runtime_env_name"), node.data.get("runtime_env_value")
            if isinstance(name, str) and isinstance(value, str):
                environment[name] = value
    capabilities: list[str] = []
    if supplied:
        if isinstance(supplied.get("services"), list):
            services.extend(copy.deepcopy(supplied["services"]))
        if isinstance(supplied.get("environment"), dict):
            environment.update({
                str(key): str(value) for key, value in supplied["environment"].items()
                if isinstance(key, str) and isinstance(value, str)
            })
        if isinstance(supplied.get("capabilities"), list):
            capabilities.extend(
                str(value).strip() for value in supplied["capabilities"]
                if isinstance(value, str) and value.strip()
            )
    deduped = list({(item.get("kind"), item.get("start"), item.get("check")): item for item in services}.values())
    return {
        "version": 1,
        "services": deduped,
        "environment": environment,
        "capabilities": list(dict.fromkeys(capabilities)),
    }


def _agent_graph_view(graph, *, include_managed: bool = False) -> dict[str, Any]:
    """Expose a workspace-level graph while retaining exact leaves on demand."""
    hidden_ids = {
        node.id for node in graph.nodes
        if not include_managed
        and node.type.value == "Package"
        and node.data.get("managed_by_dependency_set")
        and not node.setup_commands
        and node.discovered_by.value != "runtime"
        and not node.data.get("runtime_confidence")
    }
    visible_ids = {node.id for node in graph.nodes if node.id not in hidden_ids}
    return {
        "projection": {
            "hidden_manifest_managed_package_nodes": len(hidden_ids),
            "exact_access": "inspect_graph node_id=<id> or include_managed=true",
        },
        "nodes": [node.to_dict() for node in graph.nodes if node.id in visible_ids],
        "edges": [
            edge.to_dict() for edge in graph.edges
            if edge.src in visible_ids and edge.dst in visible_ids
        ],
    }


def run_react_builder(*, client, model: str, sandbox, graph, output_dir: str | Path,
                      max_turns: int, test_command: str,
                      initial_runtime: dict[str, Any] | None = None,
                      context_mode: str = "legacy",
                      compression_model: str | None = None,
                      compression_config: CompressionConfig | None = None,
                      observation_fallback_chars: int = DEFAULT_FALLBACK_CHARS,
                      observation_summary_chars: int = DEFAULT_SUMMARY_CHARS) -> dict:
    if context_mode not in {"legacy", "diet"}:
        raise ValueError("context_mode must be 'legacy' or 'diet'")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_log_path = out / "build_agent.log"
    build_log_path.write_text(
        "# ReAct Build Agent Trace\n"
        "# Full observations are recorded here; model context may contain marked truncation.\n\n",
        encoding="utf-8",
    )
    llm_log_dir = out / "llm_call_log"
    compression_log_dir = out / "compression_call_log"
    raw_steps_dir = out / "raw_steps"
    raw_observations_dir = out / "raw_observations"
    raw_observations_dir.mkdir(parents=True, exist_ok=True)
    if context_mode == "diet":
        raw_steps_dir.mkdir(parents=True, exist_ok=True)
    llm_call_index = 0
    graph = populate_setup_commands(graph)
    state = SessionState(
        graph=graph, script=render_build_script(graph),
        runtime=_initial_runtime(graph, initial_runtime),
    )
    candidate_states: dict[str, SessionState] = {}
    candidate_origins: dict[str, SessionState] = {}
    trace_events: list[Event] = []
    raw_observations: dict[int, dict[str, Any]] = {}
    initial_context = _diet_initial_user_context(
        graph, state.script, state.runtime, max_turns
    ) if context_mode == "diet" else _initial_user_context(
        graph, state.script, state.runtime, max_turns
    )
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_context},
    ]
    diet_steps: list[ReActStep] = []
    compressor_config = compression_config or CompressionConfig()
    compressor_model = compression_model or model
    context_budget = resolve_context_budget(model)
    # Defaults scale with the selected model rather than fixing every model to
    # a 12k/50k observation allowance.
    dynamic_observation_chars = max(12_000, int(context_budget.input_budget_tokens * 0.65))
    effective_fallback_chars = (
        dynamic_observation_chars
        if observation_fallback_chars == DEFAULT_FALLBACK_CHARS
        else observation_fallback_chars
    )
    effective_summary_chars = (
        dynamic_observation_chars
        if observation_summary_chars == DEFAULT_SUMMARY_CHARS
        else observation_summary_chars
    )
    usage = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0,
        "agent": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "reflection": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
    }

    def persist_usage() -> None:
        """Persist a parseable snapshot so token usage is observable mid-run."""
        write_json(out / "token_usage.json", usage)

    # Create the file immediately; subsequent LLM calls update it in place.
    persist_usage()
    reason = "turn_budget_exhausted"
    collect_passed = False
    last_terminal_collect = False
    replay_required_by_state: dict[str, bool] = {"active": False}
    failure_signature_counts: dict[str, int] = {}

    def current() -> SessionState:
        return candidate_states[state.active_candidate] if state.active_candidate else state

    def replay_key() -> str:
        return state.active_candidate or "active"

    def replay_required() -> bool:
        return replay_required_by_state.get(replay_key(), False)

    def mark_replay_required() -> None:
        replay_required_by_state[replay_key()] = True

    def selected_container():
        if state.active_candidate:
            return sandbox.candidate_containers[state.active_candidate].container
        return sandbox.container

    def execute(command: str) -> tuple[bool, str]:
        return _raw_exec(sandbox, selected_container(), command)

    def failure_focus(output: str, active: SessionState) -> str:
        signature = _failure_signature(output)
        failure_signature_counts[signature] = (
            failure_signature_counts.get(signature, 0) + 1
        )
        return _failure_focus(
            output,
            active.graph,
            failure_signature_counts[signature],
        )

    def promote() -> str:
        tx = state.active_candidate
        if not tx:
            return "FAILURE\nno active candidate"
        promoted = candidate_states.pop(tx)
        candidate_origins.pop(tx, None)
        sandbox.promote_candidate(tx)
        state.graph, state.script = promoted.graph, promoted.script
        state.graph_version, state.script_version = promoted.graph_version, promoted.script_version
        state.cursor, state.runtime, state.active_candidate = promoted.cursor, promoted.runtime, None
        replay_required_by_state["active"] = replay_required_by_state.pop(tx, False)
        return f"candidate promoted: {tx}"

    def inspect_graph(active: SessionState, action: dict) -> str:
        node_id = str(action.get("node_id", "")).strip()
        if node_id:
            node = active.graph.get(node_id)
            return json.dumps(node.to_dict() if node else None, ensure_ascii=False)
        node_ids = action.get("node_ids")
        if action.get("include_managed") is True and not any(
            key in action for key in ("node_ids", "state", "type", "workspace", "limit")
        ):
            return json.dumps(active.graph.to_dict(), ensure_ascii=False)
        nodes = list(active.graph.nodes)
        if action.get("include_managed") is not True:
            nodes = [
                node for node in nodes
                if not (
                    node.type.value == "Package"
                    and node.data.get("managed_by_dependency_set")
                    and not node.setup_commands
                    and node.discovered_by.value != "runtime"
                    and not node.data.get("runtime_confidence")
                )
            ]
        if isinstance(node_ids, list):
            wanted = {str(item) for item in node_ids}
            nodes = [node for node in nodes if node.id in wanted]
        for key, getter in (
            ("state", lambda n: n.state.value),
            ("type", lambda n: n.type.value),
            ("workspace", lambda n: n.workspace or n.data.get("project_path")),
        ):
            if action.get(key) is not None:
                nodes = [node for node in nodes if getter(node) == action[key]]
        if any(key in action for key in ("node_ids", "state", "type", "workspace", "limit")):
            limit = max(1, int(action.get("limit", len(nodes) or 1)))
            return json.dumps({"nodes": [node.to_dict() for node in nodes[:limit]]}, ensure_ascii=False)
        return json.dumps(_agent_graph_view(active.graph), ensure_ascii=False)

    def candidate_delta(tx: str) -> str:
        active = candidate_states.get(tx)
        origin = candidate_origins.get(tx)
        if active is None or origin is None:
            return "CANDIDATE_DELTA unavailable"
        before = {node.id: node.to_dict() for node in origin.graph.nodes}
        after = {node.id: node.to_dict() for node in active.graph.nodes}
        changed = sorted(
            node_id for node_id in set(before) | set(after)
            if before.get(node_id) != after.get(node_id)
        )
        script_diff = "\n".join(difflib.unified_diff(
            origin.script.splitlines(), active.script.splitlines(),
            fromfile="active/setup.sh", tofile="candidate/setup.sh", lineterm="",
        ))
        return (
            "CANDIDATE_DELTA\n"
            f"graph_changed_nodes={json.dumps(changed, ensure_ascii=False)}\n"
            f"runtime_changed={origin.runtime != active.runtime}\n"
            "setup_diff:\n" + (truncate_observation(script_diff, edge_chars=2500) or "(unchanged)")
        )

    def replay_state_difference(clean_container, active: SessionState) -> str:
        bootstrap = _runtime_commands(active.runtime)
        inventory = "python3 -m pip freeze 2>/dev/null || true"
        status = "git status --porcelain=v1 --untracked-files=all 2>/dev/null || true"
        command = f"{bootstrap}\n{inventory}\nprintf '\\n__REACT_GIT_STATUS__\\n'\n{status}" if bootstrap else (
            f"{inventory}\nprintf '\\n__REACT_GIT_STATUS__\\n'\n{status}"
        )
        _active_ok, active_output = _raw_exec(sandbox, selected_container(), command)
        _clean_ok, clean_output = _raw_exec(sandbox, clean_container, command)

        def split(value: str) -> tuple[set[str], set[str]]:
            packages, _marker, changed = value.partition("__REACT_GIT_STATUS__")
            return (
                {line.strip() for line in packages.splitlines() if line.strip()},
                {line.strip() for line in changed.splitlines() if line.strip()},
            )

        active_packages, active_files = split(active_output)
        clean_packages, clean_files = split(clean_output)
        active_only = sorted(active_packages - clean_packages)
        clean_only = sorted(clean_packages - active_packages)
        file_only = sorted(active_files - clean_files)
        return (
            "REPLAY_STATE_DIFFERENCE_HINT (diagnostic only)\n"
            f"active_only_packages={json.dumps(active_only[:80], ensure_ascii=False)}\n"
            f"clean_only_packages={json.dumps(clean_only[:80], ensure_ascii=False)}\n"
            f"active_only_worktree_changes={json.dumps(file_only[:80], ensure_ascii=False)}"
        )

    def execute_one(action: dict, turn: int) -> tuple[str, bool, bool, bool]:
        """Return observation, should_finish, operation_ok, terminal_collect."""
        nonlocal collect_passed, reason
        active = current()
        kind = str(action.get("action", "")).strip()
        terminal = False
        if kind == "run_shell":
            ok, output = execute(str(action.get("command", "")))
            active.cursor = str(action.get("command", ""))
            mutating = ok and str(action.get("effect", "")).strip() == "mutating"
            if mutating:
                mark_replay_required()
            suffix = ""
            if mutating:
                suffix = (
                    "\n\nPERSISTENCE_REQUIRED\n"
                    "This successful shell mutation changed only the selected container; it did not "
                    "update DepGraph, setup.sh, or runtime handoff. Persist the effective repair now "
                    "with commit_repair or the corresponding update actions before validation."
                )
            return (
                ("SUCCESS\n" if ok else "FAILURE\n") + output + suffix,
                False,
                ok,
                False,
            )
        if kind == "run_block":
            node_id = str(action.get("node_id", ""))
            node = active.graph.get(node_id)
            if node is None:
                return f"FAILURE\nunknown node: {node_id}", False, False, False
            ok, output = execute("\n".join(node.setup_commands))
            active.cursor = node_id
            return ("SUCCESS\n" if ok else "FAILURE\n") + output, False, ok, False
        if kind == "run_setup":
            ok, output = execute(active.script)
            active.cursor = "setup.sh"
            return ("SUCCESS\n" if ok else "FAILURE\n") + output, False, ok, False
        if kind == "run_collect":
            bootstrap = _runtime_commands(active.runtime)
            command = f"{bootstrap}\n{test_command}" if bootstrap else test_command
            ok, output = execute(command)
            active.cursor = "collect"
            collect_passed = ok
            terminal = bool(ok and not replay_required())
            suffix = ""
            if ok and replay_required():
                suffix = (
                    "\n\nFINALIZATION_PENDING\n"
                    "The current state contains mutations not yet exercised from the clean base. "
                    "If any shell-started daemon, environment variable, or capability is still "
                    "missing from runtime handoff, update_runtime once. Otherwise your next action "
                    "should be run_clean_replay. Do not create a new candidate or repeat run_collect."
                )
            elif not ok:
                suffix = "\n\n" + failure_focus(output, active)
            return (
                ("COLLECT_SUCCESS\n" if ok else "COLLECT_FAILURE\n") + output + suffix,
                False,
                ok,
                terminal,
            )
        if kind == "inspect_observation":
            try:
                wanted_turn = int(action.get("turn"))
                offset = int(action.get("offset", 0))
                limit = int(action.get("limit", 6000))
            except (TypeError, ValueError):
                return (
                    "FAILURE\ninspect_observation requires integer turn/offset/limit values",
                    False,
                    False,
                    False,
                )
            record = raw_observations.get(wanted_turn)
            if record is None:
                available = sorted(raw_observations)
                return (
                    "FAILURE\nunknown raw observation turn: "
                    f"{wanted_turn}; available_turns={available}",
                    False,
                    False,
                    False,
                )
            viewed = observation_slice(
                str(record["raw"]),
                turn=wanted_turn,
                raw_ref=str(record["raw_ref"]),
                match=str(action.get("match") or ""),
                offset=offset,
                limit=limit,
                max_limit=effective_summary_chars,
            )
            found = not viewed.startswith("OBSERVATION_MATCH_NOT_FOUND")
            return viewed, False, found, False
        if kind == "inspect_graph":
            return inspect_graph(active, action), False, True, False
        if kind == "inspect_setup":
            node_id = str(action.get("node_id", "")).strip()
            if not node_id:
                return active.script, False, True, False
            block = setup_block(active.script, node_id)
            return (block if block is not None else f"FAILURE\nunknown setup block: {node_id}"), False, block is not None, False
        if kind == "update_graph":
            updated, output, changed = _apply_graph_action(active.graph, action)
            if changed:
                active.graph, active.graph_version = updated, active.graph_version + 1
                mark_replay_required()
            return output, False, changed, False
        if kind == "recompile_setup":
            updated = render_build_script(active.graph)
            if updated == active.script:
                return "SETUP_ALREADY_COMPILED", False, True, False
            active.script, active.script_version = updated, active.script_version + 1
            mark_replay_required()
            return "SETUP_RECOMPILED_FROM_DEPGRAPH", False, True, False
        if kind == "update_setup":
            updated, output, changed = _apply_setup_action(active.script, action, turn)
            if changed:
                active.script, active.script_version = updated, active.script_version + 1
                mark_replay_required()
            return output, False, changed, False
        if kind == "update_runtime":
            updated, output, changed = _runtime_update(active.runtime, action)
            if changed:
                active.runtime = updated
                mark_replay_required()
            return output, False, changed, False
        if kind == "create_candidate":
            if state.active_candidate:
                return "FAILURE\na candidate is already active", False, False, False
            tx = f"react-{turn + 1}"
            checkpoint = f"react-active-{turn + 1}"
            sandbox.create_checkpoint(checkpoint)
            sandbox.create_candidate_container(tx, checkpoint)
            candidate_states[tx] = SessionState(
                graph=active.graph, script=active.script,
                graph_version=active.graph_version, script_version=active.script_version,
                cursor=active.cursor, runtime=copy.deepcopy(active.runtime),
            )
            candidate_origins[tx] = SessionState(
                graph=active.graph, script=active.script,
                graph_version=active.graph_version, script_version=active.script_version,
                cursor=active.cursor, runtime=copy.deepcopy(active.runtime),
            )
            replay_required_by_state[tx] = replay_required_by_state.get("active", False)
            state.active_candidate = tx
            return f"candidate created: {tx}", False, True, False
        if kind == "promote_candidate":
            output = promote()
            return output, False, not output.startswith("FAILURE"), False
        if kind == "abort_candidate":
            if not state.active_candidate:
                return "FAILURE\nno active candidate", False, False, False
            tx = state.active_candidate
            sandbox.abort_candidate(tx)
            candidate_states.pop(tx, None)
            candidate_origins.pop(tx, None)
            replay_required_by_state.pop(tx, None)
            state.active_candidate = None
            return f"candidate aborted: {tx}", False, True, False
        if kind == "validate_candidate":
            if not state.active_candidate:
                return "FAILURE\nvalidate_candidate requires an active candidate", False, False, False
            pieces: list[str] = []
            ok = True
            if action.get("run_setup", True):
                ok, output = execute(current().script)
                pieces.append(("SETUP_SUCCESS\n" if ok else "SETUP_FAILURE\n") + output)
            collect_ok = False
            if ok and action.get("run_collect", True):
                bootstrap = _runtime_commands(current().runtime)
                command = f"{bootstrap}\n{test_command}" if bootstrap else test_command
                collect_ok, output = execute(command)
                pieces.append(("COLLECT_SUCCESS\n" if collect_ok else "COLLECT_FAILURE\n") + output)
                collect_passed = collect_ok
                ok = collect_ok
                if not collect_ok:
                    pieces.append(failure_focus(output, current()))
            pieces.append(candidate_delta(state.active_candidate))
            if ok and action.get("promote_on_success", False):
                pieces.append(promote())
            terminal = bool(
                collect_ok and state.active_candidate is None and not replay_required()
            )
            if collect_ok and state.active_candidate is None and replay_required():
                pieces.append(
                    "FINALIZATION_PENDING\n"
                    "Candidate promoted, but its mutations have not been replayed from the clean "
                    "base. If runtime handoff is already complete, your next action should be "
                    "run_clean_replay. Otherwise update_runtime once and then run_clean_replay. "
                    "Do not create a new candidate or repeat run_collect."
                )
            return "\n\n".join(pieces), False, ok, terminal
        if kind == "run_clean_replay":
            tx = f"react-clean-{turn + 1}"
            try:
                handle = sandbox.create_candidate_container(tx, "base")
                ok, output = _raw_exec(sandbox, handle.container, active.script)
                pieces = [("CLEAN_SETUP_SUCCESS\n" if ok else "CLEAN_SETUP_FAILURE\n") + output]
                failed_output = output if not ok else ""
                if ok:
                    bootstrap = _runtime_commands(active.runtime)
                    command = f"{bootstrap}\n{test_command}" if bootstrap else test_command
                    ok, output = _raw_exec(sandbox, handle.container, command)
                    pieces.append(("CLEAN_COLLECT_SUCCESS\n" if ok else "CLEAN_COLLECT_FAILURE\n") + output)
                    if not ok:
                        failed_output = output
                pieces.append(replay_state_difference(handle.container, active))
                if not ok:
                    pieces.append(failure_focus(failed_output, active))
                collect_passed = ok
                if ok:
                    replay_required_by_state[replay_key()] = False
                return "\n\n".join(pieces), False, ok, ok
            finally:
                if tx in getattr(sandbox, "candidate_containers", {}):
                    sandbox.abort_candidate(tx)
        if kind == "commit_repair":
            pieces: list[str] = []
            command = action.get("command")
            if isinstance(command, str) and command.strip():
                ok, output = execute(command)
                pieces.append(("SHELL_SUCCESS\n" if ok else "SHELL_FAILURE\n") + output)
                if not ok:
                    return "\n\n".join(pieces), False, False, False
                mark_replay_required()
            setup_action = action.get("setup")
            if isinstance(setup_action, dict):
                # LLMs commonly serialize an unused optional setup field as
                # ``{"append": ""}``.  Treat that representation as absent so
                # it cannot prevent the graph half of a commit from persisting.
                empty_setup = (
                    not setup_action
                    or (
                        set(setup_action) == {"append"}
                        and isinstance(setup_action.get("append"), str)
                        and not setup_action["append"].strip()
                    )
                )
                if not empty_setup:
                    updated, output, changed = _apply_setup_action(active.script, setup_action, turn)
                    pieces.append(output)
                    if not changed:
                        return "\n\n".join(pieces), False, False, False
                    active.script, active.script_version = updated, active.script_version + 1
                    mark_replay_required()
            graph_updates = action.get("graph_updates")
            graph_additions = action.get("graph_add_nodes")
            graph_edges = action.get("graph_add_edges")
            if graph_updates or graph_additions or graph_edges:
                graph_action: dict[str, Any] = {}
                if graph_updates:
                    graph_action["updates"] = graph_updates
                if graph_additions:
                    graph_action["add_nodes"] = graph_additions
                if graph_edges:
                    graph_action["add_edges"] = graph_edges
                updated, output, changed = _apply_graph_action(active.graph, graph_action)
                pieces.append(output)
                if not changed:
                    return "\n\n".join(pieces), False, False, False
                active.graph, active.graph_version = updated, active.graph_version + 1
                mark_replay_required()
            runtime = action.get("runtime")
            if isinstance(runtime, dict):
                updated, output, changed = _runtime_update(active.runtime, runtime)
                pieces.append(output)
                if changed:
                    active.runtime = updated
                    mark_replay_required()
            return "\n\n".join(pieces) or "COMMIT_REPAIR_NOOP", False, bool(pieces), False
        if kind == "action_batch":
            actions = action.get("actions")
            if not isinstance(actions, list) or not actions:
                return "FAILURE\naction_batch.actions must be a non-empty list", False, False, False
            pieces: list[str] = []
            batch_terminal = False
            for index, subaction in enumerate(actions, start=1):
                if not isinstance(subaction, dict):
                    pieces.append(f"[{index}] FAILURE: sub-action is not an object")
                    return "\n\n".join(pieces), False, False, batch_terminal
                output, should_finish, ok, sub_terminal = execute_one(subaction, turn)
                pieces.append(f"[{index}] {subaction.get('action')}\n{output}")
                batch_terminal = batch_terminal or sub_terminal
                if should_finish:
                    return "\n\n".join(pieces), True, ok, batch_terminal
                if not ok:
                    return "\n\n".join(pieces), False, False, batch_terminal
            return "\n\n".join(pieces), False, True, batch_terminal
        if kind == "finish":
            reason = str(action.get("reason", "agent_finished"))
            pieces = [reason]
            if reason.strip().lower() == "success" and state.active_candidate:
                pieces.append(
                    "AGENT_REQUESTED_CANDIDATE_PROMOTION\n" + promote()
                )
            return "\n".join(pieces), True, True, False
        return f"UNKNOWN_ACTION\nFAILURE: unsupported action {kind!r}; choose a documented operation.", False, False, False

    def append_turn(turn: int, container: str, thought: str, action: dict, observation: str) -> None:
        visible = {key: value for key, value in action.items() if key not in {"thought", "analysis"}}
        with build_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"========== Turn {turn + 1} | Container: {container} ==========\n")
            handle.write("### Thought:\n" + (thought.strip() or "(No explicit rationale supplied.)") + "\n\n")
            handle.write("### Action:\n" + json.dumps(visible, ensure_ascii=False, indent=2) + "\n\n")
            handle.write("### Observation:\n" + (observation or "(no observation)").rstrip() + "\n\n")

    def add_usage(bucket: str, values: dict[str, Any] | None) -> None:
        values = values or {}
        if not values:
            return
        target = usage[bucket]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            amount = int(values.get(key, 0) or 0)
            usage[key] += amount
            target[key] += amount
        calls = int(values.get("api_calls", 1) or 1)
        usage["api_calls"] += calls
        target["api_calls"] += calls
        persist_usage()

    for turn in range(max_turns):
        if context_mode == "diet":
            raw_request = render_messages(
                system_prompt=SYSTEM_PROMPT,
                initial_context=initial_context,
                steps=diet_steps,
            )
            original_chars = message_chars(raw_request)
            request_messages, fallback_meta = fit_messages_to_token_budget(
                raw_request, budget=context_budget,
            )
            context_meta = {
                "mode": "diet", "original_chars": original_chars,
                "request_chars": message_chars(request_messages), **fallback_meta,
                "safety_fallback": bool(
                    fallback_meta["removed_messages"]
                    or fallback_meta["clipped_message_indexes"]
                ),
            }
        else:
            request_messages, context_meta = truncate_messages(conversation)
            context_meta = {"mode": "legacy", **context_meta}
        append_jsonl(out / "conversation_context.jsonl", {"turn": turn + 1, **context_meta})

        def audit_exchange(messages: list[dict], parsed_output: str, raw_response, retry_attempt: int) -> None:
            nonlocal llm_call_index
            llm_call_index += 1
            write_llm_exchange(
                llm_log_dir, llm_call_index, messages=messages, parsed_output=parsed_output,
                raw_response=raw_response, retry_attempt=retry_attempt,
            )

        response = ""
        action = None
        for protocol_round in range(2):
            response, call_usage, _raw = complete_with_retry(
                client, model, request_messages,
                accept=lambda text: _parse_single_json_object(text) is not None,
                retry_nudge=(
                    "FORMAT_ERROR: return exactly one valid JSON object with one documented action. "
                    "Escape shell backslashes and quotes as JSON; do not include Markdown or Observation."
                ),
                temperature=0,
                on_exchange=audit_exchange,
                **json_response_format_kwargs(model),
            )
            add_usage("agent", call_usage)
            action = _parse_single_json_object(response)
            if action is not None:
                break
            request_messages = request_messages + [{
                "role": "user",
                "content": "FORMAT_ERROR: no valid JSON object was parsed. Return exactly one JSON action now.",
            }]
            if context_mode == "diet":
                request_messages, _ = fit_messages_to_token_budget(
                    request_messages, budget=context_budget,
                )
        if action is None:
            action = {"action": "protocol_error"}
            observation = "FORMAT_ERROR\nNo valid JSON action after protocol retries; continue next turn."
            # Raw rejected completions are preserved in llm_call_log/, but must
            # never become an assistant trajectory entry: a malformed response
            # can itself be arbitrarily large (for example, hundreds of action
            # objects) and would defeat context retention on the next turn.
            rejected_response = (
                "FORMAT_ERROR: rejected non-single-JSON Agent response. "
                "Raw completion is available in llm_call_log/."
            )
            append_turn(turn, state.active_candidate or "active", rejected_response, action, observation)
            trace_events.append(Event(turn, state.active_candidate or "active", "protocol_error", action, observation))
            if context_mode == "diet":
                step = ReActStep(
                    turn=turn + 1, assistant_raw=rejected_response, action=action,
                    observation_raw=observation, observation_view=observation,
                    state={"container": state.active_candidate or "active", "protocol_error": True},
                    container=state.active_candidate or "active",
                )
                diet_steps.append(step)
                write_json(raw_steps_dir / f"{turn + 1:04d}.json", step.raw_record())
            else:
                conversation.extend([
                    {"role": "assistant", "content": rejected_response},
                    {"role": "user", "content": observation},
                ])
            continue

        container_name = state.active_candidate or "active"
        thought = str(action.get("thought") or action.get("analysis") or "")
        observation, should_finish, _ok, terminal_collect = execute_one(action, turn)
        raw_observation = observation
        raw_ref = f"raw_observations/{turn + 1:04d}.txt"
        raw_path = out / raw_ref
        raw_path.write_text(raw_observation, encoding="utf-8")
        raw_observations[turn + 1] = {
            "raw": raw_observation,
            "raw_ref": raw_ref,
            "raw_sha256": observation_sha256(raw_observation),
            "action": str(action.get("action") or ""),
        }
        observation_view, fallback_records = apply_rule_fallback(
            action=action,
            raw=raw_observation,
            turn=turn + 1,
            raw_ref=raw_ref,
            threshold_chars=effective_fallback_chars,
            summary_chars=effective_summary_chars,
        )
        for fallback_record in fallback_records:
            append_jsonl(out / "observation_fallback.jsonl", fallback_record.to_dict())
        rule_fallback_applied = any(record.applied for record in fallback_records)
        last_terminal_collect = terminal_collect
        event = Event(turn, container_name, str(action.get("action", "")), action, raw_observation)
        current().events.append(event)
        trace_events.append(event)
        append_turn(turn, container_name, thought, action, raw_observation)
        state_header = {
            "container": f"candidate:{state.active_candidate}" if state.active_candidate else "active",
            "graph_version": current().graph_version,
            "script_version": current().script_version,
            "cursor": current().cursor,
            "runtime_services": [str(item.get("kind") or "") for item in current().runtime.get("services", [])],
            "runtime_environment_keys": sorted(current().runtime.get("environment", {})),
            "collect_passed": collect_passed,
            "clean_replay_required": replay_required(),
            "remaining_turns": max_turns - turn - 1,
        }
        if collect_passed and replay_required() and not state.active_candidate:
            state_header.update({
                "phase": "FINALIZATION_PENDING",
                "next_action": (
                    "Use update_runtime only for missing persistent runtime state; "
                    "otherwise call run_clean_replay now."
                ),
                "avoid": "Do not create a candidate or repeat run_collect.",
            })
        if context_mode == "diet":
            step = ReActStep(
                turn=turn + 1, assistant_raw=response, action=action,
                observation_raw=raw_observation, observation_view=observation_view,
                state=state_header, container=container_name,
                compression=(
                    {"rule_fallback": [record.to_dict() for record in fallback_records]}
                    if rule_fallback_applied else {}
                ),
            )
            diet_steps.append(step)
            write_json(raw_steps_dir / f"{turn + 1:04d}.json", step.raw_record())

            target_index = len(diet_steps) - 1 - compressor_config.delay_turns
            if target_index >= 0:
                target = diet_steps[target_index]

                if not target.compression.get("rule_fallback"):
                    def audit_compression(messages: list[dict], parsed_output: str, raw_response, retry_attempt: int) -> None:
                        index = target.turn * 10 + retry_attempt
                        write_llm_exchange(
                            compression_log_dir, index, messages=messages,
                            parsed_output=parsed_output, raw_response=raw_response,
                            retry_attempt=retry_attempt,
                        )

                    reduced, record, reflection_usage = compress_observation(
                        client=client, model=compressor_model, target_turn=target.turn,
                        target_observation=target.observation_raw,
                        serialized_window=local_window(
                            diet_steps, target_index,
                            before=compressor_config.context_before,
                            after=compressor_config.delay_turns,
                        ),
                        config=compressor_config, on_exchange=audit_compression,
                    )
                    target.observation_view = reduced
                    target.compression = record.to_dict()
                    write_json(raw_steps_dir / f"{target.turn:04d}.json", target.raw_record())
                    append_jsonl(out / "context_compression.jsonl", record.to_dict())
                    add_usage("reflection", reflection_usage)
        else:
            conversation.extend([
                {"role": "assistant", "content": response},
                {"role": "user", "content": (
                    "### Observation\n" + truncate_observation(observation_view) +
                    "\n\n### Current state\n" + json.dumps(state_header, ensure_ascii=False, indent=2)
                )},
            ])
        # Stop on a terminal collect (an unchanged Active state, or a successful
        # clean replay) so the model cannot repeat collect or undo success.
        if terminal_collect and not state.active_candidate:
            reason = "success"
            break
        if should_finish:
            break

    if reason == "turn_budget_exhausted" and collect_passed and not state.active_candidate and last_terminal_collect:
        reason = "success"
    success = collect_passed and reason.strip().lower() == "success" and state.active_candidate is None
    (out / "setup.sh").write_text(state.script, encoding="utf-8")
    write_json(out / "final_depgraph.json", state.graph.to_dict())
    write_json(out / "runtime_handoff.json", state.runtime)
    write_json(out / "react_trace.json", [event.to_dict() for event in trace_events])
    if context_mode == "diet":
        raw_history: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_context},
        ]
        for step in diet_steps:
            raw_history.extend([
                {"role": "assistant", "content": step.assistant_raw},
                {"role": "user", "content": "### Observation\n" + step.observation_raw},
            ])
        write_json(out / "conversation_history.json", raw_history)
        write_json(out / "conversation_prompt_history.json", render_messages(
            system_prompt=SYSTEM_PROMPT, initial_context=initial_context, steps=diet_steps,
        ))
    else:
        write_json(out / "conversation_history.json", conversation)
    write_json(out / "token_usage.json", usage)
    write_json(out / "react_status.json", {
        "reason": reason, "success": success,
        "context_mode": context_mode,
        "compression_model": compressor_model if context_mode == "diet" else None,
        "compression_config": (
            {
                "delay_turns": compressor_config.delay_turns,
                "context_before": compressor_config.context_before,
                "min_target_chars": compressor_config.min_target_chars,
                "min_saved_chars": compressor_config.min_saved_chars,
            } if context_mode == "diet" else None
        ),
        "observation_fallback": {
            "actions": [
                "run_shell", "run_collect", "run_setup", "run_block",
                "validate_candidate", "run_clean_replay",
            ],
            "configured_threshold_chars": observation_fallback_chars,
            "configured_summary_chars": observation_summary_chars,
            "effective_threshold_chars": effective_fallback_chars,
            "effective_summary_chars": effective_summary_chars,
            "raw_directory": "raw_observations",
        },
        "context_budget": context_budget.to_dict() if context_mode == "diet" else None,
        "graph_version": state.graph_version, "script_version": state.script_version,
        "collect_passed": collect_passed, "active_candidate": state.active_candidate,
        "clean_replay_required": replay_required_by_state.get("active", False),
    })
    with build_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "========== Finished ==========\n"
            f"reason={reason}\ngraph_version=g{state.graph_version}; script_version=s{state.script_version}; "
            f"collect_passed={collect_passed}\n"
        )
    return {
        "script": state.script, "graph": state.graph, "runtime": state.runtime,
        "usage": usage, "reason": reason, "collect_passed": collect_passed, "success": success,
    }
