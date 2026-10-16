"""Initialization-only construction and verification; no ReAct repair loop."""
import io
from pathlib import Path
import tarfile
import time

from src.agent.artifacts import write_json
from src.agent.contracts import Event, SessionState
from src.agent.engine import _initial_runtime, _raw_exec, _runtime_commands
from src.python_deps.depgraph.build_script import render_build_script
from src.python_deps.depgraph.populate import populate_setup_commands


PYTEST_PYTHON = """if [ -x /app/.venv/bin/python ]; then
  _graph2env_test_python=/app/.venv/bin/python
else
  _graph2env_test_python=$(command -v python3)
fi
"""

PYTEST_BOOTSTRAP = PYTEST_PYTHON + """if ! "$_graph2env_test_python" -c 'import pytest' >/dev/null 2>&1; then
  if "$_graph2env_test_python" -m pip --version >/dev/null 2>&1; then
    "$_graph2env_test_python" -m pip install --break-system-packages pytest || \
      "$_graph2env_test_python" -m pip install pytest
  else
    command -v uv >/dev/null 2>&1 || \
      (python3 -m pip install --break-system-packages uv || python3 -m pip install uv)
    uv pip install --python "$_graph2env_test_python" pytest
  fi
fi
"$_graph2env_test_python" -c 'import sys, pytest; print("pytest interpreter:", sys.executable); print("pytest version:", pytest.__version__)'
"""


def _execute_initial_script(sandbox, container, script):
    data = script.encode("utf-8")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as handle:
        entry = tarfile.TarInfo("graph2env-initial-setup.sh")
        entry.size = len(data)
        handle.addfile(entry, io.BytesIO(data))
    if not container.put_archive("/tmp", archive.getvalue()):
        raise RuntimeError("could not upload the initial setup script")
    return _raw_exec(sandbox, container, "/bin/bash /tmp/graph2env-initial-setup.sh")


def run_initialization_only(*, sandbox, graph, output_dir, test_command,
                            initial_runtime=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "build_agent.log").write_text("# Initialization-only verification\n", encoding="utf-8")
    graph = populate_setup_commands(graph)
    state = SessionState(
        graph=graph, script=render_build_script(graph),
        runtime=_initial_runtime(graph, initial_runtime),
    )
    write_json(out / "initial_depgraph.json", graph.to_dict())
    (out / "initial_setup.sh").write_text(state.script, encoding="utf-8")
    write_json(out / "initial_runtime_handoff.json", state.runtime)
    return _run_initial_setup(sandbox=sandbox, state=state, out=out, test_command=test_command)


def _run_initial_setup(*, sandbox, state: SessionState, out: Path,
                       test_command: str) -> dict:
    """Verify initial artifacts from the seeded clean base, without an Agent."""
    usage = {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0,
        "agent": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
        "reflection": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0},
    }
    (out / "setup.sh").write_text(state.script, encoding="utf-8")
    write_json(out / "final_depgraph.json", state.graph.to_dict())
    write_json(out / "runtime_handoff.json", state.runtime)
    write_json(out / "token_usage.json", usage)
    status = {
        "ablation": "initialization_only", "reason": "initial_verification_pending",
        "success": False, "collect_passed": False, "clean_replay_succeeded": False,
        "clean_setup_succeeded": False, "active_candidate": None,
        "pytest_bootstrap_succeeded": False,
        "clean_replay_required": True, "graph_version": 0, "script_version": 0,
    }
    write_json(out / "react_status.json", status)
    events = []
    tx = "initial-clean"
    started = time.monotonic()
    try:
        handle = sandbox.create_candidate_container(tx, "base")
        ok, output = _execute_initial_script(sandbox, handle.container, state.script)
        status["clean_setup_succeeded"] = ok
        events.append(Event(0, tx, "initial_setup", {}, output))
        reason = "initial_setup_failed"
        if ok:
            environment = _runtime_commands({"environment": state.runtime.get("environment", {})})
            command = f"set -e\n{environment}\n{PYTEST_BOOTSTRAP}"
            ok, output = _raw_exec(sandbox, handle.container, command)
            events.append(Event(0, tx, "pytest_bootstrap", {}, output))
            status["pytest_bootstrap_succeeded"] = ok
            reason = "pytest_bootstrap_failed"
        if ok:
            bootstrap = _runtime_commands(state.runtime)
            command = f'set -e\n{bootstrap}\n{PYTEST_PYTHON}\n"$_graph2env_test_python" -m {test_command}'
            ok, output = _raw_exec(sandbox, handle.container, command)
            events.append(Event(0, tx, "initial_collect", {}, output))
            status["collect_passed"] = ok
            reason = "success" if ok else "initial_collection_failed"
        status.update({
            "reason": reason, "success": ok, "clean_replay_succeeded": ok,
            "clean_replay_required": not ok,
        })
    except Exception:
        status["reason"] = "initial_verification_error"
        raise
    finally:
        status["verification_duration_seconds"] = round(time.monotonic() - started, 3)
        write_json(out / "react_status.json", status)
        write_json(out / "react_trace.json", [event.to_dict() for event in events])
        with (out / "build_agent.log").open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(f"\n[{event.action}]\n{event.output}\n")
            handle.write(f"\nreason={status['reason']}\n")
        if tx in getattr(sandbox, "candidate_containers", {}):
            sandbox.abort_candidate(tx)
    return {
        "script": state.script, "graph": state.graph, "runtime": state.runtime,
        "usage": usage, "reason": status["reason"],
        "collect_passed": status["collect_passed"], "success": status["success"],
    }
