import io
import json
import tempfile
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.python_deps.depgraph.schema import DepGraph
from src.agent.engine import run_react_builder
from src.initialization_only import run_initialization_only, _execute_initial_script


class FakeContainer(str):
    def put_archive(self, path, data):
        self.upload_path, self.archive = path, data
        return True


class FakeSandbox:
    workdir = "/app"
    container = "active"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.commands = []
        self.candidate_containers = {}

    def create_candidate_container(self, tx, checkpoint):
        self.checkpoint = checkpoint
        handle = SimpleNamespace(container=FakeContainer("clean"))
        self.candidate_containers[tx] = handle
        return handle

    def exec_with_timeout(self, container, command, **kwargs):
        self.commands.append((container, command))
        return next(self.outcomes)

    def abort_candidate(self, tx):
        del self.candidate_containers[tx]


class InitialVerificationTests(unittest.TestCase):
    def test_large_script_is_uploaded_unchanged_and_not_passed_as_an_argument(self):
        script = "#!/bin/bash\n" + "# frozen script\n" * 20000 + "exit 0\n"
        sandbox = FakeSandbox([(0, "ok")])
        container = FakeContainer("clean")
        self.assertEqual(_execute_initial_script(sandbox, container, script), (True, "ok"))
        with tarfile.open(fileobj=io.BytesIO(container.archive)) as archive:
            self.assertEqual(archive.extractfile("graph2env-initial-setup.sh").read(), script.encode())
        self.assertEqual(container.upload_path, "/tmp")
        self.assertLess(len(sandbox.commands[0][1]), 100)

    def run_initial(self, root, outcomes):
        sandbox = FakeSandbox(outcomes)
        with patch("src.agent.engine.complete_with_retry") as complete:
            result = run_initialization_only(
                sandbox=sandbox, graph=DepGraph(), output_dir=root,
                test_command="pytest --collect-only -q",
                initial_runtime={"environment": {"TEST_CONFIG": "initial"}},
            )
            complete.assert_not_called()
        self.assertEqual(sandbox.checkpoint, "base")
        self.assertFalse(sandbox.candidate_containers)
        for initial, final in (
            ("initial_depgraph.json", "final_depgraph.json"),
            ("initial_setup.sh", "setup.sh"),
            ("initial_runtime_handoff.json", "runtime_handoff.json"),
        ):
            self.assertEqual((root / initial).read_bytes(), (root / final).read_bytes())
        self.assertEqual(result["usage"]["api_calls"], 0)
        return result, sandbox, json.loads((root / "react_status.json").read_text())

    def test_success_uses_clean_base_and_initial_runtime_without_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, sandbox, status = self.run_initial(
                Path(temporary), [(0, "setup ok"), (0, "pytest ready"), (0, "2 tests collected")],
            )
        self.assertTrue(result["success"])
        self.assertTrue(status["clean_replay_succeeded"])
        self.assertFalse(status["clean_replay_required"])
        self.assertEqual([container for container, _ in sandbox.commands], ["clean", "clean", "clean"])
        self.assertIn("export TEST_CONFIG=initial", sandbox.commands[1][1])
        self.assertTrue(status["pytest_bootstrap_succeeded"])
        self.assertIn('"$_graph2env_test_python" -m pytest --collect-only -q', sandbox.commands[2][1])

    def test_setup_failure_does_not_collect_or_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, sandbox, status = self.run_initial(Path(temporary), [(1, "missing compiler")])
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "initial_setup_failed")
        self.assertFalse(status["collect_passed"])
        self.assertEqual(len(sandbox.commands), 1)

    def test_collection_failure_is_not_certified_or_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, sandbox, status = self.run_initial(
                Path(temporary), [(0, "setup ok"), (0, "pytest ready"), (2, "ModuleNotFoundError")],
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "initial_collection_failed")
        self.assertTrue(status["clean_setup_succeeded"])
        self.assertEqual(len(sandbox.commands), 3)

    def test_pytest_install_failure_stops_collection_and_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, sandbox, status = self.run_initial(
                root, [(0, "setup ok"), (1, "pytest download failed")],
            )
            trace = json.loads((root / "react_trace.json").read_text())
            self.assertEqual([event["action"] for event in trace], ["initial_setup", "pytest_bootstrap"])
            self.assertIn("pytest download failed", (root / "build_agent.log").read_text())
        self.assertEqual(result["reason"], "pytest_bootstrap_failed")
        self.assertFalse(result["success"])
        self.assertFalse(status["pytest_bootstrap_succeeded"])
        self.assertFalse(status["collect_passed"])
        self.assertEqual(len(sandbox.commands), 2)

    def test_default_mode_still_calls_agent(self):
        sandbox = FakeSandbox([(0, "setup"), (0, "collected"), (0, ""), (0, "")])
        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.agent.engine.complete_with_retry",
            return_value=(json.dumps({"action": "run_clean_replay"}), {"api_calls": 1}, None),
        ) as complete:
            result = run_react_builder(
                client=None, model="deepseek-v4-flash", sandbox=sandbox,
                graph=DepGraph(), output_dir=temporary, max_turns=1,
                test_command="pytest --collect-only -q",
            )
        complete.assert_called_once()
        self.assertTrue(result["success"])
