"""Check standalone ablation outputs without Docker or model requests."""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts import run_initialization_only as runner


@pytest.mark.parametrize("success", [True, False])
def test_runner_preserves_initialization_mode_and_exports_only_on_success(tmp_path, success):
    output = tmp_path / "output"
    sandbox = SimpleNamespace(base_image_ref="python:3.11-slim", platform=None,
                              close=Mock())
    graph = object()
    runtime = {"environment": {}, "services": [], "capabilities": []}
    result = {"script": "#!/bin/bash\ntrue\n", "runtime": runtime,
              "usage": {"api_calls": 0}, "success": success}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(runner.os.environ, {"OPENAI_API_KEY": "test-placeholder"}))
        stack.enter_context(patch.object(runner, "OpenAI"))
        stack.enter_context(patch.object(runner, "choose_base_image", return_value=SimpleNamespace(
            image="python:3.11-slim", minor="3.11", platform_override=None)))
        stack.enter_context(patch.object(runner, "Sandbox", return_value=sandbox))
        stack.enter_context(patch.object(runner, "discover_test_dependency_intent", return_value=
                                        SimpleNamespace(needed_groups=[], pytest_addopts=[])))
        stack.enter_context(patch.object(runner, "build_advisory_for_repo", return_value=(None, graph)))
        construction = stack.enter_context(patch.object(runner, "run_initialization_only", return_value=result))
        export = stack.enter_context(patch.object(runner, "export_dockerfile"))
        code = runner.main([
            str(tmp_path), "--model", "test-model", "--base-image", "python:3.11-slim",
            "--out", str(output / "setup.sh"), "--runtime-out", str(output / "runtime.json"),
            "--usage-out", str(output / "usage.json"),
        ])
    assert code == (0 if success else 1)
    assert construction.call_args.kwargs["graph"] is graph
    assert (output / "setup.sh").read_text() == result["script"]
    assert json.loads((output / "runtime.json").read_text()) == runtime
    assert json.loads((output / "usage.json").read_text()) == {"api_calls": 0}
    initialization = json.loads((output / "initialization_usage.json").read_text())
    assert initialization["ablation"] == "initialization_only"
    if success:
        export.assert_called_once_with(output, "python:3.11-slim", None, runtime)
    else:
        export.assert_not_called()
    sandbox.close.assert_called_once()
