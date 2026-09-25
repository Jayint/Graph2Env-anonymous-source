"""The single-repository command exports a buildable Docker recipe."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_react_e2e


def test_success_exports_dockerfile_and_runtime_service(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runtime = {
        "environment": {"PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"},
        "services": [{
            "kind": "redis", "start": "redis-server --daemonize yes",
            "check": "redis-cli ping",
        }],
    }
    sandbox = SimpleNamespace(base_image_ref="python:3.11-slim", platform=None,
                              close=lambda: None)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}), \
         patch.object(sys, "argv", ["run_react_e2e.py", str(tmp_path), "--model", "test-model",
                                    "--base-image", "python:3.11-slim",
                                    "--out", str(output_dir / "setup.sh")]), \
         patch.object(run_react_e2e, "OpenAI"), \
         patch.object(run_react_e2e, "choose_base_image", return_value=SimpleNamespace(
             image="python:3.11-slim", minor="3.11", platform_override=None,
             reason="test")), \
         patch.object(run_react_e2e, "Sandbox", return_value=sandbox), \
         patch.object(run_react_e2e, "discover_test_dependency_intent",
                      return_value=SimpleNamespace(needed_groups=[], pytest_addopts=[])), \
         patch.object(run_react_e2e, "build_advisory_for_repo", return_value=(None, object())), \
         patch.object(run_react_e2e, "run_react_builder", return_value={
             "script": "#!/usr/bin/env bash\ntrue\n", "runtime": runtime,
             "usage": {}, "reason": "success", "success": True,
         }):
        assert run_react_e2e.main() == 0

    dockerfile = (output_dir / "Dockerfile").read_text()
    assert "FROM python:3.11-slim" in dockerfile
    assert "COPY . /app" in dockerfile
    assert "COPY --from=graph2env /setup.sh /tmp/setup.sh" in dockerfile
    assert "RUN bash /tmp/setup.sh" in dockerfile
    assert "ENV PATH=\"/app/.venv/bin:/usr/local/bin:/usr/bin:/bin\"" in dockerfile
    assert "COPY --from=graph2env /runtime-entrypoint.sh" in dockerfile
    assert "redis-server --daemonize yes" in (output_dir / "runtime-entrypoint.sh").read_text()
    assert subprocess.run(["bash", "-n", str(output_dir / "runtime-entrypoint.sh")]).returncode == 0
