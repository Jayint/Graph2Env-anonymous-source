from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from src.agent.engine import _raw_exec
from src.sandbox import APT_TRANSIENT_RETRY_ATTEMPTS, Sandbox
from src.synthesizer import Synthesizer

def bare_sandbox() -> Sandbox:
    sandbox = Sandbox.__new__(Sandbox)
    sandbox.workdir = "/app"
    sandbox.apt_mirror_url = None
    sandbox.apt_retries = 5
    sandbox.apt_http_timeout_seconds = 120
    sandbox.apt_https_timeout_seconds = 120
    sandbox.command_timeout_seconds = None
    sandbox._command_classifier = Synthesizer()
    return sandbox


class FakeInstallContainer:
    def __init__(self):
        self.setup_calls = 0
        self.recovery_calls = 0

    def exec_run(self, command, **_kwargs):
        rendered = " ".join(command) if isinstance(command, list) else str(command)
        if "dpkg --configure -a" in rendered:
            self.recovery_calls += 1
            return SimpleNamespace(exit_code=0, output=b"")
        self.setup_calls += 1
        if self.setup_calls == 1:
            output = (
                b"__INSTALL_FAIL__:apt-get install -y build-essential:12\n"
                b"E: Failed to fetch package 500 reading HTTP response body: "
                b"unexpected EOF\n"
            )
            return SimpleNamespace(exit_code=100, output=output)
        return SimpleNamespace(exit_code=0, output=b"setup complete\n")


class FakeTimedOutContainer:
    def __init__(self):
        self.command = None
        self.workdir = None

    def exec_run(self, command, **kwargs):
        self.command = command
        self.workdir = kwargs.get("workdir")
        return SimpleNamespace(exit_code=124, output=b"partial output\n")


class AptRetryTests(unittest.TestCase):
    def test_sandbox_command_timeout_defaults_to_ten_minutes(self):
        default = inspect.signature(Sandbox.__init__).parameters[
            "command_timeout_seconds"
        ].default
        self.assertEqual(default, 600)

    def test_bootstrap_covers_debian_sources_https_and_no_cache(self):
        command = bare_sandbox()._build_apt_bootstrap_command()
        self.assertIn("/etc/apt/sources.list.d/*.sources", command)
        self.assertIn("https://deb.debian.org", command)
        self.assertIn('Acquire::https::Pipeline-Depth "0";', command)
        self.assertIn('Acquire::https::No-Cache "true";', command)

    def test_detects_only_transient_apt_command_failures(self):
        sandbox = bare_sandbox()
        command = (
            "for attempt in 1 2; do apt-get -o Acquire::Retries=10 "
            "install -y build-essential; done"
        )
        output = "E: Failed to fetch package: unexpected EOF"
        self.assertTrue(
            sandbox._should_retry_transient_apt_failure(command, 100, output, 1)
        )
        self.assertFalse(
            sandbox._should_retry_transient_apt_failure(command, 100, output, APT_TRANSIENT_RETRY_ATTEMPTS)
        )
        self.assertFalse(
            sandbox._should_retry_transient_apt_failure("echo apt-get", 1, output, 1)
        )

    def test_setup_replay_recovers_apt_then_retries_complete_script(self):
        sandbox = bare_sandbox()
        container = FakeInstallContainer()
        result = sandbox._run_install_script_in_container(
            container, "apt-get install -y build-essential"
        )
        self.assertEqual(result.rc, 0)
        self.assertEqual(container.setup_calls, 2)
        self.assertEqual(container.recovery_calls, 1)
        self.assertIn("Transient apt failure", result.stderr)


class CommandTimeoutTests(unittest.TestCase):
    def test_sandbox_wraps_commands_with_configured_timeout(self):
        sandbox = bare_sandbox()
        sandbox.command_timeout_seconds = 600
        wrapped = sandbox._wrap_command_with_timeout("cargo --version")
        self.assertIn("timeout --foreground --kill-after=30s 600s", wrapped)


    def test_react_raw_exec_uses_sandbox_command_timeout(self):
        sandbox = bare_sandbox()
        sandbox.command_timeout_seconds = 600
        container = FakeTimedOutContainer()
        ok, output = _raw_exec(sandbox, container, "pip install -e '.[ci]'")
        self.assertFalse(ok)
        self.assertEqual(container.workdir, "/app")
        self.assertIn(
            "timeout --foreground --kill-after=30s 600s",
            container.command[-1],
        )
        self.assertIn("Command timed out after 600 seconds", output)


