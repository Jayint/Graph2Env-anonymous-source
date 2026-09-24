"""Command Executor abstraction for the dependency-graph pipeline.

The Executor is an interface so scan/resolve/probe/certify are testable with a
``FakeExecutor`` (no Docker, no network).  Two concrete implementations ship:

* ``LocalSubprocessExecutor`` runs in the current host/venv (CI-friendly).
* ``DockerExecutor`` runs a long-lived container and ``docker exec``s into it.
"""

from __future__ import annotations

import subprocess
import shlex
import typing
import uuid
from dataclasses import dataclass

# returncode used when a command exceeds its timeout.
TIMEOUT_RC = 124


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@typing.runtime_checkable
class Executor(typing.Protocol):
    def run(self, command: str, *, timeout: int = 300) -> CommandResult: ...


def _run_subprocess(command: str, *, timeout: int) -> CommandResult:
    """Run ``command`` through the shell, capturing rc/stdout/stderr."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return CommandResult(
            command=command,
            returncode=TIMEOUT_RC,
            stdout=stdout,
            stderr=stderr + f"\n[timeout after {timeout}s]",
        )
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class LocalSubprocessExecutor:
    """Executor that runs commands directly on the host/in the current venv."""

    def run(self, command: str, *, timeout: int = 300) -> CommandResult:
        return _run_subprocess(command, timeout=timeout)


class DockerExecutor:
    """Long-lived container executor used as a context manager.

    ``with DockerExecutor("python:3.11-slim") as ex:`` starts a detached
    ``sleep infinity`` container; ``ex.run(cmd)`` does ``docker exec`` into it;
    ``__exit__`` force-removes the container.  Not unit-tested (covered by the
    gated Docker integration test); kept dependency-free so import never fails
    when Docker is absent.
    """

    def __init__(
        self,
        image: str,
        *,
        network: bool = True,
        platform: str | None = None,
    ) -> None:
        self.image = image
        self.network = network
        self.platform = platform
        self.container_id: str | None = None
        self._name = f"depgraph-probe-{uuid.uuid4().hex[:12]}"

    def __enter__(self) -> "DockerExecutor":
        net = "" if self.network else "--network none "
        platform = (
            f"--platform {shlex.quote(self.platform)} " if self.platform else ""
        )
        start = _run_subprocess(
            f"docker run -d {net}{platform}--name {self._name} "
            f"{shlex.quote(self.image)} sleep infinity",
            timeout=300,
        )
        if not start.ok:
            raise RuntimeError(
                f"failed to start probe container: {start.stderr.strip()}"
            )
        self.container_id = start.stdout.strip()
        return self

    def __exit__(self, *exc: object) -> None:
        # Only tear down a container that __enter__ actually started; if the
        # context was never entered (e.g. an exception before the `with`),
        # container_id is None and there is nothing to remove.
        if self.container_id is not None:
            _run_subprocess(f"docker rm -f {self._name}", timeout=120)
        self.container_id = None
        return None

    def run(self, command: str, *, timeout: int = 300) -> CommandResult:
        if self.container_id is None:
            raise RuntimeError("DockerExecutor.run called outside its context")
        # Run through an inner shell so pipes/heredocs in `command` work.
        escaped = command.replace("'", "'\"'\"'")
        exec_cmd = f"docker exec {self._name} sh -c '{escaped}'"
        result = _run_subprocess(exec_cmd, timeout=timeout)
        # Report the user's command (not the docker wrapper) for clean evidence.
        return CommandResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
