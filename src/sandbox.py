import os
import re
import shlex
import tarfile
import tempfile
import time
import docker
from dataclasses import dataclass
from src.synthesizer import Synthesizer

PIP_TRANSIENT_RETRY_ATTEMPTS = 3
APT_TRANSIENT_RETRY_ATTEMPTS = 5

_INSTALL_FAIL_MARKER = "__INSTALL_FAIL__"


@dataclass(frozen=True)
class InstallResult:
    rc: int
    failing_command: str | None   # $BASH_COMMAND captured by the ERR trap; None on success
    lineno: int | None
    stderr: str


@dataclass(frozen=True)
class CandidateContainer:
    """A transaction-scoped container that does not replace the working container."""

    transaction_id: str
    base_checkpoint: str
    image_id: str
    container: object


@dataclass(frozen=True)
class ResolverContainer:
    """Short-lived container used only for native package-manager resolution."""

    resolver_id: str
    image_id: str
    container: object


def _wrap_with_err_trap(script: str) -> str:
    """Prepend an ERR trap so a `set -e` abort prints the failing command + line.

    render_build_script is reuse-only (its preamble already sets `set -Eeuo pipefail`);
    we add ONLY the trap line, immediately after the shebang/preamble is irrelevant —
    bash applies the most recent trap, and `-E` makes the trap inherit into functions.
    """
    trap = (
        "trap 'rc=$?; echo "
        f"\"{_INSTALL_FAIL_MARKER}:$BASH_COMMAND:$LINENO\" >&2; exit $rc' ERR\n"
    )
    return trap + script


def _parse_install_failure(output: str) -> tuple[str | None, int | None]:
    """Return (failing_command, lineno) from the FIRST install-fail marker, else (None, None)."""
    for line in (output or "").splitlines():
        if line.startswith(_INSTALL_FAIL_MARKER + ":"):
            rest = line[len(_INSTALL_FAIL_MARKER) + 1:]
            cmd, _, lineno_s = rest.rpartition(":")
            try:
                return (cmd or None), int(lineno_s)
            except ValueError:
                return (cmd or None), None
    return None, None


def _service_extra_hosts():
    if os.environ.get("DOCKERAGENT_ENABLE_SERVICE_PROVISION") == "1":
        return {"postgres": "127.0.0.1"}
    return None


class Sandbox:
    def __init__(
        self,
        base_image="ubuntu:22.04",
        workdir="/app",
        volumes=None,
        platform=None,
        seed_dir=None,
        command_timeout_seconds=600,
        apt_mirror_url=None,
        apt_retries=5,
        apt_http_timeout_seconds=120,
        apt_https_timeout_seconds=120,
        docker_client_timeout_seconds=None,
        enable_cache_volume: bool = False,
        ensure_native_platform: bool = False,
        environment: dict[str, str] | None = None,
    ):
        self.client = docker.from_env(timeout=docker_client_timeout_seconds)
        self.base_image = base_image
        self.workdir = workdir
        self.volumes = volumes  # Mapping of {local_path: {'bind': container_path, 'mode': 'rw'}}
        self.platform = platform  # Docker platform (e.g., "linux/amd64" for x86_64 emulation on ARM64)
        self.environment = dict(environment or {})
        if self.platform is None and ensure_native_platform:
            self.platform = self._daemon_native_platform()
        self.seed_dir = os.path.abspath(seed_dir) if seed_dir else None
        self.command_timeout_seconds = command_timeout_seconds
        self.docker_client_timeout_seconds = docker_client_timeout_seconds
        self.current_image = base_image
        self.container = None
        self.last_success_image = None  # 记录上一次成功状态的镜像
        self.snapshot_image_ids = set()
        # Semantic build checkpoints are independent from execute()'s rolling
        # last-success snapshot.  They remain addressable until explicitly
        # invalidated by the graph-aware incremental executor.
        self.named_checkpoints: dict[str, str] = {}
        self.candidate_containers: dict[str, CandidateContainer] = {}
        self.resolver_containers: dict[str, ResolverContainer] = {}
        self.runtime_replay_commands = []
        self.package_manager_broken_failure_streak = 0
        self._command_classifier = Synthesizer()
        self.apt_mirror_url = self._resolve_apt_mirror_url(apt_mirror_url)
        self.apt_retries = apt_retries
        self.apt_http_timeout_seconds = apt_http_timeout_seconds
        self.apt_https_timeout_seconds = apt_https_timeout_seconds
        if enable_cache_volume:
            cache = dict(self.volumes or {})
            cache.setdefault("graph2env_pip_cache", {"bind": "/root/.cache/pip", "mode": "rw"})
            cache.setdefault("graph2env_apt_cache", {"bind": "/var/cache/apt/archives", "mode": "rw"})
            self.volumes = cache
        self._setup_initial_container()

    def _container_environment_kwargs(self) -> dict:
        environment = getattr(self, "environment", None)
        return {"environment": dict(environment)} if environment else {}

    def _daemon_native_platform(self) -> str | None:
        """Docker daemon's native ``os/arch`` platform, normalized for pulls.

        An unqualified local tag may point at an emulated architecture left by
        an earlier explicit run.  Asking the daemon (rather than the host OS)
        also works for remote Docker contexts.
        """
        try:
            info = self.client.info() or {}
        except Exception:
            return None
        os_name = str(info.get("OSType", "linux")).lower()
        arch = str(info.get("Architecture", "")).lower()
        aliases = {
            "aarch64": "arm64",
            "arm64": "arm64",
            "x86_64": "amd64",
            "amd64": "amd64",
        }
        normalized = aliases.get(arch)
        return f"{os_name}/{normalized}" if os_name and normalized else None

    def _local_image_matches_platform(self) -> bool:
        """Return whether the tagged local image already matches ``self.platform``.

        Docker SDK ``images.pull`` may invoke an interactive credential helper
        even when the exact platform image is cached. Besides hanging offline
        runs, doing that once per RAT worker defeats concurrent execution.
        """
        if not self.platform:
            return True
        try:
            image = self.client.images.get(self.current_image)
        except docker.errors.DockerException:
            return False
        return self._image_matches_platform(image)

    def _image_matches_platform(self, image) -> bool:
        """Check one immutable Docker Image object against ``self.platform``."""
        if not self.platform:
            return True
        requested = self.platform.split("/", 2)
        if len(requested) < 2:
            return False
        requested_os, requested_arch = requested[0], requested[1]
        aliases = {"x86_64": "amd64", "aarch64": "arm64"}
        requested_arch = aliases.get(requested_arch, requested_arch)
        attrs = image.attrs or {}
        local_os = str(attrs.get("Os", "")).lower()
        local_arch = aliases.get(
            str(attrs.get("Architecture", "")).lower(),
            str(attrs.get("Architecture", "")).lower(),
        )
        return local_os == requested_os.lower() and local_arch == requested_arch.lower()

    def _resolved_platform_image_ref(self) -> str:
        """Resolve the selected tag to an immutable image id for this platform.

        Concurrent workers may pull different architectures under the same tag.
        Pinning the returned local image id prevents a later retag from changing
        this Sandbox's initial/reset image.
        """
        if not self.platform:
            return self.current_image
        try:
            local_image = self.client.images.get(self.current_image)
        except docker.errors.DockerException:
            local_image = None
        if local_image is not None and self._image_matches_platform(local_image):
            return local_image.id
        if not self.platform:
            return self.current_image
        try:
            print(f"[Platform] Pulling {self.current_image} for platform {self.platform}...")
            pulled = self.client.images.pull(self.current_image, platform=self.platform)
            if not self._image_matches_platform(pulled):
                raise RuntimeError(
                    f"registry returned {pulled.attrs.get('Os')}/"
                    f"{pulled.attrs.get('Architecture')} for requested {self.platform}"
                )
            return pulled.id
        except Exception as e:
            print(f"[Platform] Pull failed: {e}")
            try:
                fallback = self.client.images.get(self.current_image)
            except docker.errors.DockerException as fallback_error:
                raise RuntimeError(
                    f"cannot resolve {self.current_image} for {self.platform}"
                ) from fallback_error
            if not self._image_matches_platform(fallback):
                raise RuntimeError(
                    f"local image {self.current_image} does not match {self.platform}"
                ) from e
            return fallback.id

    def _stable_local_image_alias(self, image_ref: str) -> str | None:
        """Give a resolved image id a platform-specific, concurrency-safe tag.

        BuildKit does not reliably accept a raw local image id in ``FROM`` and
        may try to pull ``sha256:<id>`` as a registry name.  A tag derived from
        the immutable id and effective platform is stable across workers while
        remaining directly usable by the downstream RAT Docker build.
        """
        try:
            image = self.client.images.get(image_ref)
            image_id = str(image.id).removeprefix("sha256:")
            if not image_id:
                return None
            platform = re.sub(r"[^a-z0-9_.-]+", "-", (self.platform or "default").lower())
            tag = f"{image_id[:16]}-{platform}"
            if not image.tag("graph2env-v3-base", tag=tag):
                return None
            return f"graph2env-v3-base:{tag}"
        except Exception as exc:
            print(f"[Platform] Stable alias unavailable: {exc}")
            return None

    def _setup_initial_container(self):
        """Initializes the container from the base image."""
        print(f"Initializing container from {self.current_image}...")
        if self.platform:
            print(f"[Platform] Using platform: {self.platform}")
            if self._local_image_matches_platform():
                print(f"[Platform] Reusing local {self.current_image} ({self.platform}).")
        launch_image = self._resolved_platform_image_ref()
        print(f"[Platform] Resolved image: {launch_image}")
        self.base_image_alias = self._stable_local_image_alias(launch_image)
        if self.base_image_alias:
            print(f"[Platform] Stable image: {self.base_image_alias}")
        self.base_image_ref = launch_image
        self.current_image = launch_image
        _extra_hosts = _service_extra_hosts()
        self.container = self.client.containers.run(
            launch_image,
            detach=True,
            tty=True,
            working_dir=self.workdir,
            # Alpine and other minimal images do not ship bash.  Boot through
            # the POSIX shell first, then install/verify bash below because the
            # execution engine deliberately relies on bash semantics.
            command="/bin/sh",
            volumes=self.volumes,
            platform=self.platform,
            **self._container_environment_kwargs(),
            **({} if _extra_hosts is None else {"extra_hosts": _extra_hosts})
        )
        self._ensure_bash_available()
        # Ensure workdir exists
        self.container.exec_run(f"mkdir -p {self.workdir}")
        self._bootstrap_apt_if_supported()
        if self.seed_dir:
            self._seed_workdir_from_host()
        # Always keep a baseline snapshot so the first failed command can roll back
        # to the initialized workspace rather than the raw base image.
        baseline_image = self.container.commit()
        self._register_snapshot(baseline_image.id)
        self.last_success_image = baseline_image.id
        if not hasattr(self, "named_checkpoints"):
            self.named_checkpoints = {}
        self.named_checkpoints["base"] = baseline_image.id
        print(f"[Baseline Snapshot] {self.last_success_image[:12]}")

    def _ensure_bash_available(self) -> None:
        """Install bash and GNU timeout before the v3 engine uses them.

        The sandbox's command runner, install gate, and generated setup scripts
        intentionally use bash features such as ``pipefail``.  Starting a raw
        Alpine image with ``/bin/bash`` therefore fails before any graph work can
        begin.  Use the universally available ``/bin/sh`` for this one bootstrap
        step and support the common Linux package managers.
        """
        if not self.container:
            return

        command = (
            "set -e\n"
            "has_bash=0\n"
            "has_gnu_timeout=0\n"
            "command -v bash >/dev/null 2>&1 && has_bash=1\n"
            "timeout --version 2>&1 | grep -q 'GNU coreutils' && has_gnu_timeout=1 || true\n"
            "if [ \"$has_bash\" = 1 ] && [ \"$has_gnu_timeout\" = 1 ]; then exit 0; fi\n"
            "if command -v apk >/dev/null 2>&1; then\n"
            "  apk add --no-cache bash coreutils\n"
            "elif command -v apt-get >/dev/null 2>&1; then\n"
            "  apt-get update\n"
            "  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bash coreutils\n"
            "elif command -v dnf >/dev/null 2>&1; then\n"
            "  dnf install -y bash coreutils\n"
            "elif command -v yum >/dev/null 2>&1; then\n"
            "  yum install -y bash coreutils\n"
            "else\n"
            "  echo 'no supported package manager available to install bash/coreutils' >&2\n"
            "  exit 127\n"
            "fi\n"
            "command -v bash >/dev/null 2>&1\n"
            "timeout --version 2>&1 | grep -q 'GNU coreutils'\n"
        )
        result = self.container.exec_run(["/bin/sh", "-lc", command])
        output = result.output
        if isinstance(output, (bytes, bytearray)):
            output = output.decode("utf-8", errors="replace")
        if result.exit_code != 0:
            detail = (output or "").strip()
            raise RuntimeError(
                "base image does not provide bash/GNU timeout and automatic installation failed"
                + (f": {detail}" if detail else "")
            )

    def _resolve_apt_mirror_url(self, apt_mirror_url):
        configured = (
            apt_mirror_url
            or os.environ.get("GRAPH2ENV_APT_MIRROR_URL")
            or os.environ.get("APT_MIRROR_URL")
        )
        if not configured:
            return None
        return configured.rstrip("/")

    def _bootstrap_apt_if_supported(self):
        if not self.container:
            return

        bootstrap_command = self._build_apt_bootstrap_command()
        exec_result = self.container.exec_run(
            ["/bin/bash", "-lc", bootstrap_command],
            workdir=self.workdir,
        )
        exit_code = exec_result.exit_code
        output = exec_result.output.decode("utf-8", errors="replace")
        if exit_code == 0:
            if self.apt_mirror_url:
                print(f"[Apt Bootstrap] Configured retries and mirror: {self.apt_mirror_url}")
            else:
                print("[Apt Bootstrap] Configured apt retries/timeouts.")
            return

        print(
            f"[Apt Bootstrap Warning] Failed to prepare apt mirror/retry settings (exit {exit_code})."
        )
        if output.strip():
            print(output)

    def _build_apt_bootstrap_command(self):
        mirror_block = ""
        if self.apt_mirror_url:
            mirror = shlex.quote(self.apt_mirror_url)
            mirror_block = (
                f"APT_MIRROR_URL={mirror}\n"
                "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do\n"
                "  [ -f \"$file\" ] || continue\n"
                "  sed -i \"s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" \"$file\"\n"
                "  sed -i \"s|https://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" \"$file\"\n"
                "  sed -i \"s|http://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" \"$file\"\n"
                "  sed -i \"s|https://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" \"$file\"\n"
                "done\n"
                "apt-get update\n"
            )

        return (
            "set -e\n"
            "if ! command -v apt-get >/dev/null 2>&1; then\n"
            "  exit 0\n"
            "fi\n"
            "mkdir -p /etc/apt/apt.conf.d\n"
            "cat >/etc/apt/apt.conf.d/99graph2env-retries <<'EOF'\n"
            f"Acquire::Retries \"{self.apt_retries}\";\n"
            f"Acquire::http::Timeout \"{self.apt_http_timeout_seconds}\";\n"
            f"Acquire::https::Timeout \"{self.apt_https_timeout_seconds}\";\n"
            "Acquire::http::Pipeline-Depth \"0\";\n"
            "Acquire::https::Pipeline-Depth \"0\";\n"
            "Acquire::http::No-Cache \"true\";\n"
            "Acquire::https::No-Cache \"true\";\n"
            "EOF\n"
            "for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do\n"
            "  [ -f \"$file\" ] || continue\n"
            "  sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \"$file\"\n"
            "done\n"
            f"{mirror_block}"
        )

    def _seed_workdir_from_host(self):
        """Copy the host workspace into the container so rollback includes repo state."""
        if not os.path.isdir(self.seed_dir):
            raise ValueError(f"seed_dir does not exist or is not a directory: {self.seed_dir}")

        self.container.exec_run(
            ["/bin/bash", "-lc", f"rm -rf {self.workdir}/* {self.workdir}/.[!.]* {self.workdir}/..?* 2>/dev/null || true"]
        )

        # Docker accepts a file-like request body.  Keep the tar on disk and
        # reopen it for every retry instead of materialising the entire
        # workspace as one ``bytes`` object.  The old BytesIO/getvalue path
        # duplicated multi-gigabyte repositories in memory and Docker Desktop
        # aborted the request with OSError(22) before the Agent could start.
        last_error = None
        with tempfile.NamedTemporaryFile(
            prefix="graph2env-seed-", suffix=".tar"
        ) as archive_file:
            with tarfile.open(fileobj=archive_file, mode="w") as tar:
                for entry in sorted(os.listdir(self.seed_dir)):
                    entry_path = os.path.join(self.seed_dir, entry)
                    tar.add(entry_path, arcname=entry, recursive=True)
            archive_file.flush()

            for attempt in range(1, 4):
                try:
                    with open(archive_file.name, "rb") as archive_stream:
                        if self.container.put_archive(self.workdir, archive_stream):
                            return
                    last_error = RuntimeError("Docker put_archive returned false")
                except (OSError, docker.errors.DockerException) as exc:
                    last_error = exc
                if attempt < 3:
                    time.sleep(attempt)
        raise RuntimeError(
            f"Failed to copy workspace from {self.seed_dir} into container after 3 attempts: {last_error}"
        )

    def exec_readonly(self, command):
        """Run a read-only probe/extractor command with NO side effects.

        Returns (exit_code:int, output:str). Does not commit snapshots, does not
        run preflight rejection, does not retry, does not inject SYSTEM prefixes,
        and deliberately does NOT apply `set -o pipefail` (pipefail can flip the
        exit code of legitimate probe chains like `cmd | grep -q`, causing silent
        mis-certification). It wraps the command in a login shell so `&&` and `|`
        work, and returns the raw exit code untouched.
        Callers must only pass commands that do not mutate the environment.
        """
        result = self.container.exec_run(["/bin/sh", "-lc", command], workdir=self.workdir)
        output = result.output
        if isinstance(output, (bytes, bytearray)):
            output = output.decode("utf-8", errors="replace")
        exit_code = result.exit_code
        if exit_code is None:
            # Docker exec_run can yield None (streaming/detached); a probe that did not
            # produce a real exit code must never be read as success (rc==0).
            exit_code = -1
        return exit_code, output or ""

    def read_file_bytes(self, path, max_bytes):
        """Read a file's RAW bytes from the live container, capped at ``max_bytes``.

        Used by Tier-1 file capture (Fix 1) to capture the achieved file state
        losslessly. Deliberately uses a NON-login shell-free exec (``head -c`` via the
        argv form, no ``/bin/sh -lc``) so profile / MOTD / conda-init banners cannot
        pollute the bytes. Reads ``max_bytes + 1`` so the caller can detect oversize and
        fall back to replaying the edit instead of truncating. Returns ``(rc, bytes)``;
        rc != 0 means the file is unreadable (missing / permission) and the caller must
        replay the edit rather than drop it.
        """
        limit = int(max_bytes) + 1
        result = self.container.exec_run(
            ["head", "-c", str(limit), "--", path], workdir=self.workdir
        )
        output = result.output
        if output is None:
            output = b""
        elif not isinstance(output, (bytes, bytearray)):
            output = str(output).encode("utf-8", "surrogateescape")
        exit_code = result.exit_code
        if exit_code is None:
            exit_code = -1
        return exit_code, bytes(output)

    def execute(self, command):
        """
        Executes a bash command.
        Returns (success, output).
        """
        print(f"[Container ID: {self.container.short_id}]")
        print(f"Executing: {command}")

        preflight_rejection_prefix = self._get_preflight_rejection_prefix(command)
        if preflight_rejection_prefix:
            print("Command rejected before execution by sandbox preflight.")
            return False, preflight_rejection_prefix

        wrapped_command = ["/bin/bash", "-c", self._wrap_command_with_timeout(command)]
        try:
            exec_result = self.container.exec_run(
                wrapped_command,
                workdir=self.workdir
            )
        except docker.errors.DockerException as exc:
            print(f"[System] Command execution failed because the container is unusable: {exc}")
            recovery_message = self._restore_last_success_container(
                reason=f"container_exec_error: {exc}"
            )
            return False, (
                f"[SYSTEM] Command execution failed because the container became unusable: {exc}\n"
                f"{recovery_message}"
            )

        exit_code = exec_result.exit_code
        output = exec_result.output.decode('utf-8', errors='replace')
        retry_outputs = []
        attempt_index = 1
        while self._should_retry_transient_package_failure(
            command, exit_code, output, attempt_index
        ):
            manager = self._transient_package_manager(command, output)
            retry_outputs.append(
                f"[SYSTEM] Transient {manager} failure on attempt {attempt_index}; "
                f"retrying the same command.\n{output}"
            )
            attempt_index += 1
            print(
                f"[Sandbox Retry] Transient {manager} failure; "
                f"retrying attempt {attempt_index}/"
                f"{self._package_retry_attempts(manager)}."
            )
            if manager == "apt":
                self._prepare_apt_retry(self.container)
            try:
                exec_result = self.container.exec_run(
                    wrapped_command,
                    workdir=self.workdir,
                )
            except docker.errors.DockerException as exc:
                print(f"[System] Command retry failed because the container is unusable: {exc}")
                recovery_message = self._restore_last_success_container(
                    reason=f"container_exec_retry_error: {exc}"
                )
                return False, (
                    f"[SYSTEM] Command retry failed because the container became unusable: {exc}\n"
                    f"{recovery_message}"
                )
            exit_code = exec_result.exit_code
            output = exec_result.output.decode('utf-8', errors='replace')

        if retry_outputs:
            output = "\n\n".join([*retry_outputs, output])

        if self._is_timeout_exit(exit_code):
            output = (
                f"[SYSTEM] Command timed out after {self.command_timeout_seconds} seconds.\n\n"
                f"{output}"
            )
        
        # 判断是否为"信息性退出"（非真正错误）
        is_informational_exit = self._is_informational_exit(exit_code, output)
        
        # 检测输出中是否有测试失败信号（用于 Observation 前缀注入）
        test_fail_prefix = self._get_test_failure_prefix(exit_code, output)
        truncated_test_prefix = self._get_truncated_test_output_prefix(command)
        
        if exit_code == 0 or is_informational_exit:
            # Success: 保存当前成功状态
            if is_informational_exit:
                print(f"Command exited with code {exit_code} (informational, not an error).")
            else:
                print("Command succeeded.")

            if truncated_test_prefix:
                output = truncated_test_prefix + output

            self.package_manager_broken_failure_streak = 0
            self._track_runtime_command(command)
            
            # 优化：只对会对环境产生影响的指令进行 commit
            if self._should_commit(command):
                # 创建新的成功快照
                previous_snapshot = self.last_success_image
                success_image = self.container.commit()
                self._register_snapshot(success_image.id)
                self.last_success_image = success_image.id
                if previous_snapshot and previous_snapshot != self.last_success_image:
                    self._remove_snapshot_image(previous_snapshot)
                print(f"[Snapshot Created] {self.last_success_image[:12]}")
            else:
                print("[Skip Snapshot] Command is read-only or informational.")
            
            return True, output
        else:
            print(f"Command failed (exit {exit_code}). Preserving current state for agent decision.")

            failed_mutation_prefix = self._get_failed_mutating_command_prefix(
                command, exit_code
            )
            rollback_candidate_prefix = self._get_package_manager_rollback_prefix(
                command, exit_code, output
            )
            if truncated_test_prefix:
                output = truncated_test_prefix + output
            if test_fail_prefix:
                output = test_fail_prefix + output
            if failed_mutation_prefix:
                output = failed_mutation_prefix + output
            if rollback_candidate_prefix:
                output = rollback_candidate_prefix + output

            if not self._container_is_healthy():
                print("[System] Container became unhealthy after the failed command. Restoring last successful snapshot.")
                recovery_message = self._restore_last_success_container(
                    reason=f"container_unhealthy_after_exit_{exit_code}"
                )
                if output and not output.endswith("\n"):
                    output += "\n"
                output += (
                    "\n[SYSTEM] The container became unhealthy after the failed command, "
                    "so the system restored the last successful snapshot automatically.\n"
                    f"{recovery_message}"
                )

            return False, output

    def rollback(self, reason="agent_requested"):
        """Restore the container to the last successful snapshot on explicit agent request."""
        print(f"[System] Explicit rollback requested ({reason}).")
        try:
            message = self._restore_last_success_container(reason=reason)
        except docker.errors.DockerException as exc:
            return False, f"[SYSTEM] Rollback failed: {exc}"
        return True, message

    def reset_to_base(self) -> None:
        """Recreate the container fresh from base_image (NOT last_success_image).

        Distinct from rollback()/_restore_last_success_container, which restore the
        last good snapshot. Used by the Stage-2 binding-install gate so every install
        attempt runs from clean. Does NOT replay runtime services (install-only path).
        """
        if self.container is not None:
            try:
                self.container.stop()
            except docker.errors.DockerException:
                pass
            try:
                self.container.remove()
            except docker.errors.DockerException:
                pass
        _extra_hosts = _service_extra_hosts()
        base_image_ref = getattr(self, "base_image_ref", self.base_image)
        self.container = self.client.containers.run(
            base_image_ref, detach=True, tty=True, working_dir=self.workdir,
            command="/bin/sh", volumes=self.volumes, platform=self.platform,
            **self._container_environment_kwargs(),
            **({} if _extra_hosts is None else {"extra_hosts": _extra_hosts}),
        )
        self._ensure_bash_available()
        self.container.exec_run(f"mkdir -p {self.workdir}")
        self._bootstrap_apt_if_supported()
        if self.seed_dir:
            self._seed_workdir_from_host()
        self.current_image = base_image_ref

    def run_install_script(self, script: str) -> InstallResult:
        """Run an install-only setup.sh in the CURRENT container, bypassing execute()'s
        preflight (which rejects multi-step scripts). Returns InstallResult with the
        ERR-trap-localized failing command on rc!=0.

        Invariant: this does NOT commit a snapshot; the Stage-2 gate always calls
        reset_to_base() before this, so last_success_image is never relied upon here.
        """
        return self._run_install_script_in_container(self.container, script)

    def _run_install_script_in_container(self, container, script: str) -> InstallResult:
        wrapped = _wrap_with_err_trap(script)
        command = ["/bin/bash", "-c", self._wrap_command_with_timeout(wrapped)]
        retry_outputs: list[str] = []
        attempt_index = 1
        while True:
            result = container.exec_run(command, workdir=self.workdir)
            output = result.output
            if isinstance(output, (bytes, bytearray)):
                output = output.decode("utf-8", errors="replace")
            output = output or ""
            rc = result.exit_code if result.exit_code is not None else -1
            if self._is_timeout_exit(rc):
                output = (
                    f"[SYSTEM] Command timed out after "
                    f"{self.command_timeout_seconds} seconds.\n\n{output}"
                )
            failing_command, lineno = _parse_install_failure(output)
            if not self._should_retry_transient_package_failure(
                failing_command or "", rc, output, attempt_index
            ):
                break
            manager = self._transient_package_manager(
                failing_command or "", output
            )
            retry_outputs.append(
                f"[SYSTEM] Transient {manager} failure during setup replay on attempt "
                f"{attempt_index}; retrying the complete idempotent setup script.\n{output}"
            )
            attempt_index += 1
            print(
                f"[Sandbox Retry] Transient {manager} failure during setup replay; "
                f"retrying attempt {attempt_index}/"
                f"{self._package_retry_attempts(manager)}."
            )
            if manager == "apt":
                self._prepare_apt_retry(container)
        combined = "\n\n".join([*retry_outputs, output]) if retry_outputs else output
        return InstallResult(
            rc=rc,
            failing_command=failing_command,
            lineno=lineno,
            stderr=combined,
        )

    def create_candidate_container(
        self,
        transaction_id: str,
        checkpoint_name: str | None,
    ) -> CandidateContainer:
        """Fork an isolated transaction container from a named checkpoint image.

        The image already contains the verified prefix, so this method deliberately
        performs no bootstrap, repository seeding, or Build Plan command replay.
        The working container remains live and untouched until ``promote_candidate``.
        """
        if not transaction_id:
            raise ValueError("candidate transaction_id must be non-empty")
        candidates = getattr(self, "candidate_containers", None)
        if candidates is None:
            candidates = self.candidate_containers = {}
        if transaction_id in candidates:
            raise ValueError(f"candidate transaction already exists: {transaction_id}")

        base_checkpoint = checkpoint_name or "base"
        image_id = getattr(self, "named_checkpoints", {}).get(base_checkpoint)
        if image_id is None:
            raise KeyError(f"unknown candidate base checkpoint: {base_checkpoint}")
        _extra_hosts = _service_extra_hosts()
        container = self.client.containers.run(
            image_id,
            detach=True,
            tty=True,
            working_dir=self.workdir,
            command="/bin/bash",
            # Candidate transactions must not mutate working-container bind/cache
            # volumes. The verified prefix and repository are already in the image.
            volumes=None,
            platform=self.platform,
            **self._container_environment_kwargs(),
            **({} if _extra_hosts is None else {"extra_hosts": _extra_hosts}),
        )
        handle = CandidateContainer(
            transaction_id=transaction_id,
            base_checkpoint=base_checkpoint,
            image_id=image_id,
            container=container,
        )
        candidates[transaction_id] = handle
        print(
            f"[Candidate Created] {transaction_id} from "
            f"{base_checkpoint} -> {image_id[:12]}"
        )
        return handle

    def create_resolver_container(self, resolver_id: str) -> ResolverContainer:
        """Fork an isolated resolver container from the seeded base checkpoint.

        Resolver commands may create lock/cache files, so they never execute in
        the working container. The base checkpoint already contains the
        repository and apt bootstrap; no Build Plan command is replayed.
        """
        if not resolver_id:
            raise ValueError("resolver_id must be non-empty")
        if resolver_id in self.resolver_containers:
            raise ValueError(f"resolver container already exists: {resolver_id}")
        image_id = self.named_checkpoints.get("base")
        if image_id is None:
            raise KeyError("base checkpoint unavailable for resolver")
        container = self.client.containers.run(
            image_id,
            detach=True,
            tty=True,
            working_dir=self.workdir,
            command="/bin/bash",
            volumes=None,
            platform=self.platform,
            **self._container_environment_kwargs(),
        )
        handle = ResolverContainer(
            resolver_id=resolver_id,
            image_id=image_id,
            container=container,
        )
        self.resolver_containers[resolver_id] = handle
        print(f"[Resolver Created] {resolver_id} -> {image_id[:12]}")
        return handle

    def resolver_exec(
        self,
        resolver: ResolverContainer | str,
        command: str,
    ) -> tuple[int, str]:
        resolver_id = (
            resolver.resolver_id
            if isinstance(resolver, ResolverContainer)
            else str(resolver)
        )
        handle = self.resolver_containers.get(resolver_id)
        if handle is None:
            raise KeyError(f"unknown resolver container: {resolver_id}")
        result = handle.container.exec_run(
            ["/bin/bash", "-o", "pipefail", "-lc", command],
            workdir=self.workdir,
        )
        output = result.output
        if isinstance(output, (bytes, bytearray)):
            output = output.decode("utf-8", errors="replace")
        return (result.exit_code if result.exit_code is not None else -1), output or ""

    def close_resolver_container(
        self,
        resolver: ResolverContainer | str,
    ) -> None:
        resolver_id = (
            resolver.resolver_id
            if isinstance(resolver, ResolverContainer)
            else str(resolver)
        )
        handle = self.resolver_containers.pop(resolver_id, None)
        if handle is None:
            return
        try:
            handle.container.stop()
        except docker.errors.DockerException:
            pass
        try:
            handle.container.remove()
        except docker.errors.DockerException:
            pass
        print(f"[Resolver Removed] {resolver_id}")

    def _candidate(self, candidate: CandidateContainer | str) -> CandidateContainer:
        transaction_id = (
            candidate.transaction_id
            if isinstance(candidate, CandidateContainer)
            else str(candidate)
        )
        handle = getattr(self, "candidate_containers", {}).get(transaction_id)
        if handle is None:
            raise KeyError(f"unknown candidate transaction: {transaction_id}")
        return handle

    def candidate_exec_readonly(
        self,
        candidate: CandidateContainer | str,
        command: str,
    ) -> tuple[int, str]:
        handle = self._candidate(candidate)
        result = handle.container.exec_run(
            ["/bin/sh", "-lc", command], workdir=self.workdir
        )
        output = result.output
        if isinstance(output, (bytes, bytearray)):
            output = output.decode("utf-8", errors="replace")
        return (result.exit_code if result.exit_code is not None else -1), output or ""

    def candidate_run_install_script(
        self,
        candidate: CandidateContainer | str,
        script: str,
    ) -> InstallResult:
        return self._run_install_script_in_container(
            self._candidate(candidate).container, script
        )

    def abort_candidate(self, candidate: CandidateContainer | str) -> None:
        handle = self._candidate(candidate)
        try:
            handle.container.stop()
        except docker.errors.DockerException:
            pass
        try:
            handle.container.remove()
        except docker.errors.DockerException:
            pass
        self.candidate_containers.pop(handle.transaction_id, None)
        print(f"[Candidate Aborted] {handle.transaction_id}")

    def promote_candidate(self, candidate: CandidateContainer | str) -> None:
        """Atomically make a verified candidate the working container.

        Graph/manual-block state is committed by the caller only after this method
        succeeds.  The old working container is never used to replay the repair.
        """
        handle = self._candidate(candidate)
        old_container = self.container
        self.container = handle.container
        self.current_image = handle.image_id
        self.last_success_image = handle.image_id
        self.candidate_containers.pop(handle.transaction_id, None)
        if old_container is not None and old_container is not handle.container:
            try:
                old_container.stop()
            except docker.errors.DockerException:
                pass
            try:
                old_container.remove()
            except docker.errors.DockerException:
                pass
        print(f"[Candidate Promoted] {handle.transaction_id}")

    def create_checkpoint(self, name: str) -> str:
        """Commit the current container under a stable semantic checkpoint name.

        Unlike ``last_success_image``, named checkpoints are not rotated by
        ``execute()``.  The incremental plan executor owns their lifetime and
        invalidates them when a changed block breaks the corresponding prefix.
        """
        if not name or name == "base":
            raise ValueError("checkpoint name must be non-empty and cannot replace 'base'")
        image = self.container.commit()
        image_id = image.id
        self._register_snapshot(image_id)
        checkpoints = getattr(self, "named_checkpoints", None)
        if checkpoints is None:
            checkpoints = self.named_checkpoints = {}
        previous = checkpoints.get(name)
        checkpoints[name] = image_id
        if previous and previous != image_id and previous not in checkpoints.values():
            self._remove_snapshot_image(previous, force_named=True)
        print(f"[Checkpoint Created] {name} -> {image_id[:12]}")
        return name

    def restore_checkpoint(self, name: str) -> None:
        """Restore a named build checkpoint without replaying runtime services."""
        image_id = getattr(self, "named_checkpoints", {}).get(name)
        if image_id is None:
            raise KeyError(f"unknown checkpoint: {name}")
        self._replace_container_from_image(image_id, replay_runtime=False)
        self.current_image = image_id
        self.last_success_image = image_id
        print(f"[Checkpoint Restored] {name} -> {image_id[:12]}")

    def drop_checkpoint(self, name: str) -> None:
        """Drop one non-base checkpoint and its unreferenced image."""
        if name == "base":
            return
        checkpoints = getattr(self, "named_checkpoints", {})
        image_id = checkpoints.pop(name, None)
        if not image_id or image_id in checkpoints.values():
            return
        if image_id == getattr(self, "last_success_image", None):
            self.last_success_image = checkpoints.get("base")
        self._remove_snapshot_image(image_id, force_named=True)
        print(f"[Checkpoint Dropped] {name}")

    def _register_snapshot(self, image_id):
        if image_id:
            self.snapshot_image_ids.add(image_id)

    def _remove_snapshot_image(self, image_id, *, force_named=False):
        if not image_id:
            return
        if not force_named and image_id in getattr(self, "named_checkpoints", {}).values():
            return
        try:
            image = self.client.images.get(image_id)
            self.client.images.remove(image.id, force=True)
        except docker.errors.ImageNotFound:
            self.snapshot_image_ids.discard(image_id)
            return
        except docker.errors.APIError:
            return
        else:
            self.snapshot_image_ids.discard(image_id)

    def _replace_container_from_image(self, image, *, replay_runtime: bool) -> None:
        if self.container:
            try:
                self.container.stop()
            except docker.errors.DockerException:
                pass
            try:
                self.container.remove()
            except docker.errors.DockerException:
                pass

        _extra_hosts = _service_extra_hosts()
        self.container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            working_dir=self.workdir,
            command="/bin/bash",
            volumes=self.volumes,
            platform=self.platform,
            **self._container_environment_kwargs(),
            **({} if _extra_hosts is None else {"extra_hosts": _extra_hosts})
        )
        self.container.exec_run(f"mkdir -p {self.workdir}")
        if replay_runtime:
            self._replay_runtime_commands()

    def _restore_last_success_container(self, reason="rollback"):
        rollback_image = (
            self.last_success_image
            if self.last_success_image
            else getattr(self, "base_image_ref", self.base_image)
        )
        self._replace_container_from_image(rollback_image, replay_runtime=True)
        return (
            "[SYSTEM] Restored the container to the last successful snapshot "
            f"because of {reason}.\n"
            "[SYSTEM] Ephemeral runtime services were replayed when possible."
        )

    def _wrap_command_with_timeout(self, command):
        """Enforce a per-command timeout when GNU `timeout` is available in the container."""
        pipefail_command = self._wrap_command_with_pipefail(command)
        if not self.command_timeout_seconds:
            return pipefail_command

        timeout_seconds = int(self.command_timeout_seconds)
        return (
            "if timeout --version 2>&1 | grep -q 'GNU coreutils'; then "
            f"timeout --foreground --kill-after=30s {timeout_seconds}s {pipefail_command}; "
            "else "
            f"{pipefail_command}; "
            "fi"
        )

    def _wrap_command_with_pipefail(self, command):
        quoted_command = shlex.quote(command)
        # Do not use a login shell here: it can replace the container's PATH
        # and hide the project virtual environment configured by the runner.
        return f"/bin/bash -o pipefail -c {quoted_command}"

    def exec_with_timeout(self, container, command, *, workdir=None):
        """Execute a raw command in any managed container with the sandbox timeout.

        ReAct may target the active, candidate, or clean-replay container rather
        than ``self.container``.  This helper keeps those direct execution paths
        under the same per-command timeout without applying snapshot or retry
        policy.
        """
        result = container.exec_run(
            ["/bin/bash", "-c", self._wrap_command_with_timeout(command)],
            workdir=workdir or self.workdir,
        )
        output = result.output
        if isinstance(output, (bytes, bytearray)):
            output = output.decode("utf-8", errors="replace")
        output = output or ""
        exit_code = result.exit_code if result.exit_code is not None else -1
        if self._is_timeout_exit(exit_code):
            output = (
                f"[SYSTEM] Command timed out after "
                f"{self.command_timeout_seconds} seconds.\n\n{output}"
            )
        return exit_code, output

    def _is_timeout_exit(self, exit_code):
        if not self.command_timeout_seconds:
            return False
        return exit_code in {124, 137}

    def _should_retry_transient_pip_failure(self, command, exit_code, output, attempt_index):
        if attempt_index >= PIP_TRANSIENT_RETRY_ATTEMPTS:
            return False
        if exit_code == 0:
            return False
        if not self._looks_like_pip_install_command(command):
            return False

        normalized_output = (output or "").lower()
        transient_markers = (
            "readtimeouterror",
            "connection reset",
            "connection aborted",
            "connection broken",
            "remote disconnected",
            "temporary failure in name resolution",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "sslerror",
            "max retries exceeded",
            "too many 502 error responses",
            "too many 503 error responses",
            "too many 504 error responses",
            "network is unreachable",
        )
        return any(marker in normalized_output for marker in transient_markers)

    def _should_retry_transient_apt_failure(
        self, command, exit_code, output, attempt_index
    ):
        if attempt_index >= APT_TRANSIENT_RETRY_ATTEMPTS:
            return False
        if exit_code == 0 or not self._looks_like_apt_command(command):
            return False
        normalized_output = (output or "").lower()
        transient_markers = (
            "unexpected eof",
            "failed to fetch",
            "temporary failure resolving",
            "temporary failure in name resolution",
            "connection reset",
            "connection failed",
            "could not connect",
            "network is unreachable",
            "i/o timeout",
            "connection timed out",
            "reading http response body",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
        )
        return any(marker in normalized_output for marker in transient_markers)

    def _should_retry_transient_package_failure(
        self, command, exit_code, output, attempt_index
    ):
        return self._should_retry_transient_pip_failure(
            command, exit_code, output, attempt_index
        ) or self._should_retry_transient_apt_failure(
            command, exit_code, output, attempt_index
        )

    def _transient_package_manager(self, command, output=""):
        if self._looks_like_apt_command(command):
            return "apt"
        return "pip"

    def _package_retry_attempts(self, manager):
        return (
            APT_TRANSIENT_RETRY_ATTEMPTS
            if manager == "apt"
            else PIP_TRANSIENT_RETRY_ATTEMPTS
        )

    def _looks_like_apt_command(self, command):
        return bool(
            re.search(
                r"\b(?:sudo\s+)?apt(?:-get)?\b[^;&|]{0,500}"
                r"\b(?:update|install|upgrade|dist-upgrade|full-upgrade)\b",
                command or "",
                re.DOTALL,
            )
        )

    def _prepare_apt_retry(self, container):
        """Repair partial dpkg state and refresh indexes before replaying setup."""
        recovery = (
            "if command -v apt-get >/dev/null 2>&1; then "
            "apt-get clean; "
            "dpkg --configure -a >/dev/null 2>&1 || true; "
            "apt-get -o Acquire::Retries=10 --fix-missing -f install -y "
            ">/dev/null 2>&1 || true; "
            "rm -rf /var/lib/apt/lists/*; "
            "apt-get -o Acquire::Retries=10 update >/dev/null 2>&1 || true; "
            "fi"
        )
        try:
            container.exec_run(["/bin/bash", "-lc", recovery], workdir=self.workdir)
        except docker.errors.DockerException:
            pass

    def _looks_like_pip_install_command(self, command):
        for raw_segment, _ in self._command_classifier._split_shell_chain(command or ""):
            normalized = self._command_classifier._normalize_command_segment(raw_segment)
            if self._command_classifier._is_navigation_only_segment(normalized):
                continue
            if re.match(r"^(?:pip|pip2|pip3)\s+install\b", normalized):
                return True
            if re.match(r"^(?:python|python2|python3)\s+-m\s+pip\s+install\b", normalized):
                return True
            if re.match(r"^uv\s+pip\s+install\b", normalized):
                return True
        return False

    def _container_is_healthy(self):
        if not self.container:
            return False

        reload_fn = getattr(self.container, "reload", None)
        if callable(reload_fn):
            try:
                reload_fn()
            except docker.errors.DockerException:
                return False

        status = getattr(self.container, "status", None)
        if status is None:
            return True
        return status == "running"
    
    def _should_commit(self, command):
        """
        判断指令是否会对环境产生影响，从而决定是否需要 commit。
        """
        # 常见的不产生副作用的指令
        readonly_commands = [
            'ls', 'cat', 'pwd', 'echo', 'env', 'hostname', 'whoami', 
            'head', 'tail', 'grep', 'find', 'du', 'df', 'top', 'ps', 
            'date', 'which', 'type', 'file'
        ]
        
        # 获取指令的第一个单词
        first_word = command.strip().split()[0].lower() if command.strip() else ""
        
        # 如果指令在只读列表中，则不 commit
        if first_word in readonly_commands:
            return False
            
        # 默认需要 commit
        return True

    def _track_runtime_command(self, command):
        """Remember pure runtime service commands so they can be replayed after rollback."""
        normalized_command = (command or "").strip()
        if not normalized_command:
            return

        if not self._command_classifier.is_runtime_service_command(normalized_command):
            return

        runtime_key = self._runtime_service_key(normalized_command)
        runtime_action = self._runtime_service_action(normalized_command)

        if runtime_key and runtime_action == "stop":
            self.runtime_replay_commands = [
                entry for entry in self.runtime_replay_commands if entry["key"] != runtime_key
            ]
            return

        if self._command_classifier.is_persistent_setup_command(normalized_command):
            return

        if runtime_key:
            self.runtime_replay_commands = [
                entry for entry in self.runtime_replay_commands if entry["key"] != runtime_key
            ]

        self.runtime_replay_commands.append(
            {
                "key": runtime_key or normalized_command,
                "command": normalized_command,
            }
        )

    def _replay_runtime_commands(self):
        """Restore ephemeral runtime services after container rollback."""
        if not self.runtime_replay_commands:
            return

        print(
            f"[Runtime Replay] Re-running {len(self.runtime_replay_commands)} runtime command(s) after rollback."
        )
        restored_commands = []
        for entry in self.runtime_replay_commands:
            command = entry["command"]
            exec_result = self.container.exec_run(
                ["/bin/bash", "-c", self._wrap_command_with_timeout(command)],
                workdir=self.workdir,
            )
            exit_code = exec_result.exit_code
            output = exec_result.output.decode("utf-8", errors="replace")
            if exit_code == 0 or self._is_informational_exit(exit_code, output):
                print(f"[Runtime Replay] Restored: {command}")
                restored_commands.append(entry)
                continue

            print(
                f"[Runtime Replay Warning] Failed to restore runtime command (exit {exit_code}): {command}"
            )

        self.runtime_replay_commands = restored_commands

    def _runtime_service_key(self, command):
        normalized = command.strip().lower()
        service_match = re.match(r"^service\s+(\S+)\s+(?:start|restart|reload|stop)\b", normalized)
        if service_match:
            return service_match.group(1)

        executable = normalized.split()[0] if normalized else ""
        executable = executable.strip("\"'`")
        if executable:
            return executable.rsplit("/", 1)[-1]
        return None

    def _runtime_service_action(self, command):
        normalized = command.strip().lower()
        service_match = re.match(r"^service\s+\S+\s+(start|restart|reload|stop)\b", normalized)
        if service_match:
            return service_match.group(1)
        if " --daemonize " in f" {normalized} " or " -detached" in normalized or " --fork" in normalized:
            return "start"
        return "start"
    
    def _is_informational_exit(self, exit_code, output):
        """
        判断是否为信息性退出（如显示帮助信息），而非真正的错误。
        测试命令的失败（如测试未通过）不应被视为信息性退出。
        """
        # Exit code 1-2 通常是参数错误或显示帮助
        if exit_code not in [1, 2]:
            return False
        
        # 检查输出中是否包含帮助信息的特征
        help_indicators = [
            'Usage:',
            'usage:',
            '--help',
            'Options:',
            'Commands:',
            'positional arguments:',
            'optional arguments:'
        ]
        
        # 测试失败的特征（不应被误判为信息性退出）
        test_failure_indicators = [
            'failures:',
            'errors:',
            'FAILED',
            'Failed:',        # run_all / TAP 格式：Failed: 3
            'not ok',         # TAP 协议失败行
            'Test failed',
            'assertion failed',
            'expected',
            'actual',
            'diff:',
            'Traceback (most recent call last):',
            'NameError',
            'ImportError',
            'ModuleNotFoundError',
            'LoadError',
            'Gem::LoadError',
            'bundler: command not found'
        ]
        
        output_lower = output.lower()
        
        # 如果包含测试失败特征，则不是信息性退出
        if any(indicator.lower() in output_lower for indicator in test_failure_indicators):
            return False
        
        return any(indicator.lower() in output_lower for indicator in help_indicators)

    def _get_test_failure_prefix(self, exit_code, output):
        """
        检测命令输出是否包含测试失败信号。
        若是，返回注入到 Observation 头部的强制警告；否则返回空字符串。
        目的：阻止 LLM 以"核心功能通过"为由自我合理化，绕过 No Excuses Rule。
        """
        if exit_code == 0:
            return ""

        # TAP 格式失败：run_all 输出的 "Failed: N"
        tap_fail = re.search(r'Failed:\s+([1-9]\d*)', output)
        if tap_fail:
            failed_count = tap_fail.group(1)
            return (
                f"[SYSTEM] ⚠️  TEST FAILURE DETECTED: {failed_count} test(s) FAILED.\n"
                f"[SYSTEM] Per the No Excuses Rule, you CANNOT output 'Final Answer: Success' "
                f"from this failed command. Repo2Run success requires a real "
                f"`pytest --collect-only -q --disable-warnings` verification, or the Poetry "
                f"equivalent, that proves tests are collectable. Full test failures do not need "
                f"to pass if collection succeeds.\n\n"
            )

        # pytest / unittest 格式失败
        pytest_fail = re.search(r'([1-9]\d*) failed', output, re.IGNORECASE)
        if pytest_fail:
            failed_count = pytest_fail.group(1)
            return (
                f"[SYSTEM] ⚠️  TEST FAILURE DETECTED: {failed_count} test(s) FAILED.\n"
                f"[SYSTEM] Per the No Excuses Rule, you CANNOT output 'Final Answer: Success' "
                f"from this failed command. If full test execution fails, rerun the Repo2Run "
                f"collection command and use it as final proof only if collection succeeds.\n\n"
            )

        pytest_error = re.search(r'([1-9]\d*) errors?', output, re.IGNORECASE)
        if pytest_error:
            error_count = pytest_error.group(1)
            return (
                f"[SYSTEM] ⚠️  TEST FAILURE DETECTED: {error_count} test error(s) reported.\n"
                f"[SYSTEM] Per the No Excuses Rule, you CANNOT output 'Final Answer: Success' "
                f"until Repo2Run-style pytest collection succeeds without collection/import/config "
                f"errors.\n\n"
            )

        if self._command_classifier.observation_has_test_failure_signal(output):
            return (
                "[SYSTEM] ⚠️  TEST FAILURE DETECTED in command output.\n"
                "[SYSTEM] Per the No Excuses Rule, you CANNOT output 'Final Answer: Success' "
                "from this failed command. Final proof must be a successful Repo2Run-style "
                "pytest collection command, not a failed full test run.\n\n"
            )

        # 通用失败关键词。只匹配测试用例行，避免把 "Failed: 0" 当成失败。
        if (
            re.search(r'^\s*(?:FAILED|ERROR)\s+\S+', output, re.MULTILINE)
            or re.search(r'^\s*not ok\b', output, re.IGNORECASE | re.MULTILINE)
        ):
            return (
                "[SYSTEM] ⚠️  TEST FAILURE DETECTED in command output.\n"
                "[SYSTEM] Per the No Excuses Rule, you CANNOT output 'Final Answer: Success' "
                "from this failed command. Final proof must be a successful Repo2Run-style "
                "pytest collection command.\n\n"
            )

        return ""

    def _get_truncated_test_output_prefix(self, command):
        """Warn when a test command is piped through a lossy output filter."""
        if not self._command_classifier.is_truncated_test_output_command(command or ""):
            return ""

        return (
            "[SYSTEM] ⚠️  TRUNCATED TEST OUTPUT: this command pipes a test run through "
            "`head`, `tail`, or `grep`, so the Observation may omit failures, passes, or the final "
            "test summary.\n"
            "[SYSTEM] Do NOT treat this as complete verification. For final verification, "
            "run the full project test command without output-limiting pipes; long output "
            "will be handled by observation compression.\n\n"
        )

    def _get_preflight_rejection_prefix(self, command):
        """Reject commands that are too ambiguous to record/replay safely."""
        return (
            self._get_invalid_output_filter_prefix(command)
            or self._get_invalid_compound_setup_prefix(command)
        )

    def _get_invalid_output_filter_prefix(self, command):
        """Reject setup/test commands piped through lossy output filters."""
        if not self._command_pipes_setup_or_test_through_output_filter(command or ""):
            return ""

        return (
            "[SYSTEM] COMMAND REJECTED BEFORE EXECUTION: setup or test commands "
            "must not pipe output through `head`, `tail`, or `grep` because those "
            "filters can hide failures and mask the real exit status.\n"
            "[SYSTEM] The command was NOT executed and the environment was not "
            "changed. Rerun the full command without output filtering. Long output "
            "will be handled by observation compression.\n\n"
        )

    def _command_pipes_setup_or_test_through_output_filter(self, command):
        for raw_segment, _ in self._command_classifier._split_shell_chain(command or ""):
            pipeline_components = self._command_classifier._split_pipeline(raw_segment)
            if len(pipeline_components) < 2:
                continue

            saw_setup_or_test = False
            for component in pipeline_components:
                segment_type = self._classify_preflight_segment(component)
                if segment_type == "output_filter" and saw_setup_or_test:
                    return True
                if segment_type in {"setup", "test", "probe"}:
                    saw_setup_or_test = True
        return False

    def _get_invalid_compound_setup_prefix(self, command):
        """Reject multi-action setup chains that should be split into separate Actions."""
        segments = self._classified_preflight_segments(command)
        if len(segments) <= 1:
            return ""

        if self._is_allowed_apt_update_install_chain(segments):
            return ""

        setup_segments = [segment for segment in segments if segment["type"] == "setup"]
        if len(setup_segments) >= 2:
            return self._build_invalid_compound_setup_message(
                "this Action combines multiple independent setup mutations",
            )

        if len(setup_segments) == 1:
            mixed_segments = [
                segment
                for segment in segments
                if segment["type"] in {"test", "probe", "readonly"}
            ]
            if mixed_segments:
                return self._build_invalid_compound_setup_message(
                    "this Action combines a setup mutation with a verification, probe, or read-only check",
                )

        return ""

    def _classified_preflight_segments(self, command):
        segments = []
        for raw_segment, _ in self._command_classifier._split_shell_chain(command or ""):
            segment_type = self._classify_preflight_segment(raw_segment)
            if segment_type == "empty":
                continue
            segments.append({"raw": raw_segment.strip(), "type": segment_type})
        return segments

    def _classify_preflight_segment(self, raw_segment):
        cleaned_segment = self._command_classifier._strip_trailing_redirections(
            raw_segment or ""
        )
        normalized = self._command_classifier._normalize_command_segment(cleaned_segment)
        if not normalized:
            return "empty"
        if self._command_classifier._is_output_truncation_component(normalized):
            return "output_filter"
        if self._command_classifier._is_navigation_only_segment(normalized):
            return "navigation"
        if self._is_setup_preparation_segment(normalized):
            return "preparation"
        if self._command_classifier._is_test_like_segment(normalized):
            return "test"
        if self._is_probe_segment(normalized):
            return "probe"
        if self._command_classifier._is_readonly_command(cleaned_segment):
            return "readonly"
        if self._command_classifier._segment_has_meaningful_setup_activity(normalized):
            return "setup"
        return "other"

    def _is_setup_preparation_segment(self, normalized_segment):
        return (
            normalized_segment.startswith("mkdir -p ")
            or normalized_segment.startswith("mkdir --parents ")
            or normalized_segment.startswith("install -d ")
        )

    def _is_probe_segment(self, normalized_segment):
        probe_patterns = (
            r"^(?:python|python2|python3)\s+-c\b",
            r"^(?:node)\s+-e\b",
            r"^(?:ruby)\s+-e\b",
            r"^(?:php)\s+-r\b",
            r"^(?:pip|pip2|pip3|python\s+-m\s+pip|python3\s+-m\s+pip)\s+(?:check|show|list|freeze)\b",
            r"^(?:npm|yarn|pnpm)\s+(?:list|ls)\b",
        )
        return any(re.search(pattern, normalized_segment) for pattern in probe_patterns)

    def _is_allowed_apt_update_install_chain(self, segments):
        meaningful_segments = [
            segment
            for segment in segments
            if segment["type"] not in {"navigation", "preparation"}
        ]
        if len(meaningful_segments) != 2:
            return False

        first = self._command_classifier._normalize_command_segment(
            meaningful_segments[0]["raw"]
        )
        second = self._command_classifier._normalize_command_segment(
            meaningful_segments[1]["raw"]
        )
        return (
            re.match(r"^(?:apt-get|apt)\s+update\b", first) is not None
            and re.match(r"^(?:apt-get|apt)\s+install\b", second) is not None
        )

    def _build_invalid_compound_setup_message(self, reason):
        return (
            f"[SYSTEM] COMMAND REJECTED BEFORE EXECUTION: {reason}.\n"
            "[SYSTEM] The command was NOT executed and the environment was not changed. "
            "Run each setup mutation, verification, or probe as a separate Action so each "
            "state-changing step can be confirmed independently.\n\n"
        )

    def _get_failed_mutating_command_prefix(self, command, exit_code):
        """Warn when a failed setup command may have left partial environment changes."""
        if exit_code == 0:
            return ""

        if not self._command_classifier.command_mutates_environment(command or ""):
            return ""

        return (
            "[SYSTEM] FAILED SETUP MUTATION: this setup command failed after attempting "
            "to change the environment.\n"
            "[SYSTEM] It may have partially installed packages, modified files, or changed "
            "services. Do not assume useful parts of this failed command are reliably "
            "available for later steps.\n"
            "[SYSTEM] If any prefix or sub-step appears useful, rerun that prefix/sub-step "
            "as its own separate Action so it is confirmed successful. If the partial "
            "changes may have polluted the environment, use `Action: __ROLLBACK__` to "
            "restore the previous snapshot.\n\n"
        )

    def _get_package_manager_rollback_prefix(self, command, exit_code, output):
        """Warn the agent when package-manager failures strongly suggest a dirty dependency state."""
        if exit_code == 0:
            return ""

        normalized_command = (command or "").strip().lower()
        output_lower = (output or "").lower()

        if not self._looks_like_package_manager_command(normalized_command):
            self.package_manager_broken_failure_streak = 0
            return ""

        broken_state_markers = (
            "unmet dependencies",
            "fix-broken install",
            "is not going to be installed",
            "depends:",
            "correcting dependencies... done",
        )
        if not any(marker in output_lower for marker in broken_state_markers):
            self.package_manager_broken_failure_streak = 0
            return ""

        self.package_manager_broken_failure_streak += 1

        if self.package_manager_broken_failure_streak >= 2:
            return (
                "[SYSTEM] STRONG ROLLBACK CANDIDATE: package-manager recovery is still failing "
                "after repeated attempts, and the container likely remains in a broken dependency state.\n"
                "[SYSTEM] Unless you have a concrete repair step that will cleanly resolve the package state, "
                "seriously consider `Action: __ROLLBACK__` before continuing.\n\n"
            )

        return (
            "[SYSTEM] ROLLBACK CANDIDATE: this package-manager failure indicates broken dependencies "
            "or partial package state.\n"
            "[SYSTEM] Consider `Action: __ROLLBACK__` if you believe the failed install left the environment "
            "in an uncertain state, especially before trying unrelated setup work.\n\n"
        )

    def _looks_like_package_manager_command(self, normalized_command):
        package_manager_prefixes = (
            "apt ",
            "apt-get ",
            "yum ",
            "dnf ",
            "apk ",
            "pacman ",
            "zypper ",
        )
        return normalized_command.startswith(package_manager_prefixes)

    def close(self, keep_alive=False):
        """关闭容器，可选择保持容器运行以供验证"""
        for resolver in tuple(getattr(self, "resolver_containers", {}).values()):
            try:
                self.close_resolver_container(resolver)
            except docker.errors.DockerException:
                pass
        for candidate in tuple(getattr(self, "candidate_containers", {}).values()):
            try:
                self.abort_candidate(candidate)
            except (KeyError, docker.errors.DockerException):
                pass
        if self.container:
            if keep_alive:
                print(f"\n[Container Kept Alive] ID: {self.container.short_id}")
                print(f"To inspect: docker exec -it {self.container.short_id} /bin/bash")
                print(f"To stop later: docker stop {self.container.short_id}")
            else:
                try:
                    self.container.stop()
                    self.container.remove()
                    print("\n[Container Cleaned Up]")
                except docker.errors.DockerException:
                    pass

        checkpoints = getattr(self, "named_checkpoints", {})
        def _checkpoint_rank(item):
            name, _image_id = item
            match = re.match(r"exec-(\d+)-", name)
            return int(match.group(1)) if match else (-1 if name == "base" else 0)

        for name, _image_id in sorted(
            tuple(checkpoints.items()), key=_checkpoint_rank, reverse=True
        ):
            if name != "base":
                try:
                    self.drop_checkpoint(name)
                except docker.errors.DockerException:
                    pass

        # Retry all remaining snapshots after named children have been removed;
        # the baseline is deliberately last in the ordering above.
        for snapshot_id in list(self.snapshot_image_ids):
            try:
                self._remove_snapshot_image(snapshot_id, force_named=True)
                print("[Snapshot Image Cleaned]")
            except docker.errors.DockerException:
                pass
        if hasattr(self, "named_checkpoints"):
            self.named_checkpoints.clear()
