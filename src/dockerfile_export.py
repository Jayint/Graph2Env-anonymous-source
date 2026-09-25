"""Export a Docker build recipe from a single-repository construction run."""

from __future__ import annotations

import json
import re
from pathlib import Path


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IMAGE = re.compile(r"[A-Za-z0-9_./:@+-]+\Z")
_PLATFORM = re.compile(r"[A-Za-z0-9_./-]+\Z")


def export_dockerfile(output_dir: Path, base_image: str, platform: str | None,
                      runtime: dict) -> Path:
    """Write a Dockerfile that builds the supplied checkout and setup.sh."""
    if not _IMAGE.fullmatch(base_image):
        raise ValueError(f"invalid base image: {base_image!r}")
    if platform and not _PLATFORM.fullmatch(platform):
        raise ValueError(f"invalid platform: {platform!r}")

    from_line = f"FROM {base_image}"
    if platform:
        from_line = f"FROM --platform={platform} {base_image}"
    lines = [
        "# syntax=docker/dockerfile:1",
        from_line,
        "USER root",
        "WORKDIR /app",
        "COPY . /app",
        "COPY --from=graph2env /setup.sh /tmp/setup.sh",
        "RUN if ! command -v bash >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v timeout >/dev/null 2>&1; then \\",
        "      if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends bash git coreutils ca-certificates; \\",
        "      elif command -v apk >/dev/null 2>&1; then apk add --no-cache bash git coreutils ca-certificates; \\",
        "      elif command -v dnf >/dev/null 2>&1; then dnf install -y bash git coreutils ca-certificates; \\",
        "      elif command -v yum >/dev/null 2>&1; then yum install -y bash git coreutils ca-certificates; \\",
        "      else exit 127; fi; fi",
        "RUN bash /tmp/setup.sh",
    ]
    for key, value in sorted((runtime.get("environment") or {}).items()):
        if _ENV_NAME.fullmatch(key) and isinstance(value, str) and "\x00" not in value:
            escaped_value = json.dumps(value).replace("$", "\\$")
            lines.append(f"ENV {key}={escaped_value}")

    services = runtime.get("services") or []
    if services:
        entrypoint = ["#!/usr/bin/env bash", "set -e"]
        for service in services:
            start, check = service.get("start"), service.get("check")
            if not isinstance(start, str) or not isinstance(check, str) or not start.strip() or not check.strip():
                raise ValueError("runtime service needs start and check commands")
            entrypoint.extend([
                start,
                "_graph2env_ready=0",
                f"until ({check}); do",
                "  _graph2env_ready=$((_graph2env_ready + 1))",
                "  if [ \"$_graph2env_ready\" -ge 30 ]; then exit 1; fi",
                "  sleep 1",
                "done",
            ])
        entrypoint.append('exec "$@"')
        (output_dir / "runtime-entrypoint.sh").write_text("\n".join(entrypoint) + "\n", encoding="utf-8")
        lines.extend([
            "COPY --from=graph2env /runtime-entrypoint.sh /usr/local/bin/graph2env-runtime-entrypoint",
            "RUN chmod +x /usr/local/bin/graph2env-runtime-entrypoint",
            'ENTRYPOINT ["/usr/local/bin/graph2env-runtime-entrypoint"]',
        ])
    lines.append('CMD ["bash"]')
    dockerfile = output_dir / "Dockerfile"
    dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dockerfile
