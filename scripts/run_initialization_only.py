#!/usr/bin/env python3
"""Construct and verify initial environment artifacts without feedback-driven repair."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from httpx import Timeout
from openai import OpenAI
from src.initialization_only import run_initialization_only
from src.dockerfile_export import export_dockerfile
from src.envstate.base_image_selection import choose_base_image
from src.envstate.env_classifier import make_construction_classifier
from src.envstate.llm_response import complete_with_retry
from src.envstate.runtime_base import resolve_runtime_base
from src.sandbox import Sandbox
from src.python_deps.depgraph.advise import build_advisory_for_repo
from src.python_deps.depgraph.test_intent import discover_test_dependency_intent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-image", default="auto")
    parser.add_argument("--out", required=True)
    for option in ("partial-out", "runtime-out", "trace-out", "usage-out", "language-hint"):
        parser.add_argument(f"--{option}")
    # Accept the existing adapter contract; these do not enable an Agent loop.
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--execution-mode", default="incremental")
    parser.add_argument("--context-mode", default="legacy")
    parser.add_argument("--command-timeout", type=int, default=0,
                        help="per-command seconds; 0 disables the individual limit")
    args = parser.parse_args(argv)
    if args.command_timeout < 0:
        parser.error("--command-timeout must be nonnegative")
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE")
    if not key:
        print("ERROR: missing initialization LLM API key", file=sys.stderr)
        return 2
    client = OpenAI(api_key=key, base_url=base or None, max_retries=0, timeout=Timeout(120.0))
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "api_calls": 0}
    out = Path(args.out).parent
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def complete(messages):
        response, call_usage, _raw = complete_with_retry(client, args.model, messages, temperature=0)
        for name in usage:
            usage[name] += int((call_usage or {}).get(name, 0) or 0)
        return response

    if args.base_image == "auto":
        decision = resolve_runtime_base(args.repo, "python:3.11-slim")
        image, minor, platform = decision.base_image, decision.minor, None
    else:
        choice = choose_base_image(args.repo, client, args.model, explicit=args.base_image)
        image, minor, platform = choice.image, choice.minor, choice.platform_override
    print(f"[v3] base-image: {image} (py {minor}) — initialization-only")
    sandbox = Sandbox(
        base_image=image, workdir="/app", platform=platform, seed_dir=args.repo,
        command_timeout_seconds=args.command_timeout, enable_cache_volume=True,
    )
    try:
        intent = discover_test_dependency_intent(args.repo)
        _advisory, graph = build_advisory_for_repo(
            args.repo, sandbox.base_image_ref, target_python=minor,
            classify=make_construction_classifier(complete),
            needed_extras=intent.needed_groups, platform=sandbox.platform,
        )
        runtime_environment = {"PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"}
        if intent.pytest_addopts:
            runtime_environment["PYTEST_ADDOPTS"] = " ".join(intent.pytest_addopts)
        (out / "initialization_usage.json").write_text(json.dumps({
            "token_usage": usage, "duration_seconds": round(time.monotonic() - started, 3),
            "ablation": "initialization_only",
        }, indent=2) + "\n", encoding="utf-8")
        result = run_initialization_only(
            sandbox=sandbox, graph=graph, output_dir=out,
            test_command="pytest --collect-only -q --disable-warnings",
            initial_runtime={"version": 1, "services": [], "environment": runtime_environment, "capabilities": []},
        )
        Path(args.out).write_text(result["script"], encoding="utf-8")
        if args.partial_out:
            Path(args.partial_out).write_text(result["script"], encoding="utf-8")
        if args.runtime_out:
            Path(args.runtime_out).write_text(json.dumps(result["runtime"], indent=2) + "\n")
        if args.usage_out:
            Path(args.usage_out).write_text(json.dumps(result["usage"], indent=2) + "\n")
        if args.trace_out and Path(args.trace_out) != out / "react_trace.json":
            Path(args.trace_out).write_bytes((out / "react_trace.json").read_bytes())
        if result["success"]:
            export_dockerfile(out, image, sandbox.platform, result["runtime"])
        print(f"INITIALIZATION ONLY: {'PASS' if result['success'] else 'FAIL'}")
        return 0 if result["success"] else 1
    finally:
        sandbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
