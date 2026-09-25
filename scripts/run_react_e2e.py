#!/usr/bin/env python3
"""Construct an environment for one local repository with the ReAct builder."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from httpx import Timeout
from openai import OpenAI
from src.agent.engine import run_react_builder
from src.dockerfile_export import export_dockerfile
from src.envstate.base_image_selection import BaseImageChoice, choose_base_image
from src.envstate.runtime_base import resolve_runtime_base
from src.envstate.env_classifier import make_construction_classifier
from src.envstate.llm_response import complete_with_retry
from src.sandbox import Sandbox
from src.python_deps.depgraph.advise import build_advisory_for_repo
from src.python_deps.depgraph.test_intent import discover_test_dependency_intent


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--model", required=True)
    p.add_argument("--base-image", default="auto")
    p.add_argument("--out", required=True)
    p.add_argument("--trace-out", default=None)
    p.add_argument("--usage-out", default=None)
    p.add_argument("--partial-out", default=None)
    p.add_argument("--runtime-out", default=None)
    p.add_argument("--language-hint", default=None)
    p.add_argument("--execution-mode", default="incremental")
    p.add_argument(
        "--context-mode", choices=("legacy", "diet"),
        default=os.getenv("REACT_CONTEXT_MODE", "legacy"),
        help="conversation retention policy; diet enables delayed reflection compression",
    )
    p.add_argument(
        "--compression-model", default=os.getenv("REACT_COMPRESSION_MODEL"),
        help="optional OpenAI-compatible model for Diet reflection (defaults to --model)",
    )
    p.add_argument(
        "--observation-fallback-chars",
        type=int,
        default=int(os.getenv("REACT_OBSERVATION_FALLBACK_CHARS", "50000")),
        help=(
            "explicit projection threshold; the default selects a model-aware "
            "dynamic threshold in Diet mode"
        ),
    )
    p.add_argument(
        "--observation-summary-chars",
        type=int,
        default=int(os.getenv("REACT_OBSERVATION_SUMMARY_CHARS", "12000")),
        help=(
            "explicit projection/slice size; the default selects a model-aware "
            "dynamic allowance in Diet mode"
        ),
    )
    p.add_argument(
        "--max-turns", "--max-cycles", dest="max_turns", type=int, default=30
    )
    p.add_argument(
        "--command-timeout",
        type=int,
        default=int(os.getenv("REACT_COMMAND_TIMEOUT_SECONDS", "600")),
        help="maximum seconds for each command executed in the sandbox",
    )
    args = p.parse_args()
    if args.command_timeout < 1:
        p.error("--command-timeout must be positive")
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_API_BASE")
    if not key:
        print("ERROR: missing LLM API key", file=sys.stderr)
        return 2
    client = OpenAI(
        api_key=key, base_url=base or None, max_retries=0, timeout=Timeout(120.0)
    )

    def complete(messages):
        return complete_with_retry(client, args.model, messages, temperature=0)[0]

    if args.base_image == "auto":
        # Python50 must be repeatable: derive one interpreter from the root and
        # test-bearing workspace constraints instead of asking the LLM to pick.
        decision = resolve_runtime_base(args.repo, "python:3.11-slim")
        choice = BaseImageChoice(
            decision.base_image,
            decision.minor,
            None,
            f"deterministic requires-python selection: {decision.reason}",
        )
    else:
        choice = choose_base_image(
            args.repo, client, args.model, explicit=args.base_image
        )
    print(f"[v3] base-image: {choice.image} (py {choice.minor}) — {choice.reason}")
    sandbox = Sandbox(
        base_image=choice.image,
        workdir="/app",
        platform=choice.platform_override,
        seed_dir=args.repo,
        command_timeout_seconds=args.command_timeout,
        enable_cache_volume=True,
    )
    try:
        intent = discover_test_dependency_intent(args.repo)
        _advisory, graph = build_advisory_for_repo(
            args.repo,
            sandbox.base_image_ref,
            target_python=choice.minor,
            classify=make_construction_classifier(complete),
            needed_extras=intent.needed_groups,
            platform=sandbox.platform,
        )
        out = Path(args.out).parent
        runtime_environment = {"PATH": "/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"}
        if intent.pytest_addopts:
            runtime_environment["PYTEST_ADDOPTS"] = " ".join(intent.pytest_addopts)
        result = run_react_builder(
            client=client,
            model=args.model,
            sandbox=sandbox,
            graph=graph,
            output_dir=out,
            max_turns=args.max_turns,
            test_command="pytest --collect-only -q --disable-warnings",
            initial_runtime={
                "version": 1,
                "services": [],
                "environment": runtime_environment,
                "capabilities": [],
            },
            context_mode=args.context_mode,
            compression_model=args.compression_model,
            observation_fallback_chars=args.observation_fallback_chars,
            observation_summary_chars=args.observation_summary_chars,
        )
        Path(args.out).write_text(result["script"], encoding="utf-8")
        if args.partial_out:
            Path(args.partial_out).write_text(result["script"], encoding="utf-8")
        if args.usage_out:
            Path(args.usage_out).write_text(
                json.dumps(result["usage"], indent=2) + "\n"
            )
        # run_react_builder already writes the full react_trace.json in this
        # directory. Do not overwrite it with a one-field summary.
        if args.trace_out and not Path(args.trace_out).exists():
            Path(args.trace_out).write_text(
                json.dumps({"reason": result["reason"]}, indent=2) + "\n"
            )
        if args.runtime_out:
            Path(args.runtime_out).write_text(
                json.dumps(result["runtime"], indent=2) + "\n"
            )
        if result["success"]:
            export_dockerfile(out, choice.image, sandbox.platform, result["runtime"])
        print(f"REACT E2E: {'PASS' if result['success'] else 'FAIL'}")
        return 0 if result["success"] else 1
    finally:
        sandbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
