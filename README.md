# Graph2Env

Anonymous method implementation for **Graph2Env: Stateful Construction of Repository Execution Environments**.
Graph2Env analyzes a local repository, builds a dependency graph, and uses a ReAct agent to construct and repair an environment in Docker. It exports a setup script, runtime handoff, dependency graph, and execution trace. Pytest collection is used inside the builder to check the constructed environment.

## Scope

This release provides the method implementation, a single-repository construction entry point, offline unit tests, and two pinned dataset manifests. Batch benchmark runners, metric aggregation, external evaluation adapters, historical experiment outputs, and test-count certification records are not included. It is not a complete reproduction package for the paper's reported scores.

## Install

Use Python 3.11 or newer. Construction requires Git, a running Docker daemon, network access, and an OpenAI-compatible model service. Unit tests do not need Docker or API credentials.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/validate_release.py
```

Run the following commands from this project's root. No custom `PYTHONPATH` is required. `requirements-validation.txt` records the dependency versions used for release testing; it is not a cross-platform lock file.

## Configure the API

Set the API variables in your shell using your own key and endpoint:

```bash
export OPENAI_API_KEY='your-own-key'
export OPENAI_API_BASE='https://api.openai.com/v1'
```

The single-repository script reads environment variables and does not load `.env` files. Existing `OPENROUTER_API_*` and `MINIMAX_API_*` variables take precedence over `OPENAI_API_*`; clear conflicting variables when switching providers. Pass a model identifier supported by your endpoint.

## Build one repository

Clone the target repository and check out its intended commit first. The dataset manifests provide repository names and commit pins; the builder operates on the checkout supplied to it and does not select or pin a dataset row automatically.

```bash
docker info
python scripts/run_react_e2e.py --help
python scripts/run_react_e2e.py /absolute/path/to/target-repository \
  --model YOUR_MODEL_ID --base-image auto \
  --out outputs/single/setup.sh \
  --trace-out outputs/single/react_trace.json \
  --usage-out outputs/single/token_usage.json \
  --runtime-out outputs/single/runtime_handoff.json \
  --max-turns 30 --context-mode legacy --command-timeout 600
```

The final command calls the model and starts Docker containers. Use a fresh output directory for each run. Temperature is 0 for agent decisions; the default budget is 30 ReAct turns, which is not a strict total-model-call limit. Graph analysis, retries and optional context compression can issue additional requests.

Exit code 0 indicates builder success, 1 indicates unsuccessful construction, and 2 can indicate missing API credentials or invalid arguments. Passing collection does not establish that test assertions pass or reproduce paper-level EBSR/ESSR scores.

## Outputs

The output directory contains `setup.sh`, `final_depgraph.json`, `runtime_handoff.json`, `react_trace.json`, `token_usage.json`, conversation history and execution logs when emitted. Inspect the trace and runtime handoff alongside the setup script. The standalone builder does not export the external evaluator's Dockerfile or benchmark summary.

## Code map

| Location | Purpose |
| --- | --- |
| `scripts/run_react_e2e.py` | Environment construction for one local checkout |
| `src/agent/` | ReAct decisions, candidate repair, context and trajectory handling |
| `src/python_deps/depgraph/` | Dependency discovery, resolution, scheduling and certification |
| `src/envstate/` | Runtime state, base-image selection and model-response handling |
| `src/sandbox.py`, `src/synthesizer.py` | Docker execution and setup-script synthesis |
| `src/ecosystems/` | Auxiliary ecosystem analysis |
| `datasets/` | RATBench-100 and EnvBench-100 manifests |
| `tests/` | Offline regression tests for retained code |

See [datasets](datasets/README.md), and [third-party materials](THIRD_PARTY.md).
