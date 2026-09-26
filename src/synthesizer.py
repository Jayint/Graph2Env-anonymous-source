import base64
import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DEFAULT_APT_RETRIES = 5
DEFAULT_APT_HTTP_TIMEOUT_SECONDS = 120
DEFAULT_APT_HTTPS_TIMEOUT_SECONDS = 120
DEFAULT_PIP_TIMEOUT_SECONDS = 300
DEFAULT_PIP_RETRIES = 5
DEFAULT_PIP_INSTALL_REPLAY_ATTEMPTS = 3
DEFAULT_PIP_INSTALL_RETRY_DELAY_SECONDS = 5
DEFAULT_APT_INSTALL_REPLAY_ATTEMPTS = 3
DEFAULT_APT_INSTALL_RETRY_DELAY_SECONDS = 5
RECIPE_REQUIRED_KEYS = {
    "build_commands",
    "post_test_patch_commands",
    "runtime_preparation_commands",
    "test_commands",
    "excluded_commands",
    "rationale",
    "confidence",
}


SETUP_LOG_SUMMARY_SYSTEM_PROMPT = """You summarize one environment-setup trajectory for downstream Dockerfile synthesis.

The input already removed planner instructions, repository trees, and other scaffolding.
It contains only the executed step history.

Your job is not to restate every raw step. Your job is to compress the trajectory while preserving the replay-critical information.

Allowed block shapes:

1. A grouped range for read-only exploration or repeated failed attempts:
Step A-B
Type: read_only_inspection | failed_attempts
Goal: ...
Attempts:
- Step A: <command> -> <result>
- Step B: <command> -> <result>
Outcome: ...

2. A single successful state-changing step that must remain replayable:
Step N
Type: successful_state_change
Thought: ...
Action: <keep as close as possible to the original executed command; preserve multiline commands when they matter>
Observation: ...

3. A single final verification step:
Step N
Type: final_verification
Thought: ...
Action: <exact verified command when possible>
Observation: ...

Rules:
1. Preserve chronological order. This is strict: successful state-changing commands must stay in the same relative order as the agent's setup trajectory.
2. You MAY merge consecutive steps only when they are:
   - read-only exploration (`cat`, `ls`, `find`, `grep`, etc.), or
   - repeated failed attempts toward the same goal.
3. Never merge a successful state-changing step into a range block.
4. Never merge a success together with earlier failures. If steps 1-4 failed and step 5 succeeded, summarize `Step 1-4` separately and keep `Step 5` as its own block.
5. Any successful step that changes persistent state MUST stay as its own single-step block with an `Action:` line kept as close as possible to the original command. This includes package installs, file edits, generated artifacts, service setup, and source/test rewrites.
6. Preserve exact or near-exact commands for:
   - successful installs and environment changes
   - successful file edits
   - generated artifacts that later tests depend on
   - final verification commands
   - the exact relative order of successful state-changing commands from the agent's setup trajectory
7. Compress `Observation` aggressively, but preserve:
   - installed/removed/upgraded packages and versions when shown
   - system packages, services, environment variables, and working-directory assumptions
   - file edits or generated artifacts
   - whether the command failed or succeeded
   - test commands, test counts, and the first real failure if any
8. For `failed_attempts` blocks, list the distinct attempted commands and the reason they failed in concise bullets.
9. Do not reintroduce planner prompts, repository structure, or generic narration.
10. Return plain text only. Do not return JSON. Do not use Markdown fences.
"""


RECIPE_SYNTHESIS_SYSTEM_PROMPT = """You are a Dockerfile synthesis module for an environment-setup agent.

You are given two primary artifacts from one setup run:
1. `setup_log_summary_text`: a compact chronological summary of the executed setup trajectory. It may contain grouped range blocks such as `Step 12-15` for repeated failed attempts or read-only exploration, and single-step blocks for successful state-changing actions.
2. `agent_run_summary`: the structured run summary for the same setup session.

Your job is to choose which shell commands must be replayed in a fresh Docker image and return them as a structured build recipe.
The host will convert `build_commands` into Dockerfile `RUN` instructions.
Return JSON only. Do not write Markdown.

Rules:
1. `build_commands` are the persistent shell commands that should become Dockerfile `RUN` steps.
2. Prefer exact commands that were actually executed. Single-step `successful_state_change` or `final_verification` blocks in `setup_log_summary_text` are stronger replay evidence than grouped range summaries.
3. Strictly follow the agent's setup trajectory order. Emit `build_commands` in the same relative order as the successful state-changing commands appeared in `setup_log_summary_text` and `agent_run_summary`. Do not sort, group, hoist, delay, or reorder commands by package manager, dependency type, perceived importance, or convenience. If command B ran after command A in the successful trajectory, keep B after A unless A is truly read-only or intentionally excluded.
4. Do not merge independent install/setup commands into one command. Separate `pip install`, `apt-get install`, `npm install`, file edits, generated stubs, and service setup commands must remain separate when they were executed separately. Package-manager transactions and setup side effects are not algebraically mergeable, even when commands appear to install related dependencies.
5. Do not change the package manager used by a verified command. For example, `apt-get install cmake` is not equivalent to `pip install cmake`; package names across apt, pip, npm, etc. are not interchangeable. If a successful chain used `apt-get install ... && pip install ...`, preserve that chain or split it without changing package managers.
6. If you rewrite or split a command, keep it semantically equivalent and explain why in `rationale`. Splitting a compound command is safer than merging previously separate commands.
7. Use `agent_run_summary.verification_bundle` or the verified command fields as the source of truth for `runtime_preparation_commands` and `test_commands`.
8. Do not assume a dependency is already installed unless the retained `build_commands`, the run summary, or the setup log clearly prove it. If the final tests need a tool such as `pytest`, keep the installation command unless another retained build command clearly provides it.
9. If a compound command partially succeeded before a later segment failed, you MAY keep the successful persistent prefix when the setup summary proves the prefix completed and the later failure happened in a different segment. Example: if `pip install pytest && python -c ...` installed `pytest` and only the later import failed, `pip install pytest` can still be a valid `build_command`. Keep that prefix as its own command; do not merge it into a later replacement command.
10. Treat grouped `failed_attempts` or `read_only_inspection` ranges as context, not as replay instructions. Do not synthesize Dockerfile commands from those ranges unless the same command is separately validated by a successful state-changing block or the run summary.
11. Exclude read-only diagnostics, version checks, local health checks, runtime-only daemon starts, final test commands, and output-truncation helpers such as `| tail` or `| head`.
12. Commands that edit repository files may stay in `build_commands` if those edits were part of the successful environment setup and must exist in the final image. Do not omit a successful file rewrite when the final verification only succeeded after that rewrite.
13. `post_test_patch_commands` exists only for compatibility with downstream code. Default to `[]` unless the evidence clearly shows a file rewrite must happen only after a later test patch is applied.
14. `runtime_preparation_commands` is only for ephemeral actions that must run again immediately before tests because image build does not persist them.
15. `excluded_commands` should list commands you intentionally left out, with short reasons.
16. When the evidence is ambiguous, choose the smallest command set that still explains why the final verified tests could run in a fresh image.

The host code trusts your recipe semantics. It will normalize shape only and will not remove questionable `build_commands`, so be precise.

Required JSON keys:
`build_commands`, `post_test_patch_commands`, `runtime_preparation_commands`, `test_commands`, `excluded_commands`, `rationale`, `confidence`.

`confidence` must be one of: "high", "medium", "low".
"""


SETUP_LOG_SUMMARY_USER_PROMPT = """Summarize this extracted setup trajectory for later Dockerfile synthesis.

Prefer grouped `Step A-B` blocks for consecutive failed attempts or read-only exploration.
Keep successful state-changing commands and final verification as their own single-step blocks with replayable `Action:` lines.

Extracted setup trajectory:
```text
{setup_log_trajectory_text}
```
"""


RECIPE_SYNTHESIS_USER_PROMPT = """Synthesize the final reproducible build recipe directly from the setup log summary and the agent run summary.

Treat `setup_log_summary_text` and `agent_run_summary` as the primary evidence.
Choose the commands that should enter the Dockerfile.

Input JSON:
```json
{recipe_input_json}
```
"""


@dataclass
class RecipeSynthesisResult:
    recipe: Dict[str, Any]
    usage: Dict[str, int]
    raw_content: str = ""
    error: Optional[str] = None
    source: str = "llm"


@dataclass
class SetupLogSummaryResult:
    summary_text: str
    usage: Dict[str, int]
    raw_content: str = ""
    error: Optional[str] = None
    source: str = "llm"


def _quote_shell_single(text):
    return "'" + text.replace("'", "'\"'\"'") + "'"


_PIP_REQUIREMENT_WITH_SHELL_OPERATOR = re.compile(
    r"(?<!['\"A-Za-z0-9_./-])"
    r"([A-Za-z][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?(?:>=|<=|>|<)"
    r"[A-Za-z0-9][A-Za-z0-9_.!*+:-]*)"
)


def quote_shell_sensitive_package_specs(command):
    """Quote pip requirement specs whose comparison operators are shell metacharacters."""
    if not command:
        return command

    lowered = command.lower()
    if "pip" not in lowered or "install" not in lowered:
        return command

    def replace_match(match):
        return _quote_shell_single(match.group(1))

    return _PIP_REQUIREMENT_WITH_SHELL_OPERATOR.sub(replace_match, command)


def resolve_apt_mirror_url(apt_mirror_url=None):
    mirror = apt_mirror_url or os.environ.get("GRAPH2ENV_APT_MIRROR_URL") or os.environ.get("APT_MIRROR_URL")
    if not mirror:
        return None
    return mirror.rstrip("/")


def build_dockerfile_apt_bootstrap_run_instructions(
    apt_mirror_url=None,
    apt_retries=DEFAULT_APT_RETRIES,
    apt_http_timeout_seconds=DEFAULT_APT_HTTP_TIMEOUT_SECONDS,
    apt_https_timeout_seconds=DEFAULT_APT_HTTPS_TIMEOUT_SECONDS,
):
    mirror = resolve_apt_mirror_url(apt_mirror_url)
    instructions = [
        (
            "RUN printf '%s\\n' "
            f"'Acquire::Retries \"{apt_retries}\";' "
            f"'Acquire::http::Timeout \"{apt_http_timeout_seconds}\";' "
            f"'Acquire::https::Timeout \"{apt_https_timeout_seconds}\";' "
            "'Acquire::http::Pipeline-Depth \"0\";' "
            "> /etc/apt/apt.conf.d/99graph2env-retries"
        )
    ]

    if mirror:
        quoted_mirror = _quote_shell_single(mirror)
        instructions.append(
            "RUN APT_MIRROR_URL="
            f"{quoted_mirror} && "
            "if [ -f /etc/apt/sources.list ]; then "
            "sed -i "
            "\"s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" "
            "/etc/apt/sources.list; "
            "fi && "
            "find /etc/apt/sources.list.d -maxdepth 1 -name '*.list' "
            "-exec sed -i "
            "\"s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" {} + "
            "2>/dev/null || true && "
            "if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then "
            "sed -i "
            "\"s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://archive.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g; "
            "s|https://security.ubuntu.com/ubuntu|${APT_MIRROR_URL}|g\" "
            "/etc/apt/sources.list.d/ubuntu.sources; "
            "fi && "
            "apt-get update"
        )

    return instructions


def build_dockerfile_pip_bootstrap_env_instructions(
    pip_timeout_seconds=DEFAULT_PIP_TIMEOUT_SECONDS,
    pip_retries=DEFAULT_PIP_RETRIES,
):
    timeout = int(pip_timeout_seconds or DEFAULT_PIP_TIMEOUT_SECONDS)
    retries = int(pip_retries or DEFAULT_PIP_RETRIES)
    return [
        "ENV PIP_DISABLE_PIP_VERSION_CHECK=1",
        f"ENV PIP_DEFAULT_TIMEOUT={timeout}",
        f"ENV PIP_RETRIES={retries}",
    ]


def _strip_leading_run_token(command):
    """Defensive (Fix 3): drop a stray leading ``RUN `` so a full instruction handed to
    a resilient wrapper can never embed a literal ``RUN`` inside the wrapped body."""
    stripped = (command or "").strip()
    if stripped[:4].upper() == "RUN " or stripped.upper() == "RUN":
        return stripped[4:].strip()
    return stripped


def build_resilient_pip_install_run_instruction(
    command,
    max_attempts=DEFAULT_PIP_INSTALL_REPLAY_ATTEMPTS,
    retry_delay_seconds=DEFAULT_PIP_INSTALL_RETRY_DELAY_SECONDS,
):
    if not command or not command.strip():
        raise ValueError("command must be a non-empty pip install invocation")

    attempts = max(1, int(max_attempts or DEFAULT_PIP_INSTALL_REPLAY_ATTEMPTS))
    delay = max(0, int(retry_delay_seconds or DEFAULT_PIP_INSTALL_RETRY_DELAY_SECONDS))
    quoted_command = _quote_shell_single(_strip_leading_run_token(command))

    return (
        "RUN GRAPH2ENV_PIP_ATTEMPT=1; "
        f"GRAPH2ENV_PIP_MAX_ATTEMPTS={attempts}; "
        "GRAPH2ENV_PIP_STATUS=1; "
        "while [ \"$GRAPH2ENV_PIP_ATTEMPT\" -le \"$GRAPH2ENV_PIP_MAX_ATTEMPTS\" ]; do "
        f"PIP_NO_CACHE_DIR=1 /bin/sh -lc {quoted_command} && GRAPH2ENV_PIP_STATUS=0 && break; "
        "GRAPH2ENV_PIP_STATUS=$?; "
        "(python -m pip cache purge >/dev/null 2>&1 || "
        "python3 -m pip cache purge >/dev/null 2>&1 || "
        "pip cache purge >/dev/null 2>&1 || true); "
        "if [ \"$GRAPH2ENV_PIP_ATTEMPT\" -eq \"$GRAPH2ENV_PIP_MAX_ATTEMPTS\" ]; then "
        "exit \"$GRAPH2ENV_PIP_STATUS\"; "
        "fi; "
        "GRAPH2ENV_PIP_ATTEMPT=$((GRAPH2ENV_PIP_ATTEMPT + 1)); "
        f"sleep {delay}; "
        "done; "
        "exit \"$GRAPH2ENV_PIP_STATUS\""
    )


def _looks_like_apt_install_replay_command(command):
    normalized = " ".join(str(command or "").split()).strip()
    if not normalized or "GRAPH2ENV_APT_ATTEMPT" in normalized:
        return False
    return bool(
        re.search(
            r"(?:^|&&|\|\||;|\()\s*(?:sudo\s+)?apt(?:-get)?\s+install\b",
            normalized,
        )
    )


def _normalize_apt_install_replay_command(command):
    normalized = " ".join(str(command or "").split()).strip()
    normalized = re.sub(r"(^|(?:&&|\|\||;|\()\s*)sudo\s+apt", r"\1apt", normalized)
    has_update = re.search(
        r"(?:^|&&|\|\||;|\()\s*apt(?:-get)?\s+update\b",
        normalized,
    )
    if has_update:
        return normalized
    return f"apt-get update && {normalized}"


def build_resilient_apt_install_run_instruction(
    command,
    max_attempts=DEFAULT_APT_INSTALL_REPLAY_ATTEMPTS,
    retry_delay_seconds=DEFAULT_APT_INSTALL_RETRY_DELAY_SECONDS,
):
    if not command or not command.strip():
        raise ValueError("command must be a non-empty apt install invocation")

    attempts = max(1, int(max_attempts or DEFAULT_APT_INSTALL_REPLAY_ATTEMPTS))
    delay = max(0, int(retry_delay_seconds or DEFAULT_APT_INSTALL_RETRY_DELAY_SECONDS))
    replay_command = _normalize_apt_install_replay_command(_strip_leading_run_token(command))
    quoted_command = _quote_shell_single(replay_command)

    return (
        "RUN GRAPH2ENV_APT_ATTEMPT=1; "
        f"GRAPH2ENV_APT_MAX_ATTEMPTS={attempts}; "
        "GRAPH2ENV_APT_STATUS=1; "
        "while [ \"$GRAPH2ENV_APT_ATTEMPT\" -le \"$GRAPH2ENV_APT_MAX_ATTEMPTS\" ]; do "
        "rm -rf /var/lib/apt/lists/*; "
        f"DEBIAN_FRONTEND=noninteractive /bin/sh -lc {quoted_command} "
        "&& GRAPH2ENV_APT_STATUS=0 && break; "
        "GRAPH2ENV_APT_STATUS=$?; "
        "(apt-get clean >/dev/null 2>&1 || true); "
        "rm -rf /var/lib/apt/lists/*; "
        "if [ \"$GRAPH2ENV_APT_ATTEMPT\" -eq \"$GRAPH2ENV_APT_MAX_ATTEMPTS\" ]; then "
        "exit \"$GRAPH2ENV_APT_STATUS\"; "
        "fi; "
        "GRAPH2ENV_APT_ATTEMPT=$((GRAPH2ENV_APT_ATTEMPT + 1)); "
        f"sleep {delay}; "
        "done; "
        "exit \"$GRAPH2ENV_APT_STATUS\""
    )


def is_generated_apt_bootstrap_run_instruction(instruction):
    if not instruction:
        return False
    normalized = " ".join(instruction.split())
    if "99graph2env-retries" in normalized:
        return True
    return "APT_MIRROR_URL=" in normalized and "archive.ubuntu.com/ubuntu" in normalized


class Synthesizer:
    # RepoLaunch partial-pass finalize threshold (Fix 3 §0.5). Tunable single constant:
    # a partial-pass run finalizes only if passed/(passed+failed+errors) >= this.
    # Raised 0.5 -> 0.8: a partial-pass (rc!=0) run is only accepted as a working env
    # when a STRONG majority (>=80%) of tests passed (tolerates <=20% failures).
    MIN_PASS_RATIO = 0.8

    TEST_COMMAND_PATTERNS = [
        # Python
        r"^pytest\b",
        r"^py\.test\b",
        r"^python3?\s+-m\s+pytest\b",
        r"^(?:poetry|pdm|uv)\s+run\s+pytest\b",
        r"^(?:poetry|pdm|uv)\s+run\s+python3?\s+-m\s+pytest\b",
        r"^python3?\s+-m\s+unittest\b",
        r"^tox\b",
        r"^nox\b",
        r"^nosetests\b",
        r"^nose\b",
        # JavaScript / TypeScript
        r"^(?:npm|yarn|pnpm)\s+test\b",
        r"^jest\b",
        r"^mocha\b",
        r"^karma\b",
        r"^vitest\b",
        r"^cypress\b",
        # Rust / Go / Java / Ruby / PHP
        r"^cargo\s+test\b",
        r"^go\s+test\b",
        r"^(?:mvn|\.?/mvnw)\b(?:\s+-\S+)*\s+test\b",
        r"^(?:gradle|\.?/gradlew)\b(?:\s+-\S+)*\s+test\b",
        r"^bundle\s+exec\s+rspec\b",
        r"^bundle\s+exec\s+rake\b",
        r"^rake\s+test\b",
        r"^rspec\b",
        r"^(?:\./)?(?:vendor/bin/)?phpunit\b",
        r"^(?:\./)?(?:vendor/bin/)?pest\b",
        # C / C++
        r"^ctest\b",
        r"^cmake\b.*\b--target\b\s*(?:test|tests)\b",
        r"^(?:make|gmake|mingw32-make|ninja)\b.*\b(?:test|tests|check|tdd)\b",
    ]
    SAFE_READONLY_COMMANDS = {
        "cd", "ls", "cat", "echo", "pwd", "whoami", "who", "date", "cal", "df", "du",
        "free", "uname", "uptime", "w", "ps", "pgrep", "top", "dmesg", "tail", "head",
        "grep", "find", "locate", "which", "file", "stat", "cmp", "diff", "xz", "unxz",
        "sort", "wc", "tr", "cut", "paste", "tee", "awk", "env", "printenv", "hostname",
        "xargs",
        "ping", "traceroute", "ssh",
    }
    VERSION_PROBE_COMMANDS = {
        "php", "composer",
        "python", "python3", "pip", "pip3",
        "node", "npm", "yarn", "pnpm",
        "java", "javac", "mvn", "gradle", "gradlew",
        "go", "rustc", "cargo",
        "ruby", "gem", "bundle",
        "gcc", "g++", "cc", "c++", "make", "cmake", "ninja",
        "git",
    }

    def __init__(self, base_image="python:3.10", workdir="/app"):
        self.base_image = base_image
        self.workdir = workdir
        self.instructions = []
        self.build_recipe = None

    def summarize_setup_log_for_recipe(self, client, model, setup_log_trajectory_text, log_dir=None):
        """Compress the extracted setup trajectory into a synthesis-oriented summary."""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        raw_content = ""
        messages = [
            {"role": "system", "content": SETUP_LOG_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SETUP_LOG_SUMMARY_USER_PROMPT.format(
                    setup_log_trajectory_text=setup_log_trajectory_text or ""
                ),
            },
        ]
        self._log_setup_log_summary("input", messages, log_dir=log_dir)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            raw_content = response.choices[0].message.content or ""
            summary_text = raw_content.strip()
            if not summary_text:
                raise ValueError("empty setup log summary response")
            self._log_setup_log_summary(
                "output",
                {
                    "content": raw_content,
                    "summary_text": summary_text,
                    "usage": usage,
                    "source": "llm",
                    "error": None,
                    "model": model,
                },
                log_dir=log_dir,
            )
            return SetupLogSummaryResult(
                summary_text=summary_text,
                usage=usage,
                raw_content=raw_content,
                source="llm",
            )
        except Exception as exc:
            self._log_setup_log_summary(
                "output",
                {
                    "content": raw_content,
                    "summary_text": "",
                    "usage": usage,
                    "source": "llm_error",
                    "error": str(exc),
                    "model": model,
                },
                log_dir=log_dir,
            )
            return SetupLogSummaryResult(
                summary_text="",
                usage=usage,
                raw_content=raw_content,
                error=str(exc),
                source="llm_error",
            )

    def synthesize_build_recipe(self, client, model, recipe_input, log_dir=None):
        """Ask the LLM to summarize the exploratory run into a final build recipe."""
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        raw_content = ""
        recipe_input_json = json.dumps(recipe_input or {}, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": RECIPE_SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RECIPE_SYNTHESIS_USER_PROMPT.format(
                    recipe_input_json=recipe_input_json
                ),
            },
        ]
        self._log_recipe_synthesis("input", messages, log_dir=log_dir)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            raw_content = response.choices[0].message.content or ""
            recipe = self.extract_build_recipe_json(raw_content)
            recipe = self.normalize_build_recipe(recipe, recipe_input=recipe_input)
            self.apply_build_recipe(recipe)
            self._log_recipe_synthesis(
                "output",
                {
                    "content": raw_content,
                    "recipe": recipe,
                    "usage": usage,
                    "source": "llm",
                    "error": None,
                    "model": model,
                },
                log_dir=log_dir,
            )
            return RecipeSynthesisResult(
                recipe=recipe,
                usage=usage,
                raw_content=raw_content,
                source="llm",
            )
        except Exception as exc:
            fallback_recipe = self.normalize_build_recipe({}, recipe_input=recipe_input)
            if fallback_recipe.get("build_commands") and fallback_recipe.get("test_commands"):
                fallback_recipe["rationale"] = (
                    "Deterministically replayed successful trajectory setup commands because "
                    f"LLM recipe synthesis failed: {exc}"
                )
                fallback_recipe["confidence"] = "low"
                self.apply_build_recipe(fallback_recipe)
                self._log_recipe_synthesis(
                    "output",
                    {
                        "content": raw_content,
                        "recipe": fallback_recipe,
                        "usage": usage,
                        "source": "deterministic_fallback_after_llm_error",
                        "error": str(exc),
                        "model": model,
                    },
                    log_dir=log_dir,
                )
                return RecipeSynthesisResult(
                    recipe=fallback_recipe,
                    usage=usage,
                    raw_content=raw_content,
                    source="deterministic_fallback_after_llm_error",
                )

            self._log_recipe_synthesis(
                "output",
                {
                    "content": raw_content,
                    "recipe": {},
                    "usage": usage,
                    "source": "llm_error",
                    "error": str(exc),
                    "model": model,
                },
                log_dir=log_dir,
            )
            return RecipeSynthesisResult(
                recipe={},
                usage=usage,
                raw_content=raw_content,
                error=str(exc),
                source="llm_error",
            )

    def _log_setup_log_summary(self, call_type, data, log_dir=None):
        if not log_dir:
            return

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "setup_log_summary.md")

        if call_type == "input":
            with open(log_file, "w", encoding="utf-8") as file_obj:
                file_obj.write("##### LLM INPUT (setup log summary) #####\n")
                file_obj.write("================================ Human Message =================================\n\n")
                for message in data:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    if role == "system":
                        file_obj.write(f"[{role.upper()}]\n{content}\n\n")
                    elif role == "user":
                        file_obj.write(f"{content}\n\n")
                    else:
                        file_obj.write(f"[{role.upper()}]\n{content}\n\n")
            return

        with open(log_file, "a", encoding="utf-8") as file_obj:
            file_obj.write("================================ AI Message =================================\n\n")
            file_obj.write(f"{data.get('content') or ''}\n\n")
            file_obj.write("================================ Parsed Summary =================================\n\n")
            file_obj.write(f"{data.get('summary_text') or ''}\n\n")
            file_obj.write("================================ Metadata =================================\n\n")
            file_obj.write(f"- Model: {data.get('model', '')}\n")
            file_obj.write(f"- Source: {data.get('source')}\n")
            file_obj.write(f"- Error: {data.get('error') or ''}\n")
            usage = data.get("usage") or {}
            file_obj.write(f"- Prompt Tokens: {usage.get('input_tokens', 0)}\n")
            file_obj.write(f"- Completion Tokens: {usage.get('output_tokens', 0)}\n")
            file_obj.write(f"- Total Tokens: {usage.get('total_tokens', 0)}\n")

    def _log_recipe_synthesis(self, call_type, data, log_dir=None):
        if not log_dir:
            return

        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "recipe_synthesis.md")

        if call_type == "input":
            with open(log_file, "w", encoding="utf-8") as file_obj:
                file_obj.write("##### LLM INPUT (build recipe synthesis) #####\n")
                file_obj.write("================================ Human Message =================================\n\n")
                for message in data:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    if role == "system":
                        file_obj.write(f"[{role.upper()}]\n{content}\n\n")
                    elif role == "user":
                        file_obj.write(f"{content}\n\n")
                    else:
                        file_obj.write(f"[{role.upper()}]\n{content}\n\n")
            return

        with open(log_file, "a", encoding="utf-8") as file_obj:
            file_obj.write("================================ AI Message =================================\n\n")
            file_obj.write(f"{data.get('content') or ''}\n\n")
            file_obj.write("================================ Parsed Build Recipe =================================\n\n")
            file_obj.write("```json\n")
            file_obj.write(json.dumps(data.get("recipe") or {}, ensure_ascii=False, indent=2))
            file_obj.write("\n```\n\n")
            file_obj.write("================================ Metadata =================================\n\n")
            file_obj.write(f"- Model: {data.get('model', '')}\n")
            file_obj.write(f"- Source: {data.get('source')}\n")
            file_obj.write(f"- Error: {data.get('error') or ''}\n")
            usage = data.get("usage") or {}
            file_obj.write(f"- Prompt Tokens: {usage.get('input_tokens', 0)}\n")
            file_obj.write(f"- Completion Tokens: {usage.get('output_tokens', 0)}\n")
            file_obj.write(f"- Total Tokens: {usage.get('total_tokens', 0)}\n")

    def extract_build_recipe_json(self, content):
        """Extract the final complete build recipe JSON object from an LLM response."""
        if not content:
            raise ValueError("empty recipe synthesis response")

        candidate = content.strip()
        content_without_thinking = self._strip_think_blocks(candidate)
        search_regions = []
        for region in (content_without_thinking, candidate):
            for match in re.finditer(r"```json\s*(.*?)\s*```", region, re.DOTALL | re.IGNORECASE):
                fenced = match.group(1).strip()
                if fenced:
                    search_regions.append(fenced)
            if region.strip():
                search_regions.append(region.strip())

        parsed_dicts = []
        for region in search_regions:
            for json_blob in self._extract_json_object_candidates(region):
                try:
                    parsed = json.loads(json_blob)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                parsed_dicts.append(parsed)

        for parsed in reversed(parsed_dicts):
            if RECIPE_REQUIRED_KEYS.issubset(parsed.keys()):
                return parsed

        if parsed_dicts:
            raise ValueError(
                "recipe synthesis response did not contain a complete build recipe JSON object"
            )
        raise ValueError("recipe synthesis response did not contain a JSON object")

    def _strip_think_blocks(self, content):
        """Remove model-private reasoning blocks before looking for public JSON."""
        return re.sub(r"<think\b[^>]*>.*?</think>", "", content or "", flags=re.DOTALL | re.IGNORECASE)

    def normalize_build_recipe(self, recipe, recipe_input=None):
        """Normalize recipe shape while preserving the LLM's semantic choices."""
        recipe = recipe or {}
        recipe_input = recipe_input or {}
        final_bundle = self._resolve_final_verification_bundle(recipe_input)

        final_runtime_commands = self._normalize_recipe_command_list(
            final_bundle.get("runtime_preparation_commands")
        )
        final_test_commands = self._normalize_recipe_command_list(
            final_bundle.get("test_commands")
        )

        recipe_runtime_commands = self._normalize_recipe_command_list(
            recipe.get("runtime_preparation_commands")
        )
        recipe_test_commands = self._normalize_recipe_command_list(recipe.get("test_commands"))
        if "runtime_preparation_commands" in recipe and recipe_runtime_commands:
            runtime_commands = recipe_runtime_commands
        elif (
            not recipe_runtime_commands
            and final_runtime_commands
            and recipe_input.get("verification_bundle")
        ):
            runtime_commands = final_runtime_commands
        elif "runtime_preparation_commands" in recipe:
            runtime_commands = recipe_runtime_commands
        else:
            runtime_commands = final_runtime_commands
        test_commands = (
            recipe_test_commands
            if "test_commands" in recipe and recipe_test_commands
            else final_test_commands
        )

        excluded_commands = self._normalize_excluded_commands(recipe.get("excluded_commands"))

        confidence = str(recipe.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        recipe_build_commands = self._normalize_recipe_command_list(recipe.get("build_commands"))
        post_test_patch_commands = self._normalize_recipe_command_list(
            recipe.get("post_test_patch_commands")
        )
        trajectory_build_commands = self._collect_trajectory_first_build_commands(recipe_input)
        using_trajectory_build_commands = bool(trajectory_build_commands)
        if using_trajectory_build_commands:
            build_commands = trajectory_build_commands
        else:
            build_commands = recipe_build_commands
            build_commands = self._canonicalize_build_commands_against_observed_actions(
                build_commands,
                recipe_input,
            )
            build_commands = self._supplement_build_commands_with_observed_file_rewrites(
                build_commands,
                recipe_input,
            )
            build_commands = self._supplement_build_commands_with_observed_exact_version_overrides(
                build_commands,
                recipe_input,
            )
            build_commands = self._supplement_build_commands_with_backend_bootstrap(
                build_commands,
                recipe_input,
            )
            build_commands = self._supplement_build_commands_with_tox_test_dependencies(
                build_commands,
                recipe_input,
                test_commands,
            )
        build_commands = self._strip_output_truncation_helpers_from_build_commands(build_commands)
        build_commands = self._sanitize_build_commands_for_replay(build_commands)
        build_commands = self._coalesce_postgres_build_configuration_commands(build_commands)
        build_commands = self._drop_readonly_build_commands(build_commands)
        build_commands = self._drop_failed_only_build_commands(build_commands, recipe_input)
        build_commands = self._drop_excluded_build_commands(
            build_commands,
            excluded_commands,
            recipe_input=recipe_input,
        )
        post_test_patch_commands = self._drop_excluded_build_commands(
            post_test_patch_commands,
            excluded_commands,
            recipe_input=recipe_input,
        )
        build_commands, post_test_patch_commands = self._move_patch_sensitive_commands(
            build_commands,
            post_test_patch_commands,
            recipe_input.get("test_patch", ""),
        )

        return {
            "build_commands": build_commands,
            "post_test_patch_commands": post_test_patch_commands,
            "runtime_preparation_commands": runtime_commands,
            "test_commands": test_commands,
            "excluded_commands": excluded_commands,
            "rationale": str(recipe.get("rationale") or "").strip(),
            "confidence": confidence,
        }

    def _collect_trajectory_first_build_commands(self, recipe_input):
        """Replay successful persistent setup actions in their original order.

        This is intentionally a negative-filter pass: a successful command is kept
        unless it is clearly read-only, test-only, runtime-only, or otherwise not
        representable as a persistent Dockerfile build step.
        """
        collected = []
        for record in self._collect_successful_action_records(recipe_input):
            command = str(record.get("command") or "").strip()
            if not command:
                continue
            if self.is_test_command(command):
                continue
            for recordable in self._extract_recordable_setup_command_units(command):
                cleaned = self._strip_output_truncation_suffix(recordable)
                if cleaned:
                    collected.append(cleaned)
        return self._dedupe_build_commands_preserve_order_sensitive(collected)

    def _extract_recordable_setup_command_units(self, command):
        """Split one successful setup action into ordered replay units when safe."""
        recordable = self._extract_recordable_setup_commands(command)
        if not recordable:
            return []
        if "<<" in recordable[0]:
            return recordable

        units = []
        pending_navigation = []
        segments = self._split_shell_chain(recordable[0])
        index = 0
        while index < len(segments):
            raw_segment, separator = segments[index]
            normalized = self._normalize_command_segment(raw_segment)
            if not normalized:
                index += 1
                continue

            if self._is_navigation_only_segment(normalized):
                pending_navigation.append((raw_segment.strip(), self._normalize_shell_separator(separator)))
                index += 1
                continue

            if self._is_apt_update_segment(normalized) and index + 1 < len(segments):
                next_raw, _ = segments[index + 1]
                next_normalized = self._normalize_command_segment(next_raw)
                if self._is_apt_install_segment(next_normalized):
                    units.append(
                        self._join_command_unit(
                            pending_navigation + [
                                (raw_segment.strip(), "&&"),
                                (next_raw.strip(), ""),
                            ]
                        )
                    )
                    pending_navigation = []
                    index += 2
                    continue

            unit_segments = pending_navigation + [(raw_segment.strip(), "")]
            units.append(self._join_command_unit(unit_segments))
            pending_navigation = []
            index += 1

        return [unit for unit in units if unit]

    def _normalize_shell_separator(self, separator):
        if separator == "\n":
            return "\n"
        return separator.strip() if separator else ""

    def _join_command_unit(self, segments):
        if not segments:
            return ""
        rebuilt = segments[0][0]
        for index in range(1, len(segments)):
            separator = segments[index - 1][1] or "&&"
            segment = segments[index][0]
            if separator == "\n":
                rebuilt = f"{rebuilt}\n{segment}"
            else:
                rebuilt = f"{rebuilt} {separator} {segment}"
        if "\n" not in rebuilt and not self._command_rewrites_files(rebuilt):
            rebuilt = re.sub(r"\s+", " ", rebuilt)
        return rebuilt.strip()

    def _is_apt_update_segment(self, normalized_command):
        return bool(re.match(r"^(?:sudo\s+)?apt(?:-get)?\s+update\b", normalized_command or ""))

    def _is_apt_install_segment(self, normalized_command):
        return bool(re.match(r"^(?:sudo\s+)?apt(?:-get)?\s+install\b", normalized_command or ""))

    def _canonicalize_build_commands_against_observed_actions(self, build_commands, recipe_input):
        observed_setup_commands = self._collect_observed_successful_setup_commands(
            recipe_input,
            include_test_command_prefixes=True,
            include_failed_test_command_prefixes=True,
        )
        if not observed_setup_commands:
            return build_commands

        canonicalized = []
        for command in build_commands:
            replacement = self._match_observed_setup_command(command, observed_setup_commands)
            canonicalized.append(replacement or command)
        return self._dedupe_preserve_order(canonicalized)

    def _supplement_build_commands_with_observed_file_rewrites(self, build_commands, recipe_input):
        """Keep verified repository rewrites even when the LLM omits one from the recipe."""
        observed_setup_commands = self._collect_observed_successful_setup_commands(recipe_input)
        observed_setup_commands.extend(
            self._collect_promotable_file_rewrite_prefixes_from_failed_test_actions(recipe_input)
        )
        observed_setup_commands.extend(
            self._collect_promotable_file_rewrite_prefixes_from_failed_setup_chains(recipe_input)
        )
        observed_setup_commands = self._dedupe_preserve_order(observed_setup_commands)
        if not observed_setup_commands:
            return build_commands

        missing_rewrites = []
        for index, observed in enumerate(observed_setup_commands):
            if not self._is_observed_repository_file_rewrite(observed):
                continue
            if self._command_already_included(observed, build_commands):
                continue
            missing_rewrites.append((index, observed))

        if not missing_rewrites:
            return build_commands

        supplemented = []
        added_rewrites = set()
        for command in build_commands:
            observed_index = self._find_observed_setup_command_index(
                command,
                observed_setup_commands,
            )
            if observed_index is not None:
                for rewrite_index, rewrite_command in missing_rewrites:
                    if rewrite_index < observed_index and rewrite_command not in added_rewrites:
                        supplemented.append(rewrite_command)
                        added_rewrites.add(rewrite_command)
            supplemented.append(command)

        for _, rewrite_command in missing_rewrites:
            if rewrite_command not in added_rewrites:
                supplemented.append(rewrite_command)
                added_rewrites.add(rewrite_command)
        return self._dedupe_preserve_order(supplemented)

    def _supplement_build_commands_with_observed_exact_version_overrides(self, build_commands, recipe_input):
        """Keep final successful exact-version pip overrides that the LLM omitted."""
        observed_overrides = self._collect_observed_exact_version_override_commands(recipe_input)
        if not observed_overrides:
            return build_commands

        missing_overrides = [
            command for command in observed_overrides if not self._command_already_included(command, build_commands)
        ]
        if not missing_overrides:
            return build_commands

        supplemented = []
        added = set()
        for command in build_commands:
            observed_index = self._find_observed_exact_version_override_index(command, observed_overrides)
            if observed_index is not None:
                for index, override_command in enumerate(observed_overrides):
                    if index < observed_index and override_command in missing_overrides and override_command not in added:
                        supplemented.append(override_command)
                        added.add(override_command)
            supplemented.append(command)

        for override_command in observed_overrides:
            if override_command in missing_overrides and override_command not in added:
                supplemented.append(override_command)
                added.add(override_command)

        return self._dedupe_preserve_order(supplemented)

    def _supplement_build_commands_with_backend_bootstrap(self, build_commands, recipe_input):
        """Keep observed build-backend bootstrap installs before no-build-isolation installs."""
        if not build_commands:
            return build_commands

        if any(self._command_installs_backend_bootstrap(command) for command in build_commands):
            return build_commands

        bootstrap_command = self._find_observed_backend_bootstrap_command(recipe_input)
        if not bootstrap_command:
            return build_commands

        first_sensitive_index = None
        for index, command in enumerate(build_commands):
            if self._command_requires_backend_bootstrap(command):
                first_sensitive_index = index
                break

        if first_sensitive_index is None:
            return build_commands

        supplemented = []
        for index, command in enumerate(build_commands):
            if index == first_sensitive_index:
                supplemented.append(bootstrap_command)
            supplemented.append(command)
        return self._dedupe_preserve_order(supplemented)

    def _supplement_build_commands_with_tox_test_dependencies(
        self,
        build_commands,
        recipe_input,
        test_commands,
    ):
        """Retain tox-declared pytest dependencies when the recipe omits them."""
        normalized_test_commands = [self._normalize_command_segment(command) for command in test_commands or []]
        if not any("pytest" in command for command in normalized_test_commands):
            return build_commands

        tox_dependencies = self._extract_observed_tox_test_dependencies(recipe_input)
        if not tox_dependencies:
            return build_commands

        missing_dependencies = [
            dependency
            for dependency in tox_dependencies
            if not any(self._command_installs_python_dependency(command, dependency) for command in build_commands)
        ]
        if not missing_dependencies:
            return build_commands

        return self._dedupe_preserve_order(
            build_commands + [f"pip install {' '.join(missing_dependencies)}"]
        )

    def _find_observed_successful_install_command(
        self,
        recipe_input,
        package_name,
        required_flags=None,
        required_substrings=None,
    ):
        required_flags = required_flags or []
        required_substrings = required_substrings or []
        for record in self._collect_successful_action_records(recipe_input):
            command = str(record.get("command") or "").strip()
            if not command:
                continue
            for component in self._split_pipeline(command):
                cleaned = self._extract_clean_pip_install_component(component)
                if not cleaned:
                    continue
                if not self._command_installs_python_dependency(cleaned, package_name):
                    continue
                normalized_cleaned = self._normalize_command_segment(cleaned)
                if any(flag not in normalized_cleaned for flag in required_flags):
                    continue
                if any(fragment not in normalized_cleaned for fragment in required_substrings):
                    continue
                return cleaned
        return None

    def _find_observed_successful_command(self, recipe_input, predicate):
        matches = self._collect_observed_successful_commands(recipe_input, predicate)
        return matches[0] if matches else None

    def _collect_observed_successful_commands(self, recipe_input, predicate):
        matches = []
        for record in self._collect_successful_action_records(recipe_input):
            command = str(record.get("command") or "").strip()
            candidate_commands = self._extract_recordable_setup_commands(command) or [command]
            for candidate in candidate_commands:
                cleaned_candidate = self._strip_output_truncation_suffix(candidate)
                normalized = self._normalize_command_segment(cleaned_candidate)
                if not normalized:
                    continue
                if predicate(normalized, cleaned_candidate):
                    matches.append(cleaned_candidate)
        return self._dedupe_preserve_order(matches)

    def _strip_output_truncation_suffix(self, command):
        cleaned = (command or "").strip()
        while True:
            updated = re.sub(r"\s*\|\s*(?:tail|head)\b(?:[^|;&]*)$", "", cleaned).strip()
            if updated == cleaned:
                return self._strip_trailing_non_state_redirections(cleaned)
            cleaned = updated

    def _strip_trailing_non_state_redirections(self, command):
        cleaned = (command or "").strip()
        while True:
            updated = re.sub(r"\s+\d?>&\d\s*$", "", cleaned).strip()
            updated = re.sub(r"\s+(?:[12]?>|&>)\s*/dev/null\s*$", "", updated).strip()
            if updated == cleaned:
                return cleaned
            cleaned = updated

    def _strip_output_truncation_helpers_from_build_commands(self, build_commands):
        cleaned_commands = []
        for command in build_commands or []:
            cleaned = self._strip_output_truncation_suffix(command)
            cleaned_commands.append(cleaned or command)
        return self._dedupe_build_commands_preserve_order_sensitive(cleaned_commands)

    def _is_observed_repository_file_rewrite(self, command):
        if not self._command_rewrites_files(command):
            return False
        if self.is_test_command(command):
            return False

        paths = self._extract_command_file_paths(command)
        if not paths:
            return False

        relevant_paths = self._select_paths_for_repository_matching(paths)
        return any(self._path_is_repository_file(path) for path in relevant_paths)

    def _select_paths_for_repository_matching(self, paths):
        relevant_paths = [path for path in (paths or []) if path]
        explicit_paths = []
        for path in relevant_paths:
            normalized = self._normalize_patch_path(path)
            if "/" in normalized or path.startswith("/"):
                explicit_paths.append(path)
        return explicit_paths or relevant_paths

    def _path_is_repository_file(self, path):
        normalized = self._normalize_patch_path(path)
        if not normalized:
            return False
        if "/" not in normalized and not normalized.startswith("/"):
            return True
        if "/" in normalized and not normalized.startswith("/"):
            if self._path_looks_like_sed_substitution_expression(normalized):
                return False
            if normalized.startswith(
                (
                    "dev/",
                    "etc/",
                    "opt/",
                    "proc/",
                    "root/",
                    "sys/",
                    "tmp/",
                    "usr/",
                    "var/",
                    "http:/",
                    "https:/",
                )
            ):
                return False
            return True
        return (
            normalized.startswith("/app/")
            or normalized.startswith("/testbed/")
            or normalized.startswith(
                (
                    "src/",
                    "test/",
                    "tests/",
                    "app/",
                    "apps/",
                    "lib/",
                    "libs/",
                    "package/",
                    "packages/",
                    "config/",
                    "configs/",
                    "scripts/",
                )
            )
        )

    def _path_looks_like_sed_substitution_expression(self, path):
        return bool(re.match(r"^s[/#|].+[/#|].*", path or ""))

    def _command_already_included(self, command, included_commands):
        normalized_command = self._normalize_command_for_recipe_comparison(command)
        if not normalized_command:
            return False

        for included in included_commands:
            normalized_included = self._normalize_command_for_recipe_comparison(included)
            if not normalized_included:
                continue
            if normalized_command == normalized_included:
                return True
            if normalized_command in normalized_included:
                return True
        return False

    def _find_observed_setup_command_index(self, command, observed_setup_commands):
        normalized_command = self._normalize_command_for_recipe_comparison(command)
        if not normalized_command:
            return None

        for index, observed in enumerate(observed_setup_commands):
            normalized_observed = self._normalize_command_for_recipe_comparison(observed)
            if not normalized_observed:
                continue
            if normalized_command == normalized_observed:
                return index
            if normalized_command in normalized_observed or normalized_observed in normalized_command:
                return index
        return None

    def _normalize_command_for_recipe_comparison(self, command):
        command = self._strip_run_prefix((command or "").strip())
        if "\n" in command:
            return "\n".join(line.rstrip() for line in command.splitlines()).strip()
        return re.sub(r"\s+", " ", command).strip()

    def _collect_observed_successful_setup_commands(
        self,
        recipe_input,
        include_test_command_prefixes=False,
        include_failed_test_command_prefixes=False,
    ):
        observed = []
        for record in self._collect_successful_action_records(recipe_input):
            command = str(record.get("command") or "").strip()
            if not command:
                continue
            if not include_test_command_prefixes and self.is_test_command(command):
                continue
            observed.extend(self._extract_recordable_setup_commands(command))
        if include_failed_test_command_prefixes:
            observed.extend(
                self._collect_recordable_prefixes_from_failed_test_actions(recipe_input)
            )
        return self._dedupe_preserve_order(observed)

    def _sanitize_build_commands_for_replay(self, build_commands):
        sanitized = []
        for command in build_commands or []:
            stripped = self._strip_run_prefix(str(command or "").strip())
            if not stripped:
                continue
            if self.is_test_command(stripped):
                sanitized.extend(self._extract_recordable_setup_commands(stripped))
                continue
            stripped = self._normalize_echo_e_file_write_command(stripped)
            sanitized.append(stripped)
        return self._dedupe_build_commands_preserve_order_sensitive(sanitized)

    def _normalize_echo_e_file_write_command(self, command):
        match = re.match(
            r"^echo\s+-e\s+(?P<quote>['\"])(?P<body>.*)(?P=quote)\s*>\s*(?P<path>\S+)$",
            command or "",
            flags=re.DOTALL,
        )
        if not match:
            return command
        body = match.group("body")
        if not body.endswith("\\n"):
            body = f"{body}\\n"
        return f"printf '%b' {self._shell_single_quote(body)} > {match.group('path')}"

    def _coalesce_postgres_build_configuration_commands(self, build_commands):
        """Keep PostgreSQL startup and build-time psql mutations in one RUN layer."""
        commands = [str(command or "").strip() for command in build_commands or [] if str(command or "").strip()]
        coalesced = []
        index = 0
        while index < len(commands):
            command = commands[index]
            if not self._is_postgres_cluster_start_command(command):
                coalesced.append(command)
                index += 1
                continue

            group = [command]
            next_index = index + 1
            while next_index < len(commands) and self._is_postgres_build_config_command(
                commands[next_index]
            ):
                group.append(commands[next_index])
                next_index += 1

            if len(group) > 1:
                coalesced.append(" && ".join(group))
                index = next_index
                continue

            coalesced.append(command)
            index += 1

        return self._dedupe_build_commands_preserve_order_sensitive(coalesced)

    def _is_postgres_cluster_start_command(self, command):
        normalized = self._normalize_command_segment(command)
        return bool(
            re.search(r"\bpg_ctlcluster\b.*\bstart\b", normalized)
            or re.search(r"\bservice\s+postgresql\s+start\b", normalized)
        )

    def _is_postgres_build_config_command(self, command):
        normalized = self._normalize_command_segment(command)
        if re.search(r"\bpsql\b", normalized):
            return True
        if re.match(r"^(?:su\s+-\s+postgres\s+-c\s+)?(?:createdb|createuser|dropdb|dropuser)\b", normalized):
            return True
        if "postgres" in normalized and "/etc/hosts" in normalized and self._has_output_redirection(command):
            return True
        return False

    def _drop_readonly_build_commands(self, build_commands):
        filtered = []
        for command in build_commands or []:
            stripped = self._strip_run_prefix(str(command or "").strip())
            if not stripped:
                continue
            if self._is_readonly_command(stripped):
                continue
            filtered.append(stripped)
        return self._dedupe_build_commands_preserve_order_sensitive(filtered)

    def _drop_failed_only_build_commands(self, build_commands, recipe_input):
        reliable_success_commands = self._collect_observed_successful_setup_commands(
            recipe_input,
            include_failed_test_command_prefixes=True,
        )
        reliable_success_commands.extend(
            self._collect_promotable_file_rewrite_prefixes_from_failed_setup_chains(recipe_input)
        )
        reliable_success_commands = self._dedupe_preserve_order(reliable_success_commands)
        failed_commands = self._collect_failed_setup_commands(recipe_input)
        if not failed_commands:
            return build_commands

        filtered = []
        for command in build_commands or []:
            if self._command_has_reliable_success_evidence(command, reliable_success_commands):
                filtered.append(command)
                continue
            if self._command_has_failed_only_evidence(command, failed_commands):
                continue
            filtered.append(command)
        return self._dedupe_build_commands_preserve_order_sensitive(filtered)

    def _drop_excluded_build_commands(self, build_commands, excluded_commands, recipe_input=None):
        if not excluded_commands:
            return build_commands

        excluded_step_ranges = self._collect_excluded_step_ranges(excluded_commands)
        excluded_clone_directories = self._collect_excluded_clone_directories(excluded_commands)
        excluded_pip_install_names = self._collect_excluded_pip_install_package_names(excluded_commands)
        observed_records = self._collect_observed_successful_setup_command_records(recipe_input)
        observed_cursor = 0
        filtered = []
        for command in build_commands or []:
            matched_record_index = None
            matched_step_index = None
            if excluded_step_ranges and observed_records:
                matched_record_index, matched_step_index = self._find_next_observed_setup_command_record(
                    command,
                    observed_records,
                    start_index=observed_cursor,
                )
            if any(
                self._excluded_command_matches_build_command(command, excluded)
                for excluded in excluded_commands
            ):
                if matched_record_index is not None:
                    observed_cursor = matched_record_index + 1
                continue
            if self._command_installs_from_excluded_clone_directory(
                command,
                excluded_clone_directories,
            ):
                if matched_record_index is not None:
                    observed_cursor = matched_record_index + 1
                continue
            if matched_step_index is not None and self._step_index_in_ranges(
                matched_step_index,
                excluded_step_ranges,
            ):
                observed_cursor = matched_record_index + 1
                continue
            if matched_record_index is not None:
                observed_cursor = matched_record_index + 1
            filtered.append(command)
        filtered = self._drop_orphaned_uninstalls_for_excluded_installs(
            filtered,
            excluded_pip_install_names,
        )
        return self._dedupe_build_commands_preserve_order_sensitive(filtered)

    def _collect_excluded_clone_directories(self, excluded_commands):
        directories = set()
        for excluded in excluded_commands or []:
            if isinstance(excluded, dict):
                text = str(excluded.get("command") or "")
            else:
                text = str(excluded or "")
            text = self._strip_parenthesized_exclusion_note(text)
            for raw_segment, _ in self._split_shell_chain(text):
                directories.update(self._git_clone_target_directories(raw_segment))
        return directories

    def _git_clone_target_directories(self, command):
        try:
            tokens = shlex.split(str(command or ""))
        except ValueError:
            return set()
        if len(tokens) < 3 or tokens[0] != "git" or tokens[1] != "clone":
            return set()

        positional = []
        skip_next = False
        options_with_value = {
            "-b",
            "--branch",
            "-c",
            "--config",
            "--depth",
            "--filter",
            "-j",
            "--jobs",
            "-o",
            "--origin",
            "--reference",
            "--reference-if-able",
            "--separate-git-dir",
            "--template",
        }
        for token in tokens[2:]:
            if skip_next:
                skip_next = False
                continue
            if token == "--":
                continue
            if token.startswith("-"):
                if token in options_with_value:
                    skip_next = True
                continue
            positional.append(token)

        if not positional:
            return set()
        clone_target = positional[1] if len(positional) >= 2 else positional[0]
        directory = self._clone_directory_name_from_target(clone_target)
        return {directory} if directory else set()

    def _clone_directory_name_from_target(self, target):
        target = str(target or "").strip().strip("'\"")
        if not target:
            return ""
        target = target.rstrip("/")
        if ":" in target and "/" not in target.rsplit(":", 1)[-1]:
            target = target.rsplit(":", 1)[-1]
        directory = os.path.basename(target)
        if directory.endswith(".git"):
            directory = directory[:-4]
        return self._normalize_clone_directory_name(directory)

    def _normalize_clone_directory_name(self, directory):
        directory = str(directory or "").strip().strip("'\"").rstrip("/")
        if not directory:
            return ""
        return os.path.basename(directory).lower()

    def _command_installs_from_excluded_clone_directory(self, command, excluded_clone_directories):
        if not excluded_clone_directories:
            return False
        cwd = ""
        for raw_segment, _ in self._split_shell_chain(str(command or "")):
            segment = raw_segment.strip()
            if not segment:
                continue
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = []
            if tokens and tokens[0] == "cd" and len(tokens) >= 2:
                cwd = tokens[1]
                continue
            for target in self._pip_install_targets_from_segment(segment):
                if target in {".", "./"}:
                    if self._path_matches_excluded_clone_directory(cwd, excluded_clone_directories):
                        return True
                    continue
                if self._path_matches_excluded_clone_directory(target, excluded_clone_directories):
                    return True
        return False

    def _pip_install_targets_from_segment(self, command):
        try:
            tokens = shlex.split(str(command or ""))
        except ValueError:
            return []
        install_index = self._pip_install_token_index(tokens)
        if install_index is None:
            return []

        targets = []
        options_with_value = {
            "-c",
            "--constraint",
            "-r",
            "--requirement",
            "-i",
            "--index-url",
            "--extra-index-url",
            "-f",
            "--find-links",
            "--trusted-host",
            "--python-version",
            "--platform",
            "--abi",
            "--implementation",
            "--prefix",
            "--root",
            "--target",
        }
        index = install_index + 1
        while index < len(tokens):
            token = tokens[index]
            if token in {"-e", "--editable"} and index + 1 < len(tokens):
                targets.append(self._strip_requirement_extras(tokens[index + 1]))
                index += 2
                continue
            if token.startswith("-e") and len(token) > 2:
                targets.append(self._strip_requirement_extras(token[2:]))
                index += 1
                continue
            if token.startswith("--editable="):
                targets.append(self._strip_requirement_extras(token.split("=", 1)[1]))
                index += 1
                continue
            if token in options_with_value:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            targets.append(self._strip_requirement_extras(token))
            index += 1
        return targets

    def _pip_install_token_index(self, tokens):
        if len(tokens) >= 2 and tokens[0] in {"pip", "pip2", "pip3"} and tokens[1] == "install":
            return 1
        if (
            len(tokens) >= 4
            and tokens[0] in {"python", "python2", "python3"}
            and tokens[1] == "-m"
            and tokens[2] == "pip"
            and tokens[3] == "install"
        ):
            return 3
        return None

    def _strip_requirement_extras(self, target):
        return re.sub(r"\[[^\]]*\]$", "", str(target or "").strip())

    def _path_matches_excluded_clone_directory(self, path, excluded_clone_directories):
        directory = self._normalize_clone_directory_name(path)
        return bool(directory and directory in excluded_clone_directories)

    def _collect_excluded_pip_install_package_names(self, excluded_commands):
        names = set()
        for excluded in excluded_commands or []:
            if isinstance(excluded, dict):
                text = str(excluded.get("command") or "")
            else:
                text = str(excluded or "")
            names.update(self._extract_pip_install_package_names(text))
        return names

    def _drop_orphaned_uninstalls_for_excluded_installs(self, build_commands, excluded_pip_install_names):
        if not excluded_pip_install_names:
            return build_commands

        filtered = []
        commands = list(build_commands or [])
        for index, command in enumerate(commands):
            uninstall_names = self._extract_pip_uninstall_package_names(command)
            matched_uninstall_names = uninstall_names & excluded_pip_install_names
            if matched_uninstall_names and not self._later_command_installs_any_package(
                commands[index + 1 :],
                matched_uninstall_names,
            ):
                continue
            filtered.append(command)
        return filtered

    def _later_command_installs_any_package(self, commands, package_names):
        for command in commands or []:
            if self._extract_pip_install_package_names(command) & set(package_names or []):
                return True
        return False

    def _extract_pip_uninstall_package_names(self, command):
        names = set()
        for raw_segment, _ in self._split_shell_chain(str(command or "")):
            try:
                tokens = shlex.split(raw_segment)
            except ValueError:
                continue
            uninstall_index = self._pip_uninstall_token_index(tokens)
            if uninstall_index is None:
                continue
            index = uninstall_index + 1
            while index < len(tokens):
                token = tokens[index]
                if token in {"-y", "--yes"}:
                    index += 1
                    continue
                if token.startswith("-"):
                    index += 1
                    continue
                package_name = self._extract_package_name_from_requirement_token(token)
                if package_name:
                    names.add(package_name)
                index += 1
        return names

    def _pip_uninstall_token_index(self, tokens):
        if len(tokens) >= 2 and tokens[0] in {"pip", "pip2", "pip3"} and tokens[1] == "uninstall":
            return 1
        if (
            len(tokens) >= 4
            and tokens[0] in {"python", "python2", "python3"}
            and tokens[1] == "-m"
            and tokens[2] == "pip"
            and tokens[3] == "uninstall"
        ):
            return 3
        return None

    def _collect_excluded_step_ranges(self, excluded_commands):
        ranges = []
        for excluded in excluded_commands or []:
            if isinstance(excluded, dict):
                text = f"{excluded.get('command') or ''} {excluded.get('reason') or ''}"
            else:
                text = str(excluded or "")
            for match in re.finditer(r"\bSteps?\s+(\d+)(?:\s*[-–—]\s*(\d+))?", text, flags=re.IGNORECASE):
                start = int(match.group(1))
                end = int(match.group(2) or start)
                if end < start:
                    start, end = end, start
                ranges.append((start, end))
        return ranges

    def _collect_observed_successful_setup_command_records(self, recipe_input):
        records = []
        for record in self._collect_successful_action_records(recipe_input):
            step_index = record.get("step_index")
            try:
                step_index = int(step_index)
            except (TypeError, ValueError):
                step_index = None
            for command in self._extract_recordable_setup_commands(str(record.get("command") or "")):
                records.append({"step_index": step_index, "command": command})
        return records

    def _find_next_observed_setup_command_record(self, command, observed_records, start_index=0):
        normalized_command = self._normalize_command_for_recipe_comparison(command)
        if not normalized_command:
            return None, None
        for index in range(max(0, int(start_index or 0)), len(observed_records or [])):
            record = observed_records[index]
            normalized_observed = self._normalize_command_for_recipe_comparison(record.get("command"))
            if not normalized_observed:
                continue
            if normalized_command == normalized_observed:
                return index, record.get("step_index")
            if normalized_command in normalized_observed or normalized_observed in normalized_command:
                return index, record.get("step_index")
        return None, None

    def _step_index_in_ranges(self, step_index, ranges):
        try:
            step_index = int(step_index)
        except (TypeError, ValueError):
            return False
        return any(start <= step_index <= end for start, end in ranges or [])

    def _excluded_command_matches_build_command(self, command, excluded):
        if isinstance(excluded, dict):
            excluded_command = excluded.get("command")
        else:
            excluded_command = excluded

        normalized_command = self._normalize_command_for_recipe_comparison(command)
        normalized_excluded = self._normalize_command_for_recipe_comparison(excluded_command)
        if not normalized_command or not normalized_excluded:
            return False

        if normalized_command == normalized_excluded:
            return True
        normalized_excluded_base = self._strip_parenthesized_exclusion_note(normalized_excluded)
        if normalized_excluded_base != normalized_excluded and normalized_command == normalized_excluded_base:
            return True
        normalized_command_relaxed = self._normalize_command_for_exclusion_comparison(command)
        normalized_excluded_relaxed = self._normalize_command_for_exclusion_comparison(
            excluded_command
        )
        if (
            normalized_command_relaxed
            and normalized_excluded_relaxed
            and normalized_command_relaxed == normalized_excluded_relaxed
        ):
            return True
        if self._command_matches_excluded_command_segments(command, normalized_excluded):
            return True
        return False

    def _strip_parenthesized_exclusion_note(self, command):
        return re.sub(r"\s+\([^)]*\)\s*$", "", command or "").strip()

    def _command_matches_excluded_command_segments(self, command, normalized_excluded):
        normalized_excluded_base = self._strip_parenthesized_exclusion_note(normalized_excluded)
        for relaxed in (False, True):
            excluded_segments = self._normalized_shell_chain_segments_for_exclusion(
                normalized_excluded_base,
                relaxed=relaxed,
            )
            if len(excluded_segments) <= 1:
                continue
            command_segments = self._normalized_shell_chain_segments_for_exclusion(
                command,
                relaxed=relaxed,
            )
            if not command_segments or not excluded_segments:
                continue
            if len(command_segments) > len(excluded_segments):
                continue
            for start_index in range(len(excluded_segments) - len(command_segments) + 1):
                if excluded_segments[start_index : start_index + len(command_segments)] == command_segments:
                    return True
        return False

    def _normalized_shell_chain_segments_for_exclusion(self, command, relaxed=False):
        stripped = self._strip_run_prefix(str(command or "").strip())
        if not stripped:
            return []
        segments = []
        for raw_segment, _ in self._split_shell_chain(stripped):
            segment = self._strip_output_truncation_suffix(raw_segment)
            if relaxed:
                normalized = self._normalize_command_for_exclusion_comparison(segment)
            else:
                normalized = self._normalize_command_for_recipe_comparison(segment)
            if normalized:
                segments.append(normalized)
        return segments

    def _normalize_command_for_exclusion_comparison(self, command):
        command = self._strip_output_truncation_suffix(self._strip_run_prefix((command or "").strip()))
        if not command or "\n" in command:
            return self._normalize_command_for_recipe_comparison(command)
        try:
            tokens = shlex.split(command)
        except ValueError:
            return self._normalize_command_for_recipe_comparison(command)

        filtered_tokens = [
            token
            for token in tokens
            if token not in {"--no-cache-dir", "--no-cache"}
        ]
        return " ".join(filtered_tokens).strip()

    def _command_matches_any_recipe_evidence(self, command, evidence_commands):
        normalized_command = self._normalize_command_for_recipe_comparison(command)
        if not normalized_command:
            return False
        for evidence in evidence_commands or []:
            normalized_evidence = self._normalize_command_for_recipe_comparison(evidence)
            if not normalized_evidence:
                continue
            if normalized_command == normalized_evidence:
                return True
            if normalized_command in normalized_evidence or normalized_evidence in normalized_command:
                return True
        return False

    def _command_has_reliable_success_evidence(self, command, evidence_commands):
        return self._pip_install_command_matches_evidence(
            command,
            evidence_commands,
            allow_candidate_superset=True,
        ) or self._command_matches_any_recipe_evidence(command, evidence_commands)

    def _command_has_failed_only_evidence(self, command, evidence_commands):
        return self._pip_install_command_matches_evidence(
            command,
            evidence_commands,
            allow_candidate_superset=False,
        ) or self._command_matches_any_recipe_evidence(command, evidence_commands)

    def _pip_install_command_matches_evidence(
        self,
        command,
        evidence_commands,
        allow_candidate_superset,
    ):
        command_packages = self._extract_pip_install_package_names(command)
        if not command_packages:
            return False

        for evidence in evidence_commands or []:
            evidence_packages = self._extract_pip_install_package_names(evidence)
            if not evidence_packages:
                continue
            if allow_candidate_superset:
                if evidence_packages.issubset(command_packages):
                    return True
            elif command_packages.issubset(evidence_packages):
                return True
        return False

    def _extract_pip_install_package_names(self, command):
        try:
            tokens = shlex.split(self._strip_output_truncation_suffix(command), posix=True)
        except ValueError:
            return set()

        install_start = self._find_pip_install_start_index(tokens)
        if install_start is None:
            return set()

        packages = set()
        skip_next = False
        option_args = {
            "-r",
            "--requirement",
            "-c",
            "--constraint",
            "-i",
            "--index-url",
            "--extra-index-url",
            "-f",
            "--find-links",
            "--trusted-host",
        }
        start = install_start + 2
        if (
            tokens[install_start] in {"python", "python2", "python3"}
            and install_start + 3 < len(tokens)
            and tokens[install_start + 1 : install_start + 4] == ["-m", "pip", "install"]
        ):
            start = install_start + 4

        for token in tokens[start:]:
            if skip_next:
                skip_next = False
                continue
            if token in {"pip", "pip2", "pip3", "python", "python2", "python3", "-m", "install"}:
                continue
            if token in option_args:
                skip_next = True
                continue
            if token.startswith("--") and "=" in token:
                continue
            if token.startswith("-"):
                continue
            if self._token_looks_like_shell_redirection(token):
                continue

            package_name = self._extract_package_name_from_requirement_token(token)
            if package_name:
                packages.add(package_name)
        return packages

    def _extract_package_name_from_requirement_token(self, token):
        token = (token or "").strip()
        if not token:
            return ""
        if "://" in token or token.startswith(("git+", "http:", "https:", "file:")):
            egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", token)
            return egg_match.group(1).lower().replace("_", "-") if egg_match else ""
        if token.startswith((".", "/", "~")):
            return ""

        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?(?:[=<>!~].*)?$", token)
        if not match:
            return ""
        return match.group(1).lower().replace("_", "-")

    def _collect_failed_setup_commands(self, recipe_input):
        recipe_input = recipe_input or {}
        collected = []
        candidate_lists = [
            recipe_input.get("failed_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("failed_actions"),
        ]
        for records in candidate_lists:
            if not records:
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                command = str(record.get("command") or "").strip()
                if not command:
                    continue
                candidate_commands = self._extract_recordable_setup_commands(command)
                if not candidate_commands and self.is_test_command(command):
                    continue
                for candidate in candidate_commands or [command]:
                    cleaned_candidate = self._strip_output_truncation_suffix(candidate)
                    if cleaned_candidate:
                        collected.append(cleaned_candidate)
            if collected:
                break
        return self._dedupe_preserve_order(collected)

    def _collect_recordable_prefixes_from_failed_test_actions(self, recipe_input):
        recipe_input = recipe_input or {}
        collected = []
        candidate_lists = [
            recipe_input.get("failed_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("failed_actions"),
            recipe_input.get("successful_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("successful_actions"),
        ]
        for records in candidate_lists:
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                command = str(record.get("command") or "").strip()
                if not command or not self.is_test_command(command):
                    continue

                observation = str(record.get("observation_summary") or record.get("observation") or "")
                if not self._observation_proves_later_test_segment_ran(observation):
                    continue

                for prefix in self._extract_recordable_setup_commands(command):
                    if prefix:
                        collected.append(self._strip_output_truncation_suffix(prefix))
            if collected:
                break
        return self._dedupe_preserve_order(collected)

    def _collect_promotable_file_rewrite_prefixes_from_failed_test_actions(self, recipe_input):
        promotable = []
        for prefix in self._collect_recordable_prefixes_from_failed_test_actions(recipe_input):
            if self._failed_test_prefix_is_safe_file_rewrite_to_promote(prefix):
                promotable.append(prefix)
        return self._dedupe_preserve_order(promotable)

    def _collect_promotable_file_rewrite_prefixes_from_failed_setup_chains(self, recipe_input):
        recipe_input = recipe_input or {}
        collected = []
        candidate_lists = [
            recipe_input.get("failed_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("failed_actions"),
            recipe_input.get("successful_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("successful_actions"),
        ]
        for records in candidate_lists:
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                command = str(record.get("command") or "").strip()
                if not command or self.is_test_command(command):
                    continue

                observation = str(record.get("observation_summary") or record.get("observation") or "")
                if not self._observation_has_obvious_command_failure_signal(observation):
                    continue

                segments = self._split_shell_chain(command)
                if len(segments) < 2:
                    continue

                for raw_segment, separator in segments[:-1]:
                    if separator != "&&":
                        continue
                    cleaned_segment = self._strip_output_truncation_suffix(raw_segment)
                    if self._failed_test_prefix_is_safe_file_rewrite_to_promote(cleaned_segment):
                        collected.append(cleaned_segment)
        return self._dedupe_preserve_order(collected)

    def _failed_test_prefix_is_safe_file_rewrite_to_promote(self, command):
        if not self._is_observed_repository_file_rewrite(command):
            return False

        normalized = self._normalize_command_segment(command)
        if normalized.startswith(("sed ", "perl ")):
            return True
        if normalized.startswith(("python ", "python2 ", "python3 ")) and self._command_rewrites_files(command):
            return True
        return False

    def _observation_proves_later_test_segment_ran(self, observation):
        if not observation or not observation.strip():
            return False
        return (
            self._observation_has_effective_test_signal(observation)
            or self._observation_has_test_failure_signal(observation)
        )

    def _match_observed_setup_command(self, command, observed_setup_commands):
        normalized_command = self._strip_run_prefix((command or "").strip())
        if not normalized_command:
            return None

        exact_match = self._find_exact_observed_setup_command(
            normalized_command,
            observed_setup_commands,
        )
        if exact_match:
            return exact_match

        install_match = self._match_observed_cross_manager_install_command(
            normalized_command,
            observed_setup_commands,
        )
        if install_match:
            return install_match

        if not self._command_rewrites_files(normalized_command):
            return None

        mentioned_paths = self._extract_command_file_paths(normalized_command)
        if not mentioned_paths:
            return None
        relevant_paths = self._select_paths_for_repository_matching(mentioned_paths)
        if not any(self._path_is_repository_file(path) for path in relevant_paths):
            return None

        scored_candidates = []
        for observed in observed_setup_commands:
            if not self._command_rewrites_files(observed):
                continue
            overlap = 0
            for path in mentioned_paths:
                if self._command_mentions_patch_path(observed, path):
                    overlap += 1
            if overlap <= 0:
                continue
            scored_candidates.append(
                (
                    overlap,
                    int("\n" in observed),
                    int(observed.startswith(("python ", "python2 ", "python3 "))),
                    len(observed),
                    observed,
                )
            )

        if not scored_candidates:
            return None

        scored_candidates.sort(reverse=True)
        return scored_candidates[0][-1]

    def _find_exact_observed_setup_command(self, command, observed_setup_commands):
        for observed in observed_setup_commands or []:
            cleaned_observed = self._strip_output_truncation_suffix(observed)
            if command == observed or command == cleaned_observed:
                return cleaned_observed or observed
        return None

    def _match_observed_cross_manager_install_command(self, command, observed_setup_commands):
        """Undo LLM rewrites that swap package managers inside a verified install chain."""
        candidate_packages = self._extract_install_packages_by_manager(command)
        candidate_pip_packages = candidate_packages.get("pip") or set()
        if not candidate_pip_packages:
            return None

        candidates = []
        for observed in observed_setup_commands or []:
            cleaned_observed = self._strip_output_truncation_suffix(observed)
            observed_packages = self._extract_install_packages_by_manager(cleaned_observed)
            observed_pip_packages = observed_packages.get("pip") or set()
            if not observed_pip_packages:
                continue

            shared_pip_packages = candidate_pip_packages & observed_pip_packages
            if not shared_pip_packages:
                continue

            # This is intentionally conservative: only canonicalize when the
            # LLM added pip packages that were actually installed by a system
            # package manager in the same successful observed command.
            pip_packages_replaced_from_system = candidate_pip_packages - observed_pip_packages
            observed_system_packages = set()
            for manager, packages in observed_packages.items():
                if manager != "pip":
                    observed_system_packages.update(packages)
            if not pip_packages_replaced_from_system:
                continue
            if not pip_packages_replaced_from_system.issubset(observed_system_packages):
                continue
            if not observed_pip_packages.issubset(candidate_pip_packages):
                continue
            if not self._non_pip_install_packages_are_covered(candidate_packages, observed_packages):
                continue

            candidates.append(
                (
                    len(shared_pip_packages),
                    len(pip_packages_replaced_from_system),
                    -len(cleaned_observed),
                    cleaned_observed,
                )
            )

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][-1]

    def _non_pip_install_packages_are_covered(self, candidate_packages, observed_packages):
        for manager, packages in (candidate_packages or {}).items():
            if manager == "pip":
                continue
            if not packages.issubset((observed_packages or {}).get(manager, set())):
                return False
        return True

    def _extract_install_packages_by_manager(self, command):
        packages_by_manager = {}
        for raw_segment, _ in self._split_shell_chain(command or ""):
            for component in self._split_pipeline(raw_segment):
                pip_packages = self._extract_pip_install_package_names(component)
                if pip_packages:
                    packages_by_manager.setdefault("pip", set()).update(pip_packages)

                manager, system_packages = self._extract_system_install_packages(component)
                if manager and system_packages:
                    packages_by_manager.setdefault(manager, set()).update(system_packages)
        return packages_by_manager

    def _extract_system_install_packages(self, component):
        try:
            tokens = shlex.split(self._strip_trailing_redirections(component), posix=True)
        except ValueError:
            return None, set()

        tokens = self._strip_leading_env_assignment_tokens(tokens)
        if not tokens:
            return None, set()

        executable = tokens[0].rsplit("/", 1)[-1]
        if executable in {"apt", "apt-get"} and len(tokens) > 1 and tokens[1] == "install":
            return "apt", self._extract_package_tokens_after_install(tokens[2:])
        if executable in {"yum", "dnf"} and len(tokens) > 1 and tokens[1] == "install":
            return executable, self._extract_package_tokens_after_install(tokens[2:])
        if executable == "apk" and len(tokens) > 1 and tokens[1] == "add":
            return "apk", self._extract_package_tokens_after_install(tokens[2:])
        return None, set()

    def _strip_leading_env_assignment_tokens(self, tokens):
        stripped = list(tokens or [])
        while stripped and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", stripped[0]):
            stripped.pop(0)
        return stripped

    def _extract_package_tokens_after_install(self, tokens):
        packages = set()
        skip_next = False
        options_with_values = {
            "-o",
            "-c",
            "--config-file",
            "--option",
            "--root",
            "--installroot",
            "--repository",
        }
        for token in tokens or []:
            if skip_next:
                skip_next = False
                continue
            if self._token_looks_like_shell_redirection(token):
                continue
            if token in options_with_values:
                skip_next = True
                continue
            if token.startswith("-"):
                continue
            if token in {"&&", "||", ";"}:
                continue

            package_name = self._normalize_system_package_token(token)
            if package_name:
                packages.add(package_name)
        return packages

    def _normalize_system_package_token(self, token):
        normalized = (token or "").strip().lower()
        if not normalized:
            return ""
        normalized = normalized.split("=", 1)[0]
        normalized = normalized.split(":", 1)[0]
        if not re.match(r"^[a-z0-9][a-z0-9+_.-]*$", normalized):
            return ""
        return normalized

    def _extract_command_file_paths(self, command):
        command = command or ""
        raw_paths = []
        raw_paths.extend(re.findall(r"/(?:app|testbed)/[^\s'\";|)]+", command))
        raw_paths.extend(
            re.findall(
                r"(?<![A-Za-z0-9_./-])(?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+",
                command,
            )
        )
        raw_paths.extend(re.findall(r"(?:^|[\s(;&|])(?:>>|>)\s*([^\s'\";|)]+)", command))
        raw_paths.extend(
            re.findall(
                r"(?<![A-Za-z0-9_./-])(?:\./)?(?:\.[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)",
                command,
            )
        )

        normalized = []
        seen = set()
        for path in raw_paths:
            candidate = path.strip().rstrip(",);:")
            if not candidate or candidate in {">", ">>", "<<", "<<-"}:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _extract_observed_tox_test_dependencies(self, recipe_input):
        recipe_input = recipe_input or {}
        candidate_lists = [
            recipe_input.get("successful_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("successful_actions"),
        ]

        for records in candidate_lists:
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                command = self._normalize_command_segment(record.get("command") or "")
                if command not in {"cat tox.ini", "cat /app/tox.ini", "cat ./tox.ini"}:
                    continue
                observation = str(record.get("observation_summary") or "")
                dependencies = self._parse_tox_dependencies(observation)
                if dependencies:
                    return dependencies
        return []

    def _collect_observed_exact_version_override_commands(self, recipe_input):
        records = self._collect_successful_action_records(recipe_input)
        if not records:
            return []

        verification_commands = {
            self._normalize_repo2run_collect_candidate(command)
            for command in self._resolve_final_verification_bundle(recipe_input).get("test_commands") or []
            if self._normalize_repo2run_collect_candidate(command)
        }

        final_override_by_package = {}
        for record in records:
            command = str(record.get("command") or "").strip()
            if not command:
                continue

            normalized_command = self._normalize_repo2run_collect_candidate(command)
            if verification_commands and normalized_command in verification_commands:
                break

            for override in self._extract_exact_version_overrides_from_command(record):
                final_override_by_package[override["package"]] = override

        unique_commands = {}
        for override in final_override_by_package.values():
            command = override["command"]
            if command not in unique_commands:
                unique_commands[command] = {
                    "step_index": override["step_index"],
                    "command": command,
                }
            else:
                unique_commands[command]["step_index"] = min(
                    unique_commands[command]["step_index"],
                    override["step_index"],
                )

        return [
            item["command"]
            for item in sorted(
                unique_commands.values(),
                key=lambda item: item["step_index"],
            )
        ]

    def _collect_successful_action_records(self, recipe_input):
        recipe_input = recipe_input or {}
        candidate_lists = [
            recipe_input.get("successful_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("successful_actions"),
        ]

        for records in candidate_lists:
            normalized_records = []
            for position, record in enumerate(records or []):
                if not isinstance(record, dict):
                    continue
                command = str(record.get("command") or "").strip()
                if not command:
                    continue
                if self._record_has_obvious_failure_signal(record):
                    continue
                step_index = record.get("step_index")
                if not isinstance(step_index, int):
                    step_index = position
                normalized_record = dict(record)
                normalized_record["step_index"] = step_index
                normalized_record["command"] = command
                normalized_records.append(normalized_record)
            if normalized_records:
                return sorted(normalized_records, key=lambda item: item["step_index"])
        return []

    def _record_has_obvious_failure_signal(self, record):
        if not isinstance(record, dict):
            return False

        analysis = record.get("test_analysis")
        if isinstance(analysis, dict) and analysis.get("is_test_command") and not analysis.get(
            "is_effective_test_run", False
        ):
            return True

        observation = str(record.get("observation_summary") or record.get("observation") or "")
        if not observation.strip():
            return False

        command = str(record.get("command") or "")
        if not command.strip():
            return False

        if self.is_test_command(command):
            return self._observation_has_test_failure_signal(observation)

        normalized_command = self._normalize_command_segment(command)
        if self._is_setup_command(normalized_command):
            return self._observation_has_obvious_command_failure_signal(observation)
        return False

    def _observation_has_obvious_command_failure_signal(self, observation):
        normalized_observation = self._normalize_observation_text(observation).lower()
        if not normalized_observation:
            return False

        # pip emits this line with return code 0 when dependency conflicts remain.
        # It is a resolver warning, not proof that the install command failed.
        normalized_observation = re.sub(
            r"(?im)^\s*error:\s+pip's dependency resolver does not currently take into account .*$\n?",
            "",
            normalized_observation,
        )

        failure_patterns = [
            r"^\s*error:",
            r"traceback \(most recent call last\):",
            r"subprocess-exited-with-error",
            r"returned non-zero exit status",
            r"no matching distribution found",
            r"could not find a version that satisfies the requirement",
            r"failed building wheel",
            r"could not build wheels",
            r"read timed out",
            r"readtimeouterror",
            r"\[system\].*test failure detected",
            r"pluginvalidationerror",
        ]
        return any(
            re.search(pattern, normalized_observation, re.IGNORECASE | re.MULTILINE)
            for pattern in failure_patterns
        )

    def _extract_exact_version_overrides_from_command(self, command):
        overrides = []
        step_index = None
        if isinstance(command, dict):
            step_index = command.get("step_index")
            command = command.get("command")
        command = str(command or "").strip()
        if not command:
            return overrides

        for raw_segment, _ in self._split_shell_chain(command):
            for component in self._split_pipeline(raw_segment):
                cleaned_component = self._extract_clean_pip_install_component(component)
                if not cleaned_component:
                    continue
                for package, exact_spec in self._extract_exact_version_specs_from_install_component(cleaned_component):
                    overrides.append(
                        {
                            "package": package,
                            "spec": exact_spec,
                            "command": cleaned_component,
                            "step_index": step_index if isinstance(step_index, int) else 0,
                        }
                    )
        return overrides

    def _extract_clean_pip_install_component(self, component):
        component = (component or "").strip()
        if not component:
            return ""

        try:
            tokens = shlex.split(component, posix=True)
        except ValueError:
            return ""

        if not tokens:
            return ""

        install_start = self._find_pip_install_start_index(tokens)
        if install_start is None:
            return ""

        kept_tokens = []
        for token in tokens[install_start:]:
            if self._token_looks_like_shell_redirection(token):
                continue
            kept_tokens.append(token)

        cleaned = " ".join(kept_tokens).strip()
        normalized = self._normalize_command_segment(cleaned)
        if not self._normalized_segment_is_pip_install(normalized):
            return ""
        return cleaned

    def _find_pip_install_start_index(self, tokens):
        for index in range(len(tokens)):
            if tokens[index] in {"pip", "pip2", "pip3"}:
                if index + 1 < len(tokens) and tokens[index + 1] == "install":
                    return index
            if tokens[index] in {"python", "python2", "python3"}:
                if index + 3 < len(tokens) and tokens[index + 1:index + 4] == ["-m", "pip", "install"]:
                    return index
        return None

    def _token_looks_like_shell_redirection(self, token):
        if token in {">", ">>", "<", "<<", "<<<", "2>", "1>", "2>>", "1>>"}:
            return True
        return bool(re.match(r"^\d*(?:>>?|<<?|>&).*$", token or ""))

    def _extract_exact_version_specs_from_install_component(self, component):
        try:
            tokens = shlex.split(component, posix=True)
        except ValueError:
            return []

        install_start = self._find_pip_install_start_index(tokens)
        if install_start is None:
            return []

        dependency_tokens = []
        for token in tokens[install_start:]:
            if token in {"pip", "pip2", "pip3", "python", "python2", "python3", "-m", "install"}:
                continue
            if token.startswith("-"):
                continue
            if self._token_looks_like_shell_redirection(token):
                continue
            dependency_tokens.append(token)

        exact_specs = []
        seen = set()
        for token in dependency_tokens:
            match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[A-Za-z0-9_,.-]+\])?==([^=\s]+)$", token)
            if not match:
                continue
            package = match.group(1).lower()
            if package in seen:
                continue
            seen.add(package)
            exact_specs.append((package, token))
        return exact_specs

    def _normalized_segment_is_pip_install(self, normalized_command):
        normalized_command = (normalized_command or "").strip()
        return normalized_command.startswith(
            (
                "pip install",
                "pip2 install",
                "pip3 install",
                "python -m pip install",
                "python2 -m pip install",
                "python3 -m pip install",
            )
        )

    def _normalize_repo2run_collect_candidate(self, command):
        normalized = " ".join(str(command or "").split())
        while normalized.startswith("cd /app && "):
            normalized = normalized[len("cd /app && ") :].strip()
        return normalized

    def _find_observed_exact_version_override_index(self, command, observed_overrides):
        normalized_command = self._normalize_command_for_recipe_comparison(command)
        if not normalized_command:
            return None

        for index, observed in enumerate(observed_overrides):
            normalized_observed = self._normalize_command_for_recipe_comparison(observed)
            if not normalized_observed:
                continue
            if normalized_command == normalized_observed:
                return index
        return None

    def _parse_tox_dependencies(self, tox_text):
        dependencies = []
        seen = set()
        in_deps_block = False

        for raw_line in (tox_text or "").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue

            if not in_deps_block:
                if stripped.startswith("deps"):
                    in_deps_block = True
                    inline_value = stripped.partition("=")[2].strip()
                    if inline_value:
                        for token in inline_value.split():
                            if self._looks_like_plain_python_dependency(token) and token not in seen:
                                seen.add(token)
                                dependencies.append(token)
                    continue
                continue

            if line[:1].isspace():
                candidate = stripped.split("#", 1)[0].strip()
                if not candidate or ":" in candidate:
                    continue
                if self._looks_like_plain_python_dependency(candidate) and candidate not in seen:
                    seen.add(candidate)
                    dependencies.append(candidate)
                continue

            break

        return dependencies

    def _looks_like_plain_python_dependency(self, token):
        normalized = (token or "").strip()
        if not normalized:
            return False
        if normalized.startswith("-"):
            return False
        return bool(re.match(r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:[=<>!~].+)?$", normalized))

    def _command_installs_python_dependency(self, command, dependency):
        normalized_command = self._normalize_command_segment(command)
        normalized_dependency = dependency.strip().lower()
        if not self._normalized_segment_is_pip_install(normalized_command) or not normalized_dependency:
            return False
        pattern = rf"(?<![a-z0-9_.-]){re.escape(normalized_dependency)}(?![a-z0-9_.-])"
        return bool(re.search(pattern, normalized_command))

    def _find_observed_backend_bootstrap_command(self, recipe_input):
        recipe_input = recipe_input or {}
        candidate_lists = [
            recipe_input.get("successful_actions"),
            (recipe_input.get("agent_run_summary") or {}).get("successful_actions"),
        ]

        for records in candidate_lists:
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                command = str(record.get("command") or "").strip()
                if not command:
                    continue
                for raw_segment, _ in self._split_shell_chain(command):
                    segment = raw_segment.strip()
                    if not segment:
                        continue
                    if self._command_installs_backend_bootstrap(segment):
                        return segment
        return None

    def _command_requires_backend_bootstrap(self, command):
        normalized = self._normalize_command_segment(command)
        if not normalized:
            return False
        return "--no-build-isolation" in normalized

    def _command_installs_backend_bootstrap(self, command):
        normalized = self._normalize_command_segment(command)
        if not normalized:
            return False
        if "pip install" not in normalized:
            return False
        return "setuptools" in normalized and "wheel" in normalized

    def _resolve_final_verification_bundle(self, recipe_input):
        recipe_input = recipe_input or {}
        final_bundle = recipe_input.get("final_verification_bundle")
        if isinstance(final_bundle, dict) and final_bundle:
            return final_bundle

        direct_bundle = recipe_input.get("verification_bundle")
        if isinstance(direct_bundle, dict) and direct_bundle:
            return direct_bundle

        run_summary = recipe_input.get("agent_run_summary") or {}
        if not isinstance(run_summary, dict):
            return {}

        summary_bundle = run_summary.get("verification_bundle")
        if isinstance(summary_bundle, dict) and summary_bundle:
            return summary_bundle

        return {
            "runtime_preparation_commands": run_summary.get("verified_runtime_preparation_commands") or [],
            "test_commands": run_summary.get("verified_test_commands") or [],
        }

    def _move_patch_sensitive_commands(self, build_commands, post_commands, test_patch):
        patched_paths = self._extract_patch_paths(test_patch)
        if not patched_paths:
            return self._dedupe_build_commands_preserve_order_sensitive(
                (build_commands or []) + (post_commands or [])
            ), []

        kept_build_commands = []
        promoted_commands = []
        for command in build_commands:
            if self._command_should_run_after_test_patch(command, patched_paths):
                promoted_commands.append(command)
            else:
                kept_build_commands.append(command)

        return kept_build_commands, self._dedupe_preserve_order(promoted_commands + post_commands)

    def _extract_patch_paths(self, patch_text):
        paths = []
        seen = set()
        for left, right in re.findall(r"^diff --git a/(.*?) b/(.*?)$", patch_text or "", re.MULTILINE):
            for path in (left, right):
                normalized = self._normalize_patch_path(path)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    paths.append(normalized)

        for marker in re.findall(r"^(?:---|\+\+\+) [ab]/([^\t\n\r]+)", patch_text or "", re.MULTILINE):
            normalized = self._normalize_patch_path(marker)
            if normalized and normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
        return paths

    def _normalize_patch_path(self, path):
        path = (path or "").strip()
        if path in {"/dev/null", "dev/null"}:
            return ""
        return path.lstrip("./")

    def _command_should_run_after_test_patch(self, command, patched_paths):
        if not self._command_rewrites_files(command):
            return False
        return any(self._command_mentions_patch_path(command, path) for path in patched_paths)

    def _command_rewrites_files(self, command):
        normalized = command or ""
        patterns = (
            r"\bsed\b[^\n;&|]*\s-i(?:\b|['\"]|[A-Za-z])",
            r"\bperl\b[^\n;&|]*-[A-Za-z]*i[A-Za-z]*\b",
            r"\b(?:python|python2|python3)\b.*\b(?:write_text|fileinput|open\s*\()",
            r"\b(?:chmod|touch|cp|mv|rm)\b",
            r"\bcat\b[^;&|]*>",
            r">>",
        )
        return any(re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)

    def _command_mentions_patch_path(self, command, path):
        command = command or ""
        path = self._normalize_patch_path(path)
        if not path:
            return False

        variants = {
            path,
            f"./{path}",
            f"/app/{path}",
            f"/testbed/{path}",
        }
        if any(variant in command for variant in variants):
            return True

        directory, basename = os.path.split(path)
        if not directory or not basename or basename not in command:
            return False

        cd_pattern = rf"\bcd\s+(?:/app/|/testbed/|\./)?{re.escape(directory)}(?:\s|;|&&|\|\||$)"
        return bool(re.search(cd_pattern, command))

    def apply_build_recipe(self, recipe):
        """Replace recorded setup instructions with the recipe's build commands."""
        self.build_recipe = recipe or {}
        self.instructions = []
        for command in self.build_recipe.get("build_commands") or []:
            self._record_setup_instruction(command)

    def add_build_instruction(self, command: str) -> None:
        """Append a single build command as the last instruction.

        Uses the same path as apply_build_recipe (i.e. _record_setup_instruction)
        so the command is formatted, filtered for read-only no-ops, and deduplicated
        exactly like any other recipe command.  Intended for post-recipe pin layers
        that must render as the last RUN in the Dockerfile.
        """
        self._record_setup_instruction(command)

    def add_file_write_instruction(self, path: str, content: str) -> None:
        """Append ONE deterministic write of `content` to `path` (Fix 1).

        The agent's final file state is captured (file_capture.FileStateCapturer) and
        written here via a base64 payload, sidestepping heredoc/quoting hazards: the
        content is never interpreted by the shell. Idempotent on (path, content).

        Encodes with ``surrogateescape`` so a file captured byte-losslessly (raw bytes
        decoded with the same codec) round-trips to its EXACT original bytes — a
        mostly-UTF-8 file with a few odd bytes is never baked corrupted (HIGH 2)."""
        if not path:
            return
        encoded = base64.b64encode((content or "").encode("utf-8", "surrogateescape")).decode("ascii")
        parent = os.path.dirname(path)
        guard = f"mkdir -p {self._shell_single_quote(parent)} && " if parent else ""
        command = (
            f"{guard}printf '%s' {self._shell_single_quote(encoded)} "
            f"| base64 -d > {self._shell_single_quote(path)}"
        )
        instruction = f"RUN {command}"
        if instruction in self.instructions:
            return
        self.instructions.append(instruction)

    def add_env_instruction(self, name: str, value: str) -> None:
        """Prepend an `ENV name="value"` line so it persists across every RUN layer
        and into the runtime container (DROPPED_ENV). Unlike `RUN export`, a
        Dockerfile ENV survives layer boundaries and the final image's runtime env."""
        if not name:
            return
        safe = str(value).replace("\\", "\\\\").replace("$", "\\$").replace('"', '\\"')
        instruction = f'ENV {name}="{safe}"'
        if instruction in self.instructions:
            return
        self.instructions.insert(0, instruction)

    def build_fallback_recipe(self, recipe_input=None, error=None):
        """Create a conservative recipe from existing rule-based recorded instructions."""
        recipe_input = recipe_input or {}
        final_bundle = recipe_input.get("final_verification_bundle") or {}
        build_commands = []
        for instruction in self.instructions:
            if instruction.startswith("RUN "):
                build_commands.append(instruction[4:].strip())

        rationale = (
            "Fell back to the rule-recorded successful setup commands because LLM "
            "recipe synthesis failed or produced invalid JSON."
        )
        if error:
            rationale += f" Error: {error}"

        return {
            "build_commands": self._dedupe_preserve_order(build_commands),
            "post_test_patch_commands": [],
            "runtime_preparation_commands": self._normalize_recipe_command_list(
                final_bundle.get("runtime_preparation_commands")
            ),
            "test_commands": self._normalize_recipe_command_list(final_bundle.get("test_commands")),
            "excluded_commands": [],
            "rationale": rationale,
            "confidence": "low",
        }

    def _normalize_recipe_command_list(self, commands):
        if isinstance(commands, str):
            commands = [commands]

        normalized = []
        for command in commands or []:
            if not command:
                continue
            if not isinstance(command, str):
                continue
            stripped = self._strip_run_prefix(command.strip())
            if stripped:
                normalized.append(stripped)
        return self._dedupe_preserve_order(normalized)

    def _normalize_excluded_commands(self, excluded_commands):
        normalized = []
        for item in excluded_commands or []:
            if isinstance(item, str):
                command = self._strip_run_prefix(item.strip())
                if command:
                    normalized.append({"command": command, "reason": ""})
            elif isinstance(item, dict):
                command = self._strip_run_prefix(str(item.get("command", "")).strip())
                if command:
                    normalized.append({
                        "command": command,
                        "reason": str(item.get("reason", "")).strip(),
                    })
        return normalized

    def _strip_run_prefix(self, command):
        if command.lower().startswith("run "):
            return command[4:].strip()
        return command

    def _dedupe_preserve_order(self, items):
        result = []
        seen = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _dedupe_build_commands_preserve_order_sensitive(self, items):
        """Dedupe replay commands, but keep repeated file rewrites in trajectory order.

        A repeated append/patch can be necessary after a later package reinstall
        overwrites site-packages files. Dropping it changes sandbox replay state.
        """
        result = []
        seen = set()
        for item in items or []:
            if item in seen and not self._duplicate_build_command_is_order_sensitive(item):
                continue
            seen.add(item)
            result.append(item)
        return result

    def _duplicate_build_command_is_order_sensitive(self, command):
        return self._command_rewrites_files(command) or self._command_is_package_manager_mutation(command)

    def _command_is_package_manager_mutation(self, command):
        if not command or not command.strip():
            return False
        for raw_segment, _ in self._split_shell_chain(command):
            for component in self._split_pipeline(raw_segment):
                normalized = self._normalize_command_segment(component)
                if self._is_package_manager_mutation_segment(normalized):
                    return True
        return False

    def _is_package_manager_mutation_segment(self, normalized_command):
        normalized = self._strip_dev_null_redirections(normalized_command or "").strip()
        patterns = (
            r"^(?:python3?|/[\w./-]*python3?)\s+-m\s+pip\s+(?:install|uninstall|download|wheel)\b",
            r"^(?:pip3?|/[\w./-]*pip3?)\s+(?:install|uninstall|download|wheel)\b",
            r"^uv\s+pip\s+(?:install|uninstall)\b",
            r"^uv\s+(?:sync|add|remove)\b",
            r"^(?:sudo\s+)?apt(?:-get)?\s+(?:update|install|remove|purge|upgrade)\b",
            r"^(?:mamba|conda)\s+(?:install|remove|update|env\s+create)\b",
            r"^(?:npm|yarn|pnpm)\s+(?:install|add|remove|ci)\b",
        )
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _extract_first_json_object(self, text):
        candidates = self._extract_json_object_candidates(text)
        return candidates[0] if candidates else None

    def _extract_json_object_candidates(self, text):
        objects = []
        if not text:
            return objects

        position = 0
        while position < len(text):
            start = text.find("{", position)
            if start == -1:
                break

            depth = 0
            in_string = False
            escape = False
            found = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(text[start:index + 1])
                        position = index + 1
                        found = True
                        break

            if not found:
                position = start + 1

        return objects

    def record_success(self, command):
        """Records a successful bash command as a RUN instruction."""
        recordable_commands = self._extract_recordable_setup_commands(command)
        for recordable_command in recordable_commands:
            self._record_setup_instruction(recordable_command)

    def is_readonly_command(self, command):
        """Public wrapper used by the agent when tracking verification state."""
        return self._is_readonly_command(command)

    def command_mutates_environment(self, command):
        """Return True when a successful command changed the effective runtime environment."""
        return self._command_has_meaningful_setup_activity(command)

    def classify_mutation(self, command: str) -> str:
        """Coarse env-mutation class for the ActionLedger (design §13)."""
        normalized = " ".join(str(command or "").lower().split())
        if any(tok in normalized for tok in ("apt-get install", "apt install", "yum install", "apk add")):
            return "system_package_install"
        if any(tok in normalized for tok in ("apt-get remove", "apt remove", "yum remove", "apk del")):
            return "system_package_remove"
        if any(tok in normalized for tok in ("pip install", "pip3 install", "poetry install", "conda install", "npm install", "yarn add")):
            return "language_package_install"
        if "venv" in normalized or "virtualenv" in normalized:
            return "venv_change"
        if normalized.startswith(("export ", "ln -s")) or " > " in normalized or " >> " in normalized:
            return "file_or_env_change"
        return "other_mutation"

    def is_runtime_service_command(self, command):
        """Public wrapper for runtime service startup commands such as redis-server."""
        return self._command_matches_segment_predicate(command, self._is_runtime_service_segment)

    def is_runtime_healthcheck_command(self, command):
        """Public wrapper for healthcheck-only commands such as redis-cli ping."""
        return self._command_matches_segment_predicate(command, self._is_runtime_healthcheck_segment)

    def is_truncated_test_output_command(self, command):
        """Return True when a test command pipes output through a lossy filter."""
        if not command or not command.strip():
            return False

        for raw_segment, _ in self._split_shell_chain(command):
            pipeline_components = self._split_pipeline(raw_segment)
            if len(pipeline_components) < 2:
                continue

            saw_test_component = False
            for component in pipeline_components:
                normalized = self._normalize_command_segment(component)
                if not normalized:
                    continue

                if self._is_test_like_segment(normalized):
                    saw_test_component = True
                    continue

                if saw_test_component and self._is_output_truncation_component(normalized):
                    return True

        return False

    def observation_has_effective_test_signal(self, observation):
        """Expose test-output validation for agent-reported wrapper commands."""
        return self._observation_has_effective_test_signal(observation)

    def observation_has_empty_test_run_signal(self, observation):
        """Expose empty test-run detection for agent-reported wrapper commands."""
        return self._observation_has_empty_test_run_signal(observation)

    def observation_has_test_failure_signal(self, observation):
        """Expose failing test-output detection for final verification guards."""
        return self._observation_has_test_failure_signal(observation)

    def observation_has_passing_test_signal(self, observation):
        """True iff the output shows at least one test PASSED (not merely 'tests ran').

        Deliberately excludes the ambiguous 'ran N tests' / 'collected N' / 'N failed'
        signals that observation_has_effective_test_signal accepts, so a
        '5 failed, 0 passed' or a bare '--collect-only' run is NOT a passing signal."""
        if not observation:
            return False
        norm = self._normalize_observation_text(observation)
        for pat in (
            r"\b[1-9]\d*\s+passed\b",            # pytest
            r"\b[1-9]\d*\s+tests?\s+passed\b",   # AVA/tap-like summaries
            r"\b[1-9]\d*%\s+tests\s+passed\b",   # ctest
            r"test result:\s+ok\.",              # cargo (all passed)
        ):
            if re.search(pat, norm, re.IGNORECASE | re.MULTILINE):
                return True
        return False

    def observation_has_env_defect_signal(self, observation):
        """True when failures indicate a BROKEN ENVIRONMENT: a missing python dep, a
        missing native/system shared library, a missing system binary / build toolchain,
        a collection failure, or a required service that is down/unreachable. Does NOT
        match AssertionError / AttributeError / TypeError / a bare 'N failed' (pre-existing
        source bugs).

        Conservative by design (Fix 3 §5.8): when a failure could be either an env defect
        or a benign source bug we err toward env-defect=True (reject the finalize) -- a
        broken environment certified green is the worst outcome. Hardened against the
        2026-06-12 honesty audit (missing .so / DB-down phrasings / missing binaries)."""
        if not observation:
            return False
        norm = self._normalize_observation_text(observation)   # strips ANSI
        for pat in (
            # --- collection / import failures ---
            r"ERROR collecting",
            r"ImportError while importing test module",
            r"error during collection",
            r"INTERNALERROR",
            r"(?:ModuleNotFoundError|ImportError):\s+No module named\s+['\"](?!tests?\.)",
            r"ImportError:\s+cannot import name",
            # --- missing native / system shared library ---
            r"cannot open shared object file",
            r"error while loading shared libraries",
            r"Library not loaded",                       # macOS dyld
            r"undefined symbol:",
            r"version\s+`?GLIBC[^\n]*not found",
            # --- missing system binary / build toolchain / dev headers ---
            r"\bcommand not found\b",                    # ANY binary: pg_config/ffmpeg/gcc/ld/...
            r"cannot find -l\S+",                        # linker: missing -lpq etc.
            r"fatal error:\s+\S+\.h:\s+No such file",    # missing -dev header
            r"unable to execute '\S*(?:gcc|cc|clang)",
            # --- required service down / unreachable ---
            r"ConnectionRefusedError",
            r"Connection refused",
            r"could not connect to\b",
            r"can'?t connect to\b",
            r"Is the server running\b",                  # postgres
            r"could not translate host name",
            r"Name or service not known",
            r"Temporary failure in name resolution",
            r"No route to host",
            r"\bOperationalError\b",                      # DB driver: psycopg2/sqlalchemy/pymysql/MySQLdb
            r"redis(?:\.\w+)*\.ConnectionError",
            r"\bError\s+\d+\s+connecting to\b",           # redis "Error 111 connecting to"
            r"\[Errno\s+(?:111|110|113|99)\]",            # refused/timeout/unreachable/addr-unavailable
        ):
            if re.search(pat, norm, re.IGNORECASE | re.MULTILINE):
                return True
        # "collected 0 items" + a collection error (two separate re.search; no cross-line .*)
        if re.search(r"collected\s+0\s+items", norm, re.IGNORECASE) and \
           re.search(r"\berror\b", norm, re.IGNORECASE):
            return True
        return False

    def observation_pass_ratio(self, observation):
        """passed / (passed + failed + errors), or None if no countable pass/fail summary.

        Computed PER LINE and then reduced conservatively, hardened against the
        2026-06-12 code audit:
        - strips comma thousands separators so "1,000 failed" counts as 1000 (audit [6]);
        - counts singular `failed` AND plural `failures` (so "10 failures" is not read as 0);
        - when several summary lines disagree (rerun subsets, cached/log lines after the real
          summary), returns the MOST CONSERVATIVE (lowest) ratio among the lines that report
          failures -- a favourable trailing subset/log line can never inflate a mostly-failing
          run to a pass. Only if NO line reports any failure/error is a clean 1.0 returned.
        Skipped tests are excluded (mirrors compute_essr effective_total). Non-pytest
        phrasings (e.g. ctest 'N tests failed') do not match -> None -> conservative reject."""
        norm = self._normalize_observation_text(observation or "").replace(",", "")

        def _n(line, word):
            vals = [int(m) for m in re.findall(r"(\d+)\s+(?:" + word + r")\b", line, re.IGNORECASE)]
            return max(vals) if vals else 0

        failure_ratios = []
        saw_clean_pass = False
        for line in norm.splitlines():
            passed = _n(line, r"passed")
            failed = _n(line, r"failed|failures?")
            errors = _n(line, r"errors?")
            denom = passed + failed + errors
            if denom == 0:
                continue
            if failed + errors > 0:
                failure_ratios.append(passed / denom)
            elif passed > 0:
                saw_clean_pass = True

        if failure_ratios:
            return min(failure_ratios)
        if saw_clean_pass:
            return 1.0
        return None

    def observation_has_ambiguous_error_signal(self, observation):
        """True iff the output reports 'N error(s)' (pytest's collection/setup error
        category, distinct from 'failed'). A partial-pass run reporting an error is
        treated as a potential env/setup defect and rejected by the finalize gate --
        conservative (Fix 3 §5.7). Backstop for env-defects whose specific cause text
        is truncated out of the summary."""
        if not observation:
            return False
        norm = self._normalize_observation_text(observation)
        return bool(re.search(r"\b[1-9]\d*\s+errors?\b", norm, re.IGNORECASE))

    def observation_looks_like_help_text(self, observation):
        """Expose help-text detection for agent-reported wrapper commands."""
        return self._observation_looks_like_help_text(observation)

    def is_persistent_setup_command(self, command):
        """Return True when a successful command would already be replayed via Dockerfile setup."""
        return bool(self._extract_recordable_setup_commands(command))

    def _record_setup_instruction(self, command):
        """Persist a setup/build command into the generated Dockerfile."""
        if not command or not command.strip():
            return
        command = command.strip()
        if "\n" not in command:
            command = quote_shell_sensitive_package_specs(command)
        if self._is_readonly_command(command):
            return

        run_instruction = self._format_run_instruction(command)
        if (
            run_instruction in self.instructions
            and not self._duplicate_build_command_is_order_sensitive(command)
        ):
            return

        self.instructions.append(run_instruction)

    def _format_run_instruction(self, command):
        if "\n" not in command:
            return f"RUN {command}"

        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        script_path = f"/tmp/graph2env_run_{len(self.instructions) + 1}.sh"
        return (
            f"RUN printf '%s' {self._shell_single_quote(encoded)} "
            f"| base64 -d > {script_path} "
            f"&& chmod +x {script_path} "
            f"&& /bin/sh {script_path}"
        )

    def _shell_single_quote(self, value):
        return "'" + (value or "").replace("'", "'\"'\"'") + "'"

    def _render_instruction_for_dockerfile(self, instruction):
        if not instruction or not instruction.startswith("RUN "):
            return instruction

        command = instruction[4:].strip()
        if "\n" in command:
            return instruction
        if _looks_like_apt_install_replay_command(command):
            return build_resilient_apt_install_run_instruction(command)
        if self._looks_like_pip_install_command(command):
            return build_resilient_pip_install_run_instruction(command)
        return instruction

    def _extract_recordable_setup_commands(self, command):
        """Keep setup/build prefixes of successful commands while excluding the test invocation itself."""
        if not command or not command.strip():
            return []
        if self._looks_like_ephemeral_workspace_repair(command):
            return []
        if self._is_readonly_command(command):
            return []

        if not self.is_test_command(command):
            recordable_command = self._extract_recordable_command_segments(
                command,
                stop_before_test=False,
            )
            return [recordable_command] if recordable_command else []

        setup_prefix = self._extract_recordable_command_segments(
            command,
            stop_before_test=True,
        )
        return [setup_prefix] if setup_prefix else []

    def _looks_like_ephemeral_workspace_repair(self, command):
        """Skip ad-hoc artifact repair commands that only work because the workspace is already dirty."""
        normalized_command = command.strip().lower()
        if not normalized_command:
            return False

        numbered_duplicate_repair = re.search(
            r"(?:^|(?:&&|\|\||;)\s*)(?:mv|cp)\s+(['\"]?)([^'\"\s]+?)\.(\d+)\1\s+(['\"]?)\2\4(?:\s|$)",
            normalized_command,
        )
        if not numbered_duplicate_repair:
            return False

        return any(
            marker in normalized_command
            for marker in (
                "tar -x",
                "unzip ",
                "gunzip ",
                "ln -s ",
                "ln -sf ",
                "/opt/",
            )
        )
    
    def _is_readonly_command(self, command):
        """Treat safe inspection/search commands as read-only when they do not redirect output."""
        if not command or not command.strip():
            return False
        if self._has_output_redirection(self._strip_dev_null_redirections(command)):
            return False

        saw_component = False
        for raw_segment, _ in self._split_shell_chain(command):
            for component in self._split_pipeline(raw_segment):
                normalized = self._normalize_command_segment(component)
                if not normalized:
                    continue
                if not self._pipeline_component_is_safe_readonly(normalized):
                    return False
                saw_component = True
        return saw_component
    
    def is_test_command(self, command):
        """Determine whether a command runs tests."""
        if not command or not command.strip():
            return False

        # Read-only commands such as `echo "tests passed"` must never be treated as test runs.
        if self._is_readonly_command(command):
            return False

        for _, normalized in self._iter_command_segments(command):
            if self._is_test_like_segment(normalized):
                return True

        return False

    def analyze_test_run(self, command, observation=""):
        """Judge whether a successful command actually executed meaningful tests."""
        result = {
            "is_test_command": False,
            "is_effective_test_run": False,
            "confidence": "none",
            "reason": "not_test_command",
        }

        if not self.is_test_command(command):
            return result

        result["is_test_command"] = True

        if self._observation_looks_like_help_text(observation):
            result["reason"] = "help_or_usage_output"
            return result

        if self.is_truncated_test_output_command(command):
            result["reason"] = "truncated_test_output"
            return result

        if self._observation_has_test_failure_signal(observation):
            result["reason"] = "test_failure_signal"
            return result

        if self._observation_has_effective_test_signal(observation):
            result["is_effective_test_run"] = True
            result["confidence"] = "high"
            result["reason"] = "observed_test_execution_signal"
            return result

        # Some runners (notably `go test ./...`) can mix real package results with
        # informational lines such as `[no test files]`. Treat explicit positive
        # execution signals as authoritative before falling back to empty-run hints.
        if self._observation_has_empty_test_run_signal(observation):
            result["reason"] = "no_tests_executed"
            return result

        if observation and any(
            self._looks_like_test_executable(normalized)
            for _, normalized in self._iter_command_segments(command)
        ):
            result["is_effective_test_run"] = True
            result["confidence"] = "medium"
            result["reason"] = "direct_test_executable_with_output"
            return result

        result["reason"] = "no_reliable_test_execution_signal"
        return result

    def _iter_command_segments(self, command):
        """Yield normalized shell command segments split on common separators."""
        for segment, _ in self._split_shell_chain(command):
            normalized = self._normalize_command_segment(segment)
            if normalized:
                yield segment.strip(), normalized

    def _command_matches_segment_predicate(self, command, predicate):
        if not command or not command.strip():
            return False

        for _, normalized in self._iter_command_segments(command):
            if predicate(normalized):
                return True
        return False

    def _split_shell_chain(self, command):
        """Split a shell command into ordered segments while respecting quotes, escapes, and heredocs."""
        segments = []
        current = []
        in_single = False
        in_double = False
        escape = False
        index = 0

        while index < len(command):
            char = command[index]

            if escape:
                current.append(char)
                escape = False
                index += 1
                continue

            if char == "\\":
                current.append(char)
                escape = True
                index += 1
                continue

            if char == "'" and not in_double:
                in_single = not in_single
                current.append(char)
                index += 1
                continue

            if char == '"' and not in_single:
                in_double = not in_double
                current.append(char)
                index += 1
                continue

            if not in_single and not in_double:
                if command.startswith("&&", index) or command.startswith("||", index):
                    raw_segment = "".join(current).strip()
                    separator = command[index:index + 2]
                    if raw_segment:
                        segments.append((raw_segment, separator))
                    current = []
                    index += 2
                    continue

                if char in {";", "\n"}:
                    if char == "\n":
                        heredoc_descriptor = self._extract_heredoc_descriptor("".join(current))
                        if heredoc_descriptor is not None:
                            delimiter, strip_tabs = heredoc_descriptor
                            current.append(char)
                            index += 1
                            index = self._consume_heredoc_body(
                                command,
                                index,
                                current,
                                delimiter,
                                strip_tabs=strip_tabs,
                            )
                            raw_segment = "".join(current).strip()
                            if raw_segment:
                                separator = "\n" if index < len(command) else ""
                                segments.append((raw_segment, separator))
                            current = []
                            continue

                    raw_segment = "".join(current).strip()
                    if raw_segment:
                        segments.append((raw_segment, char))
                    current = []
                    index += 1
                    continue

            current.append(char)
            index += 1

        raw_segment = "".join(current).strip()
        if raw_segment:
            segments.append((raw_segment, ""))
        return segments

    def _extract_heredoc_descriptor(self, segment):
        match = re.search(
            r"<<(?P<strip>-)?\s*(?P<quote>['\"]?)(?P<delimiter>[^\s'\"`]+)(?P=quote)\s*$",
            segment or "",
        )
        if not match:
            return None
        return match.group("delimiter"), bool(match.group("strip"))

    def _consume_heredoc_body(self, command, index, current, delimiter, strip_tabs=False):
        while index < len(command):
            next_newline = command.find("\n", index)
            if next_newline == -1:
                line = command[index:]
                current.append(line)
                return len(command)

            line = command[index:next_newline]
            current.append(line)
            current.append("\n")
            index = next_newline + 1

            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                return index

        return index

    def _split_pipeline(self, segment):
        """Split a shell segment on single-pipe operators while respecting quotes/escapes."""
        components = []
        current = []
        in_single = False
        in_double = False
        escape = False
        index = 0

        while index < len(segment):
            char = segment[index]

            if escape:
                current.append(char)
                escape = False
                index += 1
                continue

            if char == "\\":
                current.append(char)
                escape = True
                index += 1
                continue

            if char == "'" and not in_double:
                in_single = not in_single
                current.append(char)
                index += 1
                continue

            if char == '"' and not in_single:
                in_double = not in_double
                current.append(char)
                index += 1
                continue

            if (
                not in_single
                and not in_double
                and char == "|"
                and not segment.startswith("||", index)
            ):
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                index += 1
                continue

            current.append(char)
            index += 1

        component = "".join(current).strip()
        if component:
            components.append(component)
        return components

    def _has_output_redirection(self, command):
        in_single = False
        in_double = False
        escape = False

        for char in command:
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == "'" and not in_double:
                in_single = not in_single
                continue

            if char == '"' and not in_single:
                in_double = not in_double
                continue

            if not in_single and not in_double and char == ">":
                return True

        return False

    def _pipeline_component_is_safe_readonly(self, normalized_command):
        if self._is_version_probe_segment(normalized_command):
            return True
        if self._is_package_manager_inspection_segment(normalized_command):
            return True
        if self._is_python_readonly_probe_segment(normalized_command):
            return True

        executable = normalized_command.split()[0]
        executable = executable.strip("\"'`")
        executable_name = executable.rsplit("/", 1)[-1]
        return executable_name in self.SAFE_READONLY_COMMANDS

    def _is_package_manager_inspection_segment(self, normalized_command):
        normalized = self._strip_dev_null_redirections(normalized_command).strip()
        patterns = (
            r"^(?:python3?|/[\w./-]*python3?)\s+-m\s+pip\s+(?:list|show|freeze|check|index\s+versions)\b",
            r"^(?:pip3?|/[\w./-]*pip3?)\s+(?:list|show|freeze|check|index\s+versions)\b",
            r"^uv\s+pip\s+(?:list|show|freeze|check)\b",
            r"^(?:poetry|pdm)\s+(?:show|list)\b",
            r"^(?:npm|yarn|pnpm)\s+(?:list|ls|why)\b",
            r"^conda\s+(?:list|info)\b",
        )
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _is_python_readonly_probe_segment(self, normalized_command):
        try:
            parts = shlex.split(normalized_command)
        except ValueError:
            return False
        if len(parts) < 3:
            return False

        executable = parts[0].strip("\"'`")
        executable_name = executable.rsplit("/", 1)[-1]
        if executable_name not in {"python", "python3"}:
            return False
        if "-c" not in parts:
            return False

        code_index = parts.index("-c") + 1
        if code_index >= len(parts):
            return False
        code = parts[code_index].strip()
        if not code:
            return False

        dangerous_markers = (
            "=",
            "open(",
            ".write",
            "write_text(",
            "touch(",
            "mkdir(",
            "unlink(",
            "remove(",
            "rmdir(",
            "rename(",
            "replace(",
            "chmod(",
            "chown(",
            "shutil",
            "subprocess",
            "os.system",
            "os.popen",
            "pip ",
            "pip.",
            "install",
            "exec(",
            "eval(",
            "__import__",
            "importlib",
        )
        lowered_code = code.lower()
        if any(marker in lowered_code for marker in dangerous_markers):
            return False

        statements = [part.strip() for part in re.split(r"[;\n]", code) if part.strip()]
        if not statements:
            return False

        imported_names = set()
        for statement in statements:
            if statement.startswith("import "):
                imported_names.update(self._extract_imported_python_names(statement))
            elif statement.startswith("from "):
                imported_names.update(self._extract_from_imported_python_names(statement))

        safe_call_pattern = re.compile(
            r"^(?:print|dir|getattr|hasattr|len|repr|str|type|bool)\s*\(",
            re.IGNORECASE,
        )
        for statement in statements:
            if statement.startswith(("import ", "from ")):
                continue
            if safe_call_pattern.match(statement):
                if not imported_names:
                    return False
                if not any(re.search(rf"\b{re.escape(name)}\b", statement) for name in imported_names):
                    return False
                continue
            return False
        return True

    def _extract_imported_python_names(self, statement):
        imported = statement[len("import "):]
        names = set()
        for item in imported.split(","):
            item = item.strip()
            if not item:
                continue
            if " as " in item:
                name = item.rsplit(" as ", 1)[-1].strip()
            else:
                name = item.split(".", 1)[0].strip()
            if name and re.match(r"^[a-z_][a-z0-9_]*$", name):
                names.add(name)
        return names

    def _extract_from_imported_python_names(self, statement):
        if " import " not in statement:
            return set()
        imported = statement.split(" import ", 1)[1]
        names = set()
        for item in imported.split(","):
            item = item.strip()
            if not item or item == "*":
                continue
            if " as " in item:
                name = item.rsplit(" as ", 1)[-1].strip()
            else:
                name = item.split(".", 1)[0].strip()
            if name and re.match(r"^[a-z_][a-z0-9_]*$", name):
                names.add(name)
        return names

    def _strip_dev_null_redirections(self, command):
        """Ignore harmless output silencing when classifying read-only probes."""
        if not command:
            return ""

        stripped = re.sub(r"\s+(?:[12]?>|&>)\s*/dev/null\b", "", command)
        stripped = re.sub(r"\s+(?:[12]?>|&>)/dev/null\b", "", stripped)
        return stripped

    def _strip_trailing_redirections(self, command):
        stripped = (command or "").strip()
        while True:
            updated = re.sub(r"\s+\d?>&\d\s*$", "", stripped).strip()
            updated = re.sub(r"\s+(?:[12]?>|&>)\s*[^\s]+$", "", updated).strip()
            if updated == stripped:
                return stripped
            stripped = updated

    def _is_version_probe_segment(self, normalized_command):
        normalized = self._strip_dev_null_redirections(normalized_command).strip()
        if not normalized:
            return False

        parts = normalized.split()
        if len(parts) != 2:
            return False

        executable = parts[0].strip("\"'`")
        executable_name = executable.rsplit("/", 1)[-1]
        if executable_name not in self.VERSION_PROBE_COMMANDS:
            return False

        return parts[1] in {"--version", "-v", "version"}

    def _normalize_command_segment(self, segment):
        normalized = segment.strip().lower()
        if not normalized:
            return ""

        normalized = re.sub(
            r"^(?:[a-z_][a-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S+)\s+)+",
            "",
            normalized,
        )
        normalized = re.sub(r"^time\s+", "", normalized)
        return normalized.strip()

    def _segment_matches_test_pattern(self, normalized_command, test_patterns):
        return any(re.match(pattern, normalized_command) for pattern in test_patterns)

    def _is_test_like_segment(self, normalized_command):
        return (
            self._segment_matches_test_pattern(normalized_command, self.TEST_COMMAND_PATTERNS)
            or self._looks_like_test_executable(normalized_command)
        )

    def _is_output_truncation_component(self, normalized_command):
        if not normalized_command:
            return False

        executable = normalized_command.split()[0]
        executable = executable.strip("\"'`")
        executable_name = executable.rsplit("/", 1)[-1]
        return executable_name in {"head", "tail", "grep", "egrep", "fgrep"}

    def _extract_recordable_command_segments(self, command, stop_before_test):
        """Rebuild a command from recordable setup/build segments only."""
        kept_segments = []
        pending_navigation = []
        has_recordable_segment = False

        for raw_segment, separator in self._split_shell_chain(command):
            normalized = self._normalize_command_segment(raw_segment)
            if not normalized:
                continue

            if stop_before_test and self._is_test_like_segment(normalized):
                break

            if self._is_runtime_only_segment(normalized):
                continue

            if self._is_navigation_only_segment(normalized):
                pending_navigation.append((raw_segment.strip(), self._normalize_shell_separator(separator)))
                continue

            if not self._segment_has_meaningful_setup_activity(normalized):
                pending_navigation = []
                continue

            has_recordable_segment = True
            if pending_navigation:
                kept_segments.extend(pending_navigation)
                pending_navigation = []
            kept_segments.append((raw_segment.strip(), self._normalize_shell_separator(separator)))

        if not has_recordable_segment:
            return None

        rebuilt_command = kept_segments[0][0]
        for index in range(1, len(kept_segments)):
            separator = kept_segments[index - 1][1] or "&&"
            segment = kept_segments[index][0]
            if separator == "\n":
                rebuilt_command = f"{rebuilt_command}\n{segment}"
            else:
                rebuilt_command = f"{rebuilt_command} {separator} {segment}"

        if "\n" not in rebuilt_command and not self._command_rewrites_files(rebuilt_command):
            rebuilt_command = re.sub(r"\s+", " ", rebuilt_command)
        rebuilt_command = re.sub(r"(?:&&|\|\||;)\s*$", "", rebuilt_command).strip()
        return rebuilt_command or None

    def _segment_has_meaningful_setup_activity(self, normalized_command):
        """Treat navigation-only prefixes as non-recordable, but keep real setup/build work."""
        if not normalized_command:
            return False

        if self._is_navigation_only_segment(normalized_command):
            return False
        if self._is_runtime_only_segment(normalized_command):
            return False
        if self._is_version_probe_segment(normalized_command):
            return False
        return not self._is_readonly_command(normalized_command)

    def _is_navigation_only_segment(self, normalized_command):
        return normalized_command.startswith(("cd ", "pushd ", "popd"))

    def _is_runtime_only_segment(self, normalized_command):
        return (
            self._is_runtime_service_segment(normalized_command)
            or self._is_runtime_healthcheck_segment(normalized_command)
        )

    def _is_runtime_service_segment(self, normalized_command):
        service_patterns = (
            r"^service\s+\S+\s+(?:start|restart|reload|stop)\b",
            r"^redis-server\b",
            r"^rabbitmq-server\b.*\b-detached\b",
            r"^memcached\b.*\b-d\b",
            r"^mongod\b.*\b--fork\b",
            r"^apache2ctl\s+start\b",
            r"^nginx\b(?:\s|$)",
            # In-image Postgres provisioning runs at RUNTIME (in the eval wrapper),
            # not baked into the image (a RUN-layer daemon dies before CMD). Match
            # the bare form and the as-postgres-user wrapped forms (runuser/su).
            r"(?:^|--\s+|\bpostgres\s+-c\s+\"?)pg_ctlcluster\b.*\bstart\b",
            r"(?:^|--\s+|\bpostgres\s+-c\s+\"?)createdb\b",
            r"(?:^|--\s+|\bpostgres\s+-c\s+\"?)createuser\b",
            # Config-binding obligation: the ALTER USER password reset needs the
            # server running, and the profile.d service-bind write is replaced by
            # the ENV bake + eval-wrapper export. Both are RUNTIME, never baked.
            r"ALTER\s+USER\s+\w+\s+(WITH\s+)?PASSWORD",
            r"/etc/profile\.d/zz_service_bind\.sh",
        )
        return any(re.search(pattern, normalized_command) for pattern in service_patterns)

    def _is_runtime_healthcheck_segment(self, normalized_command):
        healthcheck_patterns = (
            r"^redis-cli\s+ping\b",
            r"^pg_isready\b",
            r"^mysqladmin\s+ping\b",
            r"^rabbitmq-diagnostics\s+ping\b",
            r"^curl\b.*\b127\.0\.0\.1\b",
            r"^curl\b.*\blocalhost\b",
            r"^wget\b.*\b127\.0\.0\.1\b",
            r"^wget\b.*\blocalhost\b",
        )
        return any(re.search(pattern, normalized_command) for pattern in healthcheck_patterns)

    def _command_has_meaningful_setup_activity(self, command):
        """Detect whether a successful shell command materially changed the runtime environment."""
        if not command or not command.strip():
            return False
        if self._is_readonly_command(command):
            return False

        for _, normalized in self._iter_command_segments(command):
            if not normalized:
                continue

            if self._is_readonly_command(normalized):
                continue

            if self._is_runtime_service_segment(normalized):
                return True

            if self._is_setup_command(normalized):
                return True

            if self._segment_has_meaningful_setup_activity(normalized) and normalized.startswith(
                (
                    "./configure",
                    "configure ",
                    "meson ",
                    "mkdir ",
                    "rm ",
                    "cp ",
                    "mv ",
                    "ln ",
                    "chmod ",
                    "chown ",
                    "sed ",
                    "patch ",
                    "git apply",
                    "git checkout",
                    "python setup.py",
                )
            ):
                return True

        return False

    def _looks_like_test_executable(self, normalized_command):
        """Detect direct execution of built test binaries such as ./FooTests."""
        if not normalized_command:
            return False

        executable = normalized_command.split()[0]
        if not (executable.startswith("./") or executable.startswith("/") or "/" in executable):
            return False

        basename = executable.rsplit("/", 1)[-1]
        if basename in {"configure", "install-sh", "test-driver"}:
            return False

        test_suffixes = (
            "test",
            "tests",
            "unittest",
            "unittests",
            "spec",
            "specs",
            "test.exe",
            "tests.exe",
            "spec.exe",
            "specs.exe",
        )
        if basename.endswith(test_suffixes):
            return True

        return bool(
            re.search(r"(test|tests|unittest|unittests|spec|specs)", basename)
            and ("/test" in executable or "/tests" in executable or executable.startswith("./"))
        )

    def _observation_has_empty_test_run_signal(self, observation):
        """Detect successful commands that clearly did not run any tests."""
        if not observation:
            return False

        normalized = self._normalize_observation_text(observation).lower()
        empty_run_patterns = [
            r"no tests were found",
            r"no tests found",
            r"collected\s+0\s+items",
            r"ran\s+0\s+tests?",
            r"\b0\s+tests?\s+ran\b",
            r"\[no test files\]",
            r"no test cases matched",
            r"no tests to run",
            r"\b0\s+examples?,\s+0\s+failures?\b",
        ]
        return any(re.search(pattern, normalized, re.MULTILINE) for pattern in empty_run_patterns)

    def _observation_has_test_failure_signal(self, observation):
        """Detect output summaries that report nonzero test failures or errors."""
        if not observation:
            return False

        normalized_observation = self._normalize_observation_text(observation)
        failure_patterns = [
            r"\b[1-9]\d*[ \t]+(?:failed|failures?|errors?)\b",
            r"\b[1-9]\d*[ \t]+tests?[ \t]+failed\b",  # ctest "N tests failed" (audit [7])
            r"\b(?:failures?|errors?):\s*[1-9]\d*\b",
            r"\btests?[ \t]+run:\s*\d+,\s*failures?:\s*[1-9]\d*\b",
            r"\btests?[ \t]+run:\s*\d+,\s*failures?:\s*\d+,\s*errors?:\s*[1-9]\d*\b",
            r"\btest result:\s+failed\b",
            r"^\s*not ok\b",
            r"^\s*(?:FAILED|ERROR)\s+\S+",
            r"\bBUILD FAILURE\b",
            r"\bthere (?:was|were)\s+[1-9]\d*\s+(?:failure|failures|error|errors)\b",
        ]
        return any(
            re.search(pattern, normalized_observation, re.IGNORECASE | re.MULTILINE)
            for pattern in failure_patterns
        )

    def _observation_has_effective_test_signal(self, observation):
        """Detect observation text that strongly suggests real tests were executed."""
        if not observation:
            return False

        normalized_observation = self._normalize_observation_text(observation)
        # Some repositories expose their complete validation suite as
        # ``npm test -> grunt test``.  Grunt does not print a numeric
        # "N passed" summary: its binding success evidence is at least one
        # executed task followed by the terminal ``Done.`` marker.  Require
        # both markers so arbitrary npm output (or a bare "Done.") cannot
        # manufacture an effective test run.
        grunt_task_ran = re.search(
            r'^\s*Running\s+"[^"\n]+"\s+\([^)]+\)\s+task\s*$',
            normalized_observation,
            re.IGNORECASE | re.MULTILINE,
        )
        grunt_completed = re.search(
            r"^\s*Done\.\s*$",
            normalized_observation,
            re.IGNORECASE | re.MULTILINE,
        )
        if grunt_task_ran and grunt_completed:
            return True
        positive_patterns = [
            r"collected\s+[1-9]\d*\s+items",
            r"\b[1-9]\d*\s+tests?\s+collected\b",
            r"ran\s+[1-9]\d*\s+tests?",
            r"\b[1-9]\d*\s+passed\b",
            r"\b[1-9]\d*\s+tests?\s+passed\b",
            r"\b[1-9]\d*\s+failed\b",
            r"\b[1-9]\d*\s+skipped\b",
            r"tests\s+run:\s*[1-9]\d*,\s*failures:\s*\d+,\s*errors:\s*\d+,\s*skipped:\s*\d+",
            r"\bok\s+\([1-9]\d*\s+tests?,",
            r"\b[1-9]\d*\s+tests?,\s+[1-9]\d*\s+ran\b",
            r"\[=+\]\s+running\s+[1-9]\d*\s+tests?",
            r"test result:\s+(?:ok|failed)\.",
            r"\b[1-9]\d*%\s+tests\s+passed\b",
            r"^\s*ok\s+\S+\s+\d+(?:\.\d+)?s(?:\s|$)",
            r"\b[1-9]\d*\s+examples?,\s+\d+\s+failures?\b",
            r"\b[1-9]\d*\s+checks?,\s+\d+\s+ignored\b",
            r"^\s*passed:\s*[1-9]\d*\b",
            r"start\s+\d+:",
            r"suites:\s+\d+\s+of\s+[1-9]\d*\s+completed",
            r"asserts:\s+\d+\s+of\s+[1-9]\d*",
            r"^\s*#\s*subtest:",
            r"^\s*not ok\b",
        ]
        return any(
            re.search(pattern, normalized_observation, re.IGNORECASE | re.MULTILINE)
            for pattern in positive_patterns
        )

    def _observation_looks_like_help_text(self, observation):
        """Exclude `--help` or usage screens from being treated as test execution."""
        if not observation:
            return False

        normalized = self._normalize_observation_text(observation).lower()
        help_markers = [
            "usage:",
            "optional arguments:",
            "positional arguments:",
            "show this help",
        ]
        return any(marker in normalized for marker in help_markers)

    def _normalize_observation_text(self, observation):
        """Strip ANSI control codes and zero-width formatting artifacts before pattern matching."""
        normalized = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", observation)
        normalized = normalized.replace("\u200b", "")
        normalized = normalized.replace("\ufeff", "")
        return normalized

    def _looks_like_pip_install_command(self, command):
        if not command:
            return False

        normalized = " ".join(command.strip().split())
        pip_install_pattern = (
            r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
            r"(?:(?:python3?|/[\w./-]*python3?)\s+-m\s+pip|(?:pip3?|/[\w./-]*pip3?))\s+install\b"
        )
        return bool(re.match(pip_install_pattern, normalized))
    
    def _is_setup_command(self, command):
        """Determine whether a command performs environment setup or build configuration."""
        setup_keywords = [
            # Python
            'pip install', 'pip3 install', 'poetry install', 'uv pip', 'uv install',
            'conda install', 'pipenv install',
            # JavaScript/TypeScript
            'npm install', 'npm i ', 'yarn add', 'yarn install', 'pnpm install',
            # Rust
            'cargo build', 'cargo install',
            # Go
            'go mod download', 'go get ', 'go install',
            # Java
            'mvn install', 'mvn dependency:resolve', 'gradle build', 'gradlew build',
            # Ruby
            'bundle install', 'gem install',
            # PHP
            'composer install', 'composer require',
            # C/C++
            'make', 'cmake', 'ninja',
            # Dart
            'flutter pub get', 'dart pub get',
            # General
            'git clone', 'wget', 'curl', 'apt install', 'apt-get install', 'yum install',
        ]
        return any(keyword in command.lower() for keyword in setup_keywords)
    
    def generate_dockerfile(self, file_path="Dockerfile"):
        """Generates the final Dockerfile."""
        apt_bootstrap_instructions = build_dockerfile_apt_bootstrap_run_instructions()
        pip_bootstrap_instructions = build_dockerfile_pip_bootstrap_env_instructions()
        content = []
        if any("<<" in instruction for instruction in self.instructions):
            content.append("# syntax=docker/dockerfile:1")
        content.extend([
            f"FROM {self.base_image}",
            f"WORKDIR {self.workdir}",
            ""
        ])
        if pip_bootstrap_instructions:
            content.extend(pip_bootstrap_instructions)
            content.append("")
        if apt_bootstrap_instructions:
            content.extend(apt_bootstrap_instructions)
            content.append("")
        content.extend(self._render_instruction_for_dockerfile(instruction) for instruction in self.instructions)

        rendered = "\n".join(content)
        # Fix 3: final safety gate — guarantee one logical command per RUN. Repairs a
        # RUN-concatenation; rejects (raises) genuine corruption it cannot fix.
        from src.envstate.dockerfile_validate import validate_and_repair
        rendered = validate_and_repair(rendered)

        with open(file_path, "w") as f:
            f.write(rendered)

        print(f"Dockerfile successfully generated at {file_path}")
        return rendered
