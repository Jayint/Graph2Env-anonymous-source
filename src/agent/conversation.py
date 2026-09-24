"""Conversation retention and provider-safe request budgeting for ReAct."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_CONTEXT_CHARS = int(os.getenv("REACT_CONTEXT_MAX_CHARS", "150000"))
DEFAULT_OBSERVATION_CHARS = int(os.getenv("REACT_OBSERVATION_MAX_CHARS", "6000"))


@dataclass(frozen=True)
class ContextBudget:
    """Resolved input budget for one model call.

    The estimator counts UTF-8 bytes, a conservative upper bound when the
    exact provider tokenizer is unavailable.
    """

    model: str
    context_window_tokens: int
    input_budget_tokens: int
    output_reserve_tokens: int
    input_fraction: float
    estimator: str = "utf8_byte_upper_bound"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _known_context_window(model: str) -> int:
    if "deepseek-v4-flash" in str(model or "").casefold():
        return 1_048_576
    # Unknown routes must not silently inherit the 1M allowance.
    return 128_000


def resolve_context_budget(model: str) -> ContextBudget:
    """Resolve a model-aware budget, with environment overrides for gateways."""
    window = int(os.getenv("REACT_CONTEXT_WINDOW_TOKENS", str(_known_context_window(model))))
    fraction = min(0.95, max(0.50, float(os.getenv("REACT_CONTEXT_INPUT_FRACTION", "0.85"))))
    reserve = int(os.getenv("REACT_CONTEXT_OUTPUT_RESERVE_TOKENS", "16384"))
    reserve = max(1024, min(reserve, max(1024, window // 3)))
    return ContextBudget(
        model=str(model), context_window_tokens=window,
        input_budget_tokens=min(int(window * fraction), window - reserve),
        output_reserve_tokens=reserve, input_fraction=fraction,
    )


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Conservatively upper-bound request tokens without a provider tokenizer."""
    return sum(
        len(str(message.get("content") or "").encode("utf-8", errors="replace")) + 16
        for message in messages
    ) + 8


def _clip_utf8(value: str, max_bytes: int, *, label: str) -> str:
    raw = str(value or "").encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return str(value or "")
    marker = f"\n...[{label}: {len(raw) - max_bytes} or more UTF-8 bytes omitted]...\n".encode()
    payload = max(0, max_bytes - len(marker))
    head = payload // 2
    clipped = raw[:head] + marker + (raw[-(payload - head):] if payload - head else b"")
    return clipped.decode("utf-8", errors="ignore")


def fit_messages_to_token_budget(
    messages: list[dict[str, Any]], *, budget: ContextBudget,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit a request and, unlike the legacy slicer, also bound the newest turn."""
    copied = [dict(message) for message in messages]
    original_tokens = estimate_message_tokens(copied)
    removed = 0
    clipped_indexes: list[int] = []
    protected = 2 if len(copied) >= 2 and copied[0].get("role") == "system" else 1
    while estimate_message_tokens(copied) > budget.input_budget_tokens and len(copied) > protected + 2:
        del copied[protected:min(protected + 2, len(copied))]
        removed += 2

    candidates = [1] if protected >= 2 and len(copied) > 1 else []
    candidates.extend(index for index in range(len(copied) - 1, protected - 1, -1) if index not in candidates)
    for index in candidates:
        current = estimate_message_tokens(copied)
        if current <= budget.input_budget_tokens:
            break
        content = str(copied[index].get("content") or "")
        target_bytes = max(
            512,
            len(content.encode("utf-8", errors="replace")) - (current - budget.input_budget_tokens) - 64,
        )
        reduced = _clip_utf8(content, target_bytes, label="context safety projection")
        if reduced != content:
            copied[index]["content"] = reduced
            clipped_indexes.append(index)

    estimated = estimate_message_tokens(copied)
    if estimated > budget.input_budget_tokens:
        raise RuntimeError(
            "local context preflight could not fit the request: "
            f"estimated_tokens={estimated}, budget={budget.input_budget_tokens}"
        )
    return copied, {
        "original_estimated_tokens": original_tokens,
        "request_estimated_tokens": estimated,
        "removed_messages": removed,
        "clipped_message_indexes": clipped_indexes,
        **budget.to_dict(),
    }


def truncate_observation(text: str, *, edge_chars: int = DEFAULT_OBSERVATION_CHARS) -> str:
    """Keep the first and last *edge_chars* of a long command observation."""
    value = str(text or "")
    if len(value) <= edge_chars * 2:
        return value
    return (
        value[:edge_chars]
        + f"\n...[Observation truncated: {len(value) - edge_chars * 2} characters omitted]...\n"
        + value[-edge_chars:]
    )


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content") or "")) + 32 for message in messages)


def truncate_messages(
    messages: list[dict[str, Any]], *, max_chars: int = DEFAULT_CONTEXT_CHARS
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a request-sized copy while preserving system and initial context.

    Completed assistant/user pairs are removed from oldest to newest.  The
    caller's full conversation is never mutated.  If the immutable prefix alone
    exceeds the budget, its initial user payload is clipped at both ends so the
    repository/graph header and setup tail remain visible; exact artifacts are
    still available through inspect actions.
    """
    copied = [dict(message) for message in messages]
    original_chars = _message_chars(copied)
    if original_chars <= max_chars:
        return copied, {
            "original_chars": original_chars,
            "request_chars": original_chars,
            "removed_messages": 0,
        }

    protected = 2 if len(copied) >= 2 and copied[0].get("role") == "system" else 1
    removed = 0
    while _message_chars(copied) > max_chars and len(copied) > protected + 2:
        # Remove one old assistant+observation pair, mirroring Repo2Run's
        # oldest-message slicing while keeping the task prefix intact.
        del copied[protected:min(protected + 2, len(copied))]
        removed += 2

    if _message_chars(copied) > max_chars and protected >= 2:
        prefix = str(copied[1].get("content") or "")
        allowance = max(2000, max_chars - _message_chars([copied[0], *copied[2:]]) - 64)
        copied[1]["content"] = truncate_observation(prefix, edge_chars=max(1000, allowance // 2))

    request_chars = _message_chars(copied)
    return copied, {
        "original_chars": original_chars,
        "request_chars": request_chars,
        "removed_messages": removed,
    }


def setup_block(script: str, node_id: str) -> str | None:
    """Return one exact ``#@node``/``#@need``/``#@block`` setup block."""
    wanted = node_id.strip()
    if not wanted:
        return None
    pattern = re.compile(
        rf"(?ms)^(?P<head>#@(?:node|need|block)\s+{re.escape(wanted)}(?:\s|$).*?)"
        r"(?=^#@(?:node|need|block)\s+|^# ====================|\Z)"
    )
    match = pattern.search(script)
    return match.group("head").rstrip("\n") if match else None


def replace_setup_block(script: str, node_id: str, replacement: str) -> tuple[str, bool]:
    """Replace one stable-ID block without requiring byte-exact old text."""
    current = setup_block(script, node_id)
    if current is None:
        return script, False
    start = script.find(current)
    if start < 0:
        return script, False
    end = start + len(current)
    suffix = "\n" if end < len(script) and script[end:end + 1] == "\n" else ""
    updated = script[:start] + replacement.rstrip("\n") + suffix + script[end + len(suffix):]
    return updated, updated != script


def graph_summary(graph) -> dict[str, Any]:
    """Small structural header; exact graph remains available via inspect_graph."""
    by_type: dict[str, int] = {}
    by_state: dict[str, int] = {}
    workspaces: set[str] = set()
    for node in graph.nodes:
        by_type[node.type.value] = by_type.get(node.type.value, 0) + 1
        by_state[node.state.value] = by_state.get(node.state.value, 0) + 1
        workspace = node.workspace or node.data.get("project_path")
        if isinstance(workspace, str) and workspace:
            workspaces.add(workspace)
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "nodes_by_type": dict(sorted(by_type.items())),
        "nodes_by_state": dict(sorted(by_state.items())),
        "workspaces": sorted(workspaces),
    }
