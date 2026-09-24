"""Deterministic safety summaries for unusually large ReAct observations.

The raw command output is immutable audit evidence.  This module only builds a
bounded model-visible projection for a small allow-list of operations whose
output structure is known by the host.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_FALLBACK_CHARS = int(os.getenv("REACT_OBSERVATION_FALLBACK_CHARS", "50000"))
DEFAULT_SUMMARY_CHARS = int(os.getenv("REACT_OBSERVATION_SUMMARY_CHARS", "12000"))
SUPPORTED_ACTIONS = frozenset({
    "run_shell",
    "run_collect",
    "run_setup",
    "run_block",
    "validate_candidate",
    "run_clean_replay",
})

_MARKER_RE = re.compile(
    r"^(?:SUCCESS|FAILURE|COLLECT_(?:SUCCESS|FAILURE)|SETUP_(?:SUCCESS|FAILURE)|"
    r"CLEAN_SETUP_(?:SUCCESS|FAILURE)|CLEAN_COLLECT_(?:SUCCESS|FAILURE)|"
    r"FINALIZATION_PENDING|PERSISTENCE_REQUIRED|FAILURE_FOCUS|CANDIDATE_DELTA|"
    r"REPLAY_STATE_DIFFERENCE_HINT)(?:\b|$)"
)
_SUMMARY_RE = re.compile(
    r"(?:\b\d+\s+(?:tests?|items?)\s+collected\b|"
    r"\b(?:passed|failed|errors?|warnings?|skipped|deselected)\b.*\bin\s+[\d.]+s\b|"
    r"short test summary info|interrupted:|candidate promoted:|"
    r"graph_changed_nodes=|runtime_changed=|active_only_packages=|"
    r"clean_only_packages=|active_only_worktree_changes=|"
    r"successfully installed|failed to build|unable to fetch|"
    r"sub-process .+ returned an error code)",
    re.IGNORECASE,
)
_DIAGNOSTIC_RE = re.compile(
    r"(?:^E:\s|^Err:\s|^ERROR\b|^FAILED\b|\berror:|\bfatal:|Traceback|"
    r"ModuleNotFoundError|ImportError|CalledProcessError|Exception\b|"
    r"command not found|No such file or directory|connection (?:refused|failed)|"
    r"could not connect|cannot connect|unexpected EOF|Bad Gateway|"
    r"dependency conflict|resolution impossible|returned non-zero exit status)",
    re.IGNORECASE,
)
_STATE_BLOCK_MARKERS = frozenset({
    "CANDIDATE_DELTA",
    "REPLAY_STATE_DIFFERENCE_HINT (diagnostic only)",
    "FINALIZATION_PENDING",
    "PERSISTENCE_REQUIRED",
    "FAILURE_FOCUS",
})
_TEST_ITEM_RE = re.compile(r"^[^\s].*::[^\s]+$")
_BATCH_HEADER_RE = re.compile(r"(?m)^\[(\d+)\]\s+([A-Za-z_]+)\n")


@dataclass
class RuleFallbackRecord:
    turn: int
    action: str
    action_path: str
    eligible: bool
    applied: bool
    reason: str
    original_chars: int
    reduced_chars: int
    saved_chars: int
    raw_ref: str
    raw_sha256: str
    threshold_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observation_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _unique(lines: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.rstrip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _state_blocks(lines: list[str]) -> list[str]:
    selected: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() not in _STATE_BLOCK_MARKERS:
            continue
        stop = min(len(lines), index + 80)
        for cursor in range(index, stop):
            current = lines[cursor]
            if cursor > index and _MARKER_RE.match(current.strip()):
                break
            selected.append(current)
    return _unique(selected)


def _normalized_error_signature(line: str) -> str:
    value = " ".join(line.strip().split())
    value = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", value)
    value = re.sub(r"/(?:[^\s:]+/)+[^\s:]+", "<path>", value)
    return value[:800]


def _diagnostic_index(lines: list[str]) -> tuple[list[str], list[str]]:
    """Index distinct diagnostics and retain their first and final context."""
    groups: dict[str, dict[str, int]] = {}
    first_indexes: list[int] = []
    diagnostic_indexes: list[int] = []
    for index, line in enumerate(lines):
        if not _DIAGNOSTIC_RE.search(line):
            continue
        diagnostic_indexes.append(index)
        signature = _normalized_error_signature(line)
        if signature not in groups:
            groups[signature] = {"count": 0, "first": index + 1, "last": index + 1}
            first_indexes.append(index)
        groups[signature]["count"] += 1
        groups[signature]["last"] = index + 1
    index_lines = [
        f"count={meta['count']} first_line={meta['first']} last_line={meta['last']} signature={signature}"
        for signature, meta in groups.items()
    ]
    contexts: list[str] = []
    for index in first_indexes + (diagnostic_indexes[-1:] if diagnostic_indexes else []):
        before, after = (4, 12) if diagnostic_indexes and index == diagnostic_indexes[-1] else (2, 5)
        contexts.extend(
            line for line in lines[max(0, index - before):min(len(lines), index + after)]
            if not (_TEST_ITEM_RE.match(line) and not _DIAGNOSTIC_RE.search(line))
        )
    return index_lines, _unique(contexts)


def _append_section(parts: list[str], title: str, lines: list[str], *, budget: int) -> int:
    if not lines or budget <= len(title) + 2:
        return budget
    section = [title]
    used = len(title) + 1
    included = 0
    for line in lines:
        clipped = line if len(line) <= 1200 else line[:1200] + " ...[line clipped]"
        cost = len(clipped) + 1
        if used + cost > budget:
            break
        section.append(clipped)
        used += cost
        included += 1
    if included:
        parts.append("\n".join(section))
        return budget - used
    return budget


def summarize_observation(
    *, action: str, raw: str, turn: int, action_path: str, raw_ref: str,
    threshold_chars: int = DEFAULT_FALLBACK_CHARS,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
) -> tuple[str, RuleFallbackRecord]:
    """Return a bounded deterministic view when an allow-listed output is huge."""
    original = str(raw or "")
    digest = observation_sha256(original)
    eligible = action in SUPPORTED_ACTIONS
    record = RuleFallbackRecord(
        turn=turn,
        action=action,
        action_path=action_path,
        eligible=eligible,
        applied=False,
        reason="unsupported_action" if not eligible else "below_threshold",
        original_chars=len(original),
        reduced_chars=len(original),
        saved_chars=0,
        raw_ref=raw_ref,
        raw_sha256=digest,
        threshold_chars=threshold_chars,
    )
    if not eligible or len(original) <= threshold_chars:
        return original, record

    lines = original.splitlines()
    test_items = [line for line in lines if _TEST_ITEM_RE.match(line) and not _DIAGNOSTIC_RE.search(line)]
    markers = _unique([line for line in lines if _MARKER_RE.match(line.strip())])
    summaries = _unique([line for line in lines if _SUMMARY_RE.search(line)])
    states = _state_blocks(lines)
    diagnostic_index, diagnostics = _diagnostic_index(lines)
    nonprogress_tail = [
        line for line in lines[-120:]
        if line.strip()
        and not _TEST_ITEM_RE.match(line)
        and not re.match(
            r"^(?:Get|Hit|Ign):\d+|^Downloading\s|^Requirement already satisfied:|"
            r"^Collecting\s|^Preparing metadata|^Installing collected packages",
            line.strip(),
        )
    ]

    header = (
        "OBSERVATION_SAFETY_SUMMARY\n"
        "summary_source=host_deterministic_projection\n"
        f"action={action}\n"
        f"turn={turn}\n"
        f"action_path={action_path}\n"
        f"original_chars={len(original)}\n"
        f"raw_ref={raw_ref}\n"
        f"raw_sha256={digest}\n"
        f"test_item_lines_omitted={len(test_items)}\n"
        f"unique_error_signatures={len(diagnostic_index)}\n"
        "Full output is immutable and recoverable with inspect_observation."
    )
    parts = [header]
    budget = max(2000, summary_chars) - len(header) - 180
    budget = _append_section(parts, "STATUS_MARKERS", markers, budget=budget)
    budget = _append_section(parts, "ERROR_SIGNATURE_INDEX", diagnostic_index, budget=budget)
    budget = _append_section(parts, "DIAGNOSTIC_CONTEXT", diagnostics, budget=budget)
    budget = _append_section(parts, "RESULT_SUMMARIES", summaries, budget=budget)
    budget = _append_section(parts, "STATE_AND_HANDOFF", states, budget=budget)
    _append_section(parts, "NON_PROGRESS_TAIL", _unique(nonprogress_tail), budget=budget)
    reduced = "\n\n".join(parts)
    if len(reduced) >= len(original):
        record.reason = "no_savings"
        return original, record

    record.applied = True
    record.reason = "applied"
    record.reduced_chars = len(reduced)
    record.saved_chars = len(original) - len(reduced)
    return reduced, record


def apply_rule_fallback(
    *, action: dict[str, Any], raw: str, turn: int, raw_ref: str,
    threshold_chars: int = DEFAULT_FALLBACK_CHARS,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
    action_path: str = "root",
) -> tuple[str, list[RuleFallbackRecord]]:
    """Apply rules to one action, including eligible children of action_batch."""
    kind = str(action.get("action") or "")
    if kind != "action_batch":
        view, record = summarize_observation(
            action=kind,
            raw=raw,
            turn=turn,
            action_path=action_path,
            raw_ref=raw_ref,
            threshold_chars=threshold_chars,
            summary_chars=summary_chars,
        )
        return view, [record] if record.eligible else []

    actions = action.get("actions")
    if not isinstance(actions, list):
        return raw, []
    matches = list(_BATCH_HEADER_RE.finditer(raw))
    if not matches:
        return raw, []
    segments: list[tuple[re.Match[str], int, str, dict[str, Any]]] = []
    for position, match in enumerate(matches):
        index = int(match.group(1)) - 1
        end = matches[position + 1].start() if position + 1 < len(matches) else len(raw)
        body = raw[match.end():end].rstrip("\n")
        subaction = actions[index] if 0 <= index < len(actions) and isinstance(actions[index], dict) else {}
        segments.append((match, index, body, subaction))

    eligible = [segment for segment in segments if str(segment[3].get("action") or "") in SUPPORTED_ACTIONS]
    total_weight = sum(max(1, len(segment[2])) for segment in eligible)
    allocations = {
        index: max(2000, int(summary_chars * max(1, len(body)) / max(1, total_weight)))
        for _match, index, body, _subaction in eligible
    }
    pieces: list[str] = []
    records: list[RuleFallbackRecord] = []
    force_batch_projection = len(raw) > threshold_chars
    for match, index, body, subaction in segments:
        allocation = allocations.get(index, summary_chars)
        subview, subrecords = apply_rule_fallback(
            action=subaction,
            raw=body,
            turn=turn,
            raw_ref=raw_ref,
            threshold_chars=min(threshold_chars, allocation) if force_batch_projection else threshold_chars,
            summary_chars=allocation,
            action_path=f"{action_path}.actions[{index}]",
        )
        pieces.append(match.group(0).rstrip("\n") + "\n" + subview)
        records.extend(subrecords)
    return "\n\n".join(pieces), records


def observation_slice(
    raw: str,
    *,
    turn: int,
    raw_ref: str,
    match: str = "",
    offset: int = 0,
    limit: int = 6000,
    max_limit: int = 12000,
) -> str:
    """Return an exact bounded slice of a prior raw observation."""
    value = str(raw or "")
    requested = max(1, min(int(limit), max_limit))
    start = max(0, int(offset))
    literal = str(match or "")
    if literal:
        found = value.casefold().find(literal.casefold())
        if found < 0:
            return (
                "OBSERVATION_MATCH_NOT_FOUND\n"
                f"turn={turn}\nmatch={literal}\nraw_ref={raw_ref}\n"
                f"raw_chars={len(value)}\nraw_sha256={observation_sha256(value)}"
            )
        start = max(0, found - min(1000, requested // 4))
    end = min(len(value), start + requested)
    return (
        "OBSERVATION_EXACT_SLICE\n"
        f"turn={turn}\noffset={start}\nend={end}\nraw_chars={len(value)}\n"
        f"raw_ref={raw_ref}\nraw_sha256={observation_sha256(value)}\n"
        "<<<RAW_SLICE\n"
        + value[start:end]
        + "\nRAW_SLICE"
    )
