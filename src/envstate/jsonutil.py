from __future__ import annotations
import json
from typing import Any, Optional

# Bounds that keep parsing of pathological/truncated LLM output cheap. Restarting
# the brace scan at every "{" is O(n^2); a single malformed response (e.g. thousands
# of unbalanced "{") would otherwise stall the per-observation loop. Real proposals
# are small, so these caps never affect legitimate input.
_MAX_SCAN_CHARS = 200_000
_MAX_OBJECT_STARTS = 256


def extract_json_object(text: Optional[str]) -> Optional[dict[str, Any]]:
    """Extract the first balanced top-level JSON object from arbitrary LLM text.

    Tolerates a ```json fence, leading commentary, and trailing prose. Returns
    the parsed dict, or None if no valid object is found.

    Note: if the outermost object is truncated/unterminated but contains a fully
    balanced NESTED object, that inner object may be returned. Callers that need a
    specific top-level shape must validate the returned dict's keys (the Maintainer
    and Supervisor do — a missing key yields a safe no-op rather than wrong action).
    """
    if not text:
        return None
    if len(text) > _MAX_SCAN_CHARS:
        text = text[:_MAX_SCAN_CHARS]
    start = text.find("{")
    attempts = 0
    while start != -1 and attempts < _MAX_OBJECT_STARTS:
        attempts += 1
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
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
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break  # malformed; try the next '{'
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None
