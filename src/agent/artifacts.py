from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"react-builder:{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload) -> None:
    """Append one durable context/trajectory audit record."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _message_content(response: Any) -> str:
    """Extract the unmodified OpenAI-compatible choices[0].message.content."""
    if isinstance(response, dict):
        try:
            value = response["choices"][0]["message"]["content"]
            return "" if value is None else str(value)
        except (KeyError, IndexError, TypeError):
            # Unit-test doubles may deliberately contain only a readable marker.
            return str(response.get("raw") or response.get("provider_payload") or "")
    try:
        value = response.choices[0].message.content
        return "" if value is None else str(value)
    except (AttributeError, IndexError, TypeError, KeyError):
        return ""


def write_llm_exchange(log_dir: Path, index: int, *, messages: list[dict],
                       parsed_output: str, raw_response: Any, retry_attempt: int) -> Path:
    """Persist one physical call in the human-readable experiment log format."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{index}.log"
    system_content = "\n\n".join(
        str(message.get("content") or "") for message in messages
        if message.get("role") == "system"
    )
    # Keep the historical section name requested by the experiment format, but
    # include the complete non-system conversation so assistant history is not
    # silently missing now that ReAct uses an ordinary growing trajectory.
    user_content = "\n\n".join(
        f"[{str(message.get('role') or 'unknown').upper()}]\n"
        f"{str(message.get('content') or '')}"
        for message in messages if message.get("role") != "system"
    )
    text = (
        "=====LLM input======\n"
        "system content:\n"
        f"{system_content}\n\n"
        "user content:\n"
        f"{user_content}\n\n"
        "====LLM output====\n"
        f"{_message_content(raw_response)}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path
