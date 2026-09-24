"""Immutable raw trajectory and model-visible projection for ReAct Diet mode."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReActStep:
    """One executed turn.

    ``observation_raw`` is audit evidence and is never changed.  Only
    ``observation_view`` is eligible for reflection-based reduction.
    """

    turn: int
    assistant_raw: str
    action: dict[str, Any]
    observation_raw: str
    observation_view: str
    state: dict[str, Any]
    container: str
    compression: dict[str, Any] = field(default_factory=dict)

    def raw_record(self) -> dict[str, Any]:
        return asdict(self)

    def rendered_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "assistant", "content": self.assistant_raw},
            {
                "role": "user",
                "content": (
                    "### Observation\n" + self.observation_view
                    + "\n\n### Current state\n"
                    + json.dumps(self.state, ensure_ascii=False, indent=2)
                ),
            },
        ]


def message_chars(messages: list[dict[str, str]]) -> int:
    """Use the existing character-accounting convention for provider-neutral budgets."""
    return sum(len(str(message.get("content") or "")) + 32 for message in messages)


def render_messages(
    *, system_prompt: str, initial_context: str, steps: list[ReActStep]
) -> list[dict[str, str]]:
    """Render the mutable prompt projection without changing the raw ledger."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_context},
    ]
    for step in steps:
        messages.extend(step.rendered_messages())
    return messages


def local_window(steps: list[ReActStep], target_index: int, *, before: int, after: int) -> list[dict[str, Any]]:
    """Serialize the bounded AgentDiet window; only the target includes raw output."""
    start = max(0, target_index - before)
    stop = min(len(steps), target_index + after + 1)
    window: list[dict[str, Any]] = []
    for index in range(start, stop):
        step = steps[index]
        window.append({
            "turn": step.turn,
            "role": "target" if index == target_index else "context",
            "assistant_action": step.action,
            "observation_raw" if index == target_index else "observation_view": (
                step.observation_raw if index == target_index else step.observation_view
            ),
        })
    return window
