from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    turn: int
    container: str
    action: str
    payload: dict[str, Any]
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionState:
    graph: Any
    script: str
    graph_version: int = 0
    script_version: int = 0
    cursor: str = "start"
    active_candidate: str | None = None
    events: list[Event] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=lambda: {
        "version": 1, "services": [], "environment": {}, "capabilities": [],
    })
