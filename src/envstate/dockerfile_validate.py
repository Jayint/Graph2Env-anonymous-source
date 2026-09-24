"""Fix 3 — Dockerfile validation/repair: one logical command per RUN.

A malformed ``RUN`` whose body contains a bare Dockerfile directive token in command
position (``RUN echo a && RUN echo b``) makes ``/bin/sh`` try to run ``RUN`` as a
program (``RUN: not found``) or silently collapses the recipe. This module detects
such bodies (while NOT false-positiving on a heredoc body that merely contains the
word ``RUN``), repairs the common ``RUN``-concatenation case by splitting it into
separate ``RUN`` instructions, and rejects anything it cannot safely repair.
"""
from __future__ import annotations

import re
from typing import List, Tuple

DIRECTIVES = {
    "RUN", "FROM", "COPY", "ADD", "WORKDIR", "ENV", "ARG", "CMD", "ENTRYPOINT",
    "LABEL", "EXPOSE", "VOLUME", "USER", "HEALTHCHECK", "ONBUILD", "STOPSIGNAL",
    "SHELL", "MAINTAINER",
}

_HEREDOC_OPEN_RE = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_]\w*)\2")


class MalformedDockerfileError(ValueError):
    """Raised when a RUN body holds a bare directive token that cannot be repaired."""


def _keyword(stripped: str) -> str:
    match = re.match(r"([A-Za-z]+)", stripped)
    return match.group(1).upper() if match else ""


def _heredoc_opens(line: str) -> List[Tuple[str, bool]]:
    return [(m.group(3), bool(m.group(1))) for m in _HEREDOC_OPEN_RE.finditer(line)]


def _command_position_first_tokens(body: str) -> List[str]:
    """Leading bareword of each top-level command-position segment (quote-aware)."""
    tokens: List[str] = []
    buf: List[str] = []
    i, n = 0, len(body)
    in_single = in_double = escape = False
    expecting = True

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    while i < n:
        c = body[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\":
            escape = True
            if expecting:
                flush()
                expecting = False
            i += 1
            continue
        if in_single:
            if c == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if c == '"':
                in_double = False
            i += 1
            continue
        if c == "'":
            in_single = True
            if expecting:
                flush()
                expecting = False
            i += 1
            continue
        if c == '"':
            in_double = True
            if expecting:
                flush()
                expecting = False
            i += 1
            continue
        if body.startswith("&&", i) or body.startswith("||", i):
            flush()
            expecting = True
            i += 2
            continue
        if c in ";|&\n()":
            flush()
            expecting = True
            i += 1
            continue
        if c in "<>":
            flush()
            expecting = False
            i += 1
            continue
        if c.isspace():
            if buf:
                flush()
                expecting = False
            i += 1
            continue
        if expecting:
            buf.append(c)
        i += 1
    flush()
    return [t for t in tokens if t]


def _split_command_separator_segments(body: str) -> List[str]:
    """Split on top-level command separators (``;`` ``&&`` ``||`` newline). Pipes kept."""
    segments: List[str] = []
    buf: List[str] = []
    i, n = 0, len(body)
    in_single = in_double = escape = False
    while i < n:
        c = body[i]
        if escape:
            buf.append(c)
            escape = False
            i += 1
            continue
        if c == "\\":
            buf.append(c)
            escape = True
            i += 1
            continue
        if in_single:
            buf.append(c)
            if c == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            buf.append(c)
            if c == '"':
                in_double = False
            i += 1
            continue
        if c == "'":
            in_single = True
            buf.append(c)
            i += 1
            continue
        if c == '"':
            in_double = True
            buf.append(c)
            i += 1
            continue
        if body.startswith("&&", i) or body.startswith("||", i):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if c in ";\n":
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    segments.append("".join(buf))
    return [s for s in segments if s.strip()]


def _run_instruction_bodies(text: str):
    """Yield each RUN instruction's scannable body (heredoc bodies excluded)."""
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _keyword(line.strip()) != "RUN":
            i += 1
            continue
        parts: List[str] = []
        cur = line
        while True:
            opens = _heredoc_opens(cur)
            parts.append(cur)
            if opens:
                delim, strip = opens[-1]
                i += 1
                while i < n:
                    candidate = lines[i].lstrip("\t") if strip else lines[i]
                    if candidate.strip() == delim:
                        break
                    i += 1
                break
            if cur.rstrip().endswith("\\"):
                i += 1
                if i < n:
                    cur = lines[i]
                    continue
                break
            break
        joined = " ".join(p.rstrip("\\").strip() for p in parts)
        joined = re.sub(r"^RUN\s+", "", joined, flags=re.IGNORECASE)
        yield joined
        i += 1


def find_malformed_runs(text: str) -> List[str]:
    """Return the bodies of RUN instructions that hold a bare directive token."""
    problems: List[str] = []
    for body in _run_instruction_bodies(text):
        for token in _command_position_first_tokens(body):
            if token.upper() in DIRECTIVES:
                problems.append(body)
                break
    return problems


def _repair_run_concatenations(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if _keyword(line.strip()) != "RUN" or _heredoc_opens(line) or line.rstrip().endswith("\\"):
            out.append(line)
            i += 1
            continue
        indent = re.match(r"\s*", line).group(0)
        body = re.sub(r"^\s*RUN\s+", "", line, flags=re.IGNORECASE)
        malformed = any(
            t.upper() in DIRECTIVES for t in _command_position_first_tokens(body)
        )
        if not malformed:
            out.append(line)
            i += 1
            continue
        for segment in _split_command_separator_segments(body):
            segment = re.sub(r"^RUN\s+", "", segment.strip(), flags=re.IGNORECASE).strip()
            if segment:
                out.append(f"{indent}RUN {segment}")
        i += 1
    return "\n".join(out)


def validate_and_repair(text: str) -> str:
    """Return a Dockerfile with malformed RUN concatenations split into separate RUNs.

    Raises ``MalformedDockerfileError`` if a non-``RUN`` directive token remains in a
    RUN body after repair (genuine corruption we must not silently emit).
    """
    if not find_malformed_runs(text):
        return text
    repaired = _repair_run_concatenations(text)
    remaining = find_malformed_runs(repaired)
    if remaining:
        raise MalformedDockerfileError(
            "RUN body contains a bare Dockerfile directive: " + remaining[0][:200]
        )
    return repaired
