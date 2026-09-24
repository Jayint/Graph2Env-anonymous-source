"""Check-quality predicate (Stage 2): reject checks structurally incapable of detecting
a missing dependency. Pure; no envstate import; no NodeType needed."""
from __future__ import annotations

import re

_TRIVIAL_HEADS = {"true", ":", "echo", "ls", "pwd", "cd", "printf"}
# `test`/`[` are conditional: `test -f <path>` detects file absence (exit 1 when absent),
# but constant/string predicates (`test 1 = 1`, `test -n ok`) always succeed and cannot
# detect a missing dependency. They are non-trivial ONLY with a file-test operator.
_FILE_TEST_OPS = {"-e", "-f", "-d", "-r", "-w", "-x", "-s", "-L", "-h", "-b", "-c", "-p", "-S"}
_SHELL_VAR = r"[A-Za-z_][A-Za-z0-9_]*"
_QUOTED_DIRECT_VAR = rf'"(?:\${_SHELL_VAR}|\$\{{{_SHELL_VAR}\}})"'
_NONEMPTY_VAR_TEST = re.compile(
    rf"^(?:test\s+-n\s+{_QUOTED_DIRECT_VAR}|\[\s+-n\s+{_QUOTED_DIRECT_VAR}\s*\])$"
)


def check_can_detect_absence(check_command: str) -> bool:
    """False when the check is structurally trivial (would pass without the install)."""
    cmd = (check_command or "").strip()
    if not cmd:
        return False
    toks = cmd.split()
    head = toks[0]
    if head in _TRIVIAL_HEADS:
        return False
    if head in ("test", "["):
        # A file predicate detects a missing path.  A strict, double-quoted
        # direct variable ``-n`` predicate detects an unset/empty Config value.
        # Constant, single-quoted and unquoted forms remain rejected: they
        # either always succeed or are not reliable under an unset variable.
        return (
            any(t in _FILE_TEST_OPS for t in toks[1:])
            or _NONEMPTY_VAR_TEST.fullmatch(cmd) is not None
        )
    return True
