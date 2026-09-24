"""Shared deterministic repository scanning helpers for ecosystem providers."""
from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache


SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "target",
    "dist", "build", "__pycache__", ".gradle",
})

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "testdata"})
_FIXTURE_DIRS = frozenset({"fixture", "fixtures", "__fixtures__"})


def is_test_fixture_path(relative_path: str) -> bool:
    """Return whether a repository-relative path belongs to a test fixture tree."""
    parts = {
        part.lower()
        for part in relative_path.replace("\\", "/").split("/")[:-1]
    }
    return bool(parts & _TEST_DIRS) and bool(parts & _FIXTURE_DIRS)


def find_files(repo_path: str, names: set[str]) -> tuple[str, ...]:
    found: list[str] = []
    lowered = {name.lower() for name in names}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for filename in sorted(files):
            if filename.lower() not in lowered:
                continue
            rel = os.path.relpath(os.path.join(root, filename), repo_path)
            found.append(rel.replace(os.sep, "/"))
    return tuple(found)


def find_source_files(repo_path: str, root: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    base = repo_path if root == "." else os.path.join(repo_path, root)
    found: list[str] = []
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for filename in sorted(files):
            if filename.lower().endswith(suffixes):
                rel = os.path.relpath(os.path.join(current, filename), repo_path)
                found.append(rel.replace(os.sep, "/"))
    return tuple(found)


def language_requirement(language_requirements, *languages):
    wanted = set(languages)
    return next(
        (item for item in language_requirements if item.language in wanted),
        None,
    )


def numeric_version(constraint: str | None, default: str) -> str:
    if not constraint:
        return default
    match = re.search(r"\d+(?:\.\d+){0,2}", constraint)
    return match.group(0) if match else default


@lru_cache(maxsize=None)
def _tree_sitter_parser(language: str):
    try:
        from tree_sitter_languages import get_parser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return get_parser(language)
    except Exception:
        return None


def parse_syntax_tree(text: str, language: str):
    """Return a tree-sitter tree when the optional parser bundle is available."""
    parser = _tree_sitter_parser(language)
    if parser is None:
        return None
    try:
        return parser.parse(text.encode("utf-8"))
    except Exception:
        return None


def walk_syntax_tree(node):
    """Yield every syntax node in source order without recursion depth risk."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def syntax_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def quoted_string_value(raw: str) -> str | None:
    raw = raw.strip()
    if len(raw) < 2 or raw[0] not in {'"', "'", "`"} or raw[-1] != raw[0]:
        return None
    return raw[1:-1]
