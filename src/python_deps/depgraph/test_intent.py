"""Deterministic discovery of dependency groups needed by the pytest goal.

The graph keeps optional/test dependencies dormant by default.  This module
activates only groups backed by repository test evidence; it never unions every
optional extra, which would make heavyweight or mutually exclusive extras part
of every environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from src.python_deps.evidence import (
    collect_python_dependency_evidence,
    discover_test_project_roots,
)


_TEST_GROUPS = frozenset({"test", "tests", "testing", "ci"})
_DEV_GROUPS = frozenset({"dev", "development"})
_TEST_TOOL_NAMES = frozenset({"pytest", "tox", "nox", "hypothesis"})
_PYTEST_IMPORT_MODE_RE = re.compile(
    r"--import-mode(?:=|\s+)(?:prepend|append|importlib)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TestDependencyIntent:
    needed_groups: frozenset[str] = frozenset()
    pytest_addopts: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def _group_from_source(source: str) -> str:
    for marker in (
        "optional-dependencies.",
        "extras_require.",
        "dependency-groups.",
        "test-requirements.",
        "tox-deps.",
        "hatch-test-deps.",
    ):
        if marker in source:
            return source.rsplit(marker, 1)[-1]
    return ""


def _has_pytest_surface(root: Path) -> bool:
    if any((root / name).is_file() for name in ("pytest.ini", "tox.ini", "conftest.py")):
        return True
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for pattern in ("test_*.py", "*_test.py"):
            if next(tests_dir.rglob(pattern), None) is not None:
                return True
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            return False
        if "[tool.pytest" in text or "pytest.ini_options" in text:
            return True
    # Package-based monorepos often keep both pyproject.toml and tests below
    # packages/* while the repository root has no Python manifest of its own.
    if any(project_root != root for project_root in discover_test_project_roots(root)):
        return True
    return False


def _is_test_tool(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized in _TEST_TOOL_NAMES or normalized.startswith("pytest-")


def _needs_importlib_mode(root: Path) -> bool:
    """Return whether independent projects expose colliding ``tests`` packages.

    Pytest's default prepend mode imports every package-style ``tests`` directory
    as the same top-level module.  In a monorepo that contains two or more such
    projects, the first imported package shadows the rest and collection fails
    with misleading ``tests.test_*`` import errors.  Importlib mode is pytest's
    namespace-safe policy for this layout.  Keep the trigger deliberately narrow:
    namespace-style test directories and single-project repositories retain the
    repository/default pytest policy.
    """
    project_roots = discover_test_project_roots(root)
    config_roots = {root, *project_roots}
    for config_root in config_roots:
        for name in ("pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"):
            path = config_root / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if _PYTEST_IMPORT_MODE_RE.search(text):
                return False

    package_test_roots: set[Path] = set()
    for project_root in project_roots:
        tests_dir = project_root / "tests"
        if not (tests_dir / "__init__.py").is_file():
            continue
        if any(next(tests_dir.rglob(pattern), None) is not None
               for pattern in ("test_*.py", "*_test.py")):
            package_test_roots.add(tests_dir)
    return len(package_test_roots) >= 2


def _hatch_test_features(root: Path) -> set[str]:
    features: set[str] = set()
    if tomllib is None:
        return features
    for project_root in discover_test_project_roots(root):
        path = project_root / "pyproject.toml"
        if not path.is_file():
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        envs = data.get("tool", {}).get("hatch", {}).get("envs", {}) or {}
        if not isinstance(envs, dict):
            continue
        for env_name, env_data in envs.items():
            normalized = str(env_name).lower().replace("_", "-")
            if normalized not in {"test", "tests", "testing", "hatch-test"}:
                continue
            if not isinstance(env_data, dict):
                continue
            for feature in env_data.get("features", []) or []:
                if isinstance(feature, str) and feature.strip():
                    features.add(feature.strip().lower())
    return features


def discover_test_dependency_intent(repo_path: str | Path) -> TestDependencyIntent:
    """Select test-related dependency groups using static repository evidence."""
    root = Path(repo_path)
    if not _has_pytest_surface(root):
        return TestDependencyIntent()

    evidence = collect_python_dependency_evidence(root)
    grouped: dict[str, list[str]] = {}
    sources: dict[str, set[str]] = {}
    for requirement in evidence.declared_dependencies:
        if requirement.kind != "optional_dependency":
            continue
        group = _group_from_source(requirement.source).lower()
        if not group:
            continue
        grouped.setdefault(group, []).append(requirement.name)
        sources.setdefault(group, set()).add(requirement.source)

    selected: set[str] = set()
    reasons: list[str] = []
    for group, names in sorted(grouped.items()):
        if group in _TEST_GROUPS:
            selected.add(group)
        elif group in _DEV_GROUPS and any(_is_test_tool(name) for name in names):
            selected.add(group)
        else:
            continue
        reason_sources = ",".join(sorted(sources[group]))
        reasons.append(f"{group}: pytest goal + {reason_sources}")

    for group in sorted(_hatch_test_features(root)):
        if group not in grouped:
            continue
        selected.add(group)
        reason_sources = ",".join(sorted(sources[group]))
        reasons.append(f"{group}: hatch test feature + {reason_sources}")

    pytest_addopts: tuple[str, ...] = ()
    if _needs_importlib_mode(root):
        pytest_addopts = ("--import-mode=importlib",)
        reasons.append(
            "pytest importlib mode: multiple independent package-style tests roots"
        )

    return TestDependencyIntent(
        needed_groups=frozenset(selected),
        pytest_addopts=pytest_addopts,
        reasons=tuple(reasons),
    )
