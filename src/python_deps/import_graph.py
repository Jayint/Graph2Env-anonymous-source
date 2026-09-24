from __future__ import annotations

import ast
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from .models import ImportFinding


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
    "vendor",
}

SOURCE_ROOT_NAMES = {"src", "lib", "python"}
MAX_PYTHON_FILES = 1000
MAX_FILE_BYTES = 500_000


def collect_project_local_modules(repo_path: str | Path) -> list[str]:
    root = Path(repo_path)
    module_names: set[str] = set()
    for source_root in _candidate_source_roots(root):
        if not source_root.is_dir():
            continue
        for child in source_root.iterdir():
            if child.name.startswith(".") or child.name in EXCLUDED_DIRS:
                continue
            if child.is_file() and child.suffix == ".py":
                module_names.add(child.stem)
            elif child.is_dir() and _looks_like_python_module_dir(child):
                module_names.add(child.name)
    return sorted(module_names)


def scan_imports(repo_path: str | Path) -> tuple[list[ImportFinding], list[str], list[str]]:
    root = Path(repo_path)
    project_local_modules = set(collect_project_local_modules(root))
    stdlib_modules = _stdlib_module_names()
    imports_by_name: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []

    for python_file in _iter_python_files(root):
        relative_path = python_file.relative_to(root).as_posix()
        try:
            content = python_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = python_file.read_text(encoding="latin-1")
            except OSError as error:
                errors.append(f"{relative_path}: {error}")
                continue
        except OSError as error:
            errors.append(f"{relative_path}: {error}")
            continue

        try:
            discovered_imports = _imports_from_ast(content)
        except SyntaxError:
            discovered_imports = _imports_from_regex(content)
            errors.append(f"{relative_path}: syntax error; used regex import fallback")

        for import_name in discovered_imports:
            imports_by_name[import_name].add(relative_path)

    findings = []
    for import_name, source_files in sorted(imports_by_name.items()):
        top_level = import_name.split(".", 1)[0]
        if top_level in project_local_modules:
            classification = "project_local"
        elif top_level in stdlib_modules:
            classification = "stdlib"
        else:
            classification = "external"
        findings.append(
            ImportFinding(
                import_name=top_level,
                classification=classification,
                source_files=tuple(sorted(source_files)),
            )
        )

    deduped = _dedupe_findings(findings)
    return deduped, sorted(project_local_modules), errors


def collect_pydeps_evidence(repo_path: str | Path) -> dict[str, object]:
    """Record pydeps availability without making it required for Phase 1."""
    executable = shutil.which("pydeps")
    if not executable:
        return {
            "available": False,
            "ran": False,
            "reason": "pydeps executable not found",
        }
    return {
        "available": True,
        "ran": False,
        "reason": "Phase 1 uses AST scanning as primary evidence; pydeps is optional supplemental evidence",
        "executable": executable,
    }


def _candidate_source_roots(root: Path) -> list[Path]:
    roots = [root]
    for name in SOURCE_ROOT_NAMES:
        candidate = root / name
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _looks_like_python_module_dir(path: Path) -> bool:
    if (path / "__init__.py").is_file():
        return True
    try:
        return any(child.suffix == ".py" for child in path.iterdir() if child.is_file())
    except OSError:
        return False


def _iter_python_files(root: Path):
    yielded = 0
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in EXCLUDED_DIRS and not directory.startswith(".")
        ]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = Path(current_root) / filename
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path
            yielded += 1
            if yielded >= MAX_PYTHON_FILES:
                return


def _imports_from_ast(content: str) -> set[str]:
    tree = ast.parse(content)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                imports.add(node.module.split(".", 1)[0])
    return imports


def _imports_from_regex(content: str) -> set[str]:
    imports: set[str] = set()
    for line in content.splitlines():
        import_match = re.match(r"^\s*import\s+([A-Za-z_][\w.]*)", line)
        if import_match:
            imports.add(import_match.group(1).split(".", 1)[0])
            continue
        from_match = re.match(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", line)
        if from_match:
            imports.add(from_match.group(1).split(".", 1)[0])
    return imports


def _stdlib_module_names() -> set[str]:
    modules = set(getattr(sys, "stdlib_module_names", set()))
    modules.update(sys.builtin_module_names)
    modules.update({"typing", "pathlib", "dataclasses", "unittest"})
    return modules


def _dedupe_findings(findings: list[ImportFinding]) -> list[ImportFinding]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for finding in findings:
        grouped[(finding.import_name, finding.classification)].update(finding.source_files)
    return [
        ImportFinding(import_name=name, classification=classification, source_files=tuple(sorted(files)))
        for (name, classification), files in sorted(grouped.items())
    ]
