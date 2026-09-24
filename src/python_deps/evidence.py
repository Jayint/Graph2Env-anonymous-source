from __future__ import annotations

import ast
import configparser
import glob
import os
import re
import shlex
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from packaging.requirements import InvalidRequirement, Requirement

from .import_graph import collect_pydeps_evidence, scan_imports
from .import_mapping import map_import_to_package
from .models import (
    ImportPackageMapping,
    PythonDependencyEvidence,
    PythonRequirement,
    PythonVersionRequirement,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


_PROJECT_SCAN_EXCLUDES = frozenset({
    ".git", ".hg", ".svn", ".tox", ".nox", ".venv", "venv", "env",
    "node_modules", "vendor", "build", "dist", "site-packages", "__pycache__",
    "docs", "examples",
})
_MAX_PROJECT_DEPTH = 4


def _has_packaging_metadata(path: Path) -> bool:
    """Whether *path* declares an installable Python project.

    A ``pyproject.toml`` or ``setup.cfg`` may be only tool configuration
    (pytest, coverage, formatters, an empty setuptools module list, ...).  File
    presence alone is therefore not enough to justify an editable-install
    obligation.  Keep the test-project discovery and depgraph builder on one
    content-aware definition of packaging intent.
    """
    if (path / "setup.py").is_file():
        return True

    setup_cfg = path / "setup.cfg"
    if setup_cfg.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(setup_cfg, encoding="utf-8")
        except (OSError, configparser.Error):
            pass
        else:
            if parser.has_section("metadata"):
                return True

    pyproject = path / "pyproject.toml"
    if not pyproject.is_file() or tomllib is None:
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    # ``[project]`` is PEP 621 packaging intent.  ``[build-system]`` also
    # permits a backend to supply legacy or dynamic metadata.  Tool tables on
    # their own are configuration, not evidence that ``pip -e`` is valid.
    return (
        isinstance(data.get("project"), dict)
        or isinstance(data.get("build-system"), dict)
    )


def _has_python_tests(path: Path) -> bool:
    """Whether a project subtree contains pytest-shaped files.

    The walk is bounded and prunes generated/vendor directories.  It does not
    follow symlinks, so discovery cannot escape the repository or loop.
    """
    for current, dirs, files in os.walk(path, followlinks=False):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(path).parts)
        except ValueError:
            continue
        dirs[:] = [
            name for name in dirs
            if name not in _PROJECT_SCAN_EXCLUDES and not name.startswith(".")
        ]
        if depth > _MAX_PROJECT_DEPTH + 2:
            dirs[:] = []
            continue
        if any(
            (name.startswith("test_") or name.endswith("_test.py"))
            and name.endswith(".py")
            for name in files
        ):
            return True
    return False


def discover_test_project_roots(repo_path: str | Path) -> tuple[Path, ...]:
    """Discover installable Python project roots relevant to the pytest goal.

    The repository root keeps its historical treatment when it has packaging
    metadata.  Nested projects are included only when their own subtree has
    pytest-shaped tests.  This captures package-based monorepos without pulling
    unrelated docs/examples/services into the dependency closure.
    """
    root = Path(repo_path)
    found: list[Path] = []
    if _has_packaging_metadata(root):
        found.append(root)

    for current, dirs, _files in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            rel_parts = current_path.relative_to(root).parts
        except ValueError:
            continue
        dirs[:] = [
            name for name in dirs
            if name not in _PROJECT_SCAN_EXCLUDES and not name.startswith(".")
        ]
        if len(rel_parts) >= _MAX_PROJECT_DEPTH:
            dirs[:] = []
        if current_path == root or not _has_packaging_metadata(current_path):
            continue
        if _has_python_tests(current_path):
            found.append(current_path)

    return tuple(sorted(set(found), key=lambda p: (len(p.parts), str(p))))


def discover_test_requirement_files(repo_path: str | Path) -> tuple[Path, ...]:
    """Find nested requirement manifests whose own subtree contains tests.

    Some multi-implementation repositories are not packaged as installable
    projects at all; each implementation instead owns a requirements.txt next
    to its tests.  Root requirement files retain their existing collector and
    are intentionally excluded here.
    """
    root = Path(repo_path)
    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            rel_parts = current_path.relative_to(root).parts
        except ValueError:
            continue
        dirs[:] = [
            name for name in dirs
            if name not in _PROJECT_SCAN_EXCLUDES and not name.startswith(".")
        ]
        if len(rel_parts) >= _MAX_PROJECT_DEPTH:
            dirs[:] = []
        if current_path == root or not _has_python_tests(current_path):
            continue
        for name in files:
            if re.fullmatch(r"requirements[^/]*\.txt", name, flags=re.IGNORECASE):
                found.append(current_path / name)
    return tuple(sorted(set(found)))


def _prefix_source(source: str, repo_root: Path, project_root: Path) -> str:
    if project_root == repo_root:
        return source
    prefix = project_root.relative_to(repo_root).as_posix()
    return f"{prefix}/{source}"


def _prefix_new_metadata_sources(
    evidence: PythonDependencyEvidence,
    repo_root: Path,
    project_root: Path,
    starts: tuple[int, int, int],
) -> None:
    py_start, dep_start, constraint_start = starts
    evidence.python_requires[py_start:] = [
        replace(item, source=_prefix_source(item.source, repo_root, project_root))
        for item in evidence.python_requires[py_start:]
    ]
    evidence.declared_dependencies[dep_start:] = [
        replace(item, source=_prefix_source(item.source, repo_root, project_root))
        for item in evidence.declared_dependencies[dep_start:]
    ]
    evidence.constraint_dependencies[constraint_start:] = [
        replace(item, source=_prefix_source(item.source, repo_root, project_root))
        for item in evidence.constraint_dependencies[constraint_start:]
    ]


def collect_python_dependency_evidence(repo_path: str | Path) -> PythonDependencyEvidence:
    root = Path(repo_path)
    evidence = PythonDependencyEvidence(repo_path=str(root))

    collectors = (
        _collect_pyproject_metadata,
        _collect_setup_cfg_metadata,
        _collect_setup_py_metadata,
        _collect_requirements_files,
        _collect_test_bearing_nested_requirements,
        _collect_test_requirements_files,
        _collect_tox_dependencies,
        _collect_pytest_config_dependencies,
        _collect_constraints_files,
    )
    for collector in collectors:
        try:
            collector(root, evidence)
        except Exception as error:  # Evidence collection must not abort an agent run.
            evidence.collection_errors.append(f"{collector.__name__}: {error}")

    for project_root in discover_test_project_roots(root):
        if project_root == root:
            continue
        starts = (
            len(evidence.python_requires),
            len(evidence.declared_dependencies),
            len(evidence.constraint_dependencies),
        )
        for collector in collectors:
            try:
                collector(project_root, evidence)
            except Exception as error:
                rel = project_root.relative_to(root).as_posix()
                evidence.collection_errors.append(
                    f"{collector.__name__}({rel}): {error}"
                )
        _prefix_new_metadata_sources(evidence, root, project_root, starts)

    try:
        imports, project_local_modules, import_errors = scan_imports(root)
        evidence.imports.extend(imports)
        evidence.project_local_modules.extend(project_local_modules)
        evidence.collection_errors.extend(import_errors)
    except Exception as error:
        evidence.collection_errors.append(f"scan_imports: {error}")

    evidence.pydeps = collect_pydeps_evidence(root)
    evidence.import_package_mappings.extend(_build_import_mappings(evidence))
    return evidence


def _collect_pyproject_metadata(root: Path, evidence: PythonDependencyEvidence) -> None:
    path = root / "pyproject.toml"
    if not path.is_file() or tomllib is None:
        return
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    requires_python = project.get("requires-python")
    if isinstance(requires_python, str) and requires_python.strip():
        evidence.python_requires.append(
            PythonVersionRequirement(
                specifier=requires_python.strip(),
                source="pyproject.toml:project.requires-python",
            )
        )
    for requirement in project.get("dependencies", []) or []:
        _add_requirement_line(evidence.declared_dependencies, requirement, "pyproject.toml:project.dependencies")
    optional_dependencies = project.get("optional-dependencies", {}) or {}
    if isinstance(optional_dependencies, dict):
        for group, requirements in optional_dependencies.items():
            for requirement in requirements or []:
                _add_requirement_line(
                    evidence.declared_dependencies,
                    requirement,
                    f"pyproject.toml:project.optional-dependencies.{group}",
                    kind="optional_dependency",
                    trust="medium",
                )

    dependency_groups = data.get("dependency-groups", {}) or {}
    if isinstance(dependency_groups, dict):
        for group, requirements in dependency_groups.items():
            if not isinstance(requirements, list):
                continue
            for requirement in requirements:
                # PEP 735 also permits {include-group = "..."}. The included
                # group's own concrete entries are collected independently;
                # this evidence layer never expands group references twice.
                if not isinstance(requirement, str):
                    continue
                _add_requirement_line(
                    evidence.declared_dependencies,
                    requirement,
                    f"pyproject.toml:dependency-groups.{group}",
                    kind="optional_dependency",
                    trust="medium",
                )

    poetry_dependencies = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    if isinstance(poetry_dependencies, dict):
        for name, value in poetry_dependencies.items():
            if name.lower() == "python":
                if isinstance(value, str):
                    evidence.python_requires.append(
                        PythonVersionRequirement(
                            specifier=value,
                            source="pyproject.toml:tool.poetry.dependencies.python",
                        )
                    )
                continue
            specifier = value if isinstance(value, str) else ""
            evidence.declared_dependencies.append(
                PythonRequirement(
                    name=name,
                    specifier=specifier,
                    source="pyproject.toml:tool.poetry.dependencies",
                )
            )

    hatch_envs = data.get("tool", {}).get("hatch", {}).get("envs", {}) or {}
    if isinstance(hatch_envs, dict):
        for env_name, env_data in hatch_envs.items():
            normalized = str(env_name).lower().replace("_", "-")
            if normalized not in {"test", "tests", "testing", "hatch-test"}:
                continue
            if not isinstance(env_data, dict):
                continue
            for requirement in env_data.get("extra-dependencies", []) or []:
                _add_requirement_line(
                    evidence.declared_dependencies,
                    requirement,
                    "pyproject.toml:hatch-test-deps.test",
                    kind="optional_dependency",
                    trust="medium",
                )


def _collect_setup_cfg_metadata(root: Path, evidence: PythonDependencyEvidence) -> None:
    path = root / "setup.cfg"
    if not path.is_file():
        return
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if parser.has_option("options", "python_requires"):
        evidence.python_requires.append(
            PythonVersionRequirement(
                specifier=parser.get("options", "python_requires").strip(),
                source="setup.cfg:options.python_requires",
            )
        )
    if parser.has_option("options", "install_requires"):
        for line in _split_multiline_value(parser.get("options", "install_requires")):
            _add_requirement_line(evidence.declared_dependencies, line, "setup.cfg:options.install_requires")
    if parser.has_section("options.extras_require"):
        for group, value in parser.items("options.extras_require"):
            for line in _split_multiline_value(value):
                _add_requirement_line(
                    evidence.declared_dependencies,
                    line,
                    f"setup.cfg:options.extras_require.{group}",
                    kind="optional_dependency",
                    trust="medium",
                )


def _collect_setup_py_metadata(root: Path, evidence: PythonDependencyEvidence) -> None:
    path = root / "setup.py"
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    if len(content) > 250_000:
        evidence.collection_errors.append("setup.py: skipped metadata parse because file is too large")
        return
    try:
        tree = ast.parse(content)
    except SyntaxError as error:
        evidence.collection_errors.append(f"setup.py: syntax error while parsing metadata: {error}")
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if func_name != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == "python_requires":
                value = _literal_string(keyword.value)
                if value:
                    evidence.python_requires.append(
                        PythonVersionRequirement(
                            specifier=value,
                            source="setup.py:setup.python_requires",
                        )
                    )
            elif keyword.arg == "install_requires":
                for requirement in _literal_string_list(keyword.value):
                    _add_requirement_line(
                        evidence.declared_dependencies,
                        requirement,
                        "setup.py:setup.install_requires",
                    )
            elif keyword.arg == "extras_require":
                for group, requirements in _literal_extras_require(keyword.value).items():
                    for requirement in requirements:
                        _add_requirement_line(
                            evidence.declared_dependencies,
                            requirement,
                            f"setup.py:setup.extras_require.{group}",
                            kind="optional_dependency",
                            trust="medium",
                        )


def _collect_requirements_files(root: Path, evidence: PythonDependencyEvidence) -> None:
    for path in _glob_metadata_files(root, "requirements*.txt"):
        rel = _relative_source(root, path)
        name = path.name.lower()
        if name == "requirements.txt":
            kind = "dependency"
            source = rel
            trust = "high"
        else:
            # A sibling requirements-docs.txt/dev.txt/etc. is not part of the
            # runtime transaction merely because it shares a filename prefix.
            # It becomes selectable test evidence through its explicit group;
            # docs/examples remain dormant unless another authoritative
            # manifest includes them with -r.
            stem = name.removesuffix(".txt")
            group = re.sub(r"^requirements[-_]?", "", stem) or "optional"
            group = group.replace("_", "-")
            kind = "optional_dependency"
            source = f"{rel}:dependency-groups.{group}"
            trust = "medium"
        for line in _read_requirement_lines(path, repo_root=root):
            _add_requirement_line(
                evidence.declared_dependencies,
                line,
                source,
                kind=kind,
                trust=trust,
            )


def _collect_test_bearing_nested_requirements(
    root: Path, evidence: PythonDependencyEvidence
) -> None:
    for path in discover_test_requirement_files(root):
        for line in _read_requirement_lines(path, repo_root=root):
            _add_requirement_line(
                evidence.declared_dependencies,
                line,
                _relative_source(root, path),
                kind="dependency",
                trust="high",
            )


def _collect_test_requirements_files(
    root: Path, evidence: PythonDependencyEvidence
) -> None:
    """Collect nested, explicitly test-scoped requirement files.

    Root-level ``requirements*.txt`` files retain their historical treatment
    above.  Files under ``tests/`` were previously invisible to the graph even
    when tox/CI used them directly.
    """
    patterns = (
        "tests/requirements*.txt",
        "test/requirements*.txt",
        "requirements/test*.txt",
        "requirements/dev*.txt",
    )
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            source = f"{_relative_source(root, path)}:test-requirements.test"
            for line in _read_requirement_lines(path, repo_root=root):
                _add_requirement_line(
                    evidence.declared_dependencies,
                    line,
                    source,
                    kind="optional_dependency",
                    trust="medium",
                )


def _collect_tox_dependencies(root: Path, evidence: PythonDependencyEvidence) -> None:
    path = root / "tox.ini"
    if not path.is_file():
        return
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    for section in parser.sections():
        if section != "testenv" and not section.startswith("testenv:"):
            continue
        if not parser.has_option(section, "deps"):
            continue
        for line in _split_multiline_value(parser.get(section, "deps", raw=True)):
            include = _requirement_include_path(line, root)
            if include is not None:
                for included_line in _read_requirement_lines(include, repo_root=root):
                    _add_requirement_line(
                        evidence.declared_dependencies,
                        included_line,
                        "tox.ini:tox-deps.test",
                        kind="optional_dependency",
                        trust="medium",
                    )
                continue
            _add_requirement_line(
                evidence.declared_dependencies,
                line,
                "tox.ini:tox-deps.test",
                kind="optional_dependency",
                trust="medium",
            )


def _pytest_addopts_dependencies(tokens: list[str], *, asyncio_mode: bool = False) -> set[str]:
    inferred: set[str] = set()
    for token in tokens:
        if token == "--cov" or token.startswith("--cov="):
            inferred.add("pytest-cov")
        elif token == "-n" or (token.startswith("-n") and len(token) > 2):
            inferred.add("pytest-xdist")
        elif token in {"--numprocesses", "--dist"} or token.startswith(
            ("--numprocesses=", "--dist=")
        ):
            inferred.add("pytest-xdist")
        elif token == "--asyncio-mode" or token.startswith("--asyncio-mode="):
            inferred.add("pytest-asyncio")
        elif token == "--timeout" or token.startswith("--timeout="):
            inferred.add("pytest-timeout")
    if asyncio_mode:
        inferred.add("pytest-asyncio")
    return inferred


def _as_addopts_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            return shlex.split(value)
        except ValueError:
            return value.split()
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str)]
    return []


def _collect_pytest_config_dependencies(
    root: Path, evidence: PythonDependencyEvidence
) -> None:
    """Infer only high-confidence pytest plugins from explicit config options."""
    inferred: set[str] = set()
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and tomllib is not None:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        options = data.get("tool", {}).get("pytest", {}).get("ini_options", {}) or {}
        if isinstance(options, dict):
            inferred.update(
                _pytest_addopts_dependencies(
                    _as_addopts_tokens(options.get("addopts")),
                    asyncio_mode=bool(options.get("asyncio_mode")),
                )
            )

    for filename in ("pytest.ini", "setup.cfg", "tox.ini"):
        path = root / filename
        if not path.is_file():
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error:
            continue
        section = "tool:pytest" if filename == "setup.cfg" else "pytest"
        if not parser.has_section(section):
            continue
        addopts = parser.get(section, "addopts", fallback="", raw=True)
        inferred.update(
            _pytest_addopts_dependencies(
                _as_addopts_tokens(addopts),
                asyncio_mode=parser.has_option(section, "asyncio_mode"),
            )
        )

    existing = {
        re.sub(r"[-_.]+", "-", requirement.name).lower()
        for requirement in evidence.declared_dependencies
    }
    for name in sorted(inferred):
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in existing:
            continue
        evidence.declared_dependencies.append(
            PythonRequirement(
                name=name,
                source="pytest-config:test-requirements.test",
                kind="optional_dependency",
                trust="high",
            )
        )
        existing.add(normalized)


def _collect_constraints_files(root: Path, evidence: PythonDependencyEvidence) -> None:
    for path in _glob_metadata_files(root, "constraints*.txt"):
        for line in _read_requirement_lines(path, repo_root=root):
            _add_requirement_line(
                evidence.constraint_dependencies,
                line,
                _relative_source(root, path),
                kind="constraint",
            )


def _build_import_mappings(evidence: PythonDependencyEvidence) -> list[ImportPackageMapping]:
    declared_package_names = {
        requirement.name for requirement in evidence.declared_dependencies if requirement.name
    }
    mappings: list[ImportPackageMapping] = []
    for finding in evidence.imports:
        if finding.classification != "external":
            continue
        mapping = map_import_to_package(finding.import_name, declared_package_names)
        mappings.append(
            ImportPackageMapping(
                import_name=mapping.import_name,
                package_name=mapping.package_name,
                source=mapping.source,
                trust=mapping.trust,
            )
        )
    return sorted(mappings, key=lambda item: item.import_name)


def _glob_metadata_files(root: Path, pattern: str) -> list[Path]:
    matches = [
        Path(path)
        for path in glob.glob(str(root / pattern))
        if Path(path).is_file()
    ]
    return sorted(matches)


def _requirement_include_path(line: str, base: Path) -> Path | None:
    cleaned = _strip_inline_comment(line).strip().replace("{toxinidir}", str(base))
    match = re.match(r"^(?:-r|--requirement)(?:\s+|=)?(.+)$", cleaned)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parts = shlex.split(raw)
    except ValueError:
        return None
    if len(parts) != 1:
        return None
    candidate = Path(parts[0])
    return candidate if candidate.is_absolute() else base / candidate


def _read_requirement_lines(
    path: Path,
    *,
    repo_root: Path | None = None,
    _visited: frozenset[Path] = frozenset(),
) -> Iterable[str]:
    allowed_root = (repo_root or path.parent).resolve()
    try:
        resolved = path.resolve()
        resolved.relative_to(allowed_root)
    except (OSError, ValueError):
        return
    if resolved in _visited or len(_visited) >= 12 or not resolved.is_file():
        return
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = resolved.read_text(encoding="latin-1")
    visited = _visited | {resolved}
    for raw_line in content.splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue
        include = _requirement_include_path(line, resolved.parent)
        if include is not None:
            yield from _read_requirement_lines(
                include, repo_root=allowed_root, _visited=visited
            )
            continue
        if line.startswith(("-", "--")):
            continue
        yield line


def _add_requirement_line(
    target: list[PythonRequirement],
    line: object,
    source: str,
    *,
    kind: str = "dependency",
    trust: str = "high",
) -> None:
    if not isinstance(line, str):
        return
    parsed = _parse_requirement_line(line)
    if not parsed:
        return
    name, specifier, marker, extras = parsed
    target.append(
        PythonRequirement(
            name=name,
            specifier=specifier,
            marker=marker,
            extras=extras,
            source=source,
            kind=kind,
            trust=trust,
        )
    )


def _parse_requirement_line(line: str) -> tuple[str, str, str, tuple[str, ...]] | None:
    cleaned = _strip_inline_comment(line).strip()
    if not cleaned or cleaned.startswith(("-", "--")):
        return None
    if "://" in cleaned or cleaned.startswith(("git+", "hg+", "svn+")):
        egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", cleaned)
        if egg_match:
            return egg_match.group(1), cleaned, "", ()
        return None
    try:
        requirement = Requirement(cleaned)
    except InvalidRequirement:
        return None
    marker = str(requirement.marker) if requirement.marker is not None else ""
    extras = tuple(sorted(requirement.extras))
    return requirement.name, str(requirement.specifier), marker, extras


def _strip_inline_comment(line: str) -> str:
    if " #" not in line:
        return line
    return line.split(" #", 1)[0]


def _split_multiline_value(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _literal_string(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _literal_string_list(node: ast.AST) -> list[str]:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _literal_extras_require(node: ast.AST) -> dict[str, list[str]]:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError, SyntaxError):
        return {}
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for group, requirements in value.items():
        if not isinstance(group, str):
            continue
        if isinstance(requirements, str):
            result[group] = [requirements]
        elif isinstance(requirements, (list, tuple)):
            result[group] = [item for item in requirements if isinstance(item, str)]
    return result


def _relative_source(root: Path, path: Path) -> str:
    return os.path.relpath(path, root)
