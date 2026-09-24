# src/envstate/manifest.py
"""Host-side manifest parser (shallow). Reads declared deps + build system
from a checked-out repo. Pure, never raises. Uses tomllib + packaging.
"""
from __future__ import annotations

import os
import json
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 (e.g. the 3.10 benchmark box)
    import tomli as tomllib
from dataclasses import dataclass

from packaging.requirements import Requirement

from src.envstate.world_model import Fact

# Well-known extra names whose deps are treated as required (test/dev tooling).
_TEST_DEV_EXTRAS: frozenset[str] = frozenset(
    {"tests", "test", "testing", "dev", "development", "ci"}
)


@dataclass(frozen=True)
class ManifestResult:
    build_system: str             # poetry|pip|setuptools|hatchling|flit|pipenv|unknown
    required: tuple[Fact, ...]    # Fact(name, detail=specifier); declared names only
    package_managers: tuple[str, ...] = ()


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _req_fact(spec: str) -> Fact | None:
    spec = spec.strip()
    if not spec or spec.startswith("#"):
        return None
    try:
        r = Requirement(spec)
        return Fact(name=r.name, detail=str(r.specifier))
    except Exception:
        token = spec.split(";")[0].split("#")[0].strip()
        for sep in ("==", ">=", "<=", "~=", ">", "<", "[", " "):
            token = token.split(sep)[0].strip()
        return Fact(name=token) if token else None


def _parse_requirements_txt(workplace: str, filename: str, seen: set[str]) -> list[Fact]:
    path = os.path.join(workplace, filename)
    if path in seen:
        return []
    seen.add(path)
    text = _read_text(path)
    if text is None:
        return []
    facts: list[Fact] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement"):
            parts = line.split(None, 1)
            if len(parts) == 2:
                facts.extend(_parse_requirements_txt(workplace, parts[1].strip(), seen))
            continue
        if line.startswith("-"):
            continue  # other pip flags (-e, --index-url, ...)
        f = _req_fact(line)
        if f:
            facts.append(f)
    return facts


def _load_pyproject(workplace: str) -> dict | None:
    """Parse ``pyproject.toml`` to a dict, or ``None`` when absent/malformed."""
    raw = _read_text(os.path.join(workplace, "pyproject.toml"))
    if raw is None:
        return None
    try:
        return tomllib.loads(raw)
    except Exception:
        return None


def read_requires_python(workplace: str) -> str | None:
    """Return the project's declared python constraint as a RAW spec string.

    Prefers PEP 621 ``[project].requires-python`` (e.g. ``">=3.10,<3.13"``);
    falls back to poetry's ``[tool.poetry.dependencies].python`` (e.g. ``"^3.10"``).
    Returns ``None`` when neither is declared. Interpreting the string (caret
    normalization, picking a concrete minor) is choose_python_minor's job, not
    this reader's. Pure, never raises.
    """
    pyproject = _load_pyproject(workplace)
    if not pyproject:
        return None
    pep621 = pyproject.get("project", {}).get("requires-python")
    if isinstance(pep621, str) and pep621.strip():
        return pep621.strip()
    poetry_py = (
        pyproject.get("tool", {})
        .get("poetry", {})
        .get("dependencies", {})
        .get("python")
    )
    if isinstance(poetry_py, str) and poetry_py.strip():
        return poetry_py.strip()
    return None


def parse_manifests(workplace: str) -> ManifestResult:
    build_system = "unknown"
    facts: list[Fact] = []

    pyproject = None
    raw = _read_text(os.path.join(workplace, "pyproject.toml"))
    if raw is not None:
        try:
            pyproject = tomllib.loads(raw)
        except Exception:
            pyproject = None

    poetry = (pyproject or {}).get("tool", {}).get("poetry") if pyproject else None
    has_poetry_lock = os.path.exists(os.path.join(workplace, "poetry.lock"))
    has_pipfile = os.path.exists(os.path.join(workplace, "Pipfile"))
    has_setup = os.path.exists(os.path.join(workplace, "setup.py")) or \
        os.path.exists(os.path.join(workplace, "setup.cfg"))
    try:
        req_files = sorted(
            fn for fn in os.listdir(workplace)
            if fn.startswith("requirements") and fn.endswith(".txt")
        )
    except OSError:
        req_files = []

    # build_system precedence
    if has_poetry_lock or poetry:
        build_system = "poetry"
    elif pyproject and "build-system" in pyproject:
        backend = str(pyproject["build-system"].get("build-backend", ""))
        if "hatch" in backend:
            build_system = "hatchling"
        elif "flit" in backend:
            build_system = "flit"
        elif "poetry" in backend:
            build_system = "poetry"
        else:
            build_system = "setuptools"
    elif has_pipfile:
        build_system = "pipenv"
    elif req_files:
        build_system = "pip"
    elif has_setup:
        build_system = "setuptools"

    # required extraction
    if pyproject:
        for dep in (pyproject.get("project", {}).get("dependencies") or []):
            f = _req_fact(str(dep))
            if f:
                facts.append(f)
        opt_deps: dict = (pyproject.get("project", {}).get("optional-dependencies") or {})
        for extra_name, specs in opt_deps.items():
            if extra_name.lower() not in _TEST_DEV_EXTRAS:
                continue
            for spec in (specs or []):
                f = _req_fact(str(spec))
                if f:
                    facts.append(f)
        if poetry:
            for name, val in (poetry.get("dependencies") or {}).items():
                if name.lower() == "python":
                    continue
                facts.append(Fact(name=name, detail=val if isinstance(val, str) else ""))

    seen: set[str] = set()
    for fn in req_files:
        facts.extend(_parse_requirements_txt(workplace, fn, seen))

    if has_pipfile:
        ptext = _read_text(os.path.join(workplace, "Pipfile"))
        if ptext is not None:
            try:
                pip = tomllib.loads(ptext)
                for name, val in (pip.get("packages") or {}).items():
                    detail = val if isinstance(val, str) and val != "*" else ""
                    facts.append(Fact(name=name, detail=detail))
            except Exception:
                pass

    # Polyglot workspace manifests. Keep this shallow and deterministic: the
    # EcosystemProviders own full resolution, while the WorldModel receives a
    # useful repository-wide declared-dependency summary.
    managers: list[str] = []
    skip_dirs = {".git", ".venv", "venv", "node_modules", "target", "build", "dist"}
    try:
        walker = os.walk(workplace)
        for root, dirs, filenames in walker:
            dirs[:] = [item for item in dirs if item not in skip_dirs]
            names = set(filenames)
            if "package.json" in names:
                managers.append(
                    "pnpm" if "pnpm-lock.yaml" in names else
                    "yarn" if "yarn.lock" in names else "npm"
                )
                try:
                    package = json.loads(_read_text(os.path.join(root, "package.json")) or "{}")
                    for key in ("dependencies", "devDependencies"):
                        for name, version in (package.get(key) or {}).items():
                            facts.append(Fact(str(name), str(version)))
                except Exception:
                    pass
            if "Cargo.toml" in names:
                managers.append("cargo")
                try:
                    cargo = tomllib.loads(_read_text(os.path.join(root, "Cargo.toml")) or "")
                    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
                        for name, value in (cargo.get(key) or {}).items():
                            detail = value if isinstance(value, str) else str(
                                (value or {}).get("version") or ""
                            )
                            facts.append(Fact(str(name), detail))
                except Exception:
                    pass
            if "go.mod" in names:
                managers.append("go")
                go_mod = _read_text(os.path.join(root, "go.mod")) or ""
                in_require = False
                for raw_line in go_mod.splitlines():
                    line = raw_line.split("//", 1)[0].strip()
                    if line == "require (":
                        in_require = True
                        continue
                    if in_require and line == ")":
                        in_require = False
                        continue
                    if line.startswith("require "):
                        line = line[len("require "):].strip()
                    elif not in_require:
                        continue
                    match = re.match(r"(\S+)\s+(v\S+)", line)
                    if match:
                        facts.append(Fact(match.group(1), match.group(2)))
            if "pom.xml" in names:
                managers.append("maven")
            if {"build.gradle", "build.gradle.kts"} & names:
                managers.append("gradle")
    except OSError:
        pass

    # dedup by lowercased name (keep first)
    seen_names: set[str] = set()
    deduped: list[Fact] = []
    for f in facts:
        key = f.name.lower()
        if key and key not in seen_names:
            seen_names.add(key)
            deduped.append(f)

    manager_tuple = tuple(dict.fromkeys(managers))
    if manager_tuple:
        systems = ([] if build_system == "unknown" else [build_system]) + list(manager_tuple)
        build_system = "+".join(dict.fromkeys(systems))
    return ManifestResult(
        build_system=build_system,
        required=tuple(deduped),
        package_managers=manager_tuple,
    )
