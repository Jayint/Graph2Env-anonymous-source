"""Deterministic providers for repository-declared test environment variables."""
from __future__ import annotations

import configparser
import hashlib
from pathlib import Path
import re
import shlex

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from src.python_deps.depgraph.ids import TEST_NODE_ID, config_id
from src.python_deps.depgraph.schema import (
    DepGraph, DiscoveredBy, Edge, EdgeType, Layer, Node, NodeType, State, Strength,
)

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SECRET = re.compile(r"SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|CREDENTIAL", re.I)
_DENY = {"PATH", "PYTHONPATH", "HOME", "PWD", "VIRTUAL_ENV"}
_UNRESOLVED_TEMPLATE = re.compile(
    r"\{(?:env|toxinidir|toxworkdir|envtmpdir|posargs)(?::|\})", re.I
)


def _safe_pair(name: str, value) -> tuple[str, str] | None:
    name = str(name).strip()
    value = str(value).strip().strip('"').strip("'")
    if not _NAME.fullmatch(name) or name in _DENY or _SECRET.search(name):
        return None
    if (
        not value
        or "\n" in value
        or "\r" in value
        or _UNRESOLVED_TEMPLATE.search(value)
    ):
        return None
    return name, value


def _parse_assignment_lines(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in str(value or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pair = _safe_pair(*line.split("=", 1))
        if pair:
            out[pair[0]] = pair[1]
    return out


def discover_test_environment(repo_path: str | Path) -> dict[str, tuple[str, str]]:
    """Return VAR -> (value, evidence) from pytest/tox repository config."""
    root = Path(repo_path)
    out: dict[str, tuple[str, str]] = {}

    pyproject = root / "pyproject.toml"
    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
        options = (((data.get("tool") or {}).get("pytest") or {}).get("ini_options") or {})
    except (OSError, tomllib.TOMLDecodeError):
        options = {}
    for key, value in options.items():
        pair = _safe_pair(key, value)
        if pair:
            out[pair[0]] = (pair[1], "pyproject.toml:tool.pytest.ini_options")
    for name, value in _parse_assignment_lines(options.get("env", "")).items():
        out[name] = (value, "pyproject.toml:tool.pytest.ini_options.env")

    for filename in ("pytest.ini", "setup.cfg", "tox.ini"):
        path = root / filename
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        try:
            parser.read(path, encoding="utf-8")
        except (OSError, configparser.Error):
            continue
        for section in ("pytest", "tool:pytest"):
            if not parser.has_section(section):
                continue
            for key, value in parser.items(section):
                pair = _safe_pair(key, value)
                if pair:
                    out[pair[0]] = (pair[1], f"{filename}:[{section}]")
            if parser.has_option(section, "env"):
                for name, value in _parse_assignment_lines(parser.get(section, "env")).items():
                    out[name] = (value, f"{filename}:[{section}] env")
        for section in parser.sections():
            if not section.startswith("testenv") or not parser.has_option(section, "setenv"):
                continue
            for name, value in _parse_assignment_lines(parser.get(section, "setenv")).items():
                out.setdefault(name, (value, f"{filename}:[{section}] setenv"))

    # Some reusable Django apps intentionally omit pytest-django configuration
    # but ship the conventional tests/settings.py + tests/conftest.py pair.
    # This is deterministic repository evidence (and uses setdefault in the
    # generated .pth), so it is a safe fallback when no explicit setting won.
    if (
        "DJANGO_SETTINGS_MODULE" not in out
        and (root / "tests" / "settings.py").is_file()
        and (root / "tests" / "conftest.py").is_file()
    ):
        out["DJANGO_SETTINGS_MODULE"] = (
            "tests.settings",
            "tests/settings.py + tests/conftest.py convention",
        )
    return out


def _pth_commands(name: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{name}={value}".encode()).hexdigest()[:12]
    code_line = f"import os; os.environ.setdefault({name!r}, {value!r})\n"
    writer = (
        "import pathlib,site; "
        f"path=pathlib.Path(site.getsitepackages()[0])/{('rat-test-env-' + digest + '.pth')!r}; "
        f"path.write_text({code_line!r})"
    )
    check = f"import os; raise SystemExit(0 if os.environ.get({name!r}) == {value!r} else 1)"
    return "python3 -c " + shlex.quote(writer), "python3 -c " + shlex.quote(check)


def enrich_test_environment(graph: DepGraph, repo_path: str | Path) -> DepGraph:
    new = graph
    for name, (value, evidence) in discover_test_environment(repo_path).items():
        node_id = config_id(name)
        command, check = _pth_commands(name, value)
        current = new.get(node_id)
        data = {
            **(dict(current.data) if current is not None else {}),
            "provider_backed": True,
            "asset_kind": "runtime_env",
            "runtime_env_name": name,
            "runtime_env_value": value,
        }
        node = Node(
            id=node_id,
            type=NodeType.CONFIG,
            name=name,
            layer=Layer.CONFIG,
            discovered_by=(current.discovered_by if current is not None else DiscoveredBy.STATIC_SCAN),
            state=State.MISSING,
            check_command=check,
            evidence=evidence,
            chosen_fix=f"env:{name}={value}",
            fix_candidates=(f"env:{name}={value}",),
            setup_commands=(command,),
            strength=Strength.HARD,
            data=data,
        )
        new = new.with_node(node)
        if new.get(TEST_NODE_ID) is not None:
            new = new.with_edge(Edge(
                src=TEST_NODE_ID,
                dst=node_id,
                relation=EdgeType.REQUIRES,
                origin="test-config",
            ))
    return new
