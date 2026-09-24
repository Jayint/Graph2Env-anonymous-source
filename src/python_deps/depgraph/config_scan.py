"""Static Config-tier discovery: the env vars a repo reads (project-induced).

Pure (no Executor, no network): walks the repo on disk, AST-parses each in-scope
``.py`` file, and records every ``os.environ[...]`` / ``os.environ.get(...)`` /
``os.getenv(...)`` read (and the bare ``environ``/``getenv`` forms).  Mirrors the
directory-exclusion scope of ``scan.py`` so examples/docs/build don't leak.
"""

from __future__ import annotations

import ast
import os
import re

_EXCLUDED_SEGMENTS: frozenset[str] = frozenset(
    {
        "examples", "example", "docs", "doc", "build", "dist", "samples",
        "sample", "benchmarks", "benchmark", "bench", "scripts", "script",
        ".github", ".tox", "node_modules", "site-packages", ".venv", "venv",
        ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    }
)


def _is_excluded(rel: str) -> bool:
    segments = {seg.lower() for seg in re.split(r"[\\/]+", rel) if seg}
    return bool(segments & _EXCLUDED_SEGMENTS)


def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _var_from_call(call: ast.Call) -> str | None:
    """``os.getenv('X')`` / ``os.environ.get('X')`` -> ``'X'`` (first str arg)."""
    func = call.func
    name = None
    if isinstance(func, ast.Attribute):
        name = func.attr
    if name not in ("getenv", "get"):
        return None
    # environ.get must be on an `environ` object to count.
    if name == "get":
        owner = func.value
        owner_ok = (isinstance(owner, ast.Name) and owner.id == "environ") or (
            isinstance(owner, ast.Attribute) and owner.attr == "environ"
        )
        if not owner_ok:
            return None
    return _const_str(call.args[0]) if call.args else None


def _default_from_call(call: ast.Call) -> tuple[str, str] | None:
    """``os.environ.get('X', 'literal')`` / ``os.getenv('X', 'literal')`` ->
    ``('X', 'literal')``; only when BOTH the name and the default are string
    literals (f-strings / non-str defaults are not statically resolvable)."""
    var = _var_from_call(call)
    if var is None or len(call.args) < 2:
        return None
    default = _const_str(call.args[1])
    if default is None:
        return None
    return var, default


def _var_from_subscript(sub: ast.Subscript) -> str | None:
    """``os.environ['X']`` / ``environ['X']`` -> ``'X'``."""
    value = sub.value
    is_environ = (isinstance(value, ast.Name) and value.id == "environ") or (
        isinstance(value, ast.Attribute) and value.attr == "environ"
    )
    if not is_environ:
        return None
    return _const_str(sub.slice)


def _scan_source(src: str, rel: str, out: dict[str, str]) -> None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        var = None
        if isinstance(node, ast.Call):
            var = _var_from_call(node)
        elif isinstance(node, ast.Subscript):
            var = _var_from_subscript(node)
        if var and var not in out:
            snippet = " ".join((src.splitlines()[node.lineno - 1] if node.lineno else "").split())
            out[var] = f"{rel}:{getattr(node, 'lineno', 0)}  {snippet}"[:200]


def scan_env_reads(repo_path: str) -> dict[str, str]:
    """Map each env var the repo reads -> one-line evidence (file:line snippet)."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in _EXCLUDED_SEGMENTS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_path)
            if _is_excluded(rel):
                continue
            try:
                with open(full, encoding="utf-8") as fh:
                    src = fh.read()
            except OSError:
                continue
            _scan_source(src, rel, out)
    return out


def scan_env_defaults(repo_path: str) -> dict[str, str]:
    """Map env var -> its literal default from ``os.environ.get(VAR, 'literal')``.

    A lowest-precedence value hint for CONFIG nodes (``.env.example`` and curated
    package defaults win). Powers the Service-tier URL-binding inference, which
    needs a parseable value rather than ``?``.
    """
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in _EXCLUDED_SEGMENTS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_path)
            if _is_excluded(rel):
                continue
            try:
                with open(full, encoding="utf-8") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    hit = _default_from_call(node)
                    if hit and hit[0] not in out:
                        out[hit[0]] = hit[1]
    return out


_DECOUPLE_FUNCS = frozenset({"config"})           # decouple.config('X')
_ENVIRONS_METHODS = frozenset({"str", "int", "bool", "float", "list", "url"})  # env.str('X')
_SETTINGS_BASES = frozenset({"BaseSettings"})


def _framework_var_from_call(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name) and func.id in _DECOUPLE_FUNCS:
        return _const_str(call.args[0]) if call.args else None
    if isinstance(func, ast.Attribute) and func.attr in _ENVIRONS_METHODS:
        return _const_str(call.args[0]) if call.args else None
    return None


def _settings_fields(tree: ast.AST) -> list[str]:
    """Annotated field names of any class subclassing a *Settings base -> UPPER."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {b.id for b in node.bases if isinstance(b, ast.Name)} | {
            b.attr for b in node.bases if isinstance(b, ast.Attribute)
        }
        if not (base_names & _SETTINGS_BASES):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                out.append(stmt.target.id.upper())
    return out


def scan_framework_config_reads(repo_path: str) -> dict[str, str]:
    """Env vars read via pydantic-settings / decouple / environs (string-literal
    args + BaseSettings fields). NOTE: env_prefix / field aliases not yet resolved.
    """
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in _EXCLUDED_SEGMENTS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_path)
            if _is_excluded(rel):
                continue
            try:
                with open(full, encoding="utf-8") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    var = _framework_var_from_call(node)
                    if var and var not in out:
                        out[var] = f"{rel}:{getattr(node, 'lineno', 0)}  (framework config)"[:200]
            for field in _settings_fields(tree):
                out.setdefault(field, f"{rel}  (BaseSettings field)")
    return out


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_DOCKER_ENV_LINE = re.compile(r"^\s*ENV\s+(.+?)\s*$", re.IGNORECASE)


def _read_env_pairs(path: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip().startswith("#"):
                    continue
                m = _ENV_LINE.match(line)
                if m:
                    pairs[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    except OSError:
        return {}
    return pairs


def _is_dockerfile_name(fname: str) -> bool:
    return fname == "Dockerfile" or fname.startswith("Dockerfile.") or fname.startswith("Dockerfile-") or fname.endswith(".Dockerfile")


def parse_env_example(repo_path: str) -> dict[str, str]:
    """`{VAR: example_value}` from .env.example/.sample/.template (value hints)."""
    out: dict[str, str] = {}
    for fname in (".env.example", ".env.sample", ".env.template"):
        out.update(_read_env_pairs(os.path.join(repo_path, fname)))
    return out


def parse_dockerfile_env(repo_path: str) -> dict[str, str]:
    """Literal ``ENV`` pairs from Dockerfiles shipped with the repository.

    Dockerfiles are not arbitrary shell truth, but an explicit ``ENV X=value`` in
    a project-provided test/runtime image is strong evidence that ``X`` is not a
    missing user secret at benchmark time.  Only literal ENV forms are accepted:
    ``ENV KEY=value`` and ``ENV KEY value``. Empty values are ignored.
    """
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d.lower() not in _EXCLUDED_SEGMENTS]
        for fname in filenames:
            if not _is_dockerfile_name(fname):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, repo_path)
            if _is_excluded(rel):
                continue
            try:
                with open(full, encoding="utf-8") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            for line in lines:
                m = _DOCKER_ENV_LINE.match(line)
                if not m:
                    continue
                rest = m.group(1).strip()
                if "=" in rest:
                    pairs = rest.split()
                    for pair in pairs:
                        km = _ENV_LINE.match(pair)
                        if km and km.group(2).strip().strip('"').strip("'"):
                            out.setdefault(km.group(1), km.group(2).strip().strip('"').strip("'"))
                    continue
                parts = rest.split(None, 1)
                if len(parts) == 2 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", parts[0]):
                    value = parts[1].strip().strip('"').strip("'")
                    if value:
                        out.setdefault(parts[0], value)
    return out


def configured_vars(repo_path: str) -> set[str]:
    """Vars already provided at test time -> suppress as false-missing.

    Sources: a real ``.env`` file, and an ``env =`` block in pytest config
    (``pytest.ini`` / ``setup.cfg`` / ``tox.ini``; pytest-env style).
    """
    provided: set[str] = set(_read_env_pairs(os.path.join(repo_path, ".env")))
    provided.update(parse_dockerfile_env(repo_path))
    for cfg in ("pytest.ini", "setup.cfg", "tox.ini"):
        path = os.path.join(repo_path, cfg)
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        in_env = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^env\s*=", stripped):
                in_env = True
                continue
            if in_env:
                m = _ENV_LINE.match(line)
                if m and (line.startswith(" ") or line.startswith("\t")):
                    provided.add(m.group(1))
                elif stripped and not line[:1].isspace():
                    in_env = False
    return provided
