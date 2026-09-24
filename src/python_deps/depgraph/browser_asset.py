"""Evidence-gated browser binary assets for Python browser test suites.

Playwright/Patchright wheels contain the Python API, not the browser executable.
This enrichment promotes a browser binary to a replayable ``DataAsset`` only
when the repository itself supplies all of the required evidence:

* the provider is an explicitly declared, successfully resolved Package node;
* the graph has a Test goal that can own the asset; and
* test/CI configuration contains a complete, directly replayable install recipe.

The scanner deliberately accepts only the narrow ``python -m <provider>``
forms below.  It never invents a browser, provider, flag, or system-dependency
command.  Repositories without the complete recipe retain the historical soft,
advisory DataAsset behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
    Strength,
)


_PROVIDERS = ("playwright", "patchright")
_BROWSERS = ("chromium", "firefox", "webkit")
_PYTHON = r"python(?:3(?:\.\d+)?)?"
_PROVIDER = r"(?:playwright|patchright)"
_BROWSER = r"(?:chromium|firefox|webkit)"

# Only accept commands that are independently replayable in setup.sh.  In
# particular, bare ``playwright install`` and wrapper-specific forms such as
# ``uv run playwright`` are not rewritten into a guessed provider command.
_INSTALL_DEPS_RE = re.compile(
    rf"(?P<command>\b{_PYTHON}\s+-m\s+(?P<provider>{_PROVIDER})\s+"
    rf"install-deps\s+(?P<browser>{_BROWSER}))(?=\s*(?:$|[#;&|]))",
    re.IGNORECASE,
)
_INSTALL_RE = re.compile(
    rf"(?P<command>\b{_PYTHON}\s+-m\s+(?P<provider>{_PROVIDER})\s+install\s+"
    rf"(?:(?P<prefix_flag>--with-deps)\s+)?(?P<browser>{_BROWSER})"
    rf"(?:\s+(?P<suffix_flag>--with-deps))?)(?=\s*(?:$|[#;&|]))",
    re.IGNORECASE,
)

_MAX_EVIDENCE_FILES = 512
_MAX_EVIDENCE_BYTES = 1_000_000


@dataclass(frozen=True)
class BrowserInstallEvidence:
    provider: str
    browser: str
    source: str
    line: int
    command: str
    action: str
    includes_system_deps: bool = False


def _normalise_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


def _evidence_paths(root: Path) -> tuple[Path, ...]:
    """Return only test/CI surfaces, never docs/examples/Dockerfiles."""
    candidates: set[Path] = set()
    for rel in (
        "tox.ini",
        "noxfile.py",
        ".gitlab-ci.yml",
        "azure-pipelines.yml",
        ".circleci/config.yml",
    ):
        path = root / rel
        if path.is_file():
            candidates.add(path)

    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        for suffix in ("*.yml", "*.yaml"):
            candidates.update(path for path in workflows.rglob(suffix) if path.is_file())

    for dirname in ("tests", "test"):
        test_root = root / dirname
        if not test_root.is_dir():
            continue
        for suffix in ("*.py", "*.sh", "*.yml", "*.yaml", "*.ini"):
            candidates.update(path for path in test_root.rglob(suffix) if path.is_file())

    return tuple(sorted(candidates, key=lambda path: path.as_posix())[:_MAX_EVIDENCE_FILES])


def _scan_install_evidence(repo_path: str | Path) -> tuple[BrowserInstallEvidence, ...]:
    root = Path(repo_path)
    found: list[BrowserInstallEvidence] = []
    for path in _evidence_paths(root):
        try:
            if path.is_symlink():
                continue
            if path.stat().st_size > _MAX_EVIDENCE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            source = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        for line_number, raw_line in enumerate(text.splitlines(), 1):
            if raw_line.lstrip().startswith("#"):
                continue
            for match in _INSTALL_DEPS_RE.finditer(raw_line):
                if not _is_command_position(raw_line, match.start()):
                    continue
                found.append(BrowserInstallEvidence(
                    provider=match.group("provider").lower(),
                    browser=match.group("browser").lower(),
                    source=source,
                    line=line_number,
                    command=match.group("command").strip(),
                    action="install-deps",
                    includes_system_deps=True,
                ))
            for match in _INSTALL_RE.finditer(raw_line):
                if not _is_command_position(raw_line, match.start()):
                    continue
                with_deps = bool(match.group("prefix_flag") or match.group("suffix_flag"))
                found.append(BrowserInstallEvidence(
                    provider=match.group("provider").lower(),
                    browser=match.group("browser").lower(),
                    source=source,
                    line=line_number,
                    command=match.group("command").strip(),
                    action="install",
                    includes_system_deps=with_deps,
                ))
    return tuple(found)


def _is_command_position(line: str, start: int) -> bool:
    """Reject documentation/echo/string occurrences inside an allowed file.

    A command may start a shell/config line or follow a conventional CI YAML
    key.  Anything else is merely text evidence and cannot authorize mutation.
    """
    prefix = line[:start].strip().lower()
    return prefix in {"", "-", "run:", "- run:", "script:", "- script:",
                      "command:", "- command:"}


def _resolved_declared_providers(graph: DepGraph) -> dict[str, Node]:
    """Providers backed by both manifest provenance and a concrete resolution."""
    providers: dict[str, Node] = {}
    for node in graph.nodes:
        name = _normalise_dist(node.name)
        if (
            node.type is NodeType.PACKAGE
            and name in _PROVIDERS
            and bool(node.manifest_source)
            and bool(node.version)
            and node.resolution_status == "resolved"
        ):
            providers[name] = node
    return providers


def _complete_recipes(
    evidence: tuple[BrowserInstallEvidence, ...],
    providers: dict[str, Node],
) -> tuple[tuple[BrowserInstallEvidence, ...], ...]:
    """Select only evidence-complete, idempotent browser recipes.

    A single repository-authored ``install --with-deps`` command is complete.
    Otherwise both repository-authored ``install-deps`` and ``install`` commands
    must exist for the same provider/browser pair.  No missing command is
    synthesized.
    """
    recipes: list[tuple[BrowserInstallEvidence, ...]] = []
    for provider in _PROVIDERS:
        if provider not in providers:
            continue
        for browser in _BROWSERS:
            matching = tuple(
                item for item in evidence
                if item.provider == provider and item.browser == browser
            )
            combined = next(
                (item for item in matching
                 if item.action == "install" and item.includes_system_deps),
                None,
            )
            if combined is not None:
                recipes.append((combined,))
                continue
            deps = next((item for item in matching if item.action == "install-deps"), None)
            install = next((item for item in matching if item.action == "install"), None)
            if deps is not None and install is not None:
                recipes.append((deps, install))
    return tuple(recipes)


def _asset_check(provider: str, browser: str, python: str) -> str:
    """Architecture-neutral executable check resolved through the provider API."""
    return (
        f'{python} -c "import os; from {provider}.sync_api import sync_playwright; '
        f'p=sync_playwright().start(); path=p.{browser}.executable_path; p.stop(); '
        'raise SystemExit(0 if os.access(path, os.X_OK) else 1)"'
    )


def enrich_browser_assets(graph: DepGraph, repo_path: str | Path) -> DepGraph:
    """Add replayable browser DataAssets when all static gates are satisfied."""
    test_nodes = tuple(node for node in graph.nodes if node.type is NodeType.TEST)
    if not test_nodes:
        return graph
    providers = _resolved_declared_providers(graph)
    if not providers:
        return graph
    recipes = _complete_recipes(_scan_install_evidence(repo_path), providers)
    if not recipes:
        return graph

    enriched = graph
    for recipe in recipes:
        exemplar = recipe[-1]
        provider = exemplar.provider
        browser = exemplar.browser
        asset_id = f"data:browser-{provider}-{browser}"
        existing = enriched.get(asset_id)
        if existing is None:
            commands = tuple(item.command for item in recipe)
            python = commands[-1].split(None, 1)[0]
            sources = tuple(f"{item.source}:{item.line}" for item in recipe)
            provider_node = providers[provider]
            enriched = enriched.with_node(Node(
                id=asset_id,
                type=NodeType.DATA_ASSET,
                name=f"{provider} {browser} browser binary",
                layer=Layer.CONFIG,
                discovered_by=DiscoveredBy.STATIC_SCAN,
                state=State.MISSING,
                check_command=_asset_check(provider, browser, python),
                evidence="; ".join(
                    f"{source}: {command}"
                    for source, command in zip(sources, commands)
                ),
                chosen_fix=f"browser:{provider}:{browser}",
                resolved_python=provider_node.resolved_python,
                resolved_platform=provider_node.resolved_platform,
                setup_commands=commands,
                strength=Strength.HARD,
                data={
                    "provider_backed": True,
                    "asset_kind": "browser_binary",
                    "browser_provider": provider,
                    "browser_name": browser,
                    "evidence_sources": sources,
                },
            ))
        elif not (
            existing.type is NodeType.DATA_ASSET
            and existing.data.get("provider_backed") is True
            and existing.data.get("asset_kind") == "browser_binary"
        ):
            # Never overwrite an independently classified/admitted DataAsset.
            continue

        provider_node = providers[provider]
        enriched = enriched.with_edge(Edge(
            src=asset_id,
            dst=provider_node.id,
            relation=EdgeType.REQUIRES,
            origin="browser-asset",
            data={"hard": True},
        ))
        for test_node in test_nodes:
            enriched = enriched.with_edge(Edge(
                src=test_node.id,
                dst=asset_id,
                relation=EdgeType.REQUIRES,
                origin="browser-asset",
                data={"hard": True},
            ))
    return enriched
