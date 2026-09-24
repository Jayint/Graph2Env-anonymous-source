"""Static Rust toolchain enrichment for Python projects with native Rust code.

Python repositories using maturin/PyO3 often contain a Cargo workspace while
still being evaluated as Python projects.  The Python dependency resolver
cannot encode Cargo's minimum supported Rust version, so record it as an
explicit Tool node before setup.sh is compiled.
"""

from __future__ import annotations

from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from src.python_deps.depgraph.ids import tool_id
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

_RUST_TOOL_ID = tool_id("rust")


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _declared_toolchain(repo_path: str | Path) -> tuple[str, str] | None:
    """Return ``(toolchain, provenance)`` from strongest repository evidence."""

    root = Path(repo_path)
    toolchain_file = root / "rust-toolchain.toml"
    toolchain = (_read_toml(toolchain_file).get("toolchain") or {}).get("channel")
    if isinstance(toolchain, str) and toolchain.strip():
        return toolchain.strip(), "rust-toolchain.toml:[toolchain].channel"

    plain_toolchain = root / "rust-toolchain"
    try:
        for raw_line in plain_toolchain.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                return line, "rust-toolchain"
    except OSError:
        pass

    cargo_file = root / "Cargo.toml"
    cargo = _read_toml(cargo_file)
    candidates = (
        ((cargo.get("workspace") or {}).get("package") or {}).get("rust-version"),
        (cargo.get("package") or {}).get("rust-version"),
    )
    for version in candidates:
        if isinstance(version, str) and version.strip():
            return version.strip(), "Cargo.toml:rust-version"

    if cargo_file.is_file():
        return "stable", "Cargo.toml"
    return None


def _version_check(toolchain: str) -> str:
    if toolchain in {"stable", "beta", "nightly"} or toolchain.startswith("nightly-"):
        return "/root/.cargo/bin/rustc --version"
    escaped = toolchain.replace(".", r"\.")
    return (
        "/root/.cargo/bin/rustc --version "
        f"| grep -Eq '^rustc {escaped}([. -]|$)'"
    )


def enrich_rust_toolchain(graph: DepGraph, repo_path: str | Path) -> DepGraph:
    """Add a replayable, version-aware Rustup Tool required by project nodes."""

    declared = _declared_toolchain(repo_path)
    if declared is None:
        return graph
    toolchain, provenance = declared
    provider = f"rustup:{toolchain}"
    node = Node(
        id=_RUST_TOOL_ID,
        type=NodeType.TOOL,
        name="rust/cargo",
        layer=Layer.TOOLCHAIN,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=State.UNKNOWN,
        version=toolchain,
        version_constraint=toolchain,
        check_command=_version_check(toolchain),
        fix_candidates=(provider,),
        chosen_fix=provider,
        provenance=provenance,
        setup_commands=(
            "apt-get update && apt-get install -y --no-install-recommends "
            "curl ca-certificates build-essential",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            f"| sh -s -- -y --default-toolchain {toolchain}",
            "ln -sf /root/.cargo/bin/cargo /usr/local/bin/cargo",
            "ln -sf /root/.cargo/bin/rustc /usr/local/bin/rustc",
        ),
        strength=Strength.HARD,
    )
    enriched = graph.with_node(node)
    for project in graph.nodes:
        if project.type is NodeType.PROJECT:
            enriched = enriched.with_edge(
                Edge(
                    src=project.id,
                    dst=_RUST_TOOL_ID,
                    relation=EdgeType.REQUIRES,
                    origin="static-rust-toolchain",
                )
            )
    return enriched
