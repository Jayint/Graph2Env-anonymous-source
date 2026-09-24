"""Deterministic provider registry and multi-provider graph composition."""
from __future__ import annotations

from src.ecosystems.base import DepGraphFragment, ProviderTarget, merge_graphs
from src.ecosystems.go import GoProvider
from src.ecosystems.java import GradleProvider, MavenProvider
from src.ecosystems.node import NodeProvider
from src.ecosystems.rust import RustProvider
from src.python_deps.depgraph.schema import DepGraph


PROVIDERS = (
    NodeProvider(),
    RustProvider(),
    GoProvider(),
    MavenProvider(),
    GradleProvider(),
)


def build_ecosystem_fragments(
    repo_path: str,
    language_requirements,
    *,
    base_image: str,
    platform: str | None = None,
    resolver_sandbox=None,
) -> DepGraphFragment:
    """Run every matching provider and merge all workspace fragments."""
    graph = DepGraph()
    workspaces = []
    tests: list[tuple[str, str]] = []
    target = ProviderTarget(base_image=base_image, platform=platform)
    for provider in PROVIDERS:
        for workspace in provider.detect_workspaces(repo_path, language_requirements):
            fragment = provider.build_fragment(
                repo_path,
                workspace,
                target,
                sandbox=resolver_sandbox,
            )
            graph = merge_graphs(graph, fragment.graph)
            workspaces.extend(fragment.workspaces)
            tests.extend((workspace.role, command) for command in fragment.test_commands)
    role_rank = {"secondary_runtime": 0, "primary_runtime": 1, "build_tool": 2}
    ordered_tests = tuple(
        command for _role, command in sorted(
            dict.fromkeys(tests),
            key=lambda item: (role_rank.get(item[0], 0), item[1]),
        )
    )
    return DepGraphFragment(
        graph=graph,
        workspaces=tuple(workspaces),
        test_commands=ordered_tests,
    )


def discover_polyglot_test_commands(
    repo_path: str,
    language_requirements,
) -> tuple[str, ...]:
    """Discover test entrypoints without resolving or mutating dependencies."""
    commands: list[tuple[str, str]] = []
    for item in language_requirements:
        if item.language == "python" and item.role != "build_tool":
            commands.append((item.role, "python -m pytest -q"))
    for provider in PROVIDERS:
        for workspace in provider.detect_workspaces(repo_path, language_requirements):
            # Node's public test_commands needs the repository manifest; its
            # build_fragment reads it directly. Keep this discovery path pure.
            if workspace.ecosystem.value == "npm":
                import json
                import os
                from src.ecosystems.base import in_workspace, read_text
                try:
                    data = json.loads(read_text(os.path.join(
                        repo_path, workspace.manifest_file
                    )))
                except Exception:
                    data = {}
                script = str((data.get("scripts") or {}).get("test") or "")
                tests = (
                    (in_workspace(
                        workspace, f"{workspace.package_manager} test"
                    ),)
                    if script and "no test specified" not in script.lower()
                    and workspace.role != "build_tool"
                    else ()
                )
            else:
                tests = provider.test_commands(workspace)
            commands.extend((workspace.role, command) for command in tests)
    if not commands and language_requirements:
        fallback = {
            "go": "go test ./...",
            "rust": "cargo test --all-targets",
            "javascript": "npm test",
            "typescript": "npm test",
            "java": "./mvnw -B test",
        }.get(language_requirements[0].language)
        if fallback:
            commands.append(("primary_runtime", fallback))
    role_rank = {"secondary_runtime": 0, "primary_runtime": 1, "build_tool": 2}
    return tuple(
        command for _role, command in sorted(
            dict.fromkeys(commands),
            key=lambda item: (role_rank.get(item[0], 0), item[1]),
        )
    )
