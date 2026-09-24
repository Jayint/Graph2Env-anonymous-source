"""Compose the legacy Python graph with all detected non-Python providers."""
from __future__ import annotations

from dataclasses import replace

from src.python_deps.depgraph.build import build_dep_graph
from src.python_deps.depgraph.certify import certify
from src.python_deps.depgraph.ids import TEST_NODE_ID
from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Edge,
    EdgeType,
    Ecosystem,
    Layer,
    Node,
    NodeType,
    State,
)
from src.ecosystems.base import merge_graphs
from src.ecosystems.registry import (
    build_ecosystem_fragments,
    discover_polyglot_test_commands,
)


def _annotate_python_nodes(
    graph: DepGraph,
    *,
    language_role: str,
    version_constraint: str | None,
) -> DepGraph:
    annotated = graph
    for node in graph.nodes:
        changes = {}
        if node.type in {
            NodeType.IMPORT, NodeType.PACKAGE, NodeType.PROJECT, NodeType.RUNTIME,
        }:
            changes["language"] = node.language or "python"
            changes["language_role"] = node.language_role or language_role
            changes["version_constraint"] = (
                node.version_constraint or version_constraint
            )
        if node.type in {NodeType.PACKAGE, NodeType.PROJECT}:
            changes["ecosystem"] = node.ecosystem or Ecosystem.PYPI
            changes["package_manager"] = node.package_manager or "pip"
            changes["workspace"] = node.workspace or "."
        if node.type is NodeType.PROJECT:
            changes["layer"] = Layer.BUILD
        if changes:
            annotated = annotated.with_node(replace(node, **changes))
    return annotated


def _connect_build_tool_projects(graph: DepGraph) -> DepGraph:
    """Make build-tool projects explicit prerequisites of runtime projects."""
    build_tools = tuple(
        node for node in graph.nodes
        if node.type is NodeType.PROJECT and node.language_role == "build_tool"
    )
    consumers = tuple(
        node for node in graph.nodes
        if node.type is NodeType.PROJECT
        and node.language_role in {"primary_runtime", "secondary_runtime"}
    )
    connected = graph
    for consumer in consumers:
        for build_tool in build_tools:
            if consumer.id == build_tool.id:
                continue
            connected = connected.with_edge(Edge(
                src=consumer.id,
                dst=build_tool.id,
                relation=EdgeType.REQUIRES,
                origin="language_role",
                data={"hard": True, "cross_language": True},
            ))
    return connected


def build_polyglot_dep_graph(
    repo_path: str,
    container_executor,
    *,
    language_requirements,
    base_image: str,
    host_executor=None,
    target_python: str | None = None,
    target_platform: str | None = None,
    needed_extras: frozenset[str] = frozenset(),
    resolver_sandbox=None,
) -> DepGraph:
    """Build one DepGraph from Python plus every non-Python workspace."""
    python_requirement = next(
        (item for item in language_requirements if item.language == "python"),
        None,
    )
    has_python = python_requirement is not None
    if has_python:
        graph = build_dep_graph(
            repo_path,
            container_executor,
            host_executor=host_executor,
            target_python=target_python,
            target_platform=target_platform,
            needed_extras=needed_extras,
        )
        graph = _annotate_python_nodes(
            graph,
            language_role=python_requirement.role,
            version_constraint=python_requirement.version_constraint,
        )
    else:
        graph = DepGraph().with_node(Node(
            id=TEST_NODE_ID,
            type=NodeType.TEST,
            name="repo_tests_pass",
            layer=Layer.TESTS,
            discovered_by=DiscoveredBy.GOAL,
            state=State.MISSING,
        ))

    fragment = build_ecosystem_fragments(
        repo_path,
        language_requirements,
        base_image=base_image,
        platform=target_platform,
        resolver_sandbox=resolver_sandbox,
    )
    graph = merge_graphs(graph, fragment.graph)
    graph = _connect_build_tool_projects(graph)

    # Repository-free scratch certification is valid for runtime/tool presence,
    # but not for dependency/project checks that require the seeded checkout.
    for node in tuple(graph.nodes):
        if node.type in {NodeType.RUNTIME, NodeType.TOOL} and node.check_command:
            graph = certify(graph, node.id, container_executor, cycle=4)

    test_commands = discover_polyglot_test_commands(repo_path, language_requirements)
    combined_test = " && ".join(test_commands)
    test_node = graph.get(TEST_NODE_ID)
    if test_node is None:
        test_node = Node(
            id=TEST_NODE_ID,
            type=NodeType.TEST,
            name="repo_tests_pass",
            layer=Layer.TESTS,
            discovered_by=DiscoveredBy.GOAL,
            state=State.MISSING,
        )
    graph = graph.with_node(replace(
        test_node,
        check_command=combined_test or test_node.check_command,
        data={
            **dict(test_node.data),
            "test_commands": test_commands,
            "polyglot": len(language_requirements) > 1,
        },
    ))
    for project in tuple(
        node for node in graph.nodes
        if node.type is NodeType.PROJECT and node.ecosystem is not Ecosystem.PYPI
    ):
        graph = graph.with_edge(Edge(
            src=TEST_NODE_ID,
            dst=project.id,
            relation=EdgeType.REQUIRES,
            origin="provider",
            data={"hard": bool(project.setup_commands and project.check_command)},
        ))
    return graph
