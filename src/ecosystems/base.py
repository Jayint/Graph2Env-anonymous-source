"""Provider contracts and graph-normalization helpers for non-Python ecosystems."""
from __future__ import annotations

import hashlib
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Protocol

from src.python_deps.depgraph.ids import (
    dependency_set_id,
    ecosystem_import_id,
    ecosystem_package_id,
    ecosystem_project_id,
    ecosystem_runtime_id,
    ecosystem_tool_id,
    requirement_id,
)
from src.python_deps.depgraph.schema import (
    DepGraph,
    DiscoveredBy,
    Ecosystem,
    Edge,
    EdgeType,
    Layer,
    Node,
    NodeType,
    State,
    Strength,
)


@dataclass(frozen=True)
class Workspace:
    language: str
    ecosystem: Ecosystem
    root: str
    manifest_file: str
    package_manager: str
    version_constraint: str | None
    role: str
    package_manager_version: str | None = None
    command_runner: str | None = None
    # Source language and runtime are not always the same: TypeScript executes
    # on Node.js. Keep both identities so a Node engine constraint never creates
    # a fictitious `runtime:typescript:*` obligation.
    runtime_language: str | None = None
    runtime_version_constraint: str | None = None
    resolved_runtime_version: str | None = None


@dataclass(frozen=True)
class ProviderTarget:
    base_image: str
    platform: str | None = None


@dataclass(frozen=True)
class ImportRef:
    language: str
    workspace: str
    raw_specifier: str
    source_file: str
    scope: str = "runtime"
    conditional: str | None = None
    confidence: str = "high"
    package_name: str | None = None
    package_locator: str | None = None


@dataclass(frozen=True)
class RequirementRecord:
    name: str
    constraint: str
    scope: str = "runtime"
    source: str = ""
    package_locator: str | None = None


@dataclass(frozen=True)
class PackageRecord:
    name: str
    version: str | None
    locator: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeResolverEvidence:
    command: str
    status: str
    output_digest: str | None = None
    error: str | None = None
    raw_output: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True)
class ResolutionResult:
    requirements: tuple[RequirementRecord, ...] = ()
    packages: tuple[PackageRecord, ...] = ()
    status: str = "unresolved"
    error: str | None = None
    lock_file: str | None = None
    lock_digest: str | None = None
    resolution_source: str = "manifest"
    reproducibility_confidence: str = "low"
    native_evidence: NativeResolverEvidence | None = None
    class_index: tuple[tuple[str, str], ...] = ()
    scanner_imports: tuple[ImportRef, ...] = ()


@dataclass(frozen=True)
class DepGraphFragment:
    graph: DepGraph
    workspaces: tuple[Workspace, ...] = ()
    test_commands: tuple[str, ...] = ()


class EcosystemProvider(Protocol):
    ecosystems: tuple[Ecosystem, ...]

    def detect_workspaces(
        self,
        repo_path: str,
        language_requirements,
    ) -> tuple[Workspace, ...]: ...

    def detect_runtime(self, workspace: Workspace) -> str | None: ...

    def resolve(
        self,
        repo_path: str,
        workspace: Workspace,
        target: ProviderTarget,
        sandbox=None,
    ) -> ResolutionResult: ...

    def scan_imports(
        self,
        repo_path: str,
        workspace: Workspace,
        target: ProviderTarget,
    ) -> tuple[ImportRef, ...]: ...

    def dependency_commands(
        self,
        workspace: Workspace,
        resolution: ResolutionResult,
    ) -> tuple[tuple[str, ...], str]: ...

    def project_commands(self, workspace: Workspace) -> tuple[tuple[str, ...], str | None]: ...

    def test_commands(self, workspace: Workspace) -> tuple[str, ...]: ...


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeError):
        return ""


def file_digest(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return "sha256:" + hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def workspace_path(repo_path: str, workspace: Workspace) -> str:
    return repo_path if workspace.root == "." else os.path.join(repo_path, workspace.root)


def in_workspace(workspace: Workspace, command: str) -> str:
    if workspace.root == ".":
        return command
    return f"cd {shlex.quote(workspace.root)} && {command}"


def run_native_resolver_probe(
    sandbox,
    *,
    ecosystem: Ecosystem,
    workspace: Workspace,
    command: str,
    prepare_commands: tuple[str, ...] = (),
    timeout_seconds: int = 300,
) -> NativeResolverEvidence | None:
    """Run resolver logic in an isolated container and return parseable evidence."""
    if sandbox is None:
        return None
    required = (
        "create_resolver_container",
        "resolver_exec",
        "close_resolver_container",
    )
    if any(not callable(getattr(sandbox, name, None)) for name in required):
        return None
    script = " && ".join((*prepare_commands, command))
    identity = f"{ecosystem.value}:{workspace.root}:{script}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    safe_root = re.sub(r"[^a-zA-Z0-9_.-]+", "-", workspace.root).strip("-") or "root"
    resolver_id = f"{ecosystem.value}-{safe_root}-{suffix}"
    handle = None
    wrapped = f"timeout {int(timeout_seconds)}s bash -lc {shlex.quote(script)}"
    try:
        handle = sandbox.create_resolver_container(resolver_id)
        rc, output = sandbox.resolver_exec(handle, wrapped)
        digest = (
            "sha256:" + hashlib.sha256(output.encode()).hexdigest()
            if output else None
        )
        if rc == 0:
            return NativeResolverEvidence(
                command=script,
                status="resolved",
                output_digest=digest,
                raw_output=output,
            )
        tail = " ".join((output or "").strip().splitlines()[-3:])
        return NativeResolverEvidence(
            command=script,
            status="failed",
            output_digest=digest,
            error=f"resolver exited {rc}" + (f": {tail}" if tail else ""),
            raw_output=output,
        )
    except Exception as exc:
        return NativeResolverEvidence(
            command=script,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if handle is not None:
            try:
                sandbox.close_resolver_container(handle)
            except Exception:
                pass


def merge_graphs(base: DepGraph, fragment: DepGraph) -> DepGraph:
    """Merge immutable graphs, validating edges only after every node exists."""
    merged = base
    for node in fragment.nodes:
        existing = merged.get(node.id)
        if existing is None:
            merged = merged.with_node(node)
        elif existing != node:
            # Provider-qualified ids should avoid collisions. Preserve the
            # official graph if a malformed fragment still collides.
            continue
    for edge in fragment.edges:
        if merged.get(edge.src) is not None and merged.get(edge.dst) is not None:
            merged = merged.with_edge(edge)
    return merged


def _node_state(has_setup: bool) -> State:
    return State.MISSING if has_setup else State.UNKNOWN


def graph_from_resolution(
    *,
    workspace: Workspace,
    target: ProviderTarget,
    resolution: ResolutionResult,
    imports: tuple[ImportRef, ...],
    runtime_setup: tuple[str, ...],
    runtime_check: str,
    tool_name: str,
    tool_setup: tuple[str, ...],
    tool_check: str,
    dependency_commands: tuple[str, ...],
    dependency_check: str,
    project_commands: tuple[str, ...],
    project_check: str | None,
    test_commands: tuple[str, ...],
) -> DepGraphFragment:
    """Normalize one provider result into the shared four-level graph."""
    graph = DepGraph()
    runtime_language = workspace.runtime_language or workspace.language
    runtime_constraint = (
        workspace.runtime_version_constraint or workspace.version_constraint
    )
    runtime_version = workspace.resolved_runtime_version or runtime_constraint or "default"
    runtime_node_id = ecosystem_runtime_id(runtime_language, runtime_version)
    tool_node_id = ecosystem_tool_id(workspace.ecosystem, tool_name)
    deps_node_id = dependency_set_id(workspace.ecosystem, workspace.root)
    project_node_id = ecosystem_project_id(workspace.ecosystem, workspace.root)

    common = {
        "ecosystem": workspace.ecosystem,
        "workspace": workspace.root,
        "package_manager": workspace.package_manager,
        "language": workspace.language,
        "language_role": workspace.role,
        "version_constraint": workspace.version_constraint,
    }
    graph = graph.with_node(Node(
        id=runtime_node_id,
        type=NodeType.RUNTIME,
        name=f"{runtime_language} runtime",
        layer=Layer.RUNTIME,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=_node_state(bool(runtime_setup)),
        version=workspace.resolved_runtime_version or runtime_constraint,
        check_command=runtime_check,
        setup_commands=runtime_setup,
        strength=Strength.HARD if runtime_setup else Strength.SOFT,
        **{
            **common,
            "language": runtime_language,
            "version_constraint": runtime_constraint,
        },
    ))
    graph = graph.with_node(Node(
        id=tool_node_id,
        type=NodeType.TOOL,
        name=tool_name,
        layer=Layer.TOOLCHAIN,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=_node_state(bool(tool_setup)),
        check_command=tool_check,
        setup_commands=tool_setup,
        strength=Strength.HARD if tool_setup else Strength.SOFT,
        **common,
    ))
    graph = graph.with_node(Node(
        id=deps_node_id,
        type=NodeType.DEPENDENCY_SET,
        name=f"{workspace.package_manager} dependencies ({workspace.root})",
        layer=Layer.DEPENDENCIES,
        discovered_by=DiscoveredBy.RESOLVER,
        state=State.MISSING,
        check_command=dependency_check,
        setup_commands=dependency_commands,
        strength=Strength.HARD,
        lock_digest=resolution.lock_digest,
        resolution_status=resolution.status,
        resolution_error=resolution.error,
        manifest_source=workspace.manifest_file,
        data={
            "resolution_source": resolution.resolution_source,
            "reproducibility_confidence": resolution.reproducibility_confidence,
            "package_manager_version": workspace.package_manager_version,
            "command_runner": workspace.command_runner,
            "class_index_entries": len(resolution.class_index),
            "native_resolver": (
                {
                    "command": resolution.native_evidence.command,
                    "status": resolution.native_evidence.status,
                    "output_digest": resolution.native_evidence.output_digest,
                    "error": resolution.native_evidence.error,
                }
                if resolution.native_evidence else None
            ),
        },
        **common,
    ))
    graph = graph.with_node(Node(
        id=project_node_id,
        type=NodeType.PROJECT,
        name=f"{workspace.language} project ({workspace.root})",
        layer=Layer.BUILD,
        discovered_by=DiscoveredBy.STATIC_SCAN,
        state=_node_state(bool(project_commands)),
        check_command=project_check,
        setup_commands=project_commands,
        strength=Strength.HARD if project_commands else Strength.SOFT,
        provenance=workspace.manifest_file,
        **common,
    ))
    graph = graph.with_edge(Edge(
        src=tool_node_id,
        dst=runtime_node_id,
        relation=EdgeType.REQUIRES,
        origin="provider",
    ))
    graph = graph.with_edge(Edge(
        src=deps_node_id,
        dst=runtime_node_id,
        relation=EdgeType.REQUIRES,
        origin="provider",
    ))
    graph = graph.with_edge(Edge(
        src=deps_node_id,
        dst=tool_node_id,
        relation=EdgeType.REQUIRES,
        origin="provider",
    ))
    graph = graph.with_edge(Edge(
        src=project_node_id,
        dst=deps_node_id,
        relation=EdgeType.REQUIRES,
        origin="provider",
    ))

    locator_ids: dict[str, str] = {}
    name_locators: dict[str, list[str]] = {}
    for package in resolution.packages:
        node_id = ecosystem_package_id(workspace.ecosystem, package.locator)
        locator_ids[package.locator] = node_id
        name_locators.setdefault(package.name.lower(), []).append(package.locator)
        graph = graph.with_node(Node(
            id=node_id,
            type=NodeType.PACKAGE,
            name=package.name,
            layer=Layer.DEPENDENCIES,
            discovered_by=DiscoveredBy.RESOLVER,
            state=State.UNKNOWN,
            version=package.version,
            resolved_locator=package.locator,
            lock_digest=resolution.lock_digest,
            resolution_status="resolved",
            **common,
        ))
        graph = graph.with_edge(Edge(
            src=deps_node_id,
            dst=node_id,
            relation=EdgeType.DESCRIBES,
            origin="resolver",
        ))

    for package in resolution.packages:
        src = locator_ids.get(package.locator)
        if src is None:
            continue
        for dependency_locator in package.dependencies:
            dst = locator_ids.get(dependency_locator)
            if dst is not None:
                graph = graph.with_edge(Edge(
                    src=src,
                    dst=dst,
                    relation=EdgeType.REQUIRES,
                    origin="resolver",
                    data={"hard": False},
                ))

    for requirement in resolution.requirements:
        req_id = requirement_id(
            workspace.ecosystem, workspace.root, requirement.name
        )
        graph = graph.with_node(Node(
            id=req_id,
            type=NodeType.REQUIREMENT,
            name=requirement.name,
            layer=Layer.DEPENDENCIES,
            discovered_by=DiscoveredBy.STATIC_SCAN,
            state=State.UNKNOWN,
            declared_constraint=requirement.constraint,
            declared_specifier=requirement.constraint,
            manifest_source=requirement.source or workspace.manifest_file,
            resolution_status=(
                "resolved" if requirement.package_locator else resolution.status
            ),
            resolution_error=resolution.error,
            data={"scope": requirement.scope},
            **common,
        ))
        graph = graph.with_edge(Edge(
            src=project_node_id,
            dst=req_id,
            relation=EdgeType.REQUIRES,
            origin="manifest",
            data={"hard": False},
        ))
        package_locator = requirement.package_locator
        if package_locator is None:
            candidates = name_locators.get(requirement.name.lower(), ())
            package_locator = candidates[0] if candidates else None
        package_node_id = locator_ids.get(package_locator or "")
        if package_node_id is not None:
            graph = graph.with_edge(Edge(
                src=req_id,
                dst=package_node_id,
                relation=EdgeType.RESOLVES_TO,
                origin="resolver",
            ))

    for ref in imports:
        import_node_id = ecosystem_import_id(
            workspace.ecosystem, workspace.root, ref.raw_specifier
        )
        graph = graph.with_node(Node(
            id=import_node_id,
            type=NodeType.IMPORT,
            name=ref.raw_specifier,
            layer=Layer.NAMING,
            discovered_by=DiscoveredBy.STATIC_SCAN,
            state=State.UNKNOWN,
            provenance=ref.source_file,
            data={
                "scope": ref.scope,
                "conditional": ref.conditional,
                "confidence": ref.confidence,
            },
            **common,
        ))
        if ref.package_locator and ref.package_locator in locator_ids:
            package_node_id = locator_ids[ref.package_locator]
        else:
            package_name = (ref.package_name or ref.raw_specifier).lower()
            candidates = name_locators.get(package_name, ())
            package_node_id = locator_ids[candidates[0]] if candidates else None
        if package_node_id:
            graph = graph.with_edge(Edge(
                src=import_node_id,
                dst=package_node_id,
                relation=EdgeType.MAPS_TO,
                origin="static_scan",
            ))

    return DepGraphFragment(
        graph=graph,
        workspaces=(workspace,),
        test_commands=test_commands,
    )
