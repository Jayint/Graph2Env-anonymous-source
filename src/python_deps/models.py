"""Data models for the v3 depgraph stack.

This is the verbatim subset of the original ``python_deps.models`` needed by
``evidence.py``, ``import_graph.py`` and ``failure_classifier.py`` — all
transitively reached from ``depgraph/build.py`` and ``depgraph/roots.py``.

The z3-era constraint/solver classes (DependencyConstraint, ConstraintGraph,
ConstraintEdge, SolverResult, DependencyReport, PythonCandidate,
PackageCandidate, SuggestedCommand) are excluded from the v3-only branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PythonRequirement:
    name: str
    specifier: str = ""
    marker: str = ""
    extras: tuple[str, ...] = ()
    source: str = ""
    kind: str = "dependency"
    trust: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PythonVersionRequirement:
    specifier: str
    source: str
    trust: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportFinding:
    import_name: str
    classification: str
    source_files: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportPackageMapping:
    import_name: str
    package_name: str
    source: str
    trust: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependencyFailure:
    failure_type: str
    command: str = ""
    import_name: str | None = None
    package_name: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_dependency_shaped(self) -> bool:
        return self.failure_type != "not_dependency_related"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PythonDependencyEvidence:
    repo_path: str
    python_requires: list[PythonVersionRequirement] = field(default_factory=list)
    declared_dependencies: list[PythonRequirement] = field(default_factory=list)
    constraint_dependencies: list[PythonRequirement] = field(default_factory=list)
    imports: list[ImportFinding] = field(default_factory=list)
    import_package_mappings: list[ImportPackageMapping] = field(default_factory=list)
    project_local_modules: list[str] = field(default_factory=list)
    pydeps: dict[str, Any] = field(default_factory=dict)
    collection_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        import_groups: dict[str, list[str]] = {}
        for finding in self.imports:
            import_groups.setdefault(finding.classification, []).append(finding.import_name)

        return {
            "repo_path": self.repo_path,
            "python_requires": [item.to_dict() for item in self.python_requires],
            "declared_dependencies": [item.to_dict() for item in self.declared_dependencies],
            "constraint_dependencies": [item.to_dict() for item in self.constraint_dependencies],
            "imports": [item.to_dict() for item in self.imports],
            "import_graph": {
                key: sorted(set(values))
                for key, values in sorted(import_groups.items())
            },
            "import_package_mappings": [
                item.to_dict() for item in self.import_package_mappings
            ],
            "project_local_modules": sorted(set(self.project_local_modules)),
            "pydeps": dict(self.pydeps),
            "collection_errors": list(self.collection_errors),
        }
