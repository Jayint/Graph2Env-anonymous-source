"""Polyglot ecosystem providers and DepGraph composition."""

from src.ecosystems.base import (
    DepGraphFragment,
    EcosystemProvider,
    ImportRef,
    PackageRecord,
    ProviderTarget,
    RequirementRecord,
    ResolutionResult,
    Workspace,
)
from src.ecosystems.registry import (
    build_ecosystem_fragments,
    discover_polyglot_test_commands,
)

__all__ = [
    "DepGraphFragment",
    "EcosystemProvider",
    "ImportRef",
    "PackageRecord",
    "ProviderTarget",
    "RequirementRecord",
    "ResolutionResult",
    "Workspace",
    "build_ecosystem_fragments",
    "discover_polyglot_test_commands",
]
