"""Stable node-id constructors for the concrete dependency graph.

Ids are deterministic strings of the form ``<kind>:<name>`` so the same
entity discovered by different stages collapses onto a single node.
"""

from __future__ import annotations

import re

# Characters that are safe to keep verbatim inside an id segment.  Package
# names are case-sensitive (``Pillow`` != ``pillow``) and may legitimately
# contain ``.``, ``-``, ``_`` and ``==`` (version pin), so we preserve those.
_UNSAFE = re.compile(r"[^A-Za-z0-9._=+-]+")

TEST_NODE_ID = "test:repo_tests_pass"


def slug(text: str) -> str:
    """Sanitize free text into an id-safe segment (whitespace/odd chars -> '-')."""
    cleaned = _UNSAFE.sub("-", text.strip())
    return cleaned.strip("-")


def import_id(name: str) -> str:
    return f"import:{name}"


def package_id(name: str, version: str | None) -> str:
    if version:
        return f"pkg:{name}=={version}"
    return f"pkg:{name}"


def syslib_id(soname: str) -> str:
    return f"syslib:{soname}"


def tool_id(tool: str) -> str:
    return f"tool:{tool}"


def project_id(name: str) -> str:
    return f"project:{slug(name)}"


def config_id(name: str) -> str:
    return f"config:{name}"


def service_id(name: str) -> str:
    return f"service:{name}"


def data_asset_id(name: str) -> str:
    return f"data:{name}"


def runtime_id(minor: str) -> str:
    return f"runtime:python-{minor}"


def ecosystem_value(ecosystem) -> str:
    return getattr(ecosystem, "value", str(ecosystem))


def workspace_segment(workspace: str | None) -> str:
    normalized = (workspace or ".").replace("\\", "/").strip("/")
    return slug(normalized or ".")


def ecosystem_import_id(
    ecosystem,
    workspace: str,
    raw_specifier: str,
) -> str:
    return (
        f"import:{ecosystem_value(ecosystem)}:{workspace_segment(workspace)}:"
        f"{slug(raw_specifier)}"
    )


def requirement_id(ecosystem, workspace: str, name: str) -> str:
    return (
        f"req:{ecosystem_value(ecosystem)}:{workspace_segment(workspace)}:"
        f"{slug(name)}"
    )


def ecosystem_package_id(ecosystem, locator: str) -> str:
    return f"pkg:{ecosystem_value(ecosystem)}:{slug(locator)}"


def dependency_set_id(ecosystem, workspace: str) -> str:
    return f"deps:{ecosystem_value(ecosystem)}:{workspace_segment(workspace)}"


def ecosystem_project_id(ecosystem, workspace: str) -> str:
    return f"project:{ecosystem_value(ecosystem)}:{workspace_segment(workspace)}"


def ecosystem_runtime_id(language: str, version: str | None = None) -> str:
    suffix = slug(version or "default")
    return f"runtime:{slug(language)}:{suffix}"


def ecosystem_tool_id(ecosystem, tool: str) -> str:
    return f"tool:{ecosystem_value(ecosystem)}:{slug(tool)}"
