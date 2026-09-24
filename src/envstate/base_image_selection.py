"""Base-image selection facade: pick a candidate, then pin its python minor.

Composes two existing pieces into the single decision the conductor needs:

* :class:`src.image_selector.ImageSelector` — LLM picks one image from a
  language handler's candidate list (ported from the pruned agent path).
* :func:`src.envstate.runtime_base.resolve_runtime_base` — reads
  ``requires-python`` and pins the chosen tag's python minor.

Returns ONE :class:`BaseImageChoice` whose ``minor`` is meant to flow into BOTH
the container tag AND ``build_dep_graph(target_python=...)`` (one decision, two
consumers). An explicit override is honored verbatim (no LLM, no rewrite). Any
selection failure degrades to ``python:{DEFAULT_MINOR}-slim`` so a run never
dies on image selection.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from src.image_selector import ImageSelector
from src.language_handlers import LanguageRequirement, detect_languages
from src.envstate.runtime_base import (
    DEFAULT_MINOR,
    _base_image_minor,
    resolve_runtime_base,
)

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover - deterministic Dockerfile path remains
    yaml = None

_DEFAULT_IMAGE = f"python:{DEFAULT_MINOR}-slim"

# Matches a BARE `python:X.Y` or `python:X.Y.Z` tag with no variant suffix
# (no `-slim`, `-bookworm`, `-alpine`, etc. and no trailing `latest`/garbage).
_BARE_PYTHON_TAG = re.compile(r"^python:\d+\.\d+(?:\.\d+)?$")
_ALPINE_PYTHON_TAG = re.compile(
    r"^(?P<prefix>(?:[^/@]+/)*python):(?P<version>\d+\.\d+(?:\.\d+)?)(?:-alpine[^@]*)?(?P<digest>@sha256:[A-Fa-f0-9]{64})?$"
)


def _is_python_image(image: str) -> bool:
    """Whether *image* is a tagged official Python image, registry-qualified or not."""
    reference = image.split("@", 1)[0]
    name, sep, tag = reference.rpartition(":")
    return bool(sep and tag and "/" not in tag and name.split("/")[-1] == "python")


def _ensure_slim(image: str) -> str:
    """Append ``-slim`` to a BARE ``python:X.Y[.Z]`` tag; unchanged otherwise.

    The AUTO path (``ImageSelector`` + ``resolve_runtime_base``) only rewrites
    the python minor of whatever base the selector proposed — it never adds a
    variant. Left alone, a bare candidate like ``python:3.10`` resolves to the
    ~1GB buildpack-deps image (gcc + every ``-dev`` header preinstalled), which
    pre-satisfies the graph's System-tier discovery and kills its signal. Any
    image that already has a variant suffix (``-slim``, ``-bookworm``,
    ``-alpine``, ...), isn't a recognizable ``X.Y`` tag (``python:latest``), or
    isn't a ``python:`` base at all is returned unchanged.
    """
    # The v3 setup/runtime stack is Debian-oriented (bash + apt-get).  Keeping
    # an automatically selected Alpine image would make the container fail
    # before dependency reasoning starts.  Explicit --base-image overrides are
    # handled before this helper and remain verbatim.
    alpine = _ALPINE_PYTHON_TAG.match(image)
    if alpine and "-alpine" in image.split("@", 1)[0]:
        digest = alpine.group("digest") or ""
        return f"{alpine.group('prefix')}:{alpine.group('version')}-slim{digest}"
    if _BARE_PYTHON_TAG.match(image):
        return f"{image}-slim"
    return image


@dataclass(frozen=True)
class BaseImageChoice:
    """The base-image decision: the tag to boot, the python minor to resolve
    against, an optional docker ``--platform`` override, and provenance."""

    image: str
    minor: str
    platform_override: str | None
    reason: str
    languages: tuple[LanguageRequirement, ...] = ()
    primary_language: str | None = None


def _primary_language(languages: tuple[LanguageRequirement, ...]) -> str | None:
    return next(
        (item.language for item in languages if item.role == "primary_runtime"),
        languages[0].language if languages else None,
    )


def _concrete_image(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("image")
    if not isinstance(value, str):
        return None
    image = value.removeprefix("docker://").strip()
    if (
        image
        and "$" not in image
        and "{{" not in image
        and image.lower() != "scratch"
        and ":" in image
    ):
        return image
    return None


def _ci_declared_image(repo_path: str) -> tuple[str, str] | None:
    if yaml is None:
        return None
    workflow_dir = os.path.join(repo_path, ".github", "workflows")
    workflow_paths = []
    if os.path.isdir(workflow_dir):
        workflow_paths.extend(
            os.path.join(workflow_dir, name)
            for name in sorted(os.listdir(workflow_dir))
            if name.endswith((".yml", ".yaml"))
        )
    for path in workflow_paths:
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue
        for job in (data.get("jobs") or {}).values():
            if not isinstance(job, dict):
                continue
            image = _concrete_image(job.get("container"))
            if image:
                return image, os.path.relpath(path, repo_path)
            for step in job.get("steps") or ():
                if isinstance(step, dict):
                    image = _concrete_image(step.get("uses"))
                    if image and str(step.get("uses", "")).startswith("docker://"):
                        return image, os.path.relpath(path, repo_path)

    gitlab_path = os.path.join(repo_path, ".gitlab-ci.yml")
    if os.path.isfile(gitlab_path):
        try:
            with open(gitlab_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            data = {}
        candidates = [data.get("image")]
        if isinstance(data.get("default"), dict):
            candidates.append(data["default"].get("image"))
        for key, value in data.items():
            if not str(key).startswith(".") and isinstance(value, dict):
                candidates.append(value.get("image"))
        for candidate in candidates:
            image = _concrete_image(candidate)
            if image:
                return image, ".gitlab-ci.yml"

    circle_path = os.path.join(repo_path, ".circleci", "config.yml")
    if os.path.isfile(circle_path):
        try:
            with open(circle_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            data = {}
        for job in (data.get("jobs") or {}).values():
            if not isinstance(job, dict):
                continue
            docker_images = job.get("docker") or ()
            if docker_images:
                image = _concrete_image(docker_images[0])
                if image:
                    return image, ".circleci/config.yml"
    return None


def _trusted_declared_image(repo_path: str) -> tuple[str, str] | None:
    """Read a concrete repository-owned Dockerfile or CI development image."""
    for filename in ("Dockerfile", "dockerfile"):
        path = os.path.join(repo_path, filename)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read(64_000)
        except OSError:
            continue
        for match in re.finditer(r"(?im)^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)", text):
            image = match.group(1).strip()
            if (
                image
                and "$" not in image
                and image.lower() != "scratch"
                and ":" in image
            ):
                return image, filename
    return _ci_declared_image(repo_path)


_ROOT_RUNTIME_MANIFESTS = {
    "python": {"pyproject.toml", "setup.py", "requirements.txt"},
    "javascript": {"package.json"},
    "typescript": {"package.json", "tsconfig.json"},
    "rust": {"Cargo.toml"},
    "go": {"go.mod"},
    "java": {"pom.xml", "build.gradle", "build.gradle.kts"},
}


def _generic_polyglot_base(
    languages: tuple[LanguageRequirement, ...],
) -> bool:
    """Detect a repository with several equally important root runtimes."""
    runtimes = tuple(item for item in languages if item.role != "build_tool")
    if len(runtimes) < 2:
        return False
    root_manifest_owners = 0
    for item in runtimes:
        expected = _ROOT_RUNTIME_MANIFESTS.get(item.language, set())
        root_evidence = {
            path for path in item.evidence if "/" not in path.strip("/")
        }
        if root_evidence & expected:
            root_manifest_owners += 1
    return root_manifest_owners != 1


_FALLBACK_IMAGES = {
    "python": _DEFAULT_IMAGE,
    "javascript": "node:20-bookworm",
    "typescript": "node:20-bookworm",
    "rust": "rust:1.82-bookworm",
    "go": "golang:1.22-bookworm",
    "java": "eclipse-temurin:17-jdk-noble",
}


def choose_base_image(
    repo_path: str,
    client,
    model: str,
    *,
    explicit: str | None = None,
    log_dir: str | None = None,
    language_hint: str | None = None,
) -> BaseImageChoice:
    """Select + pin the base image for ``repo_path``.

    ``explicit`` (any value other than ``None``) is honored verbatim — no LLM,
    no tag rewrite; ``minor`` is read from the tag (or ``DEFAULT_MINOR``). When
    ``explicit`` is ``None`` the LLM ``ImageSelector`` picks a candidate and
    ``resolve_runtime_base`` pins its minor against ``requires-python``. Never
    raises: on any failure returns the default ``python:{DEFAULT_MINOR}-slim``.
    """
    if explicit is not None:
        languages = detect_languages(repo_path)
        primary = language_hint or _primary_language(languages)
        minor = _base_image_minor(explicit) or DEFAULT_MINOR
        return BaseImageChoice(
            explicit, minor, None, f"explicit override: {explicit}",
            languages, primary,
        )

    languages = detect_languages(repo_path)
    primary = language_hint or _primary_language(languages)
    try:
        declared = _trusted_declared_image(repo_path)
        if declared is not None:
            declared_image, declared_source = declared
            if _is_python_image(declared_image):
                # A repository-owned Dockerfile/CI image is stronger evidence than
                # an LLM choice, but a floating or incompatible Python tag still
                # needs the same requires-python guard as every other auto-selected
                # Python image.  In particular, ``python:3-slim`` must not float to
                # whichever newest Python Docker Hub serves on the day of the run.
                decision = resolve_runtime_base(repo_path, declared_image)
                image = _ensure_slim(decision.base_image)
                minor = decision.minor
                detail = f" -> pinned {decision.base_image!r} ({decision.reason})"
                normalized_note = (
                    "" if image == decision.base_image
                    else f" -> normalized to {image!r}"
                )
            else:
                image = declared_image
                minor = _base_image_minor(declared_image) or DEFAULT_MINOR
                detail = ""
                normalized_note = ""
            return BaseImageChoice(
                image=image,
                minor=minor,
                platform_override=None,
                reason=(
                    f"auto: trusted {declared_source} declares "
                    f"{declared_image!r}{detail}{normalized_note}"
                ),
                languages=languages,
                primary_language=primary,
            )
        if _generic_polyglot_base(languages):
            python_req = next(
                (item for item in languages if item.language == "python"),
                None,
            )
            version_match = re.search(
                r"\d+\.\d+", python_req.version_constraint or ""
            ) if python_req else None
            return BaseImageChoice(
                image="ubuntu:24.04",
                minor=version_match.group(0) if version_match else DEFAULT_MINOR,
                platform_override=None,
                reason=(
                    "auto: multiple equally important root runtimes; "
                    "using a language-neutral base"
                ),
                languages=languages,
                primary_language=primary,
            )
        selector = ImageSelector(client, model=model)
        selected, _handler, _docs, platform_override = selector.select_base_image(
            repo_path,
            log_dir=log_dir,
            language_hint=primary,
            language_requirements=languages,
        )
        if _is_python_image(selected):
            decision = resolve_runtime_base(repo_path, selected)
            image = _ensure_slim(decision.base_image)
            minor = decision.minor
            detail = f" -> pinned {decision.base_image!r} ({decision.reason})"
            normalized_note = (
                "" if image == decision.base_image
                else f" -> normalized to {image!r}"
            )
        else:
            image = selected
            python_req = next(
                (item for item in languages if item.language == "python"),
                None,
            )
            version_match = re.search(
                r"\d+\.\d+", python_req.version_constraint or ""
            ) if python_req else None
            minor = version_match.group(0) if version_match else DEFAULT_MINOR
            detail = ""
            normalized_note = ""
        return BaseImageChoice(
            image=image,
            minor=minor,
            platform_override=platform_override,
            reason=(
                f"auto: languages={[item.language for item in languages]!r}; "
                f"primary={primary!r}; selected {selected!r}{detail}{normalized_note}"
            ),
            languages=languages,
            primary_language=primary,
        )
    except Exception as exc:  # noqa: BLE001 — selection must never break a run
        fallback = _FALLBACK_IMAGES.get(primary or "", "ubuntu:24.04")
        logger.warning("base-image selection unavailable, using %s: %s", fallback, exc)
        return BaseImageChoice(
            fallback,
            DEFAULT_MINOR,
            None,
            f"degraded fallback for {primary or 'polyglot'}: {exc}",
            languages,
            primary,
        )
