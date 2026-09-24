# src/envstate/runtime_base.py
"""Runtime-tier base-image selection (pure).

Two responsibilities, both side-effect-free:

* :func:`choose_python_minor` — turn the relevant projects' intersected Python constraint
  (PEP 621 ``requires-python`` or poetry ``^``/``~`` form) into ONE concrete
  minor (e.g. ``"3.10"``). Policy v2: defer to the rich ImageSelector's chosen
  python (``prefer``); only clamp to satisfy ``requires-python``. Without a
  ``prefer``, keep ``default`` if it satisfies, else the nearest supported minor
  to ``default`` within the constraint.
* :func:`pin_base_python` — rewrite the python component of a ``python:X`` base
  tag to a chosen minor, preserving the OS/variant suffix the Synthesizer chose.

Undeclared / unparseable / out-of-range constraints fall back to ``prefer`` (if
given) or ``default`` so the feature is byte-identical to today when it can't
improve on the guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from src.envstate.manifest import read_requires_python

# Ascending order — policy searches within this set.
SUPPORTED_MINORS: tuple[str, ...] = (
    "3.9", "3.10", "3.11", "3.12", "3.13", "3.14",
)
DEFAULT_MINOR = "3.11"

_VERSION_TOKEN = re.compile(r"^\d+(?:\.\d+)*$")


def _split_image_digest(image: str) -> tuple[str, str | None]:
    """Return ``(name_and_tag, digest)`` for a Docker image reference."""
    reference, sep, digest = image.partition("@")
    return reference, digest if sep and digest else None


def _normalize_constraint(raw: str) -> str:
    """Normalize poetry ``^``/``~`` shorthand to a PEP 440 specifier string.

    ``^3.10`` -> ``">=3.10,<4.0"``;  ``~3.10`` -> ``">=3.10,<3.11"``. Anything
    else (already a PEP 440 specifier) is returned unchanged for SpecifierSet.
    """
    raw = raw.strip()
    if raw.startswith("^"):
        ver = raw[1:].strip()
        major = ver.split(".")[0]
        return f">={ver},<{int(major) + 1}.0"
    if raw.startswith("~"):
        ver = raw[1:].strip()
        parts = ver.split(".")
        if len(parts) >= 2:
            return f">={ver},<{parts[0]}.{int(parts[1]) + 1}"
        return f">={ver},<{int(parts[0]) + 1}"
    return raw


def _base_image_minor(base_image: str) -> str | None:
    """The python minor already in a ``python:X.Y...`` tag (the ImageSelector's chosen
    base), or None when the base is not a recognizable supported ``python:X.Y`` tag."""
    reference, _digest = _split_image_digest(base_image)
    name, sep, tag = reference.rpartition(":")
    if not sep or "/" in tag or name.split("/")[-1] != "python":
        return None
    m = re.match(r"^(\d+\.\d+)", tag.split("-")[0])
    minor = m.group(1) if m else None
    return minor if minor in SUPPORTED_MINORS else None


def _nearest_supported(satisfying: list[str], anchor: str) -> str:
    """The satisfying minor closest to ``anchor`` by position in SUPPORTED_MINORS
    (ties -> the higher minor)."""
    idx = {m: i for i, m in enumerate(SUPPORTED_MINORS)}
    a = idx.get(anchor, idx[DEFAULT_MINOR])
    return min(satisfying, key=lambda m: (abs(idx[m] - a), -idx[m]))


def choose_python_minor(
    requires_python: str | None,
    default: str = DEFAULT_MINOR,
    *,
    prefer: str | None = None,
) -> tuple[str, str]:
    """Return ``(minor, reason)``. The pin sits ON TOP of the rich ImageSelector;
    ``prefer`` is the minor the selector already chose. Policy:
      * keep ``prefer`` when it satisfies ``requires_python`` (common case — never
        override a good rich choice);
      * clamp ``prefer`` to the nearest supported satisfying minor when it violates
        the constraint (LLM out-of-bounds guard);
      * with no ``prefer``: ``default`` if it satisfies, else the nearest supported
        minor to ``default`` within the constraint;
      * ``default`` when nothing is declared or nothing supported satisfies.
    Never raises.
    """
    if not requires_python or not requires_python.strip():
        if prefer:
            return prefer, f"no requires-python; kept selector base {prefer}"
        return default, "no requires-python declared; default"
    raw = requires_python.strip()
    try:
        spec = SpecifierSet(_normalize_constraint(raw))
    except Exception:
        if prefer:
            return prefer, f"unparseable requires-python {raw!r}; kept selector base {prefer}"
        return default, f"unparseable requires-python {raw!r}; default"
    satisfying = [m for m in SUPPORTED_MINORS
                  if spec.contains(Version(f"{m}.0"), prereleases=False)]
    if not satisfying:
        return default, f"no supported minor satisfies {raw!r}; default"
    if prefer and prefer in satisfying:
        return prefer, f"kept selector base {prefer} (satisfies {raw!r})"
    if prefer:
        chosen = _nearest_supported(satisfying, prefer)
        return chosen, f"clamped selector base {prefer} into {raw!r} -> {chosen}"
    if default in satisfying:
        return default, f"default {default} satisfies {raw!r}"
    chosen = _nearest_supported(satisfying, default)
    return chosen, f"nearest supported to default within {raw!r} -> {chosen}"


def _relevant_requires_python(repo_path: str) -> tuple[str | None, int]:
    """Return the intersection of Python constraints relevant to repo tests.

    The root manifest remains the fallback and the common single-project path.
    For package-based monorepos, dependency evidence also includes only nested
    projects whose own subtree contains pytest-shaped tests.  Intersecting those
    declarations keeps the one Runtime/base-image decision valid for every local
    project that the benchmark will install and collect.

    Invalid declarations are ignored when at least one valid declaration exists;
    if every declaration is invalid, preserve the first raw value so
    :func:`choose_python_minor` retains its historical graceful fallback.
    """
    root_constraint = read_requires_python(repo_path)
    raw_constraints: list[str] = []
    if root_constraint:
        raw_constraints.append(root_constraint)

    try:
        # Lazy import keeps this Runtime helper usable in minimal/off-arm
        # environments where only ``src.envstate`` is on sys.path.
        from src.python_deps.evidence import collect_python_dependency_evidence

        evidence = collect_python_dependency_evidence(repo_path)
        raw_constraints.extend(
            item.specifier
            for item in evidence.python_requires
            if isinstance(item.specifier, str) and item.specifier.strip()
        )
    except Exception:
        # Base selection must retain its existing never-raise behavior.  The root
        # reader above is sufficient for the historical single-project path.
        pass

    unique_raw: list[str] = []
    for raw in raw_constraints:
        value = raw.strip()
        if value and value not in unique_raw:
            unique_raw.append(value)
    if not unique_raw:
        return None, 0

    normalized: list[str] = []
    for raw in unique_raw:
        try:
            value = _normalize_constraint(raw)
            SpecifierSet(value)  # validate before composing the intersection
        except Exception:
            continue
        if value not in normalized:
            normalized.append(value)

    if not normalized:
        return unique_raw[0], len(unique_raw)
    if len(normalized) == 1:
        # Preserve the historical raw value (and RuntimeBaseDecision field) for
        # an ordinary single-project checkout, including Poetry shorthand.
        return (unique_raw[0] if len(unique_raw) == 1 else normalized[0]), 1
    return ",".join(normalized), len(normalized)


def pin_base_python(base_image: str, minor: str) -> str:
    """Rewrite a ``python:X`` base tag's version to ``minor``, keeping the variant.

    ``python:3.11-slim`` + ``"3.10"`` -> ``python:3.10-slim``. Non-python bases
    (``ubuntu:22.04``) and python tags without a recognizable version component
    (``python:latest``) are returned unchanged — conservative: never make a base
    worse than the Synthesizer's choice.
    """
    reference, _digest = _split_image_digest(base_image)
    name, sep, tag = reference.rpartition(":")
    if not sep or "/" in tag:
        # no tag (bare name, or the ":" belonged to a registry:port)
        return base_image
    if name.split("/")[-1] != "python":
        return base_image
    tokens = tag.split("-")
    if not _VERSION_TOKEN.match(tokens[0]):
        return base_image
    existing_minor = ".".join(tokens[0].split(".")[:2])
    if existing_minor == minor:
        return base_image
    tokens[0] = minor
    rewritten = f"{name}:{'-'.join(tokens)}"
    # A digest identifies the original image bytes. Keeping it after changing
    # the tag would still boot the old interpreter while claiming the new one.
    # Drop the now-incompatible digest and let Docker resolve the compatible tag.
    return rewritten


@dataclass(frozen=True)
class RuntimeBaseDecision:
    """The Runtime-tier decision for one repo: the concrete python minor to
    target, the (possibly pinned) base image, and provenance for advise/logs.
    """

    minor: str            # concrete python minor, e.g. "3.10" — the resolve target
    base_image: str       # base image after pinning (unchanged when not pinnable)
    reason: str           # why this minor (which choose_python_minor branch fired)
    requires_python: str | None  # effective declared constraint intersection, or None


def resolve_runtime_base(
    repo_path: str, base_image: str, *, default: str = DEFAULT_MINOR
) -> RuntimeBaseDecision:
    """Derive the Runtime-tier base decision for ``repo_path`` (the seam entry).

    Reads and intersects the root plus test-bearing nested-project Python
    constraints, picks a concrete minor, and pins the base tag to it. The same
    ``minor`` is meant to flow into both the ``FROM`` line (``base_image``) and
    ``build_dep_graph(target_python=...)``. Pure apart from repository reads;
    never raises.
    """
    requires_python, constraint_count = _relevant_requires_python(repo_path)
    prefer = _base_image_minor(base_image)
    minor, reason = choose_python_minor(requires_python, default=default, prefer=prefer)
    if constraint_count > 1:
        reason += f"; intersected {constraint_count} relevant project constraints"
    return RuntimeBaseDecision(
        minor=minor,
        base_image=pin_base_python(base_image, minor),
        reason=reason,
        requires_python=requires_python,
    )


def screen_runtime_pin(repo_path: str, default: str = DEFAULT_MINOR) -> dict:
    """Pure pre-screen: what would the pin do for this repo, no container needed.

    ``base_would_change`` approximates "the live base would differ from the
    selector's usual default" as ``declared AND chosen != default``. It is a
    go/no-go signal for the A/B, not ground truth (the real selector may pick a
    non-default minor); e2e confirms.
    """
    requires_python, constraint_count = _relevant_requires_python(repo_path)
    minor, reason = choose_python_minor(requires_python, default=default)
    if constraint_count > 1:
        reason += f"; intersected {constraint_count} relevant project constraints"
    return {
        "requires_python": requires_python,
        "would_pin_to": minor,
        "reason": reason,
        "base_would_change": requires_python is not None and minor != default,
    }
