"""Wheel-vs-sdist build-from-source oracle (construction-enrichment cluster 1a).

Extracted from ``resolve_lock.py`` as a pure refactor — behavior preserved
exactly for the wheel/sdist tag-matching decision. Self-contained (stdlib
only) so ``resolve_lock.py -> wheel_oracle.py`` is the only import edge (no
cycle). This module has NO concept of "local source" (the synthetic resolve
root, or a path/editable dependency) at all — ``resolve_lock.py`` keeps a thin
``native_risk_from_lock`` wrapper that does the uv.lock TOML parse, the
target-aware fork selection (shared with ``parse_uv_lock``, so it must stay
there), AND filters out local-source entries with its OWN, already-existing
``_is_local_source`` BEFORE calling :func:`risk_from_packages` here.
"""

from __future__ import annotations


def _artifact_filename(artifact: dict) -> str | None:
    """Filename of an sdist/wheel lock entry (explicit, or derived from url)."""
    if not isinstance(artifact, dict):
        return None
    name = artifact.get("filename")
    if name:
        return name
    url = artifact.get("url")
    if url:
        return url.rsplit("/", 1)[-1]
    return None


def _wheel_matches_platform(filename: str | None, target_platform: str) -> bool:
    """True when ``filename`` is installable on the (linux) ``target_platform``.

    Universal wheels (``...-none-any.whl``) match every platform. Otherwise the
    target's arch token (e.g. ``x86_64`` / ``aarch64``) must appear in a *linux*
    platform tag; macOS/Windows wheels never match a linux target.
    """
    if not filename:
        return False
    low = filename.lower()
    if not low.endswith(".whl"):
        return False
    if low.endswith("-none-any.whl"):
        return True
    arch = (target_platform.split("-", 1)[0] if target_platform else "").lower()
    if not arch:
        return False
    if "linux" not in low:  # the target is linux; skip macosx_/win_ wheels.
        return False
    return arch in low


def risk_from_packages(raw_packages: list[dict], target_platform: str) -> dict[str, dict]:
    """Map ``package name -> {build_from_source, artifact, hash}`` for already
    fork-resolved, already LOCAL-SOURCE-FILTERED ``[[package]]`` TOML entries
    (one entry per name — the caller, ``resolve_lock.native_risk_from_lock``,
    is responsible for BOTH selecting the target-applicable entry when a lock
    forks a package across resolution markers, AND filtering out local-source
    entries with its own ``_is_local_source`` before ever calling this
    function; this function only decides wheel-vs-sdist per entry and has no
    concept of "local source" at all).

    A package that ships an ``sdist`` but no wheel matching ``target_platform``
    must be built from source on the target. The chosen artifact is the
    matching wheel when one exists, else the sdist.
    """
    risk: dict[str, dict] = {}
    for pkg in raw_packages:
        name = pkg.get("name")
        if not name:
            continue
        sdist = pkg.get("sdist")
        wheels = pkg.get("wheels", []) or []

        matching_wheel = next(
            (
                w
                for w in wheels
                if _wheel_matches_platform(_artifact_filename(w), target_platform)
            ),
            None,
        )
        has_sdist = isinstance(sdist, dict) and bool(sdist)
        build_from_source = has_sdist and matching_wheel is None

        if matching_wheel is not None:
            chosen = matching_wheel
        elif has_sdist:
            chosen = sdist
        elif wheels:
            chosen = wheels[0]
        else:
            chosen = None

        risk[name] = {
            "build_from_source": build_from_source,
            "artifact": _artifact_filename(chosen) if chosen else None,
            "hash": chosen.get("hash") if isinstance(chosen, dict) else None,
        }
    return risk
