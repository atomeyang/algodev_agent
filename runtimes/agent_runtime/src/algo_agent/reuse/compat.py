"""Deterministic asset compatibility checking (guide 12.2, M4).

Compares only explicit, machine-checkable fields; anything that depends on the
business distribution or unstated hardware is reported as ``unknown`` and needs
a real verification run. ``unknown`` is never silently upgraded to ``compatible``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import contracts as C
from ..schemas.enums import Compatibility
from ..storage.repository import Repository

_PRIORITY = {
    Compatibility.INCOMPATIBLE.value: 3,
    Compatibility.ADAPTATION_REQUIRED.value: 2,
    Compatibility.UNKNOWN.value: 1,
    Compatibility.COMPATIBLE.value: 0,
}


@dataclass
class ReuseRequest:
    """What the *current* task needs from a reusable asset."""

    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    device: str = ""
    license: str = ""
    upload_policy: str = "restricted"
    dependencies: Dict[str, str] = field(default_factory=dict)  # asset_id -> pinned version
    distribution: str = ""  # "" means unknown / not yet characterised
    hard_constraints: List[str] = field(default_factory=list)


def _norm(items: List[str]) -> set:
    return {str(i).strip().lower() for i in items if str(i).strip()}


def check_compatibility(request: ReuseRequest, recipe: C.RecipeSpec,
                        manifest: C.AssetManifest) -> C.CompatibilityResult:
    reasons: List[str] = []
    missing: List[str] = []
    adaptation: List[str] = []
    actions: List[str] = []
    status = Compatibility.COMPATIBLE.value

    def raise_to(new: str) -> None:
        nonlocal status
        if _PRIORITY[new] > _PRIORITY[status]:
            status = new

    # 1) interfaces ---------------------------------------------------- #
    want_in, have_in = _norm(request.inputs), _norm(recipe.inputs)
    want_out, have_out = _norm(request.outputs), _norm(recipe.outputs)
    if want_in and have_in:
        if want_in - have_in:
            reasons.append(f"recipe does not accept input(s): {sorted(want_in - have_in)}")
            raise_to(Compatibility.INCOMPATIBLE.value)
        if have_in - want_in:
            adaptation.append(f"unused recipe input(s): {sorted(have_in - want_in)}")
            raise_to(Compatibility.ADAPTATION_REQUIRED.value)
    else:
        missing.append("input contract")
        raise_to(Compatibility.UNKNOWN.value)
    if want_out and have_out:
        if have_out - want_out:
            reasons.append(f"recipe produces unexpected output(s): {sorted(have_out - want_out)}")
            raise_to(Compatibility.INCOMPATIBLE.value)
        if want_out - have_out:
            reasons.append(f"recipe does not produce required output(s): {sorted(want_out - have_out)}")
            raise_to(Compatibility.INCOMPATIBLE.value)
    else:
        missing.append("output contract")
        raise_to(Compatibility.UNKNOWN.value)

    # 2) device -------------------------------------------------------- #
    if recipe.device and request.device:
        if recipe.device.lower() != request.device.lower():
            if "device" in {c.lower() for c in request.hard_constraints}:
                reasons.append(f"device mismatch is a hard constraint: "
                               f"recipe={recipe.device} need={request.device}")
                raise_to(Compatibility.INCOMPATIBLE.value)
            adaptation.append(f"re-measure on {request.device} (recipe validated on {recipe.device})")
            raise_to(Compatibility.ADAPTATION_REQUIRED.value)
    else:
        missing.append("device profile")
        actions.append(f"re-measure latency/memory on {request.device or 'target hardware'}")
        raise_to(Compatibility.UNKNOWN.value)

    # 3) license ------------------------------------------------------- #
    if recipe.result_range or manifest.license:
        if request.license and manifest.license and request.license != manifest.license:
            allow = {l.strip().lower() for l in request.license.split(",")}
            have = {l.strip().lower() for l in manifest.license.split(",")}
            if not have <= allow:
                reasons.append(f"license incompatible: asset={manifest.license} need={request.license}")
                raise_to(Compatibility.INCOMPATIBLE.value)
        elif not (request.license or manifest.license):
            missing.append("license")
            raise_to(Compatibility.UNKNOWN.value)

    # 4) dependency versions ------------------------------------------- #
    pinned = {d if ":" in d else f"{d}" for d in manifest.dependencies}
    for dep_id, dep_ver in request.dependencies.items():
        if dep_id in pinned and f"{dep_id}:{dep_ver}" not in pinned:
            adaptation.append(f"dependency {dep_id} pinned to {dep_ver} in task but "
                              f"asset expects {sorted(d for d in pinned if d.startswith(dep_id + ':'))}")
            raise_to(Compatibility.ADAPTATION_REQUIRED.value)

    # 5) distribution -------------------------------------------------- #
    if not request.distribution:
        missing.append("business distribution")
        actions.append("run a distribution/regression check on this task's data before reuse")
        raise_to(Compatibility.UNKNOWN.value)

    # 6) manifest status ----------------------------------------------- #
    if manifest.validation_status in ("deprecated",):
        reasons.append("asset version is deprecated")
        raise_to(Compatibility.INCOMPATIBLE.value)

    if status == Compatibility.COMPATIBLE.value:
        reasons.append("all explicit fields match; still run a fresh evaluation on this task")
        actions.append("evaluate with the current task protocol and record new run_id/eval_id")
    return C.CompatibilityResult(status=status, reasons=reasons, missing=missing,
                                 adaptation=adaptation, verification_actions=actions)


def compatibility_for(repo: Repository, asset_id: str, version: str,
                      request: ReuseRequest, recipe: Optional[C.RecipeSpec] = None,
                      *, use_cache: bool = True) -> C.CompatibilityResult:
    """Look up the asset (+ its recipe) and check compatibility, optionally cached."""
    manifest = repo.get_asset(asset_id, version)
    if manifest is None:
        return C.CompatibilityResult(status=Compatibility.UNKNOWN.value,
                                     reasons=[f"asset {asset_id}:{version} not found"],
                                     missing=["asset"])
    if recipe is None:
        recipe = _recipe_for(repo, manifest)
    cache_key = f"compat:{asset_id}:{version}:{_req_key(request)}"
    if use_cache:
        hit = repo.cache_get(cache_key)
        if hit is not None:
            return C.CompatibilityResult(**hit.get("value", {}))
    result = check_compatibility(request, recipe or C.RecipeSpec(recipe_id=""),
                                 manifest)
    if use_cache:
        # dependencies make the cached judgement invalidate when pins change
        repo.cache_put(cache_key, {"value": result.model_dump()},
                       dependencies=[f"{asset_id}:{version}"] + manifest.dependencies)
    return result


def _req_key(request: ReuseRequest) -> str:
    blob = json.dumps({"i": sorted(request.inputs), "o": sorted(request.outputs),
                       "d": request.device, "l": request.license,
                       "dep": request.dependencies, "dist": request.distribution,
                       "hc": sorted(request.hard_constraints)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _recipe_for(repo: Repository, manifest: C.AssetManifest) -> Optional[C.RecipeSpec]:
    from ..paths import read_json
    candidates = [Path(ref) for ref in manifest.artifact_refs]
    candidates += [Path(manifest.checksums.get("recipe", ""))] if manifest.checksums else []
    for p in candidates:
        if str(p) and p.exists() and p.suffix in (".json", ".yaml", ".yml"):
            data = read_json(p)
            if data and isinstance(data, dict) and "recipe_id" in data:
                return C.RecipeSpec(**data)
    return None


__all__ = ["ReuseRequest", "check_compatibility", "compatibility_for"]
