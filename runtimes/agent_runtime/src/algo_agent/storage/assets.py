"""AssetRegistry: versioned, immutable reuse assets (guide 12.1, M2).

The registry owns the index in the state database and the bodies under
``shared_assets/<area>/<asset_id>/<version>/``. Asset versions are immutable:
re-registering the exact same manifest is idempotent, any different content for
an existing version is rejected. ``publish`` only advances the validation status
when a real validation record is supplied and every dependency resolves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..paths import safe_join, sha256_file, write_json
from ..schemas import contracts as C
from ..schemas.enums import AssetStatus, AssetType
from .repository import Repository

# asset_type -> shared_assets area (guide 12.1: recipes/datasets/models/evaluation/knowledge)
_LAYOUT: Dict[str, str] = {
    AssetType.RECIPE.value: "recipes",
    AssetType.DATASET.value: "datasets",
    AssetType.MODEL.value: "models",
    AssetType.EVALUATION.value: "evaluation",
    AssetType.KNOWLEDGE.value: "knowledge",
    # components normally live in algorithm_plugins/; the registry keeps only an
    # index entry + manifest so shared source never drifts.
    AssetType.COMPONENT.value: "components",
}

# strict ladder; skipping a step requires an explicit, recorded validation
_STATUS_CHAIN = [
    AssetStatus.RESEARCHED.value,
    AssetStatus.REPRODUCED.value,
    AssetStatus.BUSINESS_VALIDATED.value,
    AssetStatus.REUSABLE.value,
]


class AssetError(RuntimeError):
    def __init__(self, code: str, message: str, *, recovery: str = "", status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "recovery": self.recovery}


# Timestamps are metadata, not content: two registrations of the same manifest
# must not be judged "different" just because created_at was re-evaluated.
_VOLATILE_MANIFEST_FIELDS = ("created_at", "validated_at")


def _content_key(manifest: C.AssetManifest) -> Dict[str, Any]:
    data = manifest.model_dump()
    for key in _VOLATILE_MANIFEST_FIELDS:
        data.pop(key, None)
    return data


def _matches_existing(existing: C.AssetManifest, manifest: C.AssetManifest,
                     supplied: Dict[str, str]) -> bool:
    """True when re-registering ``manifest`` reproduces ``existing`` exactly.

    The registry derives checksums from ``--body`` files, so a stored manifest
    always carries more than a hand-written one: when the caller declares no
    checksums we compare the declared fields instead, then require every supplied
    body to reproduce the stored bytes.
    """
    have, want = _content_key(existing), _content_key(manifest)
    if not manifest.checksums:
        have.pop("checksums", None)
        want.pop("checksums", None)
    if have != want:
        return False
    stored = existing.checksums or {}
    return all(stored.get(name) == digest for name, digest in supplied.items())


class AssetRegistry:
    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo

    # ---- paths --------------------------------------------------------- #
    def area(self, asset_type: str) -> str:
        return _LAYOUT.get(asset_type, "knowledge")

    def asset_dir(self, asset_id: str, version: str, asset_type: str) -> Path:
        base = self.settings.shared_assets / self.area(asset_type)
        try:
            return safe_join(base, asset_id, version)
        except ValueError as exc:
            raise AssetError("bad_asset_id", f"unsafe asset path: {exc}") from exc

    # ---- write --------------------------------------------------------- #
    def register(self, manifest: C.AssetManifest, *, body_files: Optional[List[Path]] = None,
                 body_ref: str = "", extra: Optional[Dict[str, Any]] = None) -> C.AssetManifest:
        """Register (or idempotently re-register) one immutable asset version."""
        if not manifest.asset_id or not manifest.asset_version:
            raise AssetError("bad_manifest", "asset_id and asset_version are required")
        if manifest.asset_type not in _LAYOUT:
            raise AssetError("bad_asset_type",
                             f"unknown asset_type {manifest.asset_type!r}; expected one of {sorted(_LAYOUT)}")
        body_files = [Path(b) for b in (body_files or [])]
        for src in body_files:
            if not src.exists():
                raise AssetError("missing_body", f"asset body file not found: {src}")
        supplied = {src.name: sha256_file(src) for src in body_files}

        existing = self.repo.get_asset(manifest.asset_id, manifest.asset_version)
        if existing is not None:
            if _matches_existing(existing, manifest, supplied):
                return existing  # idempotent: same content, regardless of timestamps
            raise AssetError(
                "asset_immutable",
                f"{manifest.asset_id}:{manifest.asset_version} already exists with different content",
                recovery=f"bump asset_version (current {existing.asset_version}) and register a new version")

        target = self.asset_dir(manifest.asset_id, manifest.asset_version, manifest.asset_type)
        fresh = not target.exists()
        target.mkdir(parents=True, exist_ok=True)
        checksums = dict(manifest.checksums)
        try:
            for src in body_files:
                dest = target / src.name
                shutil.copy2(src, dest)
                checksums[dest.name] = sha256_file(dest)
            manifest.checksums = checksums
            write_json(target / "manifest.json", manifest.model_dump())
            if extra:
                write_json(target / "extra.json", extra)
            self.repo.register_asset(manifest, body_ref=body_ref or str(target))
        except Exception:
            # never leave a half-written version behind (index/body stay consistent)
            if fresh and target.exists():
                shutil.rmtree(target, ignore_errors=True)
            raise
        return manifest

    # ---- read ---------------------------------------------------------- #
    def get(self, asset_id: str, version: str) -> C.AssetManifest:
        m = self.repo.get_asset(asset_id, version)
        if m is None:
            raise AssetError("unknown_asset", f"asset {asset_id}:{version} not found",
                             recovery="run: algo-agent assets list", status=404)
        return m

    def search(self, *, asset_type: Optional[str] = None, status: Optional[str] = None,
               query: str = "") -> List[C.AssetManifest]:
        items = self.repo.search_assets(asset_type=asset_type, status=status)
        if query:
            q = query.lower()
            items = [m for m in items
                     if q in m.asset_id.lower() or q in m.source_task.lower()
                     or q in m.asset_type.lower()]
        order = {s: i for i, s in enumerate(_STATUS_CHAIN + [AssetStatus.DEPRECATED.value])}
        return sorted(items, key=lambda m: (order.get(m.validation_status, 99), m.asset_id,
                                            m.asset_version))

    # ---- status -------------------------------------------------------- #
    def set_status(self, asset_id: str, version: str, status: str, *,
                   validation: Optional[Dict[str, Any]] = None) -> C.AssetManifest:
        if status not in {s.value for s in AssetStatus}:
            raise AssetError("bad_status", f"unknown status {status!r}")
        m = self.get(asset_id, version)
        if status == AssetStatus.DEPRECATED.value:
            return self.deprecate(asset_id, version, reason=(validation or {}).get("reason", ""))
        cur, tgt = m.validation_status, status
        if cur in (AssetStatus.DEPRECATED.value,):
            raise AssetError("deprecated", f"{asset_id}:{version} is deprecated and immutable")
        if _STATUS_CHAIN.index(tgt) < _STATUS_CHAIN.index(cur):
            raise AssetError("status_regression", f"cannot move {cur} -> {tgt}")
        if tgt != cur and not validation:
            raise AssetError(
                "validation_required",
                f"advancing to {tgt} requires a real validation record",
                recovery="pass validation={run_id, eval_id, result, verified_at}")
        if tgt != cur and _STATUS_CHAIN.index(tgt) > _STATUS_CHAIN.index(cur) + 1:
            raise AssetError("skipped_step", f"must pass through {_STATUS_CHAIN[_STATUS_CHAIN.index(cur)+1]}")
        self.repo.set_asset_status(asset_id, version, tgt)
        if validation:
            d = self.asset_dir(asset_id, version, m.asset_type)
            hist = d / "validations.json"
            records = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else []
            records.append({**validation, "status": tgt})
            write_json(hist, records)
        return self.get(asset_id, version)

    def publish(self, asset_id: str, version: str,
                validation: Optional[Dict[str, Any]] = None, *,
                new_status: Optional[str] = None) -> C.AssetManifest:
        """Validate dependencies + license, then advance one status step."""
        m = self.get(asset_id, version)
        missing = [d for d in self.resolve_dependencies(asset_id, version) if not d["resolved"]]
        if missing:
            raise AssetError(
                "unresolved_dependencies",
                f"cannot publish {asset_id}:{version}; unresolved: "
                + ", ".join(f"{d['asset_id']}:{d['version']}" for d in missing),
                recovery="register the missing dependency versions first")
        if (validation or {}).get("license_blocked"):
            raise AssetError("license_blocked", "license does not permit publishing this asset")
        if m.validation_status == AssetStatus.DEPRECATED.value:
            raise AssetError("deprecated", f"{asset_id}:{version} is deprecated and immutable",
                             recovery="register a new asset_version instead of republishing")
        target = new_status or _STATUS_CHAIN[min(_STATUS_CHAIN.index(m.validation_status) + 1,
                                                  len(_STATUS_CHAIN) - 1)]
        return self.set_status(asset_id, version, target, validation=validation)

    def deprecate(self, asset_id: str, version: str, *, reason: str = "") -> C.AssetManifest:
        m = self.get(asset_id, version)
        self.repo.set_asset_status(asset_id, version, AssetStatus.DEPRECATED.value)
        d = self.asset_dir(asset_id, version, m.asset_type)
        write_json(d / "deprecated.json", {"reason": reason, "at": C.utcnow()})
        return self.get(asset_id, version)

    # ---- dependencies -------------------------------------------------- #
    def resolve_dependencies(self, asset_id: str, version: str,
                             _seen: Optional[set] = None) -> List[Dict[str, Any]]:
        """Depth-first closure; each entry says whether it resolves locally."""
        seen = _seen if _seen is not None else set()
        out: List[Dict[str, Any]] = []
        m = self.repo.get_asset(asset_id, version)
        if m is None:
            out.append({"asset_id": asset_id, "version": version, "resolved": False,
                        "reason": "missing"})
            return out
        for dep in m.dependencies:
            key = dep if ":" in dep else f"{dep}:{version}"
            did, _, dver = key.partition(":")
            if key in seen:
                continue
            seen.add(key)
            dm = self.repo.get_asset(did, dver)
            if dm is None:
                out.append({"asset_id": did, "version": dver, "resolved": False,
                            "reason": "missing"})
                continue
            out.append({"asset_id": did, "version": dver, "resolved": True,
                        "status": dm.validation_status})
            out.extend(self.resolve_dependencies(did, dver, seen))
        return out


__all__ = ["AssetRegistry", "AssetError"]
