"""Task bundle export/import and state migration (guide 12.3, M5).

A bundle is a portable, hash-verified snapshot of one task's *reproducible*
material: spec, plans, docs, coding records, recipes, protocols and the task
spec itself. Raw data, released test sets and weights are intentionally left out
as hashed external references, and credentials are filtered. Import never
overwrites an existing same-named task.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import Settings
from ..paths import LAYOUT_FILE, TaskPaths, ensure_task_dirs, safe_join, sha256_file, slugify, write_json
from ..schemas import contracts as C
from ..schemas.enums import GRAPH_VERSION, SCHEMA_VERSION, ImportStatus, TaskPhase
from ..storage import db as _db
from ..storage.repository import Repository

BUNDLE_FORMAT = "algo-agent-task-bundle/1"
MANIFEST_NAME = "bundle_manifest.json"

_SKIP_DIRS = ("data/raw", "data/releases", "data/annotations", "runs", "model_pool",
              "runtime", "tracks", "history", "deliverables/bundles", ".iqa_cache")
_SKIP_SUFFIX = {".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx",
                ".sqlite", ".db", ".pid", ".log", ".lock"}
_SECRET_FILES = {".env", ".env.local"}
_SECRET_SUFFIX = {".key", ".pem", ".p12"}
_SECRET_KEYS = ("api_key", "apikey", "secret", "password", "token", "authorization",
                "access_key", "private_key")
_REDACT = "***REDACTED***"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_task_bundle(settings: Settings, repo: Repository, task_id: str, *,
                       out_dir: Optional[Path] = None) -> Dict[str, Any]:
    spec = repo.get_spec(task_id)
    tp = TaskPaths(settings, spec.task_slug)
    state = repo.get_state(task_id)
    if not tp.root.exists():
        raise FileNotFoundError(f"task directory not found: {tp.root}")

    target = tp.output_path(out_dir) if out_dir else (
        tp.deliverables / "bundles" / f"bundle_{uuid.uuid4().hex[:12]}")
    if not target.is_relative_to(tp.deliverables) or target == tp.deliverables:
        raise ValueError("bundle output must be a subdirectory of task deliverables/")
    if target.exists():
        raise ValueError("bundle output already exists; choose a new directory")
    target.mkdir(parents=True, exist_ok=False)

    files: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    external_refs: List[Dict[str, Any]] = []
    redacted = 0

    for src in sorted(tp.root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(tp.root).as_posix()
        if src.is_relative_to(target) or src.is_symlink() or src.name == LAYOUT_FILE:
            continue
        if any(rel == d or rel.startswith(d + "/") for d in _SKIP_DIRS):
            excluded.append({"path": rel, "reason": "large/binary asset kept as external reference"})
            continue
        if src.suffix.lower() in _SKIP_SUFFIX or src.name in _SECRET_FILES \
                or src.suffix.lower() in _SECRET_SUFFIX:
            excluded.append({"path": rel, "reason": "excluded by bundle policy"})
            continue
        dest = safe_join(target, "task", rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in (".json", ".yaml", ".yml"):
            text, n = _redact_text(src.read_text(encoding="utf-8", errors="ignore"), src.suffix)
            dest.write_text(text, encoding="utf-8")
            redacted += n
        else:
            shutil.copy2(src, dest)
        files.append({"path": f"task/{rel}", "sha256": sha256_file(dest), "size": dest.stat().st_size})

    # weights + released data that were excluded become hashed external refs
    for m in repo.list_models(task_id):
        ref = tp.root / m.weights_ref if m.weights_ref else None
        if m.weights_ref:
            external_refs.append({"path": m.weights_ref, "reason": "model weights",
                                  "checksum": m.weights_hash,
                                  "resolvable": bool(ref and ref.exists())})
    ext_manifest = target / "external_refs.json"
    write_json(ext_manifest, external_refs)

    manifest = {
        "bundle_id": f"bundle_{spec.task_slug}_{int(time.time())}",
        "bundle_format": BUNDLE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "task_id": task_id,
        "task_slug": spec.task_slug,
        "source_root": str(settings.root),
        "created_at": C.utcnow(),
        "task_spec": spec.model_dump(),
        "state": {"phase": state.phase, "iteration": state.iteration,
                  "best_model_id": state.best_model_id},
        "asset_dependencies": _assets_for_task(repo, task_id),
        "licenses": sorted({a.license for a in repo.search_assets()
                                  if a.source_task == task_id and a.license}),
        "files": files,
        "external_refs": external_refs,
        "excluded": excluded,
        "credentials_filtered": redacted,
    }
    write_json(target / MANIFEST_NAME, manifest)
    write_json(target / "checksums.json", {f["path"]: f["sha256"] for f in files})
    (target / "README.md").write_text(_bundle_md(manifest), encoding="utf-8")
    repo.append_event(task_id, "bundle.export", f"task bundle written to {target}",
                      data={"bundle_id": manifest["bundle_id"], "files": len(files),
                            "external_refs": len(external_refs), "redacted": redacted})
    return {"bundle_id": manifest["bundle_id"], "dir": str(target),
            "files": len(files), "excluded": len(excluded),
            "external_refs": external_refs, "credentials_filtered": redacted,
            "manifest": str(target / MANIFEST_NAME)}


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #
def import_task_bundle(settings: Settings, repo: Repository, bundle_path: Path, *,
                       target_task_id: Optional[str] = None,
                       dry_run: bool = False) -> C.ImportResult:
    bundle = Path(bundle_path)
    mpath = bundle / MANIFEST_NAME if bundle.is_dir() else bundle
    if not mpath.exists():
        return C.ImportResult(status=ImportStatus.FAILED.value,
                              events=[f"bundle manifest not found: {mpath}"])
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return C.ImportResult(status=ImportStatus.FAILED.value,
                              events=[f"cannot read bundle manifest: {exc}"])
    root_dir = mpath.parent

    events: List[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return C.ImportResult(
            status=ImportStatus.UNSUPPORTED_STATE.value,
            bundle_id=manifest.get("bundle_id", ""), task_id=manifest.get("task_id", ""),
            events=[f"schema_version {manifest.get('schema_version')!r} != supported {SCHEMA_VERSION}"])
    if manifest.get("graph_version") != GRAPH_VERSION:
        return C.ImportResult(
            status=ImportStatus.UNSUPPORTED_STATE.value,
            bundle_id=manifest.get("bundle_id", ""), task_id=manifest.get("task_id", ""),
            events=[f"graph_version {manifest.get('graph_version')!r} != {GRAPH_VERSION}; "
                    "refusing to restore state across graph versions"])

    missing_files: List[str] = []
    bad_hash: List[str] = []
    for f in manifest.get("files", []):
        p = safe_join(root_dir, f["path"])
        if not p.exists():
            missing_files.append(f["path"])
        elif sha256_file(p) != f["sha256"]:
            bad_hash.append(f["path"])
    if missing_files or bad_hash:
        return C.ImportResult(
            status=ImportStatus.FAILED.value, bundle_id=manifest.get("bundle_id", ""),
            task_id=manifest.get("task_id", ""), missing=missing_files,
            events=[f"hash/integrity failure: {len(missing_files)} missing, {len(bad_hash)} mismatched"])

    missing_deps = [dep for dep in manifest.get("asset_dependencies", [])
                    if not _asset_resolves(repo, dep)]
    if missing_deps:
        return C.ImportResult(
            status=ImportStatus.MISSING_DEPENDENCIES.value,
            bundle_id=manifest.get("bundle_id", ""), task_id=manifest.get("task_id", ""),
            missing=missing_deps,
            events=["asset dependencies are not registered locally; "
                    "register them before importing this bundle"])

    src_spec = C.TaskSpec(**manifest["task_spec"])
    new_task_id = target_task_id or src_spec.task_id
    new_slug = slugify(new_task_id)
    dst = TaskPaths(settings, new_slug).root
    if repo.task_exists(new_task_id) or dst.exists():
        return C.ImportResult(
            status=ImportStatus.FAILED.value, bundle_id=manifest.get("bundle_id", ""),
            task_id=new_task_id,
            events=[f"a task named {new_task_id!r} already exists; same-name import never overwrites"])

    mapping: Dict[str, str] = {}
    if not dry_run:
        for f in manifest.get("files", []):
            src = safe_join(root_dir, f["path"])
            rel = f["path"]
            if rel.startswith("task/"):
                rel = rel[len("task/"):]
            if rel == LAYOUT_FILE:
                continue  # a renamed import must not register the source task's identity
            dest = safe_join(dst, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            mapping[f["path"]] = str(dest)
        spec = src_spec.model_copy(update={"task_id": new_task_id, "task_slug": new_slug})
        tp = TaskPaths(settings, new_slug)
        ensure_task_dirs(tp)
        import yaml
        tp.task_yaml.write_text(yaml.safe_dump(spec.model_dump(mode="json"), allow_unicode=True),
                                encoding="utf-8")
        repo.create_task(spec, phase=TaskPhase.DRAFT.value)
        repo.append_event(new_task_id, "bundle.import",
                          f"imported from {manifest.get('source_root')} "
                          f"(bundle {manifest.get('bundle_id')})",
                          data={"source_task": manifest.get("task_id"),
                                "source_root": manifest.get("source_root"),
                                "target_root": str(settings.root),
                                "files": len(mapping),
                                "external_refs": len(manifest.get("external_refs", []))})
        events.append("task created; new runs will be created on re-execution and linked "
                      "via source_run_id")

    events.append(f"verified {len(manifest.get('files', []))} file(s); "
                  f"{len(manifest.get('external_refs', []))} external reference(s) must be supplied")
    events.append("old experiment manifests are left unchanged")
    return C.ImportResult(status=ImportStatus.READY.value,
                          bundle_id=manifest.get("bundle_id", ""), task_id=new_task_id,
                          source_task=manifest.get("task_id", ""), mapping=mapping,
                          events=events, new_run_ids=[])


# --------------------------------------------------------------------------- #
# migration
# --------------------------------------------------------------------------- #
def migrate_state(settings: Settings, *, backup: bool = True) -> Dict[str, Any]:
    """Versioned DB migration with backup + rollback; refuses cross-graph restores."""
    db_path = settings.db_path
    if not db_path.exists():
        return {"status": "failed", "reason": f"no database at {db_path}"}
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 0
        grows = conn.execute("SELECT value FROM meta WHERE key='graph_version'").fetchone()
        graph_version = grows["value"] if grows else ""
        tasks = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]

    if current > _db.DB_SCHEMA_VERSION:
        return {"status": ImportStatus.UNSUPPORTED_STATE.value,
                "reason": f"db schema {current} is newer than supported {_db.DB_SCHEMA_VERSION}",
                "from_version": current, "to_version": _db.DB_SCHEMA_VERSION}
    if graph_version and graph_version != GRAPH_VERSION and tasks:
        return {"status": ImportStatus.UNSUPPORTED_STATE.value,
                "reason": f"graph version {graph_version} != {GRAPH_VERSION}; "
                          "refusing to restore state across graph versions",
                "from_version": current, "to_version": _db.DB_SCHEMA_VERSION}

    backup_path: Optional[Path] = None
    if backup:
        bdir = settings.state_dir / "backups"
        bdir.mkdir(parents=True, exist_ok=True)
        stamp = C.utcnow().replace(":", "").replace("-", "")
        backup_path = bdir / f"algo_agent-{stamp}.sqlite"
        shutil.copy2(db_path, backup_path)

    try:
        conn = _db.init_db(db_path)
        with conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('graph_version', ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (GRAPH_VERSION,))
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise sqlite3.DatabaseError(f"integrity_check failed: {integrity}")
    except Exception as exc:  # rollback on failure
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, db_path)
        return {"status": "failed", "reason": f"migration failed and was rolled back: {exc}",
                "backup": str(backup_path) if backup_path else ""}
    return {"status": ImportStatus.READY.value, "from_version": current,
            "to_version": _db.DB_SCHEMA_VERSION, "graph_version": GRAPH_VERSION,
            "backup": str(backup_path) if backup_path else "", "tasks": tasks}


# --------------------------------------------------------------------------- #
def _assets_for_task(repo: Repository, task_id: str) -> List[str]:
    return [f"{a.asset_id}@{a.asset_version}" for a in repo.search_assets()
            if a.source_task == task_id]


def _asset_resolves(repo: Repository, dep: str) -> bool:
    asset_id, _, version = dep.partition("@")
    return repo.get_asset(asset_id, version) is not None


def _redact_text(text: str, suffix: str) -> Tuple[str, int]:
    """Blank out secret-looking values in config/doc files; return (text, count)."""
    if suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text, 0
        n = _redact_obj(data)
        return json.dumps(data, ensure_ascii=False, indent=2), n
    if suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except Exception:
            return text, 0
        if not isinstance(data, (dict, list)):
            return text, 0
        n = _redact_obj(data)
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False), n
    lines, n = [], 0
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in _SECRET_KEYS) and "=" in line and not line.strip().startswith("#"):
            key = line.split("=", 1)[0]
            lines.append(f"{key}={_REDACT}")
            n += 1
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), n


def _redact_obj(obj: Any) -> int:
    n = 0
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                n += _redact_obj(v)
            elif isinstance(v, str) and any(sk in str(k).lower() for sk in _SECRET_KEYS) and v:
                obj[k] = _REDACT
                n += 1
    elif isinstance(obj, list):
        for v in obj:
            n += _redact_obj(v)
    return n


def _bundle_md(manifest: Dict[str, Any]) -> str:
    return (
        f"# Task bundle — {manifest.get('task_slug')}\n\n"
        f"- bundle_id: `{manifest.get('bundle_id')}`\n"
        f"- format: `{manifest.get('bundle_format')}`\n"
        f"- schema_version: {manifest.get('schema_version')} / graph_version: "
        f"{manifest.get('graph_version')}\n"
        f"- exported from: {manifest.get('source_root')}\n"
        f"- files: {len(manifest.get('files', []))}\n"
        f"- external references: {len(manifest.get('external_refs', []))}\n"
        f"- credentials filtered: {manifest.get('credentials_filtered')}\n\n"
        "Import with:\n\n```bash\nalgo-agent --root <new-root> bundle import "
        f"{manifest.get('task_slug')}_bundle\n```\n\n"
        "> Raw data, released test sets and weights are not bundled; supply them "
        "separately by checksum.\n")


__all__ = ["export_task_bundle", "import_task_bundle", "migrate_state", "BUNDLE_FORMAT"]
