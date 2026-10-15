"""Explicit, resumable consolidation that preserves historical file bytes."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Settings
from .paths import LAYOUT_FILE, TaskPaths, ensure_task_dirs, read_json, safe_join, sha256_file, slugify, write_json


def consolidate_task(settings: Settings, owner_slug: str, moves: dict[str, str],
                     members: dict[str, str], *, apply: bool = False) -> dict:
    """Move platform-relative sources into one owner; never merge/overwrite directories.

    A persisted inventory permits retry after interruption and verifies the bytes at
    the destination. Nested sources move first and are excluded from parent inventories.
    Frozen JSON/DOCX records remain unchanged; relocation metadata resolves old paths.
    """
    tp = TaskPaths(settings, owner_slug)
    owner = tp.root
    if owner != tp.owner_root:
        raise ValueError("migration owner must be a top-level task")
    items = sorted(moves.items(), key=lambda item: -len(Path(item[0]).parts))
    destinations = [safe_join(owner, new) for _, new in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError("duplicate migration destinations")
    layout = read_json(owner / LAYOUT_FILE,
                       {"schema_version": 1, "task_slug": owner_slug, "members": {}, "relocations": {}})
    if len(set(members.values())) != len(members):
        raise ValueError("duplicate member destinations")
    for member, rel in members.items():
        member_paths = TaskPaths(settings, member)
        parts = Path(rel).parts
        if (member == owner_slug or len(parts) != 2 or parts[0] not in {"tracks", "history"}
                or parts[1] != slugify(parts[1])):
            raise ValueError("member must name tracks/<name> or history/<name>")
        safe_join(owner, rel)
        if layout["members"].get(member, rel) != rel:
            raise ValueError("conflicting member registration")
        if member_paths.owner_root not in {owner, settings.root / f"{member}_experiment"}:
            raise ValueError("member already registered in another owner")
        if any(m != member and r == rel for m, r in layout["members"].items()):
            raise ValueError("member destination already registered")
    journal = safe_join(owner, "runtime", "migrations", "consolidation.json")
    previous = read_json(journal, {})
    if previous and (previous["moves"] != moves or previous["members"] != members):
        raise ValueError("another migration already owns this journal")
    records = previous.get("records", {})
    for old, new in items:
        src, dst = safe_join(settings.root, old), safe_join(owner, new)
        if src == owner or src in owner.parents or owner in src.parents:
            raise ValueError("migration sources must be outside the owner directory")
        if src.exists() and dst.exists():
            raise ValueError(f"both migration source and destination exist: {old}")
        if old in records:
            continue
        if not src.exists():
            if dst.exists():
                raise ValueError(f"destination has no migration provenance: {new}")
            continue
        nested = [safe_join(settings.root, other) for other in moves if other != old
                  and safe_join(settings.root, other).is_relative_to(src)]
        paths = [src] if src.is_file() else sorted(src.rglob("*"))
        if src.is_symlink() or any(p.is_symlink() for p in paths):
            raise ValueError(f"migration does not follow symlinks: {old}")
        records[old] = {"destination": new, "kind": "file" if src.is_file() else "directory",
                        "files": {("." if p == src else str(p.relative_to(src))): sha256_file(p)
                                  for p in paths if p.is_file()
                                  and not any(p.is_relative_to(n) for n in nested)}}
    result = {"schema_version": 1, "owner": owner_slug, "moves": moves, "members": members,
              "records": records, "status": "planned"}
    if not apply:
        return result
    write_json(journal, result)
    for old, record in records.items():
        src = safe_join(settings.root, old)
        dst = safe_join(owner, record["destination"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        for rel, digest in record["files"].items():
            path = dst if rel == "." else safe_join(dst, rel)
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"migration checksum mismatch: {path}")
    for member, rel in members.items():
        layout["members"][member] = rel
    layout["relocations"].update(moves)
    write_json(owner / LAYOUT_FILE, layout)
    ensure_task_dirs(tp)
    for member in members:
        ensure_task_dirs(TaskPaths(settings, member))
    result["status"] = "verified"
    write_json(journal, result)
    return result
