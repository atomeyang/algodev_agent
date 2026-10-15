"""Task directory layout, slug normalisation and path-traversal safety (guide 4)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import Settings

_SLUG_RE = re.compile(r"[^a-z0-9_]+")
LAYOUT_FILE = ".task_layout.json"
DIRECTORY_RULES = """# Task Directory Contract

- One algorithm task owns exactly one top-level <task_slug>_experiment directory.
- All task-generated files, including temporary files, belong inside that directory.
- Keep requirements/plans in plans/, research in docs/research/ and evidence/.
- Keep annotation schemes in data/annotations/plans/<version>/; keep raw responses,
  rejected labels, reviews and agreement reports in data/annotations/runs/<version>/<run_id>/.
- Keep data recipes, splits and immutable releases in data/; never overwrite a release.
- Keep evaluation protocols, scripts, intermediate outputs and results in evaluation/.
  Preserve development/final-test isolation and never tune on final-test feedback.
- Keep experiment code, logs and predictions in runs/<run_id>/; methods in model_pool/.
- Keep coding workspaces and temporary/cache files in runtime/, records in coding_records/.
- Related tracks live in tracks/<track>/ and old runs in history/, not sibling task roots.
- Use TaskPaths for output paths. Do not create per-task output under the platform root,
  platform runtimes/, plugin directories, shared_assets/ or local_largemodel/.
- Platform code/services and genuinely shared read-only datasets/models may live outside
  the task; record their location/version/checksum in data/raw or runtime dependencies.
- The shared scheduler database/checkpointer is platform infrastructure, not an output
  directory. Task-local state exports do not replace its live transactions.
"""


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("_", (name or "task").strip().lower()).strip("_")
    if not slug:
        slug = "task"
    if slug[0].isdigit():
        slug = "t_" + slug
    return slug[:48]


def safe_join(base: Path, *parts: str) -> Path:
    """Join under *base*, rejecting absolute parts and traversal escapes."""
    base = base.resolve()
    candidate = base
    for part in parts:
        if part is None:
            continue
        p = Path(str(part))
        if p.is_absolute():
            raise ValueError(f"absolute path not allowed: {part}")
        candidate = candidate / p
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path escapes base {base}: {resolved}")
    return resolved


def relpath(base: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(base).resolve()))


@dataclass
class TaskPaths:
    settings: Settings
    slug: str

    def __post_init__(self) -> None:
        if self.slug != slugify(self.slug) or self.slug.endswith("_experiment"):
            raise ValueError("task slug must be normalised, without the _experiment suffix")

    @property
    def owner_root(self) -> Path:
        matches = []
        for manifest in self.settings.root.glob(f"*_experiment/{LAYOUT_FILE}"):
            layout = read_json(manifest)
            if self.slug == layout.get("task_slug") or self.slug in layout.get("members", {}):
                matches.append(manifest.parent)
        if len(matches) > 1:
            raise ValueError(f"task slug registered in multiple owners: {self.slug}")
        return safe_join(self.settings.root, matches[0].name if matches else f"{self.slug}_experiment")

    @property
    def root(self) -> Path:
        owner = self.owner_root
        layout = read_json(owner / LAYOUT_FILE, {})
        member = layout.get("members", {}).get(self.slug)
        if member is not None:
            parts = Path(member).parts
            if len(parts) != 2 or parts[0] not in {"tracks", "history"} or parts[1] != slugify(parts[1]):
                raise ValueError(f"invalid task member path: {member}")
            if (self.settings.root / f"{self.slug}_experiment").exists():
                raise ValueError(f"both a member and a top-level task directory exist: {self.slug}")
            return safe_join(owner, member)
        return owner

    def output_path(self, path: str | Path) -> Path:
        """Validate even explicit output overrides, including symlink escapes."""
        path = Path(path)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(self.root.resolve())
            except ValueError:
                raise ValueError("output must stay inside the task directory") from None
        return safe_join(self.root, str(path))

    def resolve_reference(self, value: str) -> Path:
        """Resolve moved artefacts without rewriting signed/frozen historical records."""
        layout = read_json(self.owner_root / LAYOUT_FILE, {})
        for old, new in sorted(layout.get("relocations", {}).items(), key=lambda x: -len(x[0])):
            for prefix in (old, str(self.settings.root / old)):
                if value == prefix or value.startswith(prefix + "/"):
                    return safe_join(self.owner_root, new + value[len(prefix):])
        path = Path(value)
        return path if path.is_absolute() else self.output_path(path)

    @property
    def annotations(self) -> Path:
        return self.output_path("data/annotations")

    def annotation_run(self, version: str, run_id: str) -> Path:
        return safe_join(self.annotations, "runs", version, run_id)

    @property
    def runtime(self) -> Path:
        return self.output_path("runtime")

    @property
    def coding_workspaces(self) -> Path:
        return safe_join(self.runtime, "coding", "workspaces")

    @property
    def evidence(self) -> Path:
        return self.output_path("evidence")

    @property
    def task_yaml(self) -> Path:
        return safe_join(self.root, "task.yaml")

    @property
    def overview(self) -> Path:
        return safe_join(self.root, "OVERVIEW.docx")

    @property
    def plans(self) -> Path:
        return safe_join(self.root, "plans")

    @property
    def model_pool(self) -> Path:
        return safe_join(self.root, "model_pool")

    @property
    def data(self) -> Path:
        return safe_join(self.root, "data")

    @property
    def coding_records(self) -> Path:
        return safe_join(self.root, "coding_records")

    @property
    def runs(self) -> Path:
        return safe_join(self.root, "runs")

    @property
    def model_versions(self) -> Path:
        return safe_join(self.root, "model_versions")

    @property
    def evaluation(self) -> Path:
        return safe_join(self.root, "evaluation")

    @property
    def feedback(self) -> Path:
        return safe_join(self.root, "feedback")

    @property
    def docs(self) -> Path:
        return safe_join(self.root, "docs")

    @property
    def deliverables(self) -> Path:
        return safe_join(self.root, "deliverables")

    def release_dir(self, version: str) -> Path:
        return safe_join(self.data, "releases", version)

    def plan_dir(self, version: str) -> Path:
        return safe_join(self.plans, version)


def ensure_task_dirs(tp: TaskPaths) -> None:
    for path in (
        tp.root, tp.plans, tp.model_pool, tp.data, tp.data / "raw", tp.data / "recipes",
        tp.data / "splits", tp.data / "releases", tp.coding_records, tp.runs,
        tp.model_versions, tp.evaluation / "protocols", tp.evaluation / "scripts",
        tp.evaluation / "results", tp.feedback, tp.docs / "research", tp.docs / "iterations",
        tp.deliverables, tp.annotations / "plans", tp.annotations / "runs",
        tp.annotations / "reviews", tp.evaluation / "intermediate",
        tp.coding_workspaces, tp.runtime / "tmp", tp.runtime / "cache", tp.runtime / "state",
        tp.evidence / "sources", tp.evidence / "cache",
    ):
        tp.output_path(path).mkdir(parents=True, exist_ok=True)
    layout_path = tp.owner_root / LAYOUT_FILE
    if not layout_path.exists():
        write_json(layout_path, {"schema_version": 1, "task_slug": tp.slug,
                                "members": {}, "relocations": {}})
    rules = tp.output_path("DIRECTORY_RULES.md")
    if not rules.exists():
        rules.write_text(DIRECTORY_RULES, encoding="utf-8")


def register_task_member(settings: Settings, owner_slug: str, member_slug: str, track: str,
                         *, section: str = "tracks") -> None:
    """Explicit grouping only: similar names are not proof of shared task ownership."""
    owner = TaskPaths(settings, owner_slug)
    member = TaskPaths(settings, member_slug)
    if owner_slug == member_slug or track != slugify(track) or section not in {"tracks", "history"}:
        raise ValueError("invalid task member registration")
    if owner.root != owner.owner_root:
        raise ValueError("a track cannot own another task root")
    ensure_task_dirs(owner)
    path = owner.root / LAYOUT_FILE
    layout = read_json(path)
    rel = f"{section}/{track}"
    existing = layout["members"].get(member_slug)
    if existing == rel:
        return
    if existing or rel in layout["members"].values() or (owner.root / rel).exists():
        raise ValueError("task member path is already occupied")
    if member.root.exists():
        raise ValueError("existing task must be migrated before registering it as a member")
    layout["members"][member_slug] = rel
    write_json(path, layout)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


__all__: List[str] = [
    "slugify", "safe_join", "relpath", "TaskPaths", "ensure_task_dirs", "register_task_member",
    "LAYOUT_FILE", "DIRECTORY_RULES",
    "write_json", "read_json", "sha256_file", "sha256_text", "sha256_obj",
]
