"""Platform configuration and root resolution (guide 2, 4).

``--root`` overrides everything; otherwise we derive the platform root from the
environment, from an explicit marker file, or by walking up from the CWD. Fixed
absolute paths are never assumed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MARKER = ".algo_agent_root"
# A platform root is identified by its config dir + plugin dir + runtime dir.
_PLATFORM_ANCHORS = ("configs", "algorithm_plugins", "runtimes")
# Older checkouts used these top-level dirs; accepted so existing roots still load.
_LEGACY_ANCHORS = ("agent_runtime", "algorithm_plugins", "runtime_state")
# The agent source + configs used to live here; accepted for old checkouts.
_NESTED_LEGACY = ("runtimes/agent_runtime", "algorithm_plugins", "runtime_state")


def _looks_like_root(path: Path) -> bool:
    return (all((path / a).exists() for a in _PLATFORM_ANCHORS)
            or all((path / a).exists() for a in _LEGACY_ANCHORS)
            or all((path / a).exists() for a in _NESTED_LEGACY))


def resolve_root(root: Optional[str] = None, start: Optional[Path] = None) -> Path:
    if root:
        return platform_root(Path(root))
    env = os.environ.get("ALGO_ROOT")
    if env:
        return platform_root(Path(env))
    cur = (start or Path.cwd()).resolve()
    candidate = cur
    for _ in range(12):
        if (candidate / MARKER).exists() or _looks_like_root(candidate):
            return platform_root(candidate)
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return platform_root(cur)


def platform_root(path: Path) -> Path:
    """A task (or track) is never a platform, even when passed explicitly."""
    path = path.expanduser().resolve()
    task_ancestors = [p for p in (path, *path.parents) if p.name.endswith("_experiment")]
    return task_ancestors[-1].parent if task_ancestors else path


# SQLite needs POSIX advisory locks, which some shared/FUSE filesystems (e.g. an
# OSS mount) do not implement -- ``sqlite3.connect`` then fails with "disk I/O
# error". We probe the preferred location once per process and transparently fall
# back to a local directory so the platform still runs from such a checkout.
_SQLITE_CAPABLE: Dict[str, bool] = {}


def _probe_sqlite(directory: Path) -> bool:
    probe = directory / ".sqlite_probe"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(probe), timeout=5.0)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS probe(x)")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception:
        return False
    finally:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                (directory / f".sqlite_probe{suffix}").unlink()
            except OSError:
                pass


def _fallback_state_dir(preferred: Path) -> Path:
    digest = hashlib.sha1(str(preferred).encode("utf-8")).hexdigest()[:12]
    base = os.environ.get("ALGO_STATE_FALLBACK_DIR") or str(Path.home() / ".algo_agent" / "state")
    return Path(base).expanduser() / digest


def resolve_state_dir(preferred: Path, explicit: Optional[str] = None) -> Path:
    """Return a directory that can actually hold SQLite files.

    ``ALGO_STATE_DIR`` wins when set; otherwise probe ``preferred`` and fall back
    to ``~/.algo_agent/state/<hash>`` (or ``ALGO_STATE_FALLBACK_DIR``) when the
    filesystem cannot host SQLite.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    key = str(preferred)
    if key not in _SQLITE_CAPABLE:
        _SQLITE_CAPABLE[key] = _probe_sqlite(preferred)
    return preferred if _SQLITE_CAPABLE[key] else _fallback_state_dir(preferred)


@dataclass
class Settings:
    root: Path
    workflow: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = platform_root(Path(self.root))

    # ---- directories ----------------------------------------------------- #
    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def runtimes_dir(self) -> Path:
        """All generated runtime state lives under this one directory."""
        return self.root / "runtimes"

    @property
    def state_dir(self) -> Path:
        """SQLite state + checkpoints; falls back to a local dir on FUSE mounts."""
        return resolve_state_dir(self.runtimes_dir / "state",
                                 explicit=self.env_get("ALGO_STATE_DIR"))

    @property
    def db_path(self) -> Path:
        return self.state_dir / "algo_agent.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        return self.state_dir / "checkpoints.sqlite"

    @property
    def coding_workspaces(self) -> Path:
        return self.runtimes_dir / "coding" / "workspaces"

    @property
    def evidence_dir(self) -> Path:
        return self.runtimes_dir / "evidence"

    @property
    def execution_dir(self) -> Path:
        return self.runtimes_dir / "execution"

    @property
    def local_models_dir(self) -> Path:
        """Local auxiliary LLMs (e.g. Qwen3-27B) for extraction/labelling/diagnosis.

        Named ``local_largemodel/<model_id>/`` in guide section 4; the legacy
        ``local_auxmodels/`` name is accepted as an alias.
        """
        for name in ("local_largemodel", "local_auxmodels"):
            if (self.root / name).exists():
                return self.root / name
        return self.root / "local_largemodel"

    @property
    def plugins_dir(self) -> Path:
        return self.root / "algorithm_plugins"

    @property
    def shared_assets(self) -> Path:
        return self.root / "shared_assets"

    @property
    def config_path(self) -> Path:
        return self.root / "configs" / "workflow.yaml"

    # ---- convenience ----------------------------------------------------- #
    def budget_defaults(self) -> Dict[str, Any]:
        return dict(self.workflow.get("budget", {}))

    def provider_cfg(self) -> Dict[str, Any]:
        return dict(self.workflow.get("providers", {}))

    def stop_rules(self) -> Dict[str, Any]:
        return dict(self.workflow.get("stop_rules", {}))

    def upload_policy(self) -> Dict[str, Any]:
        return dict(self.workflow.get("upload_policy", {}))

    def env_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.env.get(key, os.environ.get(key, default))

    def cloud_ready(self) -> bool:
        """A cloud research credential is configured (OpenAI-compatible, e.g. DeepSeek)."""
        return bool(self.env_get("ALGO_CLOUD_API_KEY"))

    def local_ready(self) -> bool:
        """A local OpenAI-compatible endpoint is configured (e.g. Qwen3.8-27B)."""
        return bool(self.env_get("ALGO_LOCAL_BASE_URL"))

    def offline(self) -> bool:
        """Whether all model roles must use offline adapters.

        A local endpoint enables auxiliary diagnosis/conversation only; without
        cloud credentials research still uses the explicit offline template.
        """
        if str(self.env.get("ALGO_OFFLINE", os.environ.get("ALGO_OFFLINE", ""))).lower() in {"1", "true", "yes"}:
            return True
        return not (self.cloud_ready() or self.local_ready())


def _load_env_file(root: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    env_path = root / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def load_settings(root: Optional[str] = None) -> Settings:
    resolved = resolve_root(root)
    workflow: Dict[str, Any] = {}
    cfg = resolved / "configs" / "workflow.yaml"
    if not cfg.exists():
        # legacy location
        cfg = resolved / "agent_runtime" / "configs" / "workflow.yaml"
    if cfg.exists():
        workflow = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return Settings(root=resolved, workflow=workflow, env=_load_env_file(resolved))


_LEGACY_DIR_MOVES = (
    ("runtime_state", "runtimes/state"),
    ("coding_runtime", "runtimes/coding"),
    ("evidence_runtime", "runtimes/evidence"),
    ("execution_runtime", "runtimes/execution"),
    ("agent_runtime/configs", "configs"),
)


def migrate_legacy_dirs(settings: Settings) -> List[str]:
    """Move pre-``runtimes/`` scaffolding into the new layout (idempotent)."""
    moved: List[str] = []
    for old_rel, new_rel in _LEGACY_DIR_MOVES:
        old, new = settings.root / old_rel, settings.root / new_rel
        if old.exists() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.replace(new)
            moved.append(f"{old_rel} -> {new_rel}")
    return moved


def ensure_dirs(settings: Settings) -> None:
    migrate_legacy_dirs(settings)
    for path in (
        settings.docs_dir,
        settings.config_path.parent,
        settings.state_dir,
        settings.plugins_dir,
        settings.local_models_dir,
        settings.shared_assets / "recipes",
        settings.shared_assets / "datasets",
        settings.shared_assets / "models",
        settings.shared_assets / "evaluation",
        settings.shared_assets / "knowledge",
    ):
        path.mkdir(parents=True, exist_ok=True)
