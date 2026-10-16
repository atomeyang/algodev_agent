"""Platform bootstrap: idempotent ``init`` scaffolding and ``doctor`` checks.

``init`` never overwrites existing documents; it only creates what is missing
(guide 2, M0). Root resolution never depends on a fixed absolute path.
"""

from __future__ import annotations

import os
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import MARKER, Settings, ensure_dirs, load_settings
from .storage import db as _db

DEFAULT_WORKFLOW = """schema_version: "1.0"
budget:
  max_research_calls: 3
  max_coding_calls: 4
  max_fix_attempts: 3
  max_experiments: 24
  max_epochs: 200
  max_wall_clock_s: 3600
  max_cloud_tokens: 400000
  max_no_improve: 4
  max_iterations: 6
  max_research_steps: 8
  max_candidates: 3
  max_data_operations: 24
  max_added_samples: 10000
  max_annotation_samples: 1000
  max_memory_mb: 0
  max_output_file_mb: 0
search:
  backend: disabled
  sort_by: relevance
memory:
  share_lessons: false
logging:
  exchange_events: true
  emit_event_seq: true
approval:
  default_mode: scoped_auto
  require_hash: true
upload_policy:
  default: restricted
  max_read_bytes: 262144
  allowed_suffixes: [".md", ".txt", ".json", ".yaml", ".yml", ".py", ".csv", ".tex", ".bib"]
providers:
  cloud: deepseek
  local: qwen_local
  coding: codex
stop_rules:
  require_all_hard_constraints: true
  min_metric_delta: 0.0
"""

DEFAULT_ENV = """# Algo-Agent credentials example. Copy to .env. Never commit real secrets.
ALGO_CLOUD_PROVIDER=deepseek
ALGO_CLOUD_BASE_URL=https://api.deepseek.com/v1
ALGO_CLOUD_API_KEY=
ALGO_CLOUD_MODEL=deepseek-flash
ALGO_RESEARCH_MODEL=
ALGO_CHAT_MODEL=
ALGO_INTENT_MODEL=
ALGO_INTENT_LLM=1
ALGO_LOCAL_BASE_URL=http://127.0.0.1:8084/v1
ALGO_LOCAL_MODEL=qwen3.8-27b
ALGO_LOCAL_API_KEY=
ALGO_CODEX_BIN=codex
ALGO_CODEX_ALLOW_REAL=0
ALGO_CODING_MODEL=
ALGO_CODING_REASONING_EFFORT=
ALGO_CODING_TRANSPORT=local
ALGO_CODING_SSH_HOST=
ALGO_CODING_SSH_ROOT=
ALGO_CODING_SSH_PYTHON=python3
ALGO_CODING_SSH_CODEX=codex
# Optional: ALGO_SEARCH_BACKEND=arxiv (sends technical queries externally)
ALGO_SEARCH_URL=
ALGO_SERVICE_HOST=127.0.0.1
ALGO_SERVICE_PORT=8765
"""


def init_platform(settings: Settings) -> Dict[str, Any]:
    """Create platform directories and default config; never clobber existing files."""
    created: List[str] = []
    kept: List[str] = []

    settings.root.mkdir(parents=True, exist_ok=True)
    marker = settings.root / MARKER
    if not marker.exists():
        marker.write_text("algo-agent platform root\n", encoding="utf-8")
        created.append(str(marker))

    ensure_dirs(settings)

    cfg_dir = settings.config_path.parent
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if not settings.config_path.exists():
        settings.config_path.write_text(DEFAULT_WORKFLOW, encoding="utf-8")
        created.append(str(settings.config_path))
    else:
        kept.append(str(settings.config_path))

    env_example = settings.root / ".env.example"
    if not env_example.exists():
        env_example.write_text(DEFAULT_ENV, encoding="utf-8")
        created.append(str(env_example))
    else:
        kept.append(str(env_example))

    # initialize DB + schema (idempotent)
    _db.init_db(settings.db_path)

    return {"root": str(settings.root), "created": created, "kept": kept,
            "db": str(settings.db_path), "schema_version": _db.DB_SCHEMA_VERSION}


def doctor(settings: Settings) -> Dict[str, Any]:
    from .providers import provider_report

    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, level: str = "ok") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "level": level if ok else "warning"})

    py_ok = sys.version_info >= (3, 11)
    add("python", py_ok, f"{platform.python_version()} at {sys.executable}", level="error")

    add("root", settings.root.exists(), str(settings.root), level="error")
    add("marker", (settings.root / MARKER).exists(),
        f"{MARKER} present" if (settings.root / MARKER).exists() else "run init to create",
        level="warning")

    dirs_ok = all((settings.root / d).exists() for d in
                  ("configs", "runtimes", "docs", "shared_assets", "algorithm_plugins"))
    add("directories", dirs_ok, "required directories present" if dirs_ok else "run: algo-agent init")

    lmd = settings.local_models_dir
    installed = sorted(d.name for d in lmd.iterdir()
                       if d.is_dir() and not d.name.startswith(("_", "."))) if lmd.exists() else []
    add("local_models", lmd.exists(),
        f"{lmd.name}/ ({len(installed)} model(s){': ' + ', '.join(installed) if installed else ''})"
        if lmd.exists() else "run: algo-agent init", level="warning")

    try:
        conn = _db.init_db(settings.db_path)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        add("database", True, f"{settings.db_path} schema={row['value'] if row else '?'}")
        conn.close()
    except (sqlite3.Error, OSError) as exc:
        add("database", False, f"cannot open {settings.db_path}: {exc}", level="error")

    writable = os.access(settings.root, os.W_OK)
    add("writable", writable, "root writable" if writable else "root is not writable", level="error")

    try:
        import langgraph
        add("langgraph", True, f"langgraph import ok ({getattr(langgraph, '__file__', '')})")
    except Exception as exc:
        add("langgraph", False, f"langgraph unavailable: {exc}", level="error")

    providers = []
    try:
        for status in provider_report(settings):
            providers.append({"name": status.name, "available": status.available,
                              "kind": status.kind, "detail": status.detail,
                              "capabilities": status.capabilities})
            add(f"provider:{status.name}", status.available, status.detail,
                level="ok" if status.available else "notice")
    except Exception as exc:
        add("providers", False, f"provider build failed: {exc}", level="error")

    offline = settings.offline()
    add("mode", True, "OFFLINE (no cloud credential)" if offline else "cloud credential configured")

    ok = all(c["ok"] for c in checks if c["level"] == "error")
    return {"root": str(settings.root), "offline": offline, "ok": ok, "checks": checks,
            "providers": providers}


def load_or_init(root: str | None = None) -> Settings:
    settings = load_settings(root)
    if not (settings.root / MARKER).exists():
        init_platform(settings)
        settings = load_settings(str(settings.root))
    return settings


__all__ = ["init_platform", "doctor", "load_or_init", "DEFAULT_WORKFLOW", "DEFAULT_ENV"]
