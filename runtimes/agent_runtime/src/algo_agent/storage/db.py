"""SQLite storage: versioned migrations + connection helpers (guide 5, 7).

The database is the source of truth for state, events, jobs, assets and the
call ledger. WAL mode + a single writer discipline keep a single-task writer
authoritative while allowing readers/writers from service and worker threads.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Tuple

DB_SCHEMA_VERSION = 3

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, task_slug TEXT, phase TEXT, state_version INTEGER DEFAULT 0,
    iteration INTEGER DEFAULT 0, active_plan_id TEXT, best_model_id TEXT, stop_reason TEXT,
    paused INTEGER DEFAULT 0, spec_json TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY, task_id TEXT, version TEXT, parent_plan_id TEXT,
    content_hash TEXT, status TEXT, spec_json TEXT, created_at TEXT,
    UNIQUE(task_id, version));
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, task_id TEXT, plan_id TEXT, plan_hash TEXT,
    scope_json TEXT, mode TEXT, approved_at TEXT, state_version INTEGER, revoked INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_approvals_task ON approvals(task_id, revoked);
CREATE TABLE IF NOT EXISTS commands (
    command_id TEXT PRIMARY KEY, task_id TEXT, conversation_id TEXT, intent TEXT, text TEXT,
    expected_state_version INTEGER, payload_json TEXT, idempotency_key TEXT UNIQUE, created_at TEXT);
CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, kind TEXT, level TEXT,
    message TEXT, data_json TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, event_seq);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, task_id TEXT, kind TEXT, status TEXT, idempotency_key TEXT UNIQUE,
    run_id TEXT, payload_json TEXT, result_json TEXT, lease_owner TEXT, attempts INTEGER DEFAULT 0,
    pid INTEGER, heartbeat_at TEXT, created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY, task_id TEXT, run_id TEXT, spec_json TEXT, status TEXT,
    recommended INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id TEXT PRIMARY KEY, task_id TEXT, model_id TEXT, protocol_id TEXT, spec_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY, task_id TEXT, model_id TEXT, spec_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT, version TEXT, manifest_json TEXT, created_at TEXT, PRIMARY KEY(dataset_id, version));
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT PRIMARY KEY, task_id TEXT, plan_id TEXT, candidate_id TEXT, spec_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS budget (
    task_id TEXT PRIMARY KEY, reserved_json TEXT, used_json TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, task_id TEXT, role TEXT, plan_id TEXT, baseline_version TEXT,
    turns INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS invocations (
    invocation_id TEXT PRIMARY KEY, task_id TEXT, role TEXT, plan_id TEXT, session_id TEXT,
    spec_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT, asset_version TEXT, task_id TEXT, asset_type TEXT, status TEXT,
    manifest_json TEXT, body_ref TEXT, created_at TEXT, PRIMARY KEY(asset_id, asset_version));
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY, task_id TEXT, uri TEXT, source_type TEXT, version TEXT,
    checksum TEXT, scope TEXT, upload_policy TEXT, trust TEXT, locator TEXT,
    ref_json TEXT, accessed_at TEXT);
CREATE TABLE IF NOT EXISTS read_audits (
    audit_id TEXT PRIMARY KEY, task_id TEXT, role TEXT, invocation_id TEXT, refs_json TEXT,
    requested_range TEXT, actual_range TEXT, upload_bytes INTEGER, status TEXT, usage_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY, dependencies_json TEXT, value_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS ledger (
    entry_id TEXT PRIMARY KEY, task_id TEXT, role TEXT, kind TEXT, tokens_in INTEGER,
    tokens_out INTEGER, cost REAL, currency TEXT, price_version TEXT, usage_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS change_requests (
    cr_id TEXT PRIMARY KEY, task_id TEXT, plan_id TEXT, invocation_id TEXT,
    spec_json TEXT, status TEXT, created_at TEXT);
"""

_LOCK = threading.Lock()


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(_INIT_SQL)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    current = int(row["value"]) if row else 0
    # additive migrations; each is idempotent
    migrations: Iterable[Tuple[int, str]] = [
        (1, "SELECT 1"),
        (2, "CREATE INDEX IF NOT EXISTS idx_models_task ON models(task_id)"),
        (3, "CREATE INDEX IF NOT EXISTS idx_feedback_task ON feedback(task_id)"),
    ]
    for version, script in migrations:
        if version > current:
            conn.executescript(script)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(DB_SCHEMA_VERSION),),
    )
    conn.commit()


def init_db(db_path: Path | str) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = connect(db_path)
        _migrate(conn)
    return conn


__all__ = ["connect", "init_db", "DB_SCHEMA_VERSION"]
