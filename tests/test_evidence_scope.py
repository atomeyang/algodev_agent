"""Evidence reader scope rules: raw data and frozen test sets are never uploadable."""

from __future__ import annotations

import uuid

import pytest

from algo_agent.evidence import EvidenceReader
from algo_agent.paths import TaskPaths, ensure_task_dirs
from algo_agent.schemas import contracts as C


@pytest.fixture()
def scoped(settings, repo):
    slug = f"scope_{uuid.uuid4().hex[:8]}"
    tp = TaskPaths(settings, slug)
    ensure_task_dirs(tp)
    (tp.root / "notes.md").write_text("readable evidence\n", encoding="utf-8")
    (tp.data / "raw" / "raw_notes.md").write_text("raw\n", encoding="utf-8")
    (tp.data / "releases" / "v1").mkdir(parents=True, exist_ok=True)
    (tp.data / "releases" / "v1" / "notes.md").write_text("released\n", encoding="utf-8")
    (tp.root / "weights.bin").write_bytes(b"\x00\x01\x02")
    (tp.root / "big.md").write_text("x" * 400_000, encoding="utf-8")
    return {"slug": slug, "reader": EvidenceReader(settings, repo)}


def _ref(uri: str, scope: str = "task") -> C.EvidenceRef:
    return C.EvidenceRef(evidence_id=f"ev_{uuid.uuid4().hex[:6]}", uri=uri, scope=scope)


def test_scoped_markdown_is_readable(scoped):
    result = scoped["reader"].read_evidence(_ref("task://notes.md"), scoped["slug"], "researcher")
    assert result["status"] == "ok"
    assert "readable evidence" in result["content"]
    assert result["upload_bytes"] > 0


def test_raw_data_is_denied(scoped):
    result = scoped["reader"].read_evidence(_ref("task://data/raw/raw_notes.md"),
                                            scoped["slug"], "researcher")
    assert result["status"] == "denied"
    assert "raw data" in result["reason"]


def test_frozen_release_is_denied(scoped):
    result = scoped["reader"].read_evidence(_ref("task://data/releases/v1/notes.md"),
                                            scoped["slug"], "researcher")
    assert result["status"] == "denied"


def test_disallowed_suffix_is_denied(scoped):
    result = scoped["reader"].read_evidence(_ref("task://weights.bin"), scoped["slug"], "researcher")
    assert result["status"] == "denied"
    assert "suffix" in result["reason"]


def test_oversize_file_is_denied(scoped):
    result = scoped["reader"].read_evidence(_ref("task://big.md"), scoped["slug"], "researcher")
    assert result["status"] == "denied"
    assert "max_read_bytes" in result["reason"]


def test_out_of_scope_evidence_is_denied(scoped):
    result = scoped["reader"].read_evidence(_ref("task://notes.md", scope="external"),
                                            scoped["slug"], "researcher")
    assert result["status"] == "denied"
    assert "scope" in result["reason"]


def test_denied_reads_are_audited(scoped):
    import json
    import sqlite3

    ref = _ref("task://data/raw/raw_notes.md")
    reader = scoped["reader"]
    reader.read_evidence(ref, scoped["slug"], "researcher", invocation_id="inv_1")
    with sqlite3.connect(str(reader.repo.db_path)) as conn:
        rows = conn.execute(
            "SELECT refs_json, status FROM read_audits ORDER BY created_at DESC").fetchall()
    assert rows
    refs = json.loads(rows[0][0])
    assert refs == [ref.evidence_id]
    assert rows[0][1] == "denied"
