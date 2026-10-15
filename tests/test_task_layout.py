"""Task ownership, annotation isolation and lossless directory consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algo_agent.config import Settings, load_settings
from algo_agent.layout_migration import consolidate_task
from algo_agent.paths import (
    LAYOUT_FILE, TaskPaths, ensure_task_dirs, read_json, register_task_member, write_json,
)


def test_all_generated_directories_belong_to_one_task(tmp_path):
    tp = TaskPaths(Settings(tmp_path), "example")
    ensure_task_dirs(tp)
    for directory in (
        tp.annotations / "plans", tp.annotations / "runs", tp.annotations / "reviews",
        tp.evaluation / "protocols", tp.evaluation / "intermediate", tp.evaluation / "results",
        tp.coding_workspaces, tp.runtime / "tmp", tp.runtime / "state", tp.evidence / "sources",
    ):
        assert directory.is_dir() and directory.is_relative_to(tp.root)
    assert {p.name for p in tmp_path.iterdir()} == {"example_experiment"}
    assert "annotation" in (tp.root / "DIRECTORY_RULES.md").read_text()
    ensure_task_dirs(tp)


@pytest.mark.parametrize("suffix", ["iqa_experiment", "iqa_experiment/tracks/mos",
                                   "iqa_experiment/data/annotations"])
def test_task_and_nested_paths_never_become_platform_roots(tmp_path, suffix):
    assert load_settings(str(tmp_path / suffix)).root == tmp_path
    assert Settings(tmp_path / suffix).root == tmp_path


def test_root_discovery_ignores_legacy_platform_markers_inside_a_task(tmp_path, monkeypatch):
    from algo_agent.config import resolve_root
    monkeypatch.delenv("ALGO_ROOT", raising=False)
    nested = tmp_path / "iqa_experiment/history/legacy_rd/legacy_platform"
    nested.mkdir(parents=True)
    (nested / ".algo_agent_root").write_text("legacy")
    assert resolve_root(start=nested) == tmp_path


@pytest.mark.parametrize("slug", ["../escape", "/tmp/escape", "iqa_experiment", "UPPER"])
def test_invalid_task_slug_is_rejected(tmp_path, slug):
    with pytest.raises(ValueError):
        TaskPaths(Settings(tmp_path), slug)


def test_tracks_are_explicit_isolated_and_persist_across_reload(tmp_path):
    settings = Settings(tmp_path)
    register_task_member(settings, "iqa", "iqa_mos", "mos")
    tp = TaskPaths(load_settings(str(tmp_path)), "iqa_mos")
    ensure_task_dirs(tp)
    assert tp.root == tmp_path / "iqa_experiment" / "tracks" / "mos"
    assert not (tmp_path / "iqa_mos_experiment").exists()
    assert TaskPaths(settings, "iqa_other").root == tmp_path / "iqa_other_experiment"
    register_task_member(settings, "iqa", "iqa_mos", "mos")
    with pytest.raises(ValueError):
        register_task_member(settings, "iqa", "another", "mos")
    with pytest.raises(ValueError):
        register_task_member(settings, "different", "iqa_mos", "mos")


def test_task_output_paths_reject_absolute_traversal_and_symlink_escape(tmp_path):
    tp = TaskPaths(Settings(tmp_path), "one")
    ensure_task_dirs(tp)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tp.root / "escape").symlink_to(outside, target_is_directory=True)
    for path in ("../escape", outside / "a", "escape/a"):
        with pytest.raises(ValueError):
            tp.output_path(path)
    assert tp.output_path(tp.root / "valid") == tp.root / "valid"
    (tp.annotations / "runs").rmdir()
    (tp.annotations / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        ensure_task_dirs(tp)


def test_annotation_script_uses_versions_and_rejects_external_or_existing_outputs(tmp_path):
    from scripts.iqa_annotate import output_paths
    tp = TaskPaths(Settings(tmp_path), "labels")
    ensure_task_dirs(tp)
    out, report, scheme = output_paths(tp, "v2", "attempt1")
    assert out == tp.annotations / "runs/v2/attempt1/prelabels.jsonl"
    assert report.suffix == ".json"
    assert scheme == tp.annotations / "plans/v2/attempt1.json"
    with pytest.raises(ValueError):
        output_paths(tp, "v1", "attempt2", str(tmp_path / "labels.jsonl"))
    with pytest.raises(ValueError):
        output_paths(tp, "v1", "attempt2", "evaluation/protocols/labels.jsonl")
    out.parent.mkdir(parents=True)
    out.write_text("existing")
    with pytest.raises(ValueError):
        output_paths(tp, "v2", "attempt1")


def test_annotation_pilot_does_not_read_other_tasks(tmp_path):
    from algorithm_plugins.iqa.plans import annotation_pilot
    one, two = [TaskPaths(Settings(tmp_path), s) for s in ("one", "two")]
    for tp, count in ((one, 3), (two, 9)):
        write_json(tp.annotations / "runs/v1/r/prelabels.agreement.json", {"n_paired": count})
    assert annotation_pilot() == []
    assert annotation_pilot(task_root=one.root)[0]["n_paired"] == 3
    assert annotation_pilot(task_root=two.root)[0]["n_paired"] == 9


def test_consolidation_preserves_bytes_and_resolves_old_references(tmp_path):
    settings = Settings(tmp_path)
    old = tmp_path / "iqa_mos_experiment"
    old.mkdir()
    original = b'{"phase":"accepted","path":"original unmodified evidence"}\n'
    (old / "summary.json").write_bytes(original)
    legacy = tmp_path / "iqa_rd/iqa_rd_experiment"
    legacy.mkdir(parents=True)
    (legacy / "result.json").write_text('{"score":0.9}')
    (legacy.parent / "README.md").write_text("legacy")
    moves = {"iqa_mos_experiment": "tracks/mos", "iqa_rd/iqa_rd_experiment": "history/legacy_rd",
             "iqa_rd": "history/legacy_rd/legacy_platform"}
    members = {"iqa_mos": "tracks/mos", "iqa_rd": "history/legacy_rd"}
    preview = consolidate_task(settings, "iqa", moves, members)
    assert preview["status"] == "planned" and old.exists()
    result = consolidate_task(settings, "iqa", moves, members, apply=True)
    assert result["status"] == "verified"
    tp = TaskPaths(settings, "iqa_mos")
    assert (tp.root / "summary.json").read_bytes() == original
    for prefix in ("iqa_mos_experiment", str(old)):
        assert tp.resolve_reference(prefix + "/summary.json") == tp.root / "summary.json"
    assert not old.exists() and not (tmp_path / "iqa_rd").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"iqa_experiment"}
    assert consolidate_task(settings, "iqa", moves, members, apply=True)["status"] == "verified"


def test_consolidation_refuses_to_overwrite_or_follow_symlinks(tmp_path):
    settings = Settings(tmp_path)
    old = tmp_path / "labels"
    old.mkdir()
    destination = tmp_path / "iqa_experiment/data/annotations"
    destination.mkdir(parents=True)
    with pytest.raises(ValueError, match="both"):
        consolidate_task(settings, "iqa", {"labels": "data/annotations"}, {}, apply=True)
    (old / "link").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlink"):
        consolidate_task(settings, "iqa", {"labels": "history/labels"}, {}, apply=True)


def test_registered_paths_cannot_create_a_second_top_level_root(tmp_path):
    settings = Settings(tmp_path)
    register_task_member(settings, "iqa", "iqa_mos", "mos")
    (tmp_path / "iqa_mos_experiment").mkdir()
    with pytest.raises(ValueError, match="both"):
        TaskPaths(settings, "iqa_mos").root


def test_historical_job_paths_are_rebound_without_rewriting_the_job(tmp_path):
    from algo_agent.execution.method_worker import load_job
    settings = Settings(tmp_path)
    register_task_member(settings, "iqa", "iqa_mos", "mos")
    tp = TaskPaths(settings, "iqa_mos")
    layout = read_json(tp.owner_root / LAYOUT_FILE)
    layout["relocations"]["iqa_mos_experiment"] = "tracks/mos"
    write_json(tp.owner_root / LAYOUT_FILE, layout)
    old = tmp_path / "iqa_mos_experiment"
    path = tp.root / "runs/old/job.json"
    write_json(path, {"task_slug": "iqa_mos", "task_root": str(old),
                     "out_dir": str(old / "runs/old"), "release_dir": str(old / "data/releases/v1"),
                     "method_entry": str(old / "model_pool/m/method.py")})
    original = path.read_bytes()
    job = load_job(path)
    assert Path(job["task_root"]) == tp.root
    assert Path(job["method_entry"]) == tp.root / "model_pool/m/method.py"
    assert path.read_bytes() == original


def test_migration_member_conflict_is_detected_before_any_move(tmp_path):
    settings = Settings(tmp_path)
    register_task_member(settings, "iqa", "member", "original")
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValueError, match="conflicting"):
        consolidate_task(settings, "iqa", {"labels": "history/labels"},
                         {"member": "tracks/different"}, apply=True)
    assert (tmp_path / "labels").exists()


def test_controller_track_creation_and_state_snapshot(tmp_path, monkeypatch):
    from algo_agent.service.controller import Controller
    monkeypatch.setenv("ALGO_OFFLINE", "1")
    ctl = Controller(Settings(tmp_path), inline=True)
    ctl.create_task("RMSE <= 1", "parent", demo=True)
    spec = ctl.create_task("RMSE <= 1", "child", demo=True, parent_task="parent", track="variant")
    tp = TaskPaths(ctl.settings, spec.task_slug)
    assert tp.root == tmp_path / "parent_experiment/tracks/variant"
    assert read_json(tp.runtime / "state/snapshot.json")["task"]["task_id"] == "child"
    assert (tp.root / "task.yaml").exists()
    assert not (tmp_path / "child_experiment").exists()


def test_coding_workspaces_are_task_local(demo_repo, settings):
    tp = TaskPaths(settings, "demo_task")
    assert list(tp.coding_workspaces.glob("*/coding_contract.json"))
    assert not settings.coding_workspaces.exists()
