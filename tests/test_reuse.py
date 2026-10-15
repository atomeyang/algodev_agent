"""Reuse surface: compatibility honesty, reproduction, bundles, migration (guide 12)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from algo_agent.reuse import (
    ReuseRequest,
    check_compatibility,
    compatibility_for,
    export_reproduction,
    export_task_bundle,
    import_task_bundle,
    migrate_state,
    validate_reproduction,
)
from algo_agent.schemas import contracts as C
from algo_agent.schemas.enums import Compatibility, ImportStatus
from algo_agent.storage.db import DB_SCHEMA_VERSION


def _recipe(**kw) -> C.RecipeSpec:
    data = {"recipe_id": "r1", "version": "v1", "inputs": ["rgb_image"],
            "outputs": ["quality_score"], "device": "mobile", "license": "synthetic"}
    data.update(kw)
    return C.RecipeSpec(**data)


def _manifest(**kw) -> C.AssetManifest:
    data = {"asset_id": "a1", "asset_version": "v1", "asset_type": "recipe",
            "license": "synthetic", "validation_status": "reproduced"}
    data.update(kw)
    return C.AssetManifest(**data)


# ---- compatibility -------------------------------------------------------- #
def test_unknown_is_never_silently_compatible():
    request = ReuseRequest(inputs=["rgb_image"], outputs=["quality_score"])
    result = check_compatibility(request, _recipe(), _manifest())
    assert result.status == Compatibility.UNKNOWN.value
    assert "device profile" in result.missing
    assert "business distribution" in result.missing


def test_all_explicit_fields_matching_reports_compatible():
    request = ReuseRequest(inputs=["rgb_image"], outputs=["quality_score"], device="mobile",
                           license="synthetic", distribution="mobile_photos")
    result = check_compatibility(request, _recipe(), _manifest())
    assert result.status == Compatibility.COMPATIBLE.value
    assert result.verification_actions


def test_interface_mismatch_is_incompatible():
    request = ReuseRequest(inputs=["audio_waveform"], outputs=["quality_score"], device="mobile",
                           license="synthetic", distribution="mobile_photos")
    result = check_compatibility(request, _recipe(), _manifest())
    assert result.status == Compatibility.INCOMPATIBLE.value


def test_recipe_device_mismatch_needs_adaptation():
    request = ReuseRequest(inputs=["rgb_image"], outputs=["quality_score"], device="edge_tpu",
                           license="synthetic", distribution="mobile_photos")
    result = check_compatibility(request, _recipe(), _manifest())
    assert result.status == Compatibility.ADAPTATION_REQUIRED.value


def test_compatibility_for_missing_asset_stays_unknown(repo):
    result = compatibility_for(repo, "ghost", "v1", ReuseRequest())
    assert result.status == Compatibility.UNKNOWN.value
    assert "asset" in result.missing


def test_status_change_invalidates_cached_compatibility(settings, repo, tmp_path):
    """A cached "compatible" verdict must not survive a deprecation (guide 12.2)."""
    from algo_agent.storage import AssetRegistry

    recipe = {"recipe_id": "r_dep", "version": "v1", "inputs": ["rgb_image"],
              "outputs": ["quality_score"], "device": "mobile",
              "applicable_distribution": "mobile_photos"}
    recipe_path = tmp_path / "recipe_dep.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    asset_id = f"unit_cache_{uuid.uuid4().hex[:8]}"
    reg = AssetRegistry(settings, repo)
    reg.register(C.AssetManifest(asset_id=asset_id, asset_version="v1", asset_type="recipe",
                                 license="synthetic", artifact_refs=[str(recipe_path)]))
    request = ReuseRequest(inputs=["rgb_image"], outputs=["quality_score"], device="mobile",
                           license="synthetic", distribution="mobile_photos")

    assert compatibility_for(repo, asset_id, "v1", request).status == Compatibility.COMPATIBLE.value
    reg.deprecate(asset_id, "v1", reason="superseded")
    result = compatibility_for(repo, asset_id, "v1", request)
    assert result.status == Compatibility.INCOMPATIBLE.value
    assert any("deprecated" in r for r in result.reasons)


# ---- reproduction --------------------------------------------------------- #
def test_reproduction_package_exports_and_validates(settings, demo_report, demo_repo):
    exported = export_reproduction(settings, demo_repo, "demo_task")
    assert exported["files"]
    assert Path(exported["dir"]).is_dir()

    result = validate_reproduction(settings, demo_repo, "demo_task")
    assert result["status"] != "failed", result
    assert result["status"] in {"validated", "blocked", "mismatch"}
    if result["status"] == "validated":
        assert not result["blockers"] and not result["mismatches"]


def test_reproduction_detects_out_of_tolerance_results(settings, demo_report, demo_repo):
    export_reproduction(settings, demo_repo, "demo_task")
    expected = validate_reproduction(settings, demo_repo, "demo_task")["expected_results"]
    if not expected:
        pytest.skip("demo task recorded no expected results")
    key = next(iter(expected))
    result = validate_reproduction(settings, demo_repo, "demo_task",
                                   actual_metrics={key: float(expected[key]) + 10.0})
    assert result["status"] == "mismatch"
    assert any(key in m for m in result["mismatches"])


# ---- bundles / migration -------------------------------------------------- #
def test_bundle_export_import_is_hashed_and_no_overwrite(settings, demo_report, demo_repo):
    exported = export_task_bundle(settings, demo_repo, "demo_task")
    assert exported["files"] > 0
    new_task = f"imported_{uuid.uuid4().hex[:8]}"

    first = import_task_bundle(settings, demo_repo, Path(exported["dir"]),
                               target_task_id=new_task)
    assert first.status == ImportStatus.READY.value, first.events
    assert demo_repo.task_exists(new_task)
    from algo_agent.paths import TaskPaths
    assert TaskPaths(settings, new_task).root.name == f"{new_task}_experiment"
    assert TaskPaths(settings, "demo_task").root.name == "demo_task_experiment"

    again = import_task_bundle(settings, demo_repo, Path(exported["dir"]),
                               target_task_id=new_task)
    assert again.status == ImportStatus.FAILED.value
    assert "already exists" in " ".join(again.events)


def test_bundle_dry_run_does_not_create_the_task(settings, demo_report, demo_repo):
    exported = export_task_bundle(settings, demo_repo, "demo_task")
    probe = f"dryrun_{uuid.uuid4().hex[:8]}"
    result = import_task_bundle(settings, demo_repo, Path(exported["dir"]),
                                target_task_id=probe, dry_run=True)
    assert result.status == ImportStatus.READY.value
    assert not demo_repo.task_exists(probe)


def test_migrate_reports_schema_version(settings, demo_report):
    result = migrate_state(settings, backup=True)
    assert result["status"] == ImportStatus.READY.value
    assert result["to_version"] == DB_SCHEMA_VERSION


def test_export_outputs_stay_in_task_and_do_not_include_previous_bundles(settings, demo_repo):
    from algo_agent.paths import TaskPaths
    tp = TaskPaths(settings, "demo_task")
    first = export_task_bundle(settings, demo_repo, "demo_task")
    second = export_task_bundle(settings, demo_repo, "demo_task")
    assert Path(first["dir"]).is_relative_to(tp.deliverables)
    assert Path(second["dir"]).is_relative_to(tp.deliverables)
    assert first["files"] == second["files"]
    manifest = json.loads(Path(second["manifest"]).read_text())
    assert not any("deliverables/bundles/" in f["path"] or "runtime/" in f["path"]
                   for f in manifest["files"])
    for exporter in (export_task_bundle, export_reproduction):
        with pytest.raises(ValueError):
            exporter(settings, demo_repo, "demo_task", out_dir=settings.root / "outside")
