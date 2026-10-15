"""Asset registry: immutability, idempotency, publish ladder (guide 12.1)."""

from __future__ import annotations

import pytest

from algo_agent.schemas import contracts as C
from algo_agent.storage import AssetError, AssetRegistry


def _manifest(asset_id: str, version: str = "v1", **kw) -> C.AssetManifest:
    data = {"asset_id": asset_id, "asset_version": version, "asset_type": "recipe",
            "source_task": "unit_task", "license": "synthetic"}
    data.update(kw)
    return C.AssetManifest(**data)


def test_register_is_idempotent_across_fresh_manifests(settings, repo):
    reg = AssetRegistry(settings, repo)
    first = reg.register(_manifest("unit_idem"))
    # a brand-new manifest object has a different created_at; content is identical
    second = reg.register(_manifest("unit_idem"))
    assert first.asset_id == second.asset_id == "unit_idem"
    assert second.validation_status == "researched"


def test_register_is_idempotent_with_body_files(settings, repo, tmp_path):
    body = tmp_path / "recipe.json"
    body.write_text('{"recipe_id": "r1"}', encoding="utf-8")
    reg = AssetRegistry(settings, repo)
    first = reg.register(_manifest("unit_body_idem"), body_files=[body])
    assert list(first.checksums) == ["recipe.json"]
    # same manifest + same body -> idempotent, and works without re-supplying the body
    assert reg.register(_manifest("unit_body_idem"), body_files=[body]).checksums == first.checksums
    assert reg.register(_manifest("unit_body_idem")).checksums == first.checksums


def test_register_rejects_a_changed_body_file(settings, repo, tmp_path):
    body = tmp_path / "recipe.json"
    body.write_text('{"recipe_id": "r1"}', encoding="utf-8")
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_body_changed"), body_files=[body])
    body.write_text('{"recipe_id": "r2"}', encoding="utf-8")
    with pytest.raises(AssetError) as exc:
        reg.register(_manifest("unit_body_changed"), body_files=[body])
    assert exc.value.code == "asset_immutable"


def test_register_rejects_changed_content_for_existing_version(settings, repo):
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_immutable"))
    with pytest.raises(AssetError) as exc:
        reg.register(_manifest("unit_immutable", license="proprietary"))
    assert exc.value.code == "asset_immutable"
    assert "bump asset_version" in exc.value.recovery


def test_publish_requires_validation_and_advances_one_step(settings, repo):
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_publish"))
    with pytest.raises(AssetError) as exc:
        reg.publish("unit_publish", "v1")
    assert exc.value.code == "validation_required"

    record = {"run_id": "run_1", "eval_id": "eval_1", "result": "rmse 0.2",
              "verified_at": C.utcnow()}
    published = reg.publish("unit_publish", "v1", validation=record)
    assert published.validation_status == "reproduced"

    with pytest.raises(AssetError) as exc:
        reg.publish("unit_publish", "v1", validation=record, new_status="reusable")
    assert exc.value.code == "skipped_step"


def test_publish_blocks_on_unresolved_dependencies(settings, repo):
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_dep_parent", dependencies=["unit_dep_missing:v1"]))
    with pytest.raises(AssetError) as exc:
        reg.publish("unit_dep_parent", "v1", validation={"run_id": "r", "eval_id": "e",
                                                         "result": "ok"})
    assert exc.value.code == "unresolved_dependencies"


def test_resolve_dependencies_reports_local_versions(settings, repo):
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_child"))
    reg.register(_manifest("unit_parent", dependencies=["unit_child:v1"]))
    deps = reg.resolve_dependencies("unit_parent", "v1")
    assert deps == [{"asset_id": "unit_child", "version": "v1", "resolved": True,
                     "status": "researched"}]


def test_deprecate_is_terminal_for_status_changes(settings, repo):
    reg = AssetRegistry(settings, repo)
    reg.register(_manifest("unit_deprecated"))
    reg.deprecate("unit_deprecated", "v1", reason="superseded")
    with pytest.raises(AssetError) as exc:
        reg.publish("unit_deprecated", "v1", validation={"run_id": "r", "eval_id": "e"})
    assert exc.value.code == "deprecated"


def test_unknown_asset_raises_not_found(settings, repo):
    reg = AssetRegistry(settings, repo)
    with pytest.raises(AssetError) as exc:
        reg.get("unit_missing_asset", "v1")
    assert exc.value.status == 404
