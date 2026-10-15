"""Small real image fixtures; no live annotation service or GPU is used."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from algo_agent.execution.data_pipeline import prepare_development, prepare_final, verify_release
from algo_agent.paths import TaskPaths, sha256_file
from algo_agent.schemas import contracts as C
from tests.test_auto_process import ctrl, _plan


def _fixture(tmp_path, monkeypatch, ctrl):
    root = tmp_path / "koniq"
    image_dir = root / "512x384"
    image_dir.mkdir(parents=True)
    rows = []
    for i, split in enumerate(["training"] * 8 + ["validation"] * 3 + ["test"] * 3):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (12, 16, 3), dtype=np.uint8)).save(
            image_dir / f"{i}.jpg")
        rows.append(f"{i}.jpg,{30 + i * 2},{split}")
    (root / "koniq10k_distributions_sets.csv").write_text("image_name,MOS,set\n" + "\n".join(rows))
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    monkeypatch.setenv("KONIQ_DIR", str(root))
    spec = ctrl.create_task("", "iqa_data", auto_process=True, task_spec={
        "task_id": "iqa_data", "task_slug": "iqa_data", "domain": "iqa",
        "hard_constraints": [{"name": "rom_mb_fp16", "value": 60, "unit": "MB"}],
        "primary_metric": {"name": "plcc", "threshold": 0.5, "direction": "max"}})
    return spec, image_dir


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_iqa_augmentation_and_annotation_rubric_are_executable(tmp_path, monkeypatch, ctrl):
    from algorithm_plugins.iqa import annotation
    spec, image_dir = _fixture(tmp_path, monkeypatch, ctrl)
    called = []
    def annotate(path, **kwargs):
        called.append(kwargs)
        return {"ok": True, "schema_valid": True, "quality_score_0_100": 55.0, "confidence_0_1": 0.95,
                "defects": {}, "model_id": "fake_vlm", "prompt_version": "test",
                "usage": {}, "path": path}
    monkeypatch.setattr(annotation, "annotate", annotate)
    monkeypatch.setattr(annotation, "endpoint_config",
                        lambda: {"base_url": "http://127.0.0.1:1234", "model_id": "fake_vlm"})
    plan = _plan(ctrl, spec, [
        {"operation_id": "flip", "kind": "augment", "hypothesis": "mirror invariance",
         "params": {"count": 2, "label_preserving": True}},
        {"operation_id": "label", "kind": "annotate", "hypothesis": "consistent severity rubric",
         "params": {"limit": 2, "rubric": "Distinguish texture from blur.", "min_confidence": 0.9}},
        {"operation_id": "stress_flip", "kind": "augment", "split": "development_test",
         "hypothesis": "test mirror invariance", "params": {"count": 1, "label_preserving": True}},
    ])
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    manifest = prepare_development(tp, spec, plan, "v1", ctrl.repo)
    assert manifest.splits == {"train": 10, "development_test": 4, "final_test": 3}
    assert len(called) == 2 and "texture" in called[0]["rubric"]
    schemes = list((tp.annotations / "plans/v1").glob("*.json"))
    assert len(schemes) == 1
    assert "texture" in json.loads(schemes[0].read_text())["operation"]["params"]["rubric"]
    responses = list((tp.annotations / "runs/v1").glob("*/responses.jsonl"))
    assert len(responses) == 1 and len(_read(responses[0])) == 2
    assert _read(responses[0])[0]["response"]["quality_score_0_100"] == 55.0
    train = _read(tp.release_dir("v1") / "dev.jsonl")
    assert train[0]["label_provenance"]["label_kind"] == "weak"
    assert train[0]["label_provenance"]["review_required"] is True
    development = _read(tp.release_dir("v1") / "eval_inputs.jsonl")
    assert not {r["path"] for r in train} & {r["path"] for r in development}
    assert all(set(r) == {"path", "image"} for r in development)
    final_version = prepare_final(tp, spec, "v1")
    assert len(_read(tp.release_dir(final_version) / "eval_inputs.jsonl")) == 3
    assert not (tp.release_dir(final_version) / "dev.jsonl").exists()


def test_iqa_external_image_edits_invalidate_release(tmp_path, monkeypatch, ctrl):
    spec, images = _fixture(tmp_path, monkeypatch, ctrl)
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    plan = _plan(ctrl, spec)
    prepare_development(tp, spec, plan, "v1", ctrl.repo)
    (images / "0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source image changed"):
        verify_release(tp, "v1")


def test_rejected_annotation_responses_survive_failed_data_preparation(tmp_path, monkeypatch, ctrl):
    from algorithm_plugins.iqa import annotation
    spec, _ = _fixture(tmp_path, monkeypatch, ctrl)
    monkeypatch.setattr(annotation, "endpoint_config", lambda: {"base_url": "http://127.0.0.1:1234"})
    monkeypatch.setattr(annotation, "annotate",
                        lambda path, **kw: {"ok": False, "path": path, "error": "invalid label"})
    plan = _plan(ctrl, spec, [
        {"operation_id": "reject", "kind": "annotate", "hypothesis": "rubric pilot",
         "params": {"limit": 2}}])
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    with pytest.raises(ValueError, match="no usable weak labels"):
        prepare_development(tp, spec, plan, "v1", ctrl.repo)
    responses = list((tp.annotations / "runs/v1").glob("*/responses.jsonl"))
    assert len(responses) == 1
    assert len(_read(responses[0])) == 2
    assert not any(r["accepted_as_weak_label"] for r in _read(responses[0]))
    assert not (tp.release_dir("v1") / "manifest.json").exists()


def test_iqa_independent_final_extension_is_frozen(tmp_path, monkeypatch, ctrl):
    spec, images = _fixture(tmp_path, monkeypatch, ctrl)
    path = images / "new.jpg"
    Image.fromarray(np.full((12, 16, 3), 77, dtype=np.uint8)).save(path)
    source = tmp_path / "labels.jsonl"
    source.write_text(json.dumps({"path": str(path), "image": path.name, "subject": "new",
                                 "score": 60.0, "defects": {}}) + "\n")
    spec.data_sources = [C.DataSource(source_id="independent", path=str(source),
                                      checksum=sha256_file(source), label_kind="human", independent=True)]
    ctrl.repo.save_spec(spec)
    plan = _plan(ctrl, spec, [{"operation_id": "coverage", "kind": "extend", "split": "final_test",
                              "source_id": "independent", "hypothesis": "improve final coverage"}])
    tp = TaskPaths(ctrl.settings, spec.task_slug)
    manifest = prepare_development(tp, spec, plan, "v1", ctrl.repo)
    assert manifest.splits["final_test"] == 4
    assert not any(r["path"] == str(path) for r in _read(tp.release_dir("v1") / "eval_inputs.jsonl"))
    version = prepare_final(tp, spec, "v1")
    assert len(_read(tp.release_dir(version) / "eval_inputs.jsonl")) == 4


def test_iqa_copied_test_image_cannot_be_added_to_training(tmp_path, monkeypatch, ctrl):
    import shutil
    spec, images = _fixture(tmp_path, monkeypatch, ctrl)
    path = images / "copied.jpg"
    shutil.copy2(images / "13.jpg", path)
    source = tmp_path / "leak.jsonl"
    source.write_text(json.dumps({"path": str(path), "image": path.name, "subject": "renamed",
                                 "score": 56.0, "defects": {}}) + "\n")
    spec.data_sources = [C.DataSource(source_id="leak", path=str(source),
                                      checksum=sha256_file(source), label_kind="human")]
    ctrl.repo.save_spec(spec)
    plan = _plan(ctrl, spec, [{"operation_id": "more", "kind": "extend", "source_id": "leak",
                              "hypothesis": "more samples"}])
    with pytest.raises(ValueError, match="overlap"):
        prepare_development(TaskPaths(ctrl.settings, spec.task_slug), spec, plan, "v1", ctrl.repo)


def test_iqa_mos_head_can_reload_without_refitting(tmp_path):
    import joblib
    from algorithm_plugins.iqa.methods.score_mos_transfer import _fit_predict_ridge
    rng = np.random.default_rng(7)
    X, E = rng.normal(size=(20, 4)), rng.normal(size=(4, 4))
    y = X[:, 0] * 3 + 40
    artifact = tmp_path / "head.joblib"
    predicted, _ = _fit_predict_ridge(X, y, E, list(range(15)), list(range(15, 20)), [0.1, 1],
                                     artifact_path=artifact)
    saved = joblib.load(artifact)
    actual = saved["model"].predict(E) * saved["sd"] + saved["mu"]
    assert np.allclose(actual, predicted)


def test_iqa_mos_head_automatic_subprocess_roundtrip(tmp_path, monkeypatch, ctrl):
    spec, _ = _fixture(tmp_path, monkeypatch, ctrl)
    source = tmp_path / "features.npz"
    np.savez(source, names=np.array([f"{i}.jpg" for i in range(14)]),
             feat=np.array([[i / 100, 1.0, 0.0, 0.0] for i in range(14)], dtype=np.float32))
    monkeypatch.setenv("IQA_FEATURE_NPZ", str(source))
    boundary = ctrl.run(spec.task_id)
    assert boundary["state"]["phase"] == "accepted", boundary
    evaluations = ctrl.repo.list_evals(spec.task_id)
    assert {e.stage for e in evaluations} == {"development", "final"}
    assert all(e.passed for e in evaluations)
    model = ctrl.repo.get_model(boundary["state"]["best_model_id"])
    assert model.weights_ref.endswith("mos_head.joblib")
