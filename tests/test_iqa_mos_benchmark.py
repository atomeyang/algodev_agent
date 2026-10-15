# -*- coding: utf-8 -*-
"""Real human-MOS (KonIQ-10k) track tests.

Torch-free: the exporter, the quality-only metric semantics, the method
registration and the plugin wiring are all verifiable without a GPU.  The heavy
feature extraction / head fit is exercised by the real research loop; here we
use a tiny KonIQ-shaped fixture pointed at by ``KONIQ_DIR``.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from algorithm_plugins.iqa import config as C
from algorithm_plugins.iqa import data as D
from algorithm_plugins.iqa import metrics as M
from algorithm_plugins.iqa import plans as P
from algorithm_plugins.iqa.methods import METHOD_FILES
from algorithm_plugins.iqa.methods import score_mos_transfer as MT
from algorithm_plugins.iqa.plugin import PLUGIN, _METHOD_ALIASES, _VARIANT_CONFIG

MOS_REPORT = ("轻量双输出 IQA 模型研发：输出 0-100 质量分与缺陷多标签；"
              "主指标 PLCC >= 0.85（真实 MOS，KonIQ-10k 官方测试集），SRCC >= 0.85；"
              "ROM < 60MB fp16；参考 MobileCLIP2-S0 方案。")


def _write_jpg(path):
    import cv2
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(48, 64, 3)).astype("uint8")
    cv2.imwrite(str(path), img)


def _fake_tp(tmp_path):
    return types.SimpleNamespace(
        release_dir=lambda v: tmp_path / "data" / "releases" / v,
        evaluation=tmp_path / "evaluation", slug="iqa_mos")


def _koniq_fixture(tmp_path, monkeypatch):
    """A 9-image KonIQ stand-in: 4 training + 2 validation + 3 test."""
    root = tmp_path / "koniq10k"
    img_dir = root / "512x384"
    img_dir.mkdir(parents=True)
    rows = []
    mos = {"training": 70.0, "validation": 55.0, "test": 40.0}
    for i, st in enumerate(["training"] * 4 + ["validation"] * 2 + ["test"] * 3):
        name = f"img{i:02d}.jpg"
        _write_jpg(img_dir / name)
        rows.append((name, mos[st] + i, st))
    csv = root / "koniq10k_distributions_sets.csv"
    csv.write_text("image_name,c1,c2,c3,c4,c5,c_total,MOS,SD,set\n"
                   + "\n".join(f"{n},0,0,0,0,0,0,{m},{0.5},{s}" for n, m, s in rows) + "\n",
                   encoding="utf-8")
    monkeypatch.setenv("KONIQ_DIR", str(root))
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    return root


def test_config_quality_only_follows_the_benchmark_env(monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    assert C.benchmark_source() == "koniq" and C.quality_only() is True
    monkeypatch.setenv("IQA_BENCHMARK", "pseudo")
    assert C.benchmark_source() == "pseudo" and C.quality_only() is False
    monkeypatch.setenv("IQA_BENCHMARK", "bogus")
    assert C.benchmark_source() == "pseudo"  # unknown -> default


def test_koniq_export_isolates_test_labels(tmp_path, monkeypatch):
    _koniq_fixture(tmp_path, monkeypatch)
    tp = _fake_tp(tmp_path)
    man = D.export_benchmark(tp, "v1")

    assert man["quality_only"] is True and man["synthetic"] is False
    assert man["splits"] == {"dev": 6, "eval": 3}

    rel = tp.release_dir("v1")
    dev = D.load_rows(rel / "dev.jsonl")
    assert len(dev) == 6
    assert all(isinstance(r["score"], float) for r in dev)
    assert all(not any(v["present"] for v in r["defects"].values()) for r in dev)
    # eval release carries inputs only -- no score/defect ground truth
    ev = D.load_rows(rel / "eval_inputs.jsonl")
    assert len(ev) == 3 and all(set(r) == {"path", "image"} for r in ev)

    split = json.loads((rel / "dev_split.json").read_text())
    assert split["n_train"] == 4 and split["n_val"] == 2
    assert split["group"] == "official_koniq_split"

    proto = json.loads((tp.evaluation / "protocols" / "v1" / "protocol.json").read_text())
    assert proto["quality_only"] is True and proto["primary_metric"] == "plcc"
    assert proto["floors"] == {"plcc": pytest.approx(C.QUALITY_PLCC_FLOOR),
                               "srcc": pytest.approx(C.QUALITY_SRCC_FLOOR)}
    labels = D.load_rows(tp.evaluation / "protocols" / "v1" / "eval_labels.jsonl")
    assert len(labels) == 3 and all("score" in r for r in labels)


def test_quality_only_metrics_drop_the_defect_gates():
    labeled = [{"path": f"i{i}.jpg", "score": 40.0 + i * 5, "defects": {}} for i in range(10)]
    preds = [{"path": f"i{i}.jpg", "score": 40.0 + i * 5, "defects": {}} for i in range(10)]
    rep = M.evaluate_predictions(labeled, preds, floors={"plcc": 0.85, "srcc": 0.85},
                                 quality_only=True)
    assert set(rep["gates"]) == {"plcc_ok", "srcc_ok", "grade_ok"}
    assert rep["quality_only"] is True and rep["pass"] is True
    assert rep["plcc"] == pytest.approx(1.0)
    assert "no defect annotations" in rep["coverage_note"]
    # a rank-degenerate prediction fails the score gates
    flat = [{"path": f"i{i}.jpg", "score": 50.0, "defects": {}} for i in range(10)]
    bad = M.evaluate_predictions(labeled, flat, floors={"plcc": 0.85, "srcc": 0.85},
                                 quality_only=True)
    assert bad["pass"] is False and bad["gates"]["plcc_ok"] is False


def test_mos_head_is_registered_and_materialised(tmp_path):
    for name in ("mobileclip2_mos_head", "mobileclip2_mos_mlp"):
        assert METHOD_FILES[name] == "score_mos_transfer.py"
        assert _METHOD_ALIASES[name] == "score_mos_transfer.py"
    assert _VARIANT_CONFIG["mobileclip2_mos_head"]["head"] == "ridge"
    assert _VARIANT_CONFIG["mobileclip2_mos_mlp"]["head"] == "mlp"
    rel = PLUGIN.materialise_method(tmp_path, "mobileclip2_mos_head")
    assert rel == "model_pool/mobileclip2_mos_head/method.py"
    body = (tmp_path / rel).read_text()
    assert "score_mos_transfer" in body and "def run(ctx, config):" in body


def test_mos_head_l2_and_prebuilt_feature_mapping(tmp_path, monkeypatch):
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    n = MT._l2(x)
    assert np.allclose(np.linalg.norm(n[0]), 1.0)
    assert np.isfinite(n[1]).all()  # zero row guarded
    npz = tmp_path / "feat.npz"
    np.savez_compressed(npz, feat=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                        names=np.array(["a.jpg", "b.jpg"]))
    monkeypatch.setenv("IQA_FEATURE_NPZ", str(npz))
    rows = [{"path": "/x/b.jpg", "image": "b.jpg"}, {"path": "/y/a.jpg", "image": "a.jpg"}]
    got = MT._prebuilt_features(rows)
    assert got is not None and np.allclose(got[0], [3.0, 4.0]) and np.allclose(got[1], [1.0, 2.0])
    # a missing name -> no fast path (falls back to extraction)
    monkeypatch.setenv("IQA_FEATURE_NPZ", str(tmp_path / "missing.npz"))
    assert MT._prebuilt_features(rows) is None


def test_plugin_parses_the_real_mos_track(monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    spec = PLUGIN.parse_requirement(MOS_REPORT, "iqa_mos", "iqa_mos")
    assert spec.primary_metric.name == "plcc" and spec.primary_metric.threshold == pytest.approx(0.85)
    assert [(m.name, m.threshold) for m in spec.secondary_metrics] == [("srcc", pytest.approx(0.85))]
    # no defect / hardcase / grade gates on a benchmark that has no annotations
    assert not any(m.name in ("defect_macro_f1", "hardcase_macro_f1", "grade_agreement")
                   for m in spec.secondary_metrics)
    design = PLUGIN.design(spec, iteration=1)
    assert [c.method for c in design["candidates"]] == ["mobileclip2_mos_head"]
    assert "real human MOS" in design["report_markdown"]
    # the defect-oriented reality check must not appear on this track
    assert "eval positives: blur 150" not in design["report_markdown"]
    assert "类别正样本覆盖" not in design["report_markdown"]
    # escalation ladder is the MOS one
    nxt = PLUGIN.design(spec, iteration=2, history=[{"methods": ["mobileclip2_mos_head"]}])
    assert [c.method for c in nxt["candidates"]] == ["mobileclip2_mos_mlp"]


def test_mos_track_pseudo_status_and_constraints(monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    spec = PLUGIN.parse_requirement(MOS_REPORT, "iqa_mos", "iqa_mos")
    # ROM ok + no per-defect F1 -> no "dual output incomplete" failure on this track
    assert PLUGIN.check_constraints(spec, {"rom_mb": 22.9, "rom_ok": 1.0}) == []
    monkeypatch.setenv("IQA_BENCHMARK", "pseudo")
    problems = PLUGIN.check_constraints(spec, {"rom_mb": 22.9, "rom_ok": 1.0})
    assert any("dual output incomplete" in p for p in problems)
    # precision status shows only the MOS gates
    monkeypatch.setenv("IQA_BENCHMARK", "koniq")
    status = P.precision_status([{"metrics": {"plcc": 0.89, "srcc": 0.87},
                                  "best_model": "m_mobileclip2_mos_head"}],
                                iteration=2, primary_metric="plcc", primary_floor=0.85)
    assert [r["metric"] for r in status["rows"] if r["metric"] != "rom_mb"] == ["plcc", "srcc"]
    assert status["quality_only"] is True


@pytest.mark.parametrize("q_only", [True, False])
def test_mos_track_data_plan_has_annotation_and_augmentation(q_only, monkeypatch):
    monkeypatch.setenv("IQA_BENCHMARK", "koniq" if q_only else "pseudo")
    plan = P.data_plan([], {"support": {d: 0 for d in C.DEFECTS}, "n_eval": 2015})
    # the local auxiliary annotator is always declared
    assert plan["annotation"]["local_model"]["model_id"] == "qwen3.8-27b"
    assert plan["annotation"]["local_model"]["path"] == "local_largemodel/qwen3.8-27b"
    assert bool(plan["real_mos"]) is q_only
