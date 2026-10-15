# -*- coding: utf-8 -*-
"""Score-track tests: degradation features, the three-view score head, wiring.

Torch-free: the features and the head helper functions are pure numpy/sklearn,
so the score-track contract is verifiable without a GPU.  The end-to-end head
run (sklearn fits) is exercised by the real research loop, not here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from algorithm_plugins.iqa import features as F
from algorithm_plugins.iqa.methods import METHOD_FILES
from algorithm_plugins.iqa.methods import score_moe as SM
from algorithm_plugins.iqa.plugin import (PLUGIN, _METHOD_ALIASES, _VARIANT_CONFIG,
                                          _score_primary_metric)
from algorithm_plugins.iqa import plans as P


def _write_image(path):
    import cv2
    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, size=(64, 96, 3))).astype("uint8")
    img[8:40, 8:60] = 240  # a bright block -> non-degenerate dark-channel / glare stats
    cv2.imwrite(str(path), img)
    return str(path)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_feature_set_is_occlusion_aware_and_versioned():
    names = F.feature_names()
    assert len(names) == 82
    assert F.FEATURE_VERSION == "v2-occ"
    assert "jpeg_blockiness" in names
    # occlusion/blocking block added in v2
    assert any(n.startswith("dk_") for n in names)
    assert any(n.startswith("dark_") for n in names)


def test_rich_features_are_deterministic_on_a_synthetic_image(tmp_path):
    p = _write_image(tmp_path / "a.jpg")
    a = F.rich(p)
    b = F.rich(p)
    names = F.feature_names()
    for n in names:
        assert n in a, n
        assert np.isfinite(float(a[n])), n
        assert float(a[n]) == float(b[n]), f"non-deterministic feature {n}"
    assert a["path"] == p


def test_extract_many_cache_roundtrip_and_version_invalidation(tmp_path):
    p = _write_image(tmp_path / "b.jpg")
    cache = tmp_path / "feat.json"
    X1, names1 = F.extract_many([p], workers=1, cache_path=cache)
    assert X1.shape == (1, len(F.feature_names()))
    blob = json.loads(cache.read_text())
    assert blob["version"] == F.FEATURE_VERSION and p in blob["rows"]
    # a stale version must force re-extraction, not silent reuse
    blob["version"] = "v0-stale"
    cache.write_text(json.dumps(blob))
    X2, names2 = F.extract_many([p], workers=1, cache_path=cache)
    assert names2 == names1
    assert json.loads(cache.read_text())["version"] == F.FEATURE_VERSION
    assert np.allclose(X1, X2)


# --------------------------------------------------------------------------- #
# three-view score head
# --------------------------------------------------------------------------- #
def test_score_moe_is_registered_everywhere():
    assert METHOD_FILES["score_moe"] == "score_moe.py"
    assert METHOD_FILES["score_moe_meta"] == "score_moe.py"
    assert _METHOD_ALIASES["score_moe"] == "score_moe.py"
    assert _METHOD_ALIASES["score_moe_meta"] == "score_moe.py"
    # core head is pixel-only; the escalated variant adds the metadata scalars
    assert _VARIANT_CONFIG["score_moe"]["use_metadata"] is False
    assert _VARIANT_CONFIG["score_moe_meta"]["use_metadata"] is True


def test_score_moe_materialises_a_runnable_entry(tmp_path):
    rel = PLUGIN.materialise_method(tmp_path, "score_moe")
    assert rel == "model_pool/score_moe/method.py"
    body = (tmp_path / rel).read_text()
    assert "score_moe" in body and "def run(ctx, config):" in body


def test_score_moe_level_and_family_semantics():
    assert SM._level_of({"good": True}) == 0
    assert SM._level_of({"good": False, "severity": 1}) == 1
    assert SM._level_of({"good": False, "severity": 0}) == 3  # unlabelled 废片 -> heavy bucket
    assert SM._family_of({"good": True}) == "good"
    assert SM._family_of({"good": False, "raw_defect": "blur"}) == "blur"
    assert SM._family_of({"good": False, "raw_defect": None}) == "none"
    assert SM._family_of({"good": False, "raw_defect": "occlusion"}) == "none"


def test_score_moe_simplex_weights_sum_to_one():
    w = SM._simplex(0.25)
    assert len(w) == 15  # C(6,2)
    for trip in w:
        assert abs(sum(trip) - 1.0) < 1e-9
        assert all(0.0 <= x <= 1.0 for x in trip)


def test_score_moe_plcc_is_pearson_and_guards_constant_vectors():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert SM._plcc(x, x) == pytest.approx(1.0)
    assert SM._plcc(x, np.zeros(4)) == 0.0     # constant -> guarded
    assert SM._plcc([1.0], [1.0]) == 0.0       # too short -> guarded


def _mini_release(tmp_path):
    """A 6-row frozen-release stand-in (4 dev + 2 eval) to run the head end to end."""
    import types

    d = tmp_path / "release"
    d.mkdir()
    _write_image(d / "g1.jpg"); _write_image(d / "g2.jpg")
    _write_image(d / "b1.jpg"); _write_image(d / "b2.jpg")
    _write_image(d / "e1.jpg"); _write_image(d / "e2.jpg")
    dev = [
        {"path": str(d / "g1.jpg"), "image": "g1.jpg", "good": True, "score": 90.0,
         "raw_defect": None, "severity": 0, "subject": "s1"},
        {"path": str(d / "g2.jpg"), "image": "g2.jpg", "good": True, "score": 92.0,
         "raw_defect": None, "severity": 0, "subject": "s2"},
        {"path": str(d / "b1.jpg"), "image": "b1.jpg", "good": False, "score": 28.0,
         "raw_defect": "blur", "severity": 3, "subject": "s3"},
        {"path": str(d / "b2.jpg"), "image": "b2.jpg", "good": False, "score": 66.0,
         "raw_defect": "over_exp", "severity": 1, "subject": "s4"},
    ]
    (d / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in dev), encoding="utf-8")
    ev = [{"path": str(d / "e1.jpg"), "image": "e1.jpg"},
          {"path": str(d / "e2.jpg"), "image": "e2.jpg"}]
    (d / "eval_inputs.jsonl").write_text("\n".join(json.dumps(r) for r in ev), encoding="utf-8")
    ctx = types.SimpleNamespace(data_dir=d, out_dir=tmp_path / "out",
                                task_root=tmp_path / "cache", split="test", seed=0)
    return ctx


def test_score_moe_runs_end_to_end_on_a_mini_release(tmp_path):
    ctx = _mini_release(tmp_path)
    res = SM.run(ctx, {"cv_folds": 2, "ensemble_seeds": 1, "feature_workers": 1,
                       "min_family_rows": 1})
    preds = json.loads((ctx.out_dir / "pred_test.json").read_text())
    assert res["n_eval"] == 2 and len(preds) == 2
    assert all(0.0 <= p["score"] <= 100.0 for p in preds)
    assert set(res["blend_weights"]) == {"global_reg", "cls_expect", "moe"}
    assert abs(sum(res["blend_weights"].values()) - 1.0) < 1e-6
    assert res["synthetic"] is False and res["rom_params_millions"] == 0.0
    # the head must not read the eval `score` field: the loader strips it,
    # so no eval-derived quality hint leaks into the returned payload.
    assert "eval_diagnostic" not in res


# --------------------------------------------------------------------------- #
# metric-driven reporting
# --------------------------------------------------------------------------- #
def test_score_primary_metric_parses_the_plcc_gate():
    assert _score_primary_metric("主指标 PLCC >= 0.85，缺陷 Macro-F1 >= 0.85") == ("plcc", 0.85)
    assert _score_primary_metric("primary metric SRCC 0.90") == ("srcc", 0.90)
    assert _score_primary_metric("缺陷 Macro-F1 >= 0.85，PLCC/SRCC >= 0.90") is None


def test_precision_status_primary_plcc_leads_the_table():
    status = P.precision_status(history=[], iteration=1, primary_metric="plcc")
    assert status["rows"][0]["metric"] == "plcc"
    assert status["rows"][0]["role"] == "primary gate"
    assert status["rows"][0]["primary"] is True
    assert any("primary metric is PLCC" in n for n in status["notes"])


def test_precision_status_primary_floor_overrides_the_default():
    """A PLCC-primary run gates at 0.85, not the frozen-protocol default 0.90."""
    hist = [{"metrics": {"plcc": 0.67, "srcc": 0.68}, "best_model": "m_score_moe"}]
    status = P.precision_status(history=hist, iteration=2,
                                primary_metric="plcc", primary_floor=0.85)
    row = status["rows"][0]
    assert row["metric"] == "plcc" and row["floor"] == pytest.approx(0.85)
    assert row["value"] == pytest.approx(0.67) and row["verdict"] == "fail"
    # without an explicit floor it keeps the protocol default
    default = P.precision_status(history=hist, iteration=2, primary_metric="plcc")
    assert default["rows"][0]["floor"] == pytest.approx(0.90)
