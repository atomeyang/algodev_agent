# -*- coding: utf-8 -*-
"""Hierarchical mixture-of-experts score head (the honest PLCC frontier).

Why this method exists
----------------------
The frozen benchmark's score ground truth is a *severity-quantised* pseudo-MOS
(``iqa_agent/benchmark.py``): defective images score 66 / 46 / 28 for
light/medium/heavy and good images score 88.0-95.9 from a **filename hash**.
So predicting the score is really a **4-level ordinal problem** (good, light,
medium, heavy) whose top level carries unlearnable, pixel-independent noise, and
PLCC is a *ranking* statistic over those levels.

Two facts, both measured on the frozen split (see ``docs/iqa_run_report.md``),
drive the design:

1. a single global regressor tops out around PLCC 0.66;
2. the binding constraint is the **good-vs-defective gate**: perfect routing
   (``oracle`` gate, diagnostic only) lifts PLCC to ~0.85, while the best real
   gate has ROC-AUC ~0.85 / accuracy ~0.83-0.89.

So instead of betting on one model, this head fits three *complementary* views
and picks their blend by **subject-group cross-validation on dev only**:

* ``global_reg`` -- ExtraTrees + RandomForest regression ensemble on the score;
* ``cls_expect`` -- a 4-class classifier whose posteriors give
  ``E[score] = sum_k P(level=k) * level_score`` (the PLCC-optimal decoding of a
  probabilistic 4-level model);
* ``moe`` -- the explicit factorisation: a gate ``P(defective|x)``, a coarse
  degradation-family router (``blur/over_exp/under_exp/flare/none``) trained on
  dev ``raw_defect`` labels only, and one severity expert per family.

Features are the 82-d degradation-aware stats (``features.rich``) plus the 16-d
calibrated physical measurements plus two cheap metadata scalars (file size),
so the deployed artefact stays zero-ROM, CPU-only and fully explainable.  The
defect side reuses the zero-ROM calibrated measurement rule, so the method is a
drop-in replacement for ``score_ordinal_sft`` on the dual-output contract.

**No eval label is ever read**: the frozen ``eval`` split is used for prediction
only.  Family/severity supervision comes exclusively from ``dev.jsonl`` and, for
``score_ordinal_sft_synth``, from on-the-fly re-degradation (never here).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import common, measure_lib

DEFAULTS: Dict[str, Any] = {
    "feature_workers": 10,
    "cv_folds": 4,
    "ensemble_seeds": 3,
    "min_family_rows": 15,
    "blend_step": 0.1,
    "use_metadata": True,
}

MEAS_KEYS = ["gray_world_dev", "lab_a_mean", "lab_b_mean", "cast_mag", "bright_ratio",
             "largest_blob_ratio", "hf_ratio", "laplacian_var", "tenengrad", "mean_lum",
             "over_ratio", "p1", "p99", "std_lum", "under_ratio", "glare_ratio"]

# pseudo-MOS level means, estimated from dev at fit time (fallback only)
LEVEL_SCORES_FALLBACK = {0: 90.0, 1: 66.0, 2: 46.0, 3: 28.0}


# --------------------------------------------------------------------------- #
# targets / families (labels only -- never filename tokens as features)
# --------------------------------------------------------------------------- #
def _level_of(row: Dict[str, Any]) -> int:
    if row.get("good"):
        return 0
    sev = int(row.get("severity") or 0)
    return 3 if sev == 0 else sev


def _family_of(row: Dict[str, Any]) -> str:
    """Coarse degradation family from the *label* field (dev only)."""
    if row.get("good"):
        return "good"
    defect = row.get("raw_defect")
    if defect in ("blur", "over_exp", "under_exp", "flare"):
        return defect
    return "none"          # occlusion / noise / haze all carry raw_defect=None


def _level_means(level: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    out = dict(LEVEL_SCORES_FALLBACK)
    for lv in (0, 1, 2, 3):
        sel = level == lv
        if sel.sum() > 0:
            out[lv] = float(np.mean(y[sel]))
    return out


# --------------------------------------------------------------------------- #
# features: 82-d degradation-aware stats + 16-d measurements + metadata
# --------------------------------------------------------------------------- #
def _rich_matrix(paths, cache: Path, workers: int):
    from .. import features as F
    key = hashlib.sha1(("rich82|" + "|".join(paths)).encode()).hexdigest()[:16]
    return F.extract_many(paths, workers=workers, cache_path=cache / f"rich_{key}.json")


def _measure_matrix(paths, cache: Path):
    cache_file = cache / "measure_feat.json"
    blob: Dict[str, Any] = {}
    if cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
    dirty = False
    rows = []
    for p in paths:
        rec = blob.get(p)
        if rec is None:
            try:
                m = measure_lib.measure(p)["global"]
            except Exception:
                m = {}
            rec = [float(m.get(k, 0.0) or 0.0) for k in MEAS_KEYS]
            blob[p] = rec
            dirty = True
        rows.append(rec)
    if dirty:
        cache_file.write_text(json.dumps(blob), encoding="utf-8")
    return np.asarray(rows, dtype=np.float64)


def _meta_matrix(paths) -> np.ndarray:
    rows = []
    for p in paths:
        try:
            sz = float(os.path.getsize(p))
        except Exception:
            sz = 0.0
        rows.append([sz, float(np.log1p(sz))])
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------- #
# model pieces
# --------------------------------------------------------------------------- #
def _global_reg(X, y, seeds: int, Xte):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    preds = []
    for s in range(seeds):
        preds.append(ExtraTreesRegressor(
            n_estimators=700, min_samples_leaf=1, max_features=0.5,
            n_jobs=-1, random_state=s).fit(X, y).predict(Xte))
    preds.append(RandomForestRegressor(
        n_estimators=900, min_samples_leaf=1, max_features=0.4,
        n_jobs=-1, random_state=0).fit(X, y).predict(Xte))
    return np.mean(preds, axis=0)


def _cls_expect_from_scores(X, level, y, Xte):
    from sklearn.ensemble import RandomForestClassifier
    means = _level_means(np.asarray(level), np.asarray(y))
    clf = RandomForestClassifier(
        n_estimators=1200, min_samples_leaf=1, max_features=0.3,
        n_jobs=-1, random_state=0).fit(X, np.asarray(level))
    cls = list(clf.classes_)
    P = clf.predict_proba(Xte)
    return P @ np.array([means.get(int(c), LEVEL_SCORES_FALLBACK.get(int(c), 50.0)) for c in cls])


def _fit_moe(X, y, level, family, Xte, min_rows: int, seed: int = 0):
    """Fit gate + family router + per-family experts; return (hard, soft, info)."""
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                  HistGradientBoostingClassifier)

    info: Dict[str, Any] = {}
    defective = (np.asarray(level) != 0).astype(int)
    gate = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=6,
        random_state=seed).fit(X, defective)

    fam = np.asarray(family)
    sel = fam != "good"
    classes = sorted(set(fam[sel].tolist()))
    router = ExtraTreesClassifier(
        n_estimators=800, min_samples_leaf=1, max_features=0.6,
        n_jobs=-1, random_state=seed).fit(X[sel], fam[sel])

    experts: Dict[str, Any] = {}
    for f in sorted(set(fam.tolist())):
        tr = fam == f
        if tr.sum() < min_rows:
            continue
        experts[f] = ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=2, max_features=0.5,
            n_jobs=-1, random_state=seed).fit(X[tr], y[tr])

    good_expert = experts.get("good")
    base = None
    if good_expert is None:
        gm = y[np.asarray(level) == 0]
        base = float(np.mean(gm)) if gm.size else 90.0

    gp = gate.predict(Xte)
    rp = router.predict(Xte)
    hard_fam = np.where(gp == 0, "good", rp.astype(object))
    hard = np.zeros(len(Xte))
    for f, m in experts.items():
        s = hard_fam == f
        if s.sum():
            hard[s] = m.predict(Xte[s])
    if good_expert is None:
        hard[gp == 0] = base

    # soft route (posterior weighted) -- reported but not blended into `hard`
    gp_p = gate.predict_proba(Xte)[:, 1]
    rp_p = router.predict_proba(Xte)
    rcls = list(router.classes_)
    soft = np.zeros(len(Xte))
    for f, m in experts.items():
        if f == "good":
            continue
        soft += gp_p * rp_p[:, rcls.index(f)] * m.predict(Xte)
    soft += (1.0 - gp_p) * (good_expert.predict(Xte) if good_expert is not None else base)

    info.update({
        "n_families": len(experts),
        "families": {f: int((fam == f).sum()) for f in sorted(set(fam.tolist()))},
        "router_classes": classes,
    })
    return hard, soft, info


def _plcc(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _simplex(step: float):
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((i / n, j / n, k / n))
    return out


# --------------------------------------------------------------------------- #
def run(ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**DEFAULTS, **(config or {})}
    C = common.plugin_config()
    dev = common.load_dev(ctx.data_dir)
    inputs = common.load_eval_inputs(ctx.data_dir)
    cache = Path(common.cache_dir(ctx))
    t0 = time.time()
    workers = int(cfg.get("feature_workers", 10))

    dev_rows = [r for r in dev if r.get("score") is not None]
    eval_paths = [r["path"] for r in inputs]

    all_paths = [r["path"] for r in dev_rows] + eval_paths
    X_rich, _names = _rich_matrix(all_paths, cache, workers)
    X_meas = _measure_matrix(all_paths, cache)
    parts = [X_rich, X_meas]
    if cfg.get("use_metadata", True):
        parts.append(_meta_matrix(all_paths))
    X_all = np.hstack(parts).astype(np.float64)
    nd = len(dev_rows)
    Xd, Xe = X_all[:nd], X_all[nd:]
    yd = np.array([float(r["score"]) for r in dev_rows])
    ld = np.array([_level_of(r) for r in dev_rows])
    fdl = [_family_of(r) for r in dev_rows]
    gd = [str(r.get("subject") or r.get("image") or i) for i, r in enumerate(dev_rows)]
    min_rows = int(cfg.get("min_family_rows", 15))
    seeds = max(1, int(cfg.get("ensemble_seeds", 3)))

    # ---- subject-group CV on dev: choose the blend of the three views ----
    from sklearn.model_selection import GroupKFold
    n_splits = min(int(cfg.get("cv_folds", 4)), max(2, len(set(gd))))
    folds = list(GroupKFold(n_splits=n_splits).split(Xd, ld, gd))
    oof_reg = np.zeros(nd); oof_cls = np.zeros(nd); oof_moe = np.zeros(nd)
    for tr, te in folds:
        Xtr, Xte = Xd[tr], Xd[te]
        oof_reg[te] = _global_reg(Xtr, yd[tr], max(1, seeds - 1), Xte)
        oof_cls[te] = _cls_expect_from_scores(Xtr, ld[tr], yd[tr], Xte)
        oof_moe[te] = _fit_moe(Xtr, yd[tr], ld[tr], [fdl[i] for i in tr],
                               Xte, min_rows, seed=0)[0]
    mask = np.concatenate([te for _, te in folds])
    comp_cv = {"global_reg": _plcc(oof_reg[mask], yd[mask]),
               "cls_expect": _plcc(oof_cls[mask], yd[mask]),
               "moe": _plcc(oof_moe[mask], yd[mask])}
    best = (-1.0, (0.0, 1.0, 0.0), 0.0)
    for w in _simplex(float(cfg.get("blend_step", 0.1))):
        p = w[0] * oof_reg + w[1] * oof_cls + w[2] * oof_moe
        c = _plcc(p[mask], yd[mask])
        if c > best[0]:
            best = (c, w, 0.0)
    cv_best, w_reg, w_cls, w_moe = best[0], best[1][0], best[1][1], best[1][2]

    # ---- refit on all dev, predict the frozen eval split (no eval labels) ----
    reg_pred = _global_reg(Xd, yd, seeds, Xe)
    cls_pred = _cls_expect_from_scores(Xd, ld, yd, Xe)
    hard, soft, moe_info = _fit_moe(Xd, yd, ld, fdl, Xe, min_rows, seed=0)
    preds = w_reg * reg_pred + w_cls * cls_pred + w_moe * hard
    preds = np.clip(preds, 0.0, 100.0)

    # ---- defect side: zero-ROM calibrated measurement rule (dual output) ----
    thr = C.measure_thresholds()
    defect_map = measure_lib.measure_defects_cached(eval_paths, thr, C.DEFECTS, cache_dir=cache)
    rows = [common.make_pred(r["path"], float(preds[i]), defect_map[r["path"]])
            for i, r in enumerate(inputs)]
    common.write_preds(ctx, rows)

    return {
        "method": "score_moe", "split": getattr(ctx, "split", "test"),
        "seed": int(getattr(ctx, "seed", 0) or 0), "n_eval": len(rows),
        "synthetic": False, "device": "cpu",
        "rom_params_millions": 0.0, "rom": C.rom_estimate(0.0),
        "feature_dim": int(X_all.shape[1]), "n_train_real": nd, "n_train_synth": 0,
        "blend_weights": {"global_reg": round(w_reg, 3), "cls_expect": round(w_cls, 3),
                          "moe": round(w_moe, 3)},
        "cv": {"per_component": {k: round(v, 4) for k, v in comp_cv.items()},
               "blend": round(cv_best, 4), "n_folds": n_splits},
        "moe": moe_info,
        "wall_clock_s": round(time.time() - t0, 1),
        "metrics": {"n_eval": len(rows)},
        "note": ("three-view score head (global ET/RF regression; 4-class soft "
                 "E[score]; coarse-family MoE) blended with weights chosen by "
                 "subject-group CV on dev only.  Family labels come from the dev "
                 "`raw_defect` field, never from filename tokens; the frozen eval "
                 "split is never read for labels."),
    }


if __name__ == "__main__":
    raise SystemExit("run through algo_agent.execution.method_worker")
