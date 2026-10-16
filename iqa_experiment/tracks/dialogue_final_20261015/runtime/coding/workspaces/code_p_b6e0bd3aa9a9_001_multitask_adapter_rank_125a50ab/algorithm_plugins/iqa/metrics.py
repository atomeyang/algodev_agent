# -*- coding: utf-8 -*-
"""Pure-numpy IQA acceptance metrics (frozen protocol code).

This mirrors ``iqa_agent/tools/iqa_model.evaluate`` bit-for-bit on the shared
keys (per-defect F1 / defect macro-F1 / hardcase macro-F1 / PLCC / SRCC) so the
agent's evaluator agrees with the reference harness, and adds the two things the
agent loop needs:

* ``grade_agreement`` -- the 分档一致率 required by the report;
* ``class_support`` / ``defect_macro_f1_cov`` -- an explicit, honest account of
  which defect classes are actually measurable on the frozen eval split.

No scipy/sklearn dependency: pearson/spearman are implemented directly so the
evaluator stays reproducible and offline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import (
    DEFECTS,
    DEFECT_MACROF1_FLOOR,
    GRADE_AGREEMENT_FLOOR,
    GRADE_BINS,
    HARDCASE_F1_FLOOR,
    MIN_CLASS_SUPPORT,
    SCORE_PLCC_FLOOR,
    SCORE_SRCC_FLOOR,
    rom_estimate as _rom_estimate,
)

try:  # numpy is a hard dependency of the platform
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# --------------------------------------------------------------------------- #
# statistics (numpy-only pearson / spearman)
# --------------------------------------------------------------------------- #
def rankdata(values: Sequence[float]) -> List[float]:
    """Average-rank transform (ties share the mean rank), 1-based."""
    a = [float(v) for v in values]
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    xs = [float(v) for v in x[:n]]
    ys = [float(v) for v in y[:n]]
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [v - mx for v in xs]
    dy = [v - my for v in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if den <= 0:
        return 0.0
    return num / den


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _index_preds(preds: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index predictions by absolute path, with an image-name fallback."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in preds or []:
        if not isinstance(p, dict):
            continue
        for key in ("path", "image"):
            v = p.get(key)
            if v:
                out.setdefault(str(v), p)
                out.setdefault(str(v).split("/")[-1], p)
    return out


def _present(row: Dict[str, Any], defect: str) -> int:
    d = (row.get("defects") or {}).get(defect)
    if isinstance(d, dict):
        return int(bool(d.get("present", 0)))
    return int(bool(d)) if d is not None else 0


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def grade_of(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    for lo, name in GRADE_BINS:
        if float(score) >= lo:
            return name
    return GRADE_BINS[-1][1]


# --------------------------------------------------------------------------- #
# headline evaluation
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    labeled: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
    *,
    defects: Sequence[str] = tuple(DEFECTS),
    min_support: int = MIN_CLASS_SUPPORT,
    floors: Optional[Dict[str, float]] = None,
    quality_only: bool = False,
) -> Dict[str, Any]:
    """Compute the full IQA acceptance report from stored predictions.

    ``quality_only=True`` is the real-MOS track (e.g. KonIQ-10k): the benchmark
    ships scores only, no defect annotations, so the defect Macro-F1 / hardcase
    gates are not applicable and the acceptance decision rests on the
    score-quality gates (PLCC / SRCC / grade agreement).

    ``labeled`` rows need ``path`` + per-defect ``{present: 0/1}`` + optional
    ``score`` / ``hardcase``.  ``preds`` rows need ``path`` + per-defect
    ``{present: 0/1}`` + optional ``score``.
    """
    by = _index_preds(preds)
    defects = list(defects)
    tp = {d: 0 for d in defects}
    fp = {d: 0 for d in defects}
    fn = {d: 0 for d in defects}
    gt_pos = {d: 0 for d in defects}
    hc_tp = {d: 0 for d in defects}
    hc_fp = {d: 0 for d in defects}
    hc_fn = {d: 0 for d in defects}
    ys: List[float] = []
    ps: List[float] = []
    g_agree = 0
    g_total = 0
    matched = 0

    for s in labeled or []:
        key = str(s.get("path") or s.get("image") or "")
        p = by.get(key) or by.get(key.split("/")[-1])
        if not p:
            continue
        matched += 1
        sc = s.get("score")
        pc = p.get("score")
        if sc is not None and pc is not None:
            ys.append(float(sc))
            ps.append(float(pc))
            gs, gp = grade_of(float(sc)), grade_of(float(pc))
            if gs is not None and gp is not None:
                g_total += 1
                g_agree += int(gs == gp)
        for d in defects:
            gt = _present(s, d)
            pr = _present(p, d)
            gt_pos[d] += int(gt)
            if gt and pr:
                tp[d] += 1
            elif pr and not gt:
                fp[d] += 1
            elif gt and not pr:
                fn[d] += 1
            if s.get("hardcase"):
                if gt and pr:
                    hc_tp[d] += 1
                elif pr and not gt:
                    hc_fp[d] += 1
                elif gt and not pr:
                    hc_fn[d] += 1

    per_defect = {d: _f1(tp[d], fp[d], fn[d]) for d in defects}
    hardcase = {d: _f1(hc_tp[d], hc_fp[d], hc_fn[d]) for d in defects}
    precision = {d: (tp[d] / (tp[d] + fp[d]) if (tp[d] + fp[d]) else 0.0) for d in defects}
    recall = {d: (tp[d] / (tp[d] + fn[d]) if (tp[d] + fn[d]) else 0.0) for d in defects}

    macro_all = sum(per_defect.values()) / len(defects) if defects else 0.0
    covered = [d for d in defects if gt_pos[d] >= int(min_support)]
    macro_cov = (sum(per_defect[d] for d in covered) / len(covered)) if covered else 0.0
    hc_covered = [d for d in defects if gt_pos[d] >= int(min_support)]
    hardcase_macro = sum(hardcase.values()) / len(defects) if defects else 0.0
    hardcase_macro_cov = (sum(hardcase[d] for d in hc_covered) / len(hc_covered)) if hc_covered else 0.0

    if len(ys) > 2 and _std(ys) > 0 and _std(ps) > 0:
        plcc = pearson(ys, ps)
        srcc = spearman(ys, ps)
    else:
        plcc = srcc = 0.0
    grade_agreement = (g_agree / g_total) if g_total else 0.0

    f = dict(floors or {})
    plcc_floor = float(f.get("plcc", SCORE_PLCC_FLOOR))
    srcc_floor = float(f.get("srcc", SCORE_SRCC_FLOOR))
    macro_floor = float(f.get("defect_macro_f1", DEFECT_MACROF1_FLOOR))
    hc_floor = float(f.get("hardcase_macro_f1", HARDCASE_F1_FLOOR))
    grade_floor = float(f.get("grade_agreement", GRADE_AGREEMENT_FLOOR))

    gates = {
        "plcc_ok": plcc >= plcc_floor,
        "srcc_ok": srcc >= srcc_floor,
        "defect_ok": macro_all >= macro_floor,
        "defect_ok_covered": macro_cov >= macro_floor,
        "hardcase_ok": hardcase_macro_cov >= hc_floor,
        "grade_ok": grade_agreement >= grade_floor,
    }
    if quality_only:
        # no defect annotations ship with a MOS-only benchmark -> those gates
        # are inapplicable rather than failed; gate on the score metrics only.
        gates = {"plcc_ok": gates["plcc_ok"], "srcc_ok": gates["srcc_ok"],
                 "grade_ok": gates["grade_ok"]}
    report: Dict[str, Any] = {
        "per_defect_f1": {d: round(v, 4) for d, v in per_defect.items()},
        "per_defect_precision": {d: round(v, 4) for d, v in precision.items()},
        "per_defect_recall": {d: round(v, 4) for d, v in recall.items()},
        "class_support": {d: int(gt_pos[d]) for d in defects},
        "covered_defects": covered,
        "defect_macro_f1": round(macro_all, 4),
        "defect_macro_f1_cov": round(macro_cov, 4),
        "hardcase_macro_f1": round(hardcase_macro, 4),
        "hardcase_macro_f1_cov": round(hardcase_macro_cov, 4),
        "plcc": round(plcc, 4),
        "srcc": round(srcc, 4),
        "grade_agreement": round(grade_agreement, 4),
        "n_labeled": len(labeled or []),
        "n_matched": matched,
        "n_scored": len(ys),
        "gates": gates,
        "floors": {"plcc": plcc_floor, "srcc": srcc_floor,
                   "defect_macro_f1": macro_floor, "hardcase_macro_f1": hc_floor,
                   "grade_agreement": grade_floor},
        "quality_only": bool(quality_only),
        "coverage_note": ("real-MOS benchmark ships no defect annotations; defect Macro-F1 "
                          "and hardcase gates are not applicable"
                          if quality_only else _coverage_note(gt_pos, covered, defects, min_support)),
        "pass": (bool(gates["plcc_ok"] and gates["srcc_ok"]) if quality_only
                 else bool(gates["plcc_ok"] and gates["defect_ok"] and gates["hardcase_ok"])),
    }
    if len(ys) <= 2 or _std(ys) <= 0 or _std(ps) <= 0:
        report["score_note"] = "degenerate prediction/GT variance; PLCC set to 0"
    return report


def _std(values: Sequence[float]) -> float:
    if np is not None:
        return float(np.std(values)) if len(values) else 0.0
    n = len(values)
    if n == 0:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


def _coverage_note(gt_pos: Dict[str, int], covered: List[str], defects: Sequence[str],
                   min_support: int) -> str:
    missing = [d for d in defects if d not in covered]
    if not missing:
        return "all defect classes have usable support on the frozen eval split"
    return (f"classes with < {min_support} GT positives are not measurable on this frozen "
            f"split: {missing} (their F1 is scored 0 by convention, which caps the all-class "
            f"macro-F1 at {(len(covered) / len(defects)):.2f} even at perfect F1 elsewhere)")


# --------------------------------------------------------------------------- #
# flatten to EvalResult.metrics (Dict[str, float])
# --------------------------------------------------------------------------- #
def flatten_report(report: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k in ("defect_macro_f1", "defect_macro_f1_cov", "hardcase_macro_f1",
              "hardcase_macro_f1_cov", "plcc", "srcc", "grade_agreement",
              "n_matched", "n_scored", "n_labeled"):
        v = report.get(k)
        if v is not None:
            flat[k] = float(v)
    for d, v in (report.get("per_defect_f1") or {}).items():
        flat[f"f1_{d}"] = float(v)
        flat[f"support_{d}"] = float((report.get("class_support") or {}).get(d, 0))
    return flat


def slices_from_report(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    slices: Dict[str, Dict[str, float]] = {}
    for d in (report.get("per_defect_f1") or {}):
        slices[d] = {
            "f1": float(report["per_defect_f1"][d]),
            "precision": float((report.get("per_defect_precision") or {}).get(d, 0.0)),
            "recall": float((report.get("per_defect_recall") or {}).get(d, 0.0)),
            "gt_pos": float((report.get("class_support") or {}).get(d, 0)),
        }
    return slices


rom_estimate = _rom_estimate

__all__ = [
    "evaluate_predictions", "flatten_report", "slices_from_report", "grade_of",
    "pearson", "spearman", "rankdata", "rom_estimate",
]
