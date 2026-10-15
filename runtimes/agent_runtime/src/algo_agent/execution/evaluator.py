"""Independent Evaluator: frozen metric code computes results from stored
predictions. The Model never computes the headline metric (guide 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..paths import TaskPaths, read_json, safe_join, write_json
from ..schemas import contracts as C


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _slice_masks(X: np.ndarray, split_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    feature = int(split_meta.get("slice_feature", 0))
    threshold = float(split_meta.get("slice_threshold", 0.0))
    low = X[:, feature] < threshold
    return {"low_light": low, "high_light": ~low, "all": np.ones(len(X), dtype=bool)}


def evaluate_run(tp: TaskPaths, run_id: str, protocol: C.EvalProtocol, model_id: str,
                 resources: Optional[Dict[str, Any]] = None, synthetic: bool = True) -> C.EvalResult:
    run_dir = safe_join(tp.runs, run_id)
    pred_path = run_dir / "pred_test.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions missing for run {run_id}: {pred_path}")
    release = tp.release_dir(protocol.dataset_version)
    label_path = release / "test.npz"
    if protocol.stage in {"development", "final"}:
        label_path = tp.evaluation / "protocols" / protocol.dataset_version / (
            "development_labels.npz" if protocol.stage == "development" else "final_labels.npz")
    test = np.load(label_path)
    X, y = test["X"], test["y"]
    y_pred = np.load(pred_path)
    if y_pred.shape != y.shape or not len(y) or not np.isfinite(y_pred).all():
        raise ValueError("predictions must be finite and match the evaluation sample shape")
    split_meta = read_json(release / "split.json", {}) or read_json(tp.data / "splits" / "split.json", {})

    metrics = {"rmse": _rmse(y, y_pred)}
    slices: Dict[str, Dict[str, float]] = {}
    for name, mask in _slice_masks(X, split_meta).items():
        if mask.sum() > 0:
            slices[name] = {"rmse": _rmse(y[mask], y_pred[mask]), "n": int(mask.sum())}
            metrics.setdefault(f"rmse_{name}", slices[name]["rmse"])

    threshold = protocol.threshold
    passed: Optional[bool] = None
    if threshold is not None:
        overall = metrics["rmse"]
        passed = overall <= threshold if protocol.direction == "min" else overall >= threshold

    eval_id = f"eval_{run_id}"
    out_dir = safe_join(tp.evaluation, "results", eval_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "slices.json", slices)
    write_json(out_dir / "resources.json", resources or {})
    result = C.EvalResult(
        eval_id=eval_id, task_id=tp.slug, model_id=model_id, protocol_id=protocol.protocol_id,
        dataset_version=protocol.dataset_version, metrics=metrics, slices=slices,
        resources=resources or {}, failures=[k for k, v in slices.items() if v.get("rmse", 0) > (threshold or 1e9)],
        passed=passed, synthetic=synthetic, stage=protocol.stage,
        notes="frozen protocol; metrics computed from stored test predictions",
    )
    write_json(out_dir / "eval_result.json", result.model_dump())
    return result


__all__ = ["evaluate_run"]
