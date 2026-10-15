"""CPU-only synthetic tabular regression methods.

The data is deterministic (fixed seed) and deliberately contains a
low-light slice where the target depends on an interaction term. A plain linear
model therefore under-fits that slice, which lets the demo exercise a real
feedback -> re-plan -> re-train -> re-evaluate loop. Results are marked
``synthetic_demo`` and must not be read as image/audio/real-business validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

METHODS = ["baseline_mean", "ridge", "gbr", "ridge_interaction"]
SLICE_FEATURE = 0
SLICE_THRESHOLD = 0.0
INTERACTION_FEATURE = 2


def slice_masks(X: np.ndarray) -> Dict[str, np.ndarray]:
    low = X[:, SLICE_FEATURE] < SLICE_THRESHOLD
    return {"low_light": low, "high_light": ~low, "all": np.ones(len(X), dtype=bool)}


def build_dataset(out_dir: Path | str, seed: int = 7, n: int = 600) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    w = np.array([1.5, -2.0, 0.8, 0.2, 1.2, -0.5])
    y = X @ w + 0.3 * rng.normal(size=n)
    low = X[:, SLICE_FEATURE] < SLICE_THRESHOLD
    y[low] += 2.0 * X[low, INTERACTION_FEATURE]
    perm = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    idx = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}
    splits = {}
    for name, ii in idx.items():
        path = out_dir / f"{name}.npz"
        np.savez(path, X=X[ii], y=y[ii])
        splits[name] = len(ii)
    (out_dir / "split.json").write_text(json.dumps({"seed": seed, "counts": splits,
                                                    "slice_feature": SLICE_FEATURE,
                                                    "slice_threshold": SLICE_THRESHOLD}, indent=2), encoding="utf-8")
    return {"seed": seed, "counts": splits, "features": X.shape[1],
            "slice": "low_light", "synthetic": True}


def _load(data_dir: Path, split: str):
    d = np.load(Path(data_dir) / f"{split}.npz")
    return d["X"], d["y"]


def _save_preds(out_dir: Path, split: str, y_pred: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"pred_{split}.npy", y_pred)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _baseline_mean(Xtr, ytr, Xev):
    return np.full(len(Xev), float(np.mean(ytr)))


def _ridge(Xtr, ytr, Xev, alpha=1.0):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    return m.predict(Xev), m


def _gbr(Xtr, ytr, Xev, seed=0, n_estimators=120, max_depth=2):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed).fit(Xtr, ytr)
    return m.predict(Xev), m


def _add_interaction(X):
    return np.column_stack([X, X[:, INTERACTION_FEATURE] * (X[:, SLICE_FEATURE] < SLICE_THRESHOLD).astype(float)])


def _ridge_interaction(Xtr, ytr, Xev, alpha=1.0):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(_add_interaction(Xtr), ytr)
    return m.predict(_add_interaction(Xev)), m


def _select_alpha(Xtr, ytr, alphas, *, interaction: bool = False, seed: int = 0, folds: int = 3) -> float:
    """Deterministic k-fold selection over the approved alpha grid.

    The plan's ``change_scope.parameter_space`` grants a small search grid; the
    method itself performs the search on the *train* split only and never looks
    at the frozen evaluation split (guide 8).
    """
    from sklearn.linear_model import Ridge

    grid = [float(a) for a in alphas]
    if len(grid) == 1:
        return grid[0]
    X = _add_interaction(Xtr) if interaction else Xtr
    idx = np.arange(len(ytr))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    parts = np.array_split(idx, max(2, int(folds)))
    best_alpha, best_score = grid[0], None
    for alpha in grid:
        scores = []
        for i in range(len(parts)):
            val = parts[i]
            tr = np.concatenate([parts[j] for j in range(len(parts)) if j != i])
            if len(tr) == 0 or len(val) == 0:
                continue
            model = Ridge(alpha=alpha).fit(X[tr], ytr[tr])
            scores.append(_rmse(ytr[val], model.predict(X[val])))
        if not scores:
            continue
        score = float(np.mean(scores))
        if best_score is None or score < best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def _resolve_alpha(config: Dict[str, Any], Xtr, ytr, *, interaction: bool, seed: int) -> float:
    spec = config.get("alpha", 1.0)
    if isinstance(spec, (list, tuple)):
        return _select_alpha(Xtr, ytr, spec, interaction=interaction, seed=seed)
    return float(spec)


def run(method: str, ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one method for one split; returns metrics + artifact refs."""
    data_dir = Path(getattr(ctx, "data_dir"))
    out_dir = Path(getattr(ctx, "out_dir"))
    split = getattr(ctx, "split", "val")
    seed = int(getattr(ctx, "seed", 0) or 0)
    Xtr, ytr = _load(data_dir, "train")
    Xev, yev = _load(data_dir, split)
    weights = None

    if method == "baseline_mean":
        y_pred = _baseline_mean(Xtr, ytr, Xev)
        weights = float(np.mean(ytr))
    elif method == "ridge":
        alpha = _resolve_alpha(config, Xtr, ytr, interaction=False, seed=seed)
        y_pred, weights = _ridge(Xtr, ytr, Xev, alpha=alpha)
    elif method == "gbr":
        y_pred, weights = _gbr(Xtr, ytr, Xev, seed=seed,
                        n_estimators=int(config.get("n_estimators", 120)),
                        max_depth=int(config.get("max_depth", 2)))
    elif method == "ridge_interaction":
        alpha = _resolve_alpha(config, Xtr, ytr, interaction=True, seed=seed)
        y_pred, weights = _ridge_interaction(Xtr, ytr, Xev, alpha=alpha)
    else:
        raise ValueError(f"unknown method {method}")

    _save_preds(out_dir, split, y_pred)
    import joblib
    joblib.dump({"method": method, "model": weights}, out_dir / "model.joblib")
    masks = slice_masks(Xev)
    metrics = {"rmse": _rmse(yev, y_pred)}
    for name, mask in masks.items():
        if mask.sum() > 0:
            metrics[f"rmse_{name}"] = _rmse(yev[mask], y_pred[mask])
    result = {
        "method": method, "split": split, "seed": seed, "config": config,
        "metrics": metrics, "n_eval": int(len(yev)), "synthetic": True, "weights_ref": "model.joblib",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def predict(ctx, config: Dict[str, Any], weights_path) -> Dict[str, Any]:
    import joblib
    artifact = joblib.load(weights_path)
    with np.load(Path(ctx.data_dir) / "test.npz") as data:
        X = data["X"]
    method, model = artifact["method"], artifact["model"]
    if method == "baseline_mean":
        prediction = np.full(len(X), model)
    else:
        prediction = model.predict(_add_interaction(X) if method == "ridge_interaction" else X)
    _save_preds(Path(ctx.out_dir), "test", prediction)
    return {"method": method, "status": "succeeded", "n_eval": len(prediction), "inference_only": True}
