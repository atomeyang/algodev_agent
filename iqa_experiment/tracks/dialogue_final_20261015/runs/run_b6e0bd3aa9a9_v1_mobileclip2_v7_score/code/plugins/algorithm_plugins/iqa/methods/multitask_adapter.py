# -*- coding: utf-8 -*-
"""MobileCLIP2-S0 + multi-task adapter heads (the main plan-A1 candidate).

Frozen MobileCLIP2-S0 visual encoder (11.41 M params, ~22.8 MB fp16) with two
lightweight trainable heads on the shared 512-d embedding:

* a score head  -> continuous 0-100 quality score (report plan A1);
* a defect head -> 5 independent logits for blur / over_exp / under_exp /
  flare / white_balance (multi-label, class-weighted BCE).

Training uses only the *dev* split (subject-group-disjoint train/val); the
frozen eval split is touched once, for prediction, and the headline metrics are
computed by the independent evaluator, never here.

Everything runs on GPU when available; encoder features are cached so the
self-optimisation loop can re-train heads in seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from . import clip_backbone, common

DEFAULTS: Dict[str, Any] = {
    "epochs": 60,
    "lr": 2e-3,
    "weight_decay": 1e-2,
    "hidden": 256,
    "dropout": 0.15,
    "batch_size": 64,
    "bce_weight": 1.0,
    "l1_weight": 0.5,
    "pos_weight_scale": 1.0,
    "min_pos_weight": 1.0,
    "max_pos_weight": 20.0,
    "defect_threshold": 0.5,
    "tune_thresholds": True,
    "image_size": 256,
    "feature_batch": 32,
}


def _pos_weight(d: np.ndarray, scale: float, lo: float, hi: float) -> np.ndarray:
    pos = d.sum(axis=0).astype(float)
    neg = d.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    w = np.clip(w * float(scale), lo, hi)
    return w.astype(np.float32)


def _macro_f1(y: np.ndarray, p: np.ndarray, classes: List[int]) -> float:
    f1s = []
    for c in classes:
        yt, yp = y[:, c], p[:, c]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _best_thresholds(y: np.ndarray, prob: np.ndarray, classes: List[int]) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    grid = np.arange(0.05, 0.96, 0.025)
    for c in classes:
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            p = (prob[:, c] >= t).astype(int)
            yt = y[:, c]
            tp = int(((yt == 1) & (p == 1)).sum())
            fp = int(((yt == 0) & (p == 1)).sum())
            fn = int(((yt == 1) & (p == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def _plcc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) == 0 or float(np.std(b)) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    cfg = {**DEFAULTS, **(config or {})}
    C = common.plugin_config()
    dev = common.load_dev(ctx.data_dir)
    inputs = common.load_eval_inputs(ctx.data_dir)
    split = common.load_dev_split(ctx.data_dir)
    device = common.resolve_device() or torch.device("cpu")
    cache = Path(common.cache_dir(ctx))
    t_start = time.time()

    pretrained = C.pretrained_ckpt()
    if not Path(pretrained).exists():
        raise FileNotFoundError(f"MobileCLIP2-S0 pretrained weights missing: {pretrained}")

    dev_paths = [r["path"] for r in dev]
    eval_paths = [r["path"] for r in inputs]

    encoder, preprocess = clip_backbone.build_encoder(pretrained, int(cfg["image_size"]))
    encoder = encoder.to(device)
    X_dev = clip_backbone.extract_features(encoder, preprocess, dev_paths, device,
                                           cache_dir=cache, tag="mobileclip2s0_256",
                                           batch_size=int(cfg["feature_batch"]))
    X_eval = clip_backbone.extract_features(encoder, preprocess, eval_paths, device,
                                            cache_dir=cache, tag="mobileclip2s0_256",
                                            batch_size=int(cfg["feature_batch"]))

    n = len(dev)
    tr_idx = np.asarray(split.get("train") or list(range(n)), dtype=int)
    va_idx = np.asarray(split.get("val") or list(range(n)), dtype=int)
    dim = X_dev.shape[1]

    y_score = np.asarray([float(r.get("score", 0.0)) for r in dev], dtype=np.float32)
    y_def = np.asarray([common.defect_vector(r, C.DEFECTS) for r in dev], dtype=np.float32)
    y_scaled = (y_score - 50.0) / 25.0
    classes = [i for i, d in enumerate(C.DEFECTS) if int(y_def[:, i].sum()) > 0]

    mu, sd = X_dev[tr_idx].mean(axis=0, keepdims=True), X_dev[tr_idx].std(axis=0, keepdims=True) + 1e-6
    Xt = torch.tensor((X_dev - mu) / sd, dtype=torch.float32, device=device)
    Xe = torch.tensor((X_eval - mu) / sd, dtype=torch.float32, device=device)
    Yt = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    Dt = torch.tensor(y_def, dtype=torch.float32, device=device)
    tr = torch.tensor(tr_idx, dtype=torch.long, device=device)
    va = torch.tensor(va_idx, dtype=torch.long, device=device)

    class MultiTaskHead(nn.Module):
        def __init__(self, in_dim: int, hidden: int, dropout: float, n_def: int):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
            self.score = nn.Linear(hidden, 1)
            self.defect = nn.Linear(hidden, n_def)

        def forward(self, x):
            h = self.trunk(x)
            return self.score(h).squeeze(-1), self.defect(h)

    torch.manual_seed(int(getattr(ctx, "seed", 0) or 0))
    head = MultiTaskHead(dim, int(cfg["hidden"]), float(cfg["dropout"]), len(C.DEFECTS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    pw = torch.tensor(_pos_weight(y_def[tr_idx], float(cfg["pos_weight_scale"]),
                                  float(cfg["min_pos_weight"]), float(cfg["max_pos_weight"])),
                      dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    n_tr = len(tr_idx)
    bs = max(8, min(int(cfg["batch_size"]), n_tr))
    history: List[Dict[str, Any]] = []
    best = {"sel": -1.0, "state": None, "epoch": 0, "thr": {}, "ab": (1.0, 0.0)}

    for epoch in range(1, int(cfg["epochs"]) + 1):
        head.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            idx = tr[perm[i:i + bs]]
            s, dlog = head(Xt[idx])
            loss = (float(cfg["l1_weight"]) * nn.functional.smooth_l1_loss(s, Yt[idx])
                    + float(cfg["bce_weight"]) * bce(dlog, Dt[idx]))
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            sv, dv = head(Xt[va])
        sv_np, dv_np = sv.cpu().numpy(), torch.sigmoid(dv).cpu().numpy()
        yv, dval = y_score[va_idx], y_def[va_idx]
        thr = _best_thresholds(dval, dv_np, classes) if cfg.get("tune_thresholds") else {c: float(cfg["defect_threshold"]) for c in classes}
        pred = np.zeros_like(dval)
        for c in classes:
            pred[:, c] = (dv_np[:, c] >= thr.get(c, 0.5)).astype(int)
        val_macro = _macro_f1(dval, pred, classes)
        val_plcc = _plcc(yv, sv_np)
        a, b = np.polyfit(sv_np, yv, 1) if len(sv_np) > 1 else (1.0, 0.0)
        sel = val_macro + max(0.0, val_plcc)
        history.append({"epoch": epoch, "loss": float(loss.item()), "val_macro_f1": round(val_macro, 4),
                        "val_plcc": round(val_plcc, 4)})
        if sel > best["sel"]:
            best = {"sel": sel, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "thr": thr, "ab": (float(a), float(b)),
                    "val_macro_f1": round(val_macro, 4), "val_plcc": round(val_plcc, 4)}

    head.load_state_dict(best["state"])
    head.eval()
    with torch.no_grad():
        se, de = head(Xe)
    se_np, de_np = se.cpu().numpy(), torch.sigmoid(de).cpu().numpy()
    a, b = best["ab"]
    score_pred = np.clip(a * se_np + b, 0.0, 100.0)

    rows = []
    for i, r in enumerate(inputs):
        defects = {}
        for ci, d in enumerate(C.DEFECTS):
            if ci in classes:
                present = int(de_np[i, ci] >= float(best["thr"].get(ci, 0.5)))
            else:
                present = 0  # no training signal for this class on dev
            defects[d] = {"present": present, "severity": present,
                          "prob": round(float(de_np[i, ci]), 4)}
        defects["white_balance"]["direction"] = "neutral"
        rows.append(common.make_pred(r["path"], float(score_pred[i]), defects))
    pred_path = common.write_preds(ctx, rows)

    head_params = sum(p.numel() for p in head.parameters())
    # deployment drops the text tower -> count the visual encoder only
    encoder_params = sum(p.numel() for p in encoder.visual.parameters())
    rom = C.rom_estimate((encoder_params + head_params) / 1e6)
    result = {
        "method": "multitask_adapter", "split": getattr(ctx, "split", "test"),
        "seed": int(getattr(ctx, "seed", 0) or 0), "n_eval": len(rows),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "synthetic": False, "device": str(device),
        "rom_params_millions": round((encoder_params + head_params) / 1e6, 3),
        "rom": rom,
        "best_epoch": best["epoch"],
        "internal_val": {"macro_f1": best.get("val_macro_f1"), "plcc": best.get("val_plcc")},
        "defect_thresholds": {C.DEFECTS[c]: round(float(t), 3) for c, t in best["thr"].items()},
        "trained_defect_classes": [C.DEFECTS[c] for c in classes],
        "score_calibration": {"a": round(a, 4), "b": round(b, 4)},
        "wall_clock_s": round(time.time() - t_start, 1),
        "metrics": {"val_macro_f1": best.get("val_macro_f1", 0.0), "val_plcc": best.get("val_plcc", 0.0)},
        "config": cfg, "pred_path": pred_path,
        "note": "frozen MobileCLIP2-S0 encoder + trained score/defect heads; "
                "headline metrics computed by the independent evaluator",
    }
    (Path(ctx.out_dir) / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (Path(ctx.out_dir) / "model_card.json").write_text(json.dumps(
        {"architecture": "MobileCLIP2-S0 (frozen) + MultiTaskHead(512->256->{score,5xdefect})",
         "encoder_params_millions": round(encoder_params / 1e6, 3),
         "head_params": head_params, "rom": rom}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit("run through algo_agent.execution.method_worker")
