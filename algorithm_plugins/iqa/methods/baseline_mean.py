# -*- coding: utf-8 -*-
"""Trivial reference: predict the training mean score, all defects absent.

Establishes the floor every real candidate must beat and exercises the whole
loop with zero GPU cost.
"""

from __future__ import annotations

from typing import Any, Dict

from . import common


def run(ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    C = common.plugin_config()
    dev = common.load_dev(ctx.data_dir)
    inputs = common.load_eval_inputs(ctx.data_dir)
    scores = [float(r["score"]) for r in dev if r.get("score") is not None]
    mean = sum(scores) / len(scores) if scores else 50.0
    rows = [common.make_pred(r["path"], mean, common.all_negative(C.DEFECTS)) for r in inputs]
    common.write_preds(ctx, rows)
    return {"method": "baseline_mean", "split": getattr(ctx, "split", "test"),
            "seed": int(getattr(ctx, "seed", 0) or 0), "n_eval": len(rows),
            "train_mean_score": round(mean, 3), "synthetic": False,
            "metrics": {"train_mean_score": round(mean, 3)},
            "note": "constant-score / all-negative-defect floor"}


if __name__ == "__main__":
    raise SystemExit("run through algo_agent.execution.method_worker")
