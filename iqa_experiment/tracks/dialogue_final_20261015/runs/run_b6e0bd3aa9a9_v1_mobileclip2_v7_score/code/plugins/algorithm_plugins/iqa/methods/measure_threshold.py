# -*- coding: utf-8 -*-
"""Deterministic measurement baseline (the distilled deployable defect head).

Exactly the head the iqa_agent engine uses before the multi-task model ships:
per-defect decisions come from calibrated physical features + F1-maximising
thresholds, and the score is a fixed linear blend of physical quality proxies.
Zero model ROM, fully explainable.
"""

from __future__ import annotations

from typing import Any, Dict

from . import common
from . import measure_lib


def run(ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    C = common.plugin_config()
    thr = C.measure_thresholds()
    inputs = common.load_eval_inputs(ctx.data_dir)
    paths = [r["path"] for r in inputs]
    cache_dir = common.cache_dir(ctx)
    defect_map = measure_lib.measure_defects_cached(paths, thr, C.DEFECTS, cache_dir=cache_dir)
    rows = []
    for r in inputs:
        m = measure_lib.measure(r["path"])["global"]
        rows.append(common.make_pred(r["path"], measure_lib.mock_score(m), defect_map[r["path"]]))
    common.write_preds(ctx, rows)
    return {"method": "measure_threshold", "split": getattr(ctx, "split", "test"),
            "seed": int(getattr(ctx, "seed", 0) or 0), "n_eval": len(rows),
            "synthetic": False, "rom_params_millions": 0.0,
            "metrics": {"n_eval": len(rows)},
            "note": "calibrated physical-feature thresholds + linear quality proxy"}


if __name__ == "__main__":
    raise SystemExit("run through algo_agent.execution.method_worker")
