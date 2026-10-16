# -*- coding: utf-8 -*-
"""Real trained score head (iqa_adapter_v7) + measurement defect head.

This is the *reproduction baseline* the iqa_agent engine reported (macro-F1
0.424, PLCC 0.476 on the frozen eval): the KONIQ-trained MobileCLIP2-S0 adapter
provides a real continuous score, while the deployable defect head is the
zero-ROM calibrated measurement rule. It proves the frozen protocol reproduces
the reference numbers inside algo_agent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import clip_backbone, common, measure_lib


def run(ctx, config: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    cfg = dict(config or {})
    C = common.plugin_config()
    inputs = common.load_eval_inputs(ctx.data_dir)
    device = common.resolve_device() or torch.device("cpu")
    t0 = time.time()

    pretrained = C.pretrained_ckpt()
    ckpt = Path(cfg.get("score_ckpt") or C.adapter_ckpt())
    if not Path(pretrained).exists() or not ckpt.exists():
        raise FileNotFoundError(f"missing backbone ({pretrained}) or adapter ({ckpt})")

    adapter, info = clip_backbone.load_trained_adapter(ckpt, pretrained, device)
    encoder = adapter.model

    cache = Path(common.cache_dir(ctx))
    paths = [r["path"] for r in inputs]
    feats = clip_backbone.extract_features(encoder, adapter.preprocess_val, paths, device,
                                           cache_dir=cache, tag="mobileclip2s0_lora_256",
                                           batch_size=int(cfg.get("feature_batch", 32)))
    with torch.no_grad():
        scores = adapter.head(torch.tensor(feats, dtype=torch.float32, device=device)).cpu().numpy()
    scores = np.clip(scores * 100.0, 0.0, 100.0)

    thr = C.measure_thresholds()
    defect_map = measure_lib.measure_defects_cached(paths, thr, C.DEFECTS, cache_dir=cache)

    rows = [common.make_pred(r["path"], float(scores[i]), defect_map[r["path"]])
            for i, r in enumerate(inputs)]
    common.write_preds(ctx, rows)

    visual_params = sum(p.numel() for p in adapter.model.visual.parameters())
    head_params = sum(p.numel() for p in adapter.head.parameters())
    rom = C.rom_estimate((visual_params + head_params) / 1e6)
    return {
        "method": "mobileclip2_v7_score", "split": getattr(ctx, "split", "test"),
        "seed": int(getattr(ctx, "seed", 0) or 0), "n_eval": len(rows), "synthetic": False,
        "device": str(device), "rom_params_millions": round((visual_params + head_params) / 1e6, 3),
        "rom": rom, "ckpt": str(ckpt), "ckpt_info": info,
        "wall_clock_s": round(time.time() - t0, 1),
        "metrics": {"n_eval": len(rows)},
        "note": "trained MobileCLIP2-S0 adapter score + measurement defect head (reproduction baseline)",
    }


if __name__ == "__main__":
    raise SystemExit("run through algo_agent.execution.method_worker")
