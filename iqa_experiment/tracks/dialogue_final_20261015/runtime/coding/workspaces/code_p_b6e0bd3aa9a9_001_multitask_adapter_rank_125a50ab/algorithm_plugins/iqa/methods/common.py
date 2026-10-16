# -*- coding: utf-8 -*-
"""Shared helpers for IQA method implementations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PLUGIN_DIR = Path(__file__).resolve().parent.parent      # algorithm_plugins/iqa
REPO_ROOT = PLUGIN_DIR.parent.parent                     # .../algo_agent


def ensure_plugin_importable() -> None:
    for p in (str(REPO_ROOT), str(PLUGIN_DIR.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def plugin_config():
    ensure_plugin_importable()
    from algorithm_plugins.iqa import config as C
    return C


def load_rows(path: Path | str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_dev(data_dir: Path) -> List[Dict[str, Any]]:
    return load_rows(Path(data_dir) / "dev.jsonl")


def load_dev_split(data_dir: Path) -> Dict[str, Any]:
    p = Path(data_dir) / "dev_split.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def load_eval_inputs(data_dir: Path) -> List[Dict[str, Any]]:
    for name in ("eval_inputs.jsonl", "eval.jsonl"):
        p = Path(data_dir) / name
        if p.exists():
            rows = load_rows(p)
            return [{"path": r.get("path"), "image": r.get("image")} for r in rows]
    raise FileNotFoundError(f"no eval inputs in {data_dir}")


def defect_vector(row: Dict[str, Any], defects) -> List[int]:
    return [int(bool((row.get("defects") or {}).get(d, {}).get("present", 0))) for d in defects]


def defect_severity(row: Dict[str, Any], defects) -> List[float]:
    return [float((row.get("defects") or {}).get(d, {}).get("severity", 0) or 0) for d in defects]


def make_pred(path: str, score: Optional[float], defects: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {"path": path, "image": Path(path).name,
            "score": (None if score is None else round(float(score), 3)),
            "defects": defects}


def write_preds(ctx, rows: List[Dict[str, Any]]) -> str:
    out_dir = Path(ctx.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    (out_dir / "pred_test.json").write_text(payload, encoding="utf-8")
    (out_dir / f"pred_{getattr(ctx, 'split', 'test')}.json").write_text(payload, encoding="utf-8")
    return str(out_dir / "pred_test.json")


def resolve_device(prefer_gpu: bool = True):
    """Return a torch device, or None when torch/CUDA is unavailable."""
    try:
        import torch
    except Exception:
        return None
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def all_negative(defects) -> Dict[str, Dict[str, Any]]:
    out = {d: {"present": 0, "severity": 0} for d in defects}
    out["white_balance"]["direction"] = "neutral"
    return out


def cache_dir(ctx) -> str:
    """Per-task feature/measure cache directory (off the release, reusable)."""
    d = Path(getattr(ctx, "task_root", Path.cwd())) / ".iqa_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


__all__ = ["cache_dir", "plugin_config", "load_rows", "load_dev", "load_dev_split", "load_eval_inputs",
           "defect_vector", "defect_severity", "make_pred", "write_preds", "resolve_device",
           "all_negative", "REPO_ROOT", "PLUGIN_DIR"]
