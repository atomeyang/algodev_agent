# -*- coding: utf-8 -*-
"""IQA domain configuration for the algo_agent business plugin.

Single source of truth for the light dual-output IQA task: the defect taxonomy,
the acceptance floors, the ROM budget and the resolution of the real local
assets (frozen benchmark, pretrained MobileCLIP2-S0 / trained adapter).

The parent workspace (``.../QAA``) holds the real IQA material produced by the
``iqa_agent`` research engine.  Everything resolves local-first with an env
override (``QAA_ROOT``) so the plugin also runs when the workspace is moved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

# ---- defect taxonomy (from the IQA requirement report) --------------------
DEFECTS: List[str] = ["blur", "over_exp", "under_exp", "flare", "white_balance"]
DEFECT_ZH: Dict[str, str] = {
    "blur": "模糊", "over_exp": "过曝", "under_exp": "欠曝",
    "flare": "炫光", "white_balance": "白平衡",
}
SEVERITY: Dict[int, str] = {0: "none", 1: "light", 2: "medium", 3: "heavy"}
WB_DIRECTIONS: List[str] = ["cool", "warm", "green", "magenta"]

# ---- acceptance floors (report ch.4 / ch.10) ------------------------------
ROM_BUDGET_MB: float = float(os.environ.get("ROM_BUDGET_MB", 60.0))
SCORE_PLCC_FLOOR: float = 0.90
SCORE_SRCC_FLOOR: float = 0.90
DEFECT_MACROF1_FLOOR: float = 0.85
HARDCASE_F1_FLOOR: float = 0.85
GRADE_AGREEMENT_FLOOR: float = 0.90
DEPLOY_PARAMS_MILLIONS: float = float(os.environ.get("DEPLOY_PARAMS_MILLIONS", 11.46))

# ---- real-MOS (KonIQ-10k) track floors ------------------------------------
# The requirement report's PLCC/SRCC target is defined against *real* human MOS
# (the trained adapter reproduced KonIQ val PLCC 0.909 / test 0.818 there).  On
# that quality-only track the defect F1 / hardcase / grade gates do not apply
# (KonIQ ships no defect annotations), so the acceptance floors are the two
# rank-correlation metrics the user asked for.
QUALITY_PLCC_FLOOR: float = float(os.environ.get("IQA_QUALITY_PLCC_FLOOR", 0.85))
QUALITY_SRCC_FLOOR: float = float(os.environ.get("IQA_QUALITY_SRCC_FLOOR", 0.85))

# A defect class needs at least this many ground-truth positives on the frozen
# eval split to be *measurable*; below it the class F1 is undefined (the
# convention below scores 0) and the all-class macro is structurally capped.
MIN_CLASS_SUPPORT: int = 5

# score -> grade bucket for the 分档一致率 (grade agreement) metric
GRADE_BINS = [(85.0, "excellent"), (70.0, "good"), (50.0, "fair"), (0.0, "poor")]

# ---- workspace / asset resolution -----------------------------------------
PLUGIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLUGIN_DIR.parent.parent            # .../algo_agent
_WORKSPACE_DEFAULT = REPO_ROOT.parent           # .../QAA


def qaa_root() -> Path:
    """Parent workspace that holds the real IQA assets (env-overridable)."""
    return Path(os.environ.get("QAA_ROOT", str(_WORKSPACE_DEFAULT)))


def iqa_agent_dir() -> Path:
    return qaa_root() / "iqa_agent"


def benchmark_dir() -> Path:
    return iqa_agent_dir() / "_work" / "benchmark"


def registry_dir() -> Path:
    return iqa_agent_dir() / "_work" / "registry"


def dev_jsonl() -> Path:
    return benchmark_dir() / "dev.jsonl"


def eval_jsonl() -> Path:
    return benchmark_dir() / "eval.jsonl"


def measure_thresholds_path() -> Path:
    return registry_dir() / "measure_thresholds.json"


def baseline_report_path() -> Path:
    return registry_dir() / "baseline.json"


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


def pretrained_ckpt() -> Path:
    """MobileCLIP2-S0 open_clip pretrained weights (encoder backbone)."""
    return _first_existing(
        iqa_agent_dir() / "models" / "mobileclip2_s0.pt",
        qaa_root() / "chuanyin_mobileclip2_final" / "runs" / "mobileclip2_s0.pt",
    )


def adapter_ckpt() -> Path:
    """Trained single-task IQA adapter (KONIQ-trained score head)."""
    return _first_existing(
        iqa_agent_dir() / "models" / "iqa_adapter_v7" / "best.pt",
        qaa_root() / "chuanyin_mobileclip2_final" / "runs" / "iqa_adapter_v7" / "best.pt",
    )


# Calibrated measurement thresholds (F1-maximising on dev). Shipping the values
# here keeps the plugin runnable even when the parent workspace is absent.
DEFAULT_MEASURE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "blur": {"feature": "laplacian_var", "direction": "lt", "threshold": 492.852},
    "over_exp": {"feature": "over_ratio", "direction": "gt", "threshold": 0.04857},
    "under_exp": {"feature": "mean_lum", "direction": "lt", "threshold": 63.398},
    "flare": {"feature": "largest_blob_ratio", "direction": "gt", "threshold": 0.193904},
    "white_balance": {"feature": "cast_mag", "direction": "gt", "threshold": 0.5479},
}


def measure_thresholds() -> Dict[str, Dict[str, Any]]:
    thr = {k: dict(v) for k, v in DEFAULT_MEASURE_THRESHOLDS.items()}
    path = measure_thresholds_path()
    try:
        import json
        cal = json.loads(path.read_text(encoding="utf-8"))
        for d, v in cal.items():
            if d in thr and isinstance(v, dict) and "threshold" in v:
                thr[d] = {"feature": v.get("feature", thr[d]["feature"]),
                          "direction": v.get("direction", thr[d]["direction"]),
                          "threshold": float(v["threshold"])}
    except Exception:
        pass
    return thr


def rom_estimate(params_millions: float, budget_mb: float | None = None) -> Dict[str, Any]:
    """fp16 ROM estimate identical to iqa_agent.tools.budget.check."""
    budget = ROM_BUDGET_MB if budget_mb is None else budget_mb
    mb = float(params_millions) * 2.0
    return {"ok": bool(mb < budget), "rom_mb": round(mb, 2), "budget_mb": budget,
            "headroom_mb": round(budget - mb, 2), "basis": "params_fp16_estimate"}


def benchmark_source() -> str:
    """Which IQA benchmark backs the task.

    ``pseudo`` (default) -- the frozen ``iqa_agent`` testset benchmark whose
    score ground truth is a *severity-quantised pseudo-MOS* (a stand-in used
    until real MOS arrives).  ``koniq`` -- the real-MOS KonIQ-10k benchmark
    (human MOS, official train/val/test split) that the requirement report's
    PLCC/SRCC target is actually defined against.
    """
    src = (os.environ.get("IQA_BENCHMARK") or "pseudo").strip().lower()
    return src if src in ("pseudo", "koniq") else "pseudo"


def quality_only() -> bool:
    """True when the active benchmark ships no defect annotations (MOS only)."""
    return benchmark_source() == "koniq"


def koniq_dir() -> Path:
    return Path(os.environ.get("KONIQ_DIR", str(qaa_root() / "IQADataPart" / "koniq10k")))


def koniq_sets_csv() -> Path:
    return koniq_dir() / "koniq10k_distributions_sets.csv"


def koniq_image_dir() -> Path:
    return _first_existing(koniq_dir() / "512x384", koniq_dir() / "1024x768")


def assets_available() -> Dict[str, bool]:
    return {
        "benchmark_source": benchmark_source(),
        "dev_jsonl": dev_jsonl().exists(),
        "eval_jsonl": eval_jsonl().exists(),
        "pretrained_ckpt": pretrained_ckpt().exists(),
        "adapter_ckpt": adapter_ckpt().exists(),
        "koniq_sets_csv": koniq_sets_csv().exists(),
        "koniq_images": koniq_image_dir().exists(),
    }


__all__ = [
    "DEFECTS", "DEFECT_ZH", "SEVERITY", "WB_DIRECTIONS", "GRADE_BINS",
    "ROM_BUDGET_MB", "SCORE_PLCC_FLOOR", "SCORE_SRCC_FLOOR", "DEFECT_MACROF1_FLOOR",
    "HARDCASE_F1_FLOOR", "GRADE_AGREEMENT_FLOOR", "DEPLOY_PARAMS_MILLIONS",
    "MIN_CLASS_SUPPORT", "QUALITY_PLCC_FLOOR", "QUALITY_SRCC_FLOOR", "qaa_root", "iqa_agent_dir", "benchmark_dir", "registry_dir",
    "dev_jsonl", "eval_jsonl", "measure_thresholds", "measure_thresholds_path",
    "baseline_report_path", "pretrained_ckpt", "adapter_ckpt", "rom_estimate",
    "assets_available", "PLUGIN_DIR", "REPO_ROOT",
    "benchmark_source", "quality_only", "koniq_dir", "koniq_sets_csv", "koniq_image_dir",
]
