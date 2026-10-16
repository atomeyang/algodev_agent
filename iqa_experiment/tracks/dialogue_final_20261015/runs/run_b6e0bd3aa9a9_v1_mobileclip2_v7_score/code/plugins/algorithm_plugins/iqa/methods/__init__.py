"""IQA method implementations (each exposes ``run(ctx, config) -> dict``)."""

METHOD_FILES = {
    "baseline_mean": "baseline_mean.py",
    "measure_threshold": "measure_threshold.py",
    "mobileclip2_v7_score": "mobileclip2_v7_score.py",
    "multitask_adapter": "multitask_adapter.py",
    # score-metric-driven heads (PLCC/SRCC), added for the score optimisation track
    "score_ordinal_sft": "score_ordinal_sft.py",
    "score_ordinal_sft_lite": "score_ordinal_sft.py",
    # three-view (regression + 4-class soft-E + coarse-family MoE) score head
    "score_moe": "score_moe.py",
    "score_moe_meta": "score_moe.py",
    # real human-MOS (KonIQ-10k) track: frozen MobileCLIP2-S0 + light head
    "mobileclip2_mos_head": "score_mos_transfer.py",
    "mobileclip2_mos_mlp": "score_mos_transfer.py",
}

__all__ = ["METHOD_FILES"]
