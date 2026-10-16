# -*- coding: utf-8 -*-
"""Business plugin: light dual-output Image Quality Assessment (IQA).

Domain owner for the requirement "ROM < 60 MB fp16, dual output (0-100 quality
score + 5 defect multi-labels), defect Macro-F1 >= 0.85, PLCC/SRCC >= 0.90,
grade agreement >= 0.90" (IQA model selection & dataset report, plan A1).

Delegates the real work to ``algorithm_plugins.iqa.methods.*`` and the frozen
metrics to ``algorithm_plugins.iqa.metrics``; the graph nodes stay domain
agnostic and fall back to the synthetic-tabular demo when this hook set is
absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common.plugin import MethodContext  # noqa: F401  (documented contract)

from . import config as C
from . import data as D
from . import metrics as M
from .methods import METHOD_FILES

# Strong, unambiguous IQA signatures: a single hit is enough to claim the
# domain. Deliberately excludes generic terms like '质量分' (quality score),
# which also appear in non-IQA image-quality tasks.
_STRONG_KEYWORDS = [
    "iqa", "image quality", "画质", "图像质量", "无参考质量", "双输出", "dual output",
    "mobileclip", "q-align", "轻量双输出",
]
# Defect-taxonomy terms: >= 2 co-occurring, or 1 alongside 缺陷 (defect),
# indicates the 5-class IQA multi-label task rather than a generic regression.
_DEFECT_TERMS = [
    "模糊", "过曝", "欠曝", "炫光", "白平衡",
    "blur", "over_exp", "under_exp", "flare", "white_balance",
]

# method name -> implementation file (variants share one implementation)
_METHOD_ALIASES: Dict[str, str] = {
    "baseline_mean": "baseline_mean.py",
    "measure_threshold": "measure_threshold.py",
    "mobileclip2_v7_score": "mobileclip2_v7_score.py",
    "multitask_adapter": "multitask_adapter.py",
    "multitask_adapter_balanced": "multitask_adapter.py",
    "multitask_adapter_long": "multitask_adapter.py",
    # score-metric (PLCC/SRCC) driven heads
    "score_ordinal_sft": "score_ordinal_sft.py",
    "score_ordinal_sft_lite": "score_ordinal_sft.py",
    "score_ordinal_sft_synth": "score_ordinal_sft.py",
    # three-view blend head (best honest PLCC frontier)
    "score_moe": "score_moe.py",
    "score_moe_meta": "score_moe.py",
    # real human-MOS (KonIQ-10k) track heads
    "mobileclip2_mos_head": "score_mos_transfer.py",
    "mobileclip2_mos_mlp": "score_mos_transfer.py",
}

# per-variant hyper-parameters handed to the method via ExperimentSpec.config
_VARIANT_CONFIG: Dict[str, Dict[str, Any]] = {
    "multitask_adapter": {"epochs": 40, "lr": 2e-3, "bce_weight": 1.0, "pos_weight_scale": 1.0},
    "multitask_adapter_balanced": {"epochs": 60, "lr": 1.2e-3, "bce_weight": 2.5,
                                   "pos_weight_scale": 2.0, "max_pos_weight": 40.0, "dropout": 0.25},
    "multitask_adapter_long": {"epochs": 120, "lr": 1.5e-3, "bce_weight": 1.5,
                               "pos_weight_scale": 1.5, "tune_thresholds": True},
    # degradation-aware ordinal score head; the ladder widens the severity
    # supervision and the model-selection grid as iterations accumulate.
    "score_ordinal_sft": {"use_synthetic": None, "ensemble_seeds": 3,
                          "synth_defects": ["blur", "over_exp", "under_exp", "white_balance"]},
    "score_ordinal_sft_lite": {"use_synthetic": False, "ensemble_seeds": 3},
    "score_ordinal_sft_synth": {"use_synthetic": True, "ensemble_seeds": 5,
                                "synth_defects": ["blur", "over_exp", "under_exp",
                                                  "white_balance", "flare"]},
    # three-view blend; `_meta` keeps the file-size scalars, the plain variant
    # drops them so the ablation isolates their (small) contribution.
    # iteration-1 core head: degradation features + physical measurements only.
    "score_moe": {"use_metadata": False, "ensemble_seeds": 3, "cv_folds": 4,
                  "blend_step": 0.1},
    # escalated head: adds the cheap file-size scalars and a finer blend grid.
    "score_moe_meta": {"use_metadata": True, "ensemble_seeds": 4, "cv_folds": 4,
                       "blend_step": 0.05},
    # real-MOS heads: ridge with train/val-CV alpha selection (primary), then an
    # MLP escalation if the ridge frontier is short of the PLCC floor.
    "mobileclip2_mos_head": {"head": "ridge", "image_size": 256,
                             "alphas": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "mobileclip2_mos_mlp": {"head": "mlp", "image_size": 256, "hidden": [512, 256],
                            "mlp_max_iter": 600},
}


_SCORE_PRIMARY_RE = None


def _score_primary_metric(text: str):
    """Detect an explicit "the primary gate is PLCC/SRCC >= x" statement.

    Deterministic and offline; returns ``(metric_name, threshold)`` or ``None``.
    """
    global _SCORE_PRIMARY_RE
    if _SCORE_PRIMARY_RE is None:
        _SCORE_PRIMARY_RE = [
            re.compile(r"(主指标|首要指标|primary\s+metric|主门禁|first\s+gate|主要指标)[^0-9]{0,40}"
                       r"(plcc|srcc|pearson|spearman)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)", re.I),
            re.compile(r"(plcc|srcc|pearson|spearman)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)"
                       r"[^0-9]{0,20}(为主指标|作为主指标|as\s+the\s+primary)", re.I),
        ]
    for pat in _SCORE_PRIMARY_RE:
        m = pat.search(text or "")
        if m:
            groups = m.groups()
            name = next((g.lower() for g in groups if g and g.lower() in
                         ("plcc", "srcc", "pearson", "spearman")), "plcc")
            thr = next((float(g) for g in groups if g and re.fullmatch(r"[0-9]+(\.[0-9]+)?", g)), 0.85)
            return ("srcc" if name in ("srcc", "spearman") else "plcc"), thr
    return None


def _primary_floor_line(spec) -> str:
    """First acceptance-floor bullet, driven by the task's gated metric."""
    pm = getattr(spec, "primary_metric", None)
    name = str(getattr(pm, "name", "") or "")
    thr = getattr(pm, "threshold", None)
    if name in ("plcc", "srcc"):
        lo = float(thr) if thr is not None else C.SCORE_PLCC_FLOOR
        return (f"- **{name.upper()} >= {lo:g}  (primary gate this run)**, "
                "with defect Macro-F1 etc. as secondary gates")
    lo = float(thr) if thr is not None else C.DEFECT_MACROF1_FLOOR
    return f"- defect Macro-F1 >= {lo:g}  (primary)"


def _secondary_floor_line(spec) -> str:
    """Secondary acceptance floors, without re-listing the primary metric."""
    pm = getattr(spec, "primary_metric", None)
    name = str(getattr(pm, "name", "") or "")
    parts: list[str] = []
    if name != "plcc":
        parts.append(f"PLCC >= {C.SCORE_PLCC_FLOOR}")
    if name != "srcc":
        parts.append(f"SRCC >= {C.SCORE_SRCC_FLOOR}")
    parts += [f"defect Macro-F1 >= {C.DEFECT_MACROF1_FLOOR}",
              f"hardcase Macro-F1 >= {C.HARDCASE_F1_FLOOR}",
              f"grade agreement >= {C.GRADE_AGREEMENT_FLOOR}"]
    return "- secondary floors: " + ", ".join(parts)


def _reality_lines(quality_only: bool) -> List[str]:
    if quality_only:
        return [
            "- benchmark: real human MOS (KonIQ-10k), official train/val/test split",
            "- PLCC/SRCC are scored on the official **test** split; defect annotations do not exist "
            "here, so the defect / hardcase / grade gates are reported as inapplicable",
        ]
    return [
        "- eval positives: blur 150, over_exp 75, under_exp 75, flare 9, **white_balance 0**",
        "- the all-class Macro-F1 is therefore capped at 0.8 even with perfect F1 on the other four "
        "classes; the report also tracks `defect_macro_f1_cov` over classes with support.",
    ]


class IqaPlugin:
    domain = "iqa"

    # ------------------------------------------------------------------ #
    # detection / parsing
    # ------------------------------------------------------------------ #
    def matches(self, text: str) -> bool:
        low = (text or "").lower()
        if any(k in low for k in _STRONG_KEYWORDS):
            return True
        defect_hits = sum(1 for k in _DEFECT_TERMS if k in low)
        if defect_hits >= 2:
            return True
        return "\u7f3a\u9677" in low and defect_hits >= 1

    def parse_requirement(self, text: str, task_id: str, slug: str, *, demo: bool = False,
                          synthetic: bool = False, **_: Any):
        from algo_agent.schemas import contracts as CT

        # A requirement that explicitly names the score metric as the *primary*
        # gate ("主指标 PLCC >= 0.85" / "primary metric PLCC 0.85") switches the
        # task onto the score track: PLCC becomes the gated metric and the
        # defect Macro-F1 floors stay as secondary gates.
        score_primary = _score_primary_metric(text)
        if score_primary:
            primary = CT.MetricTarget(name=score_primary[0], direction="max",
                                      threshold=score_primary[1], unit="r",
                                      protocol_id="iqa_frozen_v1", measured_on="eval")
        else:
            # default track: the defect Macro-F1 gate is primary, PLCC/SRCC secondary
            primary = CT.MetricTarget(name="defect_macro_f1", direction="max",
                                      threshold=C.DEFECT_MACROF1_FLOOR, unit="f1",
                                      protocol_id="iqa_frozen_v1", measured_on="eval")
        if C.quality_only():
            # Real-MOS track (KonIQ-10k): no defect annotations ship, so the
            # dual-output defect gates are inapplicable.  The gated secondary
            # metric is the companion rank correlation at the same floor.
            secondary = []
            if primary.name != "plcc":
                secondary.append(CT.MetricTarget(name="plcc", direction="max",
                                                 threshold=C.QUALITY_PLCC_FLOOR, unit="r"))
            if primary.name != "srcc":
                secondary.append(CT.MetricTarget(name="srcc", direction="max",
                                                 threshold=C.QUALITY_SRCC_FLOOR, unit="r"))
        else:
            secondary = [
                CT.MetricTarget(name="plcc", direction="max", threshold=C.SCORE_PLCC_FLOOR, unit="r"),
                CT.MetricTarget(name="srcc", direction="max", threshold=C.SCORE_SRCC_FLOOR, unit="r"),
                CT.MetricTarget(name="hardcase_macro_f1", direction="max", threshold=C.HARDCASE_F1_FLOOR),
                CT.MetricTarget(name="grade_agreement", direction="max", threshold=C.GRADE_AGREEMENT_FLOOR),
            ]
        rom = CT.MetricTarget(name="rom_mb", direction="min", threshold=C.ROM_BUDGET_MB, unit="MB",
                              protocol_id="fp16_from_params", measured_on="export")
        constraints = [
            CT.HardConstraint(name="rom_mb_fp16", value=C.ROM_BUDGET_MB, unit="MB",
                              measurement_scope="fp16 deployment package (encoder + heads; text tower dropped)",
                              blocking=True),
            CT.HardConstraint(name="dual_output", value="score+defects", unit="",
                              measurement_scope="inference API", blocking=True),
            CT.HardConstraint(name="latency_ms", value=120.0, unit="ms",
                              measurement_scope="single 256x256 crop, mobile-class device", blocking=False),
        ]
        if C.quality_only():
            unknowns = [
                f"active benchmark = real human MOS ({C.benchmark_source()}); the report's "
                "PLCC/SRCC target is defined here",
                "KonIQ ships no defect annotations -> the defect head is not measurable on "
                "this track (dual output kept as a product-level constraint)",
                "no on-device latency measurement in this run (ROM estimate only)",
            ]
        else:
            unknowns = [
                "white_balance has no positives on the frozen eval split -> not measurable here",
                "pseudo-MOS (severity-quantised) used as score ground truth, not real MOS",
                "no on-device latency measurement in this run (ROMM estimate only)",
            ]
        return CT.TaskSpec(
            task_id=task_id, task_slug=slug, domain="iqa", business=text.strip(),
            inputs=["rgb_image_256"], outputs=["quality_score_0_100", "defect_multilabel_5"],
            target_hardware="mobile", precision="fp16",
            hard_constraints=constraints, primary_metric=primary, secondary_metrics=secondary,
            budget=CT.Budget(max_iterations=4, max_experiments=24, max_coding_calls=0,
                             max_epochs=200, max_wall_clock_s=5400.0),
            unknowns=unknowns, demo=False, synthetic=False, source="natural_language")

    # ---- base protocol -------------------------------------------------
    def validate_requirement(self, spec: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        if not spec.get("primary_metric", {}).get("name"):
            problems.append("primary metric name is required")
        names = {c.get("name") for c in spec.get("hard_constraints", [])}
        if "rom_mb_fp16" not in names:
            problems.append("ROM hard constraint missing")
        return problems

    def validate_dataset(self, manifest: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        splits = manifest.get("splits", {})
        if splits.get("eval", 0) < 100:
            problems.append("frozen eval split too small")
        if not manifest.get("checksums"):
            problems.append("dataset release has no checksums")
        return problems

    def default_slices(self) -> List[str]:
        return list(C.DEFECTS) + ["hardcase"]

    def deployment_scope(self) -> Dict[str, Any]:
        return {"device": "mobile", "runtime": "fp16 (ONNX/TFLite export out of scope here)",
                "rom_budget_mb": C.ROM_BUDGET_MB, "params_millions": C.DEPLOY_PARAMS_MILLIONS,
                "text_tower": "dropped at deployment", "note": "ROM measured as fp16 params estimate"}

    # ------------------------------------------------------------------ #
    # research / design  (drives candidates + protocol)
    # ------------------------------------------------------------------ #
    def slices(self, spec) -> List[str]:
        return list(C.DEFECTS) + ["hardcase"]

    def data_sources(self, spec) -> List[str]:
        return ["qaa_iqa_frozen_benchmark", "iqa_agent_dev_split", "iqa_agent_measure_registry"]

    def design(self, spec, *, iteration: int = 1, feedback: Optional[List[Dict[str, Any]]] = None,
               slices: Optional[List[str]] = None, methods_hint: Optional[List[str]] = None,
               history: Optional[List[Dict[str, Any]]] = None,
               constraints: Optional[List[Dict[str, Any]]] = None,
               task_root: Optional[str] = None) -> Dict[str, Any]:
        from algo_agent.schemas import contracts as CT

        feedback = feedback or []
        weak = self._weak_defects(feedback)
        if not weak:
            weak = self._weak_from_history(history)
        iteration = int(iteration or 1)

        def cand(method: str, family: str, desc: str, gain: str, cost: str, baseline: bool = False,
                 source: str = "local", failure: Optional[List[str]] = None) -> CT.Candidate:
            return CT.Candidate(candidate_id=method, method=method, family=family, description=desc,
                                expected_gain=gain, cost_estimate=cost, is_baseline=baseline,
                                source=source, failure_conditions=failure or [])

        # Methods already run in earlier iterations, so the loop escalates
        # instead of re-proposing the same variant (from the enriched history).
        tried: set = set()
        for entry in history or []:
            tried.update(entry.get("methods") or [])

        primary_name = str(getattr(getattr(spec, "primary_metric", None), "name", "") or "")
        score_driven = primary_name in ("plcc", "srcc")

        if methods_hint:
            methods = list(methods_hint)
        elif score_driven and C.quality_only():
            # Real-MOS track: frozen MobileCLIP2-S0 features + a light head is
            # the strongest honest frontier (measured KonIQ test PLCC ~0.89);
            # escalate ridge -> MLP only if the CV-selected ridge falls short.
            ladder = ["mobileclip2_mos_head", "mobileclip2_mos_mlp"]
            if methods_hint:
                methods = list(methods_hint)
            elif iteration <= 1:
                methods = ["mobileclip2_mos_head"]
            else:
                methods = [next((m for m in ladder if m not in tried), ladder[-1])]
        elif score_driven:
            # pseudo-MOS PLCC/SRCC track: escalate along the score-specific
            # ladder instead of re-proposing the defect-oriented variants.
            ladder = ["score_moe", "score_moe_meta", "score_ordinal_sft_synth",
                      "score_ordinal_sft_lite"]
            if iteration <= 1:
                methods = ["mobileclip2_v7_score", "score_moe"]
            else:
                methods = [next((m for m in ladder if m not in tried), ladder[-1])]
        elif iteration <= 1:
            methods = ["measure_threshold", "mobileclip2_v7_score", "multitask_adapter"]
        elif weak and ("flare" in weak or "white_balance" in weak or "under_exp" in weak):
            # rare/weak classes first, escalating to a longer mixed-loss schedule
            ladder = ["multitask_adapter_balanced", "multitask_adapter_long"]
            methods = [next((m for m in ladder if m not in tried), ladder[-1])]
        else:
            methods = ["multitask_adapter_long"]

        catalog = {
            "baseline_mean": (cand("baseline_mean", "baseline",
                                   "Constant training-mean score, all defects absent.",
                                   "reference floor only", "negligible", baseline=True,
                                   failure=["cannot express any defect or score variation"]), {}),
            "measure_threshold": (cand("measure_threshold", "rule",
                                       "Calibrated physical features (Laplacian/over-under ratio/blob/cast) "
                                       "with F1-maximising thresholds; zero-ROM distilled defect head.",
                                       "reproduces the dev-calibrated reference defect F1", "65 s CPU",
                                       failure=["rare classes (flare) and wb direction are not separable"]), {}),
            "mobileclip2_v7_score": (cand("mobileclip2_v7_score", "transfer",
                                          "Trained MobileCLIP2-S0 adapter (KONIQ) for the score + measurement "
                                          "defect head. Reproduction baseline for the frozen protocol.",
                                          "real continuous score (PLCC), 22.9 MB fp16", "~2 min GPU",
                                          failure=["defect head is still the rule head; domain shift on pseudo-MOS"]), {}),
            "multitask_adapter": (cand("multitask_adapter", "finetune",
                                       "Frozen MobileCLIP2-S0 + trainable score head + 5-way multi-label "
                                       "defect head trained on the dev split (GPU).",
                                       "targets the defect Macro-F1 gate directly", "~3-5 min GPU",
                                       failure=["rare-class imbalance", "wb unlabelled on dev"]), {}),
            "multitask_adapter_balanced": (cand("multitask_adapter_balanced", "finetune+class-balance",
                                               "Multi-task adapter with class-balanced BCE (pos_weight x2) and "
                                               "lower LR to lift the weak/rare defect classes found in feedback.",
                                               "raises rare-class recall (flare/under_exp)", "~5 min GPU",
                                               failure=["over-predicting rare classes can hurt precision"]), {}),
            "score_ordinal_sft": (cand("score_ordinal_sft", "score-head+ordinal",
                                       "Degradation-aware ordinal score head (multi-scale blur-sigma, "
                                       "clipping/gain, WB gains, noise, dark channel, glare) with "
                                       "subject-group CV model selection; synthetic severity ladder "
                                       "switched on only when it measurably lifts CV PLCC.",
                                       "directly targets the PLCC/SRCC gate (ranking, not MSE)",
                                       "~2-5 min CPU",
                                       failure=["score GT is a severity-quantised pseudo-MOS, so the "
                                                "ceiling is a 4-level ordinal problem, not a continuous one"]), {}),
            "score_ordinal_sft_lite": (cand("score_ordinal_sft_lite", "score-head+ordinal",
                                            "Same ordinal score head trained on the labelled dev split only "
                                            "(no synthetic ladder). Isolates the value of the synthetic SFT.",
                                            "cheap ablation for the synthetic ladder", "~1-2 min CPU",
                                            failure=["410 training rows -> dev->eval generalisation gap"]), {}),
            "score_moe": (cand("score_moe", "score-head+MoE",
                                "Three-view score head: global ET/RF regression, a 4-class "
                                "posterior E[score], and a coarse-degradation-family MoE "
                                "(gate + router + per-family experts), blended with weights "
                                "chosen by subject-group CV on dev only.",
                                "best honest PLCC frontier; auditable decomposition",
                                "~2-4 min CPU",
                                failure=["PLCC is capped by the good-vs-defective gate "
                                         "(ROC-AUC ~0.85) and the filename-hash noise in the "
                                         "good-image pseudo-MOS, so ~0.85 needs ~0.86 4-level "
                                         "accuracy and is at the structural ceiling"]), {}),
            "score_moe_meta": (cand("score_moe_meta", "score-head+MoE+meta",
                                    "Same three-view head with a finer blend grid and one "
                                    "more ensemble seed; escalates the MoE variant.",
                                    "small extra CV-selected gain", "~4-6 min CPU",
                                    failure=["marginal over the plain score_moe variant"]), {}),
            "score_ordinal_sft_synth": (cand("score_ordinal_sft_synth", "score-head+ordinal+SFT",
                                             "Ordinal score head with the synthetic severity ladder forced on "
                                             "and a wider model/threshold grid, for the iteration where the "
                                             "lite variant plateaus.",
                                             "more exactly-labelled severity supervision", "~5-8 min CPU",
                                             failure=["synthetic operators differ from the eval corpus' "
                                                      "unknown degradation pipeline"]), {}),
            "mobileclip2_mos_head": (cand("mobileclip2_mos_head", "transfer+ridge-head",
                                          "Frozen MobileCLIP2-S0 visual tower + L2-normalised features "
                                          "with a ridge MOS head; regularisation chosen on the official "
                                          "KonIQ train/val split, head refit on all dev rows. Reuses the "
                                          "exact deployable backbone already inside the ROM budget.",
                                          "targets the real-MOS PLCC/SRCC gate directly; measured KonIQ "
                                          "test PLCC ~0.89", "~1 run (~3-5 min GPU first time, cached)",
                                          failure=["frozen features cap the frontier (~0.89); a LoRA "
                                                   "finetune could go higher at more cost"]), {}),
            "mobileclip2_mos_mlp": (cand("mobileclip2_mos_mlp", "transfer+mlp-head",
                                         "Same frozen MobileCLIP2-S0 features with a compact MLP "
                                         "(512,256) MOS head. Escalation when the ridge frontier "
                                         "is short of the PLCC floor.",
                                         "non-linear head can recover residual feature structure",
                                         "~1-3 min CPU after cached features",
                                         failure=["small dev set -> mild overfit vs ridge (measured "
                                                  "KonIQ test PLCC ~0.887, slightly below ridge)"]), {}),
            "multitask_adapter_long": (cand("multitask_adapter_long", "finetune+regularised",
                                            "Multi-task adapter with a longer schedule and mixed score/defect loss "
                                            "to improve score PLCC while holding defect F1.",
                                            "improves score ranking + hardcase F1", "~6 min GPU",
                                            failure=["longer training may overfit the small dev split"]), {}),
        }
        candidates = []
        for m in methods:
            if m in catalog:
                candidates.append(catalog[m][0])
            else:
                candidates.append(cand(m, "unknown", m, "n/a", "unknown"))

        weak_txt = ", ".join(weak) if weak else "none identified yet"
        hypotheses = [
            f"Defect Macro-F1 is dominated by the {', '.join(C.DEFECTS[:3])} classes; "
            f"the frozen eval cannot measure white_balance (0 positives).",
            f"Weak classes carried over from feedback: {weak_txt}.",
            "A frozen MobileCLIP2-S0 encoder with light multi-task heads reaches the ROM budget "
            "with headroom while beating the rule baseline on defect F1.",
        ]
        if iteration > 1:
            hypotheses.append("Class-balanced BCE + threshold tuning recovers rare-class recall "
                              "without regressing the frequent classes.")

        recipe = CT.DataRecipe(
            recipe_id=f"iqa_recipe_v{iteration}",
            description="Frozen iqa_agent benchmark: 410 labelled dev rows (train/val, subject-disjoint) "
                        "+ 610-image frozen eval (labels held out of the method's data dir). "
                        "Label-gap classes are filled offline: local_largemodel/qwen3.8-27b proposes "
                        "candidate defect labels (scripts/iqa_annotate.py, prompt iqa-prelabel-v1) and "
                        "every candidate goes through human review before it can enter a dataset "
                        "version; classes with no measurable support receive severity-matched "
                        "augmentation (see the data plan in the research report).",
            source="qaa_iqa_frozen_benchmark + local_vlm_prelabels(qwen3.8-27b) + severity-matched augmentation",
            transforms=["shortest_side_resize_256", "bicubic", "imagenet_free_normalisation(0/1)",
                        "human-reviewed local-VLM pre-labels (provenance: model_id+prompt_version)",
                        "train-split-only augmentation for zero/low-support classes"],
            splits={"train": 0.75, "val": 0.25}, seed=7,
            isolation="subject-group-disjoint train/val; eval labels never exposed to the method; "
                      "augmentation never touches the frozen eval in the same protocol version")
        protocol = CT.EvalProtocol(
            protocol_id=f"iqa_frozen_v{iteration}",
            primary_metric=primary_name or "defect_macro_f1",
            direction="max",
            threshold=float(getattr(getattr(spec, "primary_metric", None), "threshold",
                                    C.DEFECT_MACROF1_FLOOR) or C.DEFECT_MACROF1_FLOOR),
            slices=list(C.DEFECTS) + ["hardcase"], frozen=True, dataset_version=f"v{iteration}",
            notes="frozen metric code in algorithm_plugins/iqa/metrics.py (parity-checked against "
                  "iqa_agent.tools.iqa_model.evaluate); ROM from fp16 params.")

        from . import plans as P

        status = P.precision_status(
            history, iteration, primary_metric=primary_name or None,
            primary_floor=getattr(getattr(spec, "primary_metric", None), "threshold", None))
        opt_plan = P.optimization_plan(weak, status, iteration)
        dplan = P.data_plan(weak, status, task_root=Path(task_root) if task_root else None)
        report = self._render_report(spec, candidates, iteration, feedback, weak,
                                     status=status, optimization_plan=opt_plan, data_plan=dplan)
        return {"report_markdown": report, "candidates": candidates, "hypotheses": hypotheses,
                "precision_status": status, "optimization_plan": opt_plan, "data_plan": dplan,
                "controls": ["identical frozen eval split and metric code for every candidate",
                             "ROM hard constraint checked on every exported model",
                             "compare against measure_threshold and the prior best"],
                "counterexamples": ["flare positives are only 9 in eval -> wide confidence interval",
                                    "pseudo-MOS is quantised, so PLCC has a low ceiling"],
                "unknowns": ["white_balance unmeasurable on the frozen eval (0 positives)",
                             "on-device latency not measured"],
                "evidence_refs": [str(C.benchmark_dir()), str(C.baseline_report_path()),
                                  "IQA轻量双输出模型选型与数据集构建深度调研报告.docx"],
                "data_recipe": recipe, "eval_protocol": protocol,
                "cost": {"cloud_tokens": 0, "currency": "USD", "source": "offline"},
                "status": "completed", "offline": True}

    def _weak_defects(self, feedback: List[Dict[str, Any]]) -> List[str]:
        weak: List[str] = []
        for fb in feedback or []:
            blob = " ".join(str(fb.get(k, "")) for k in ("scenario", "actual", "expected", "category"))
            for d in C.DEFECTS:
                if d in blob and d not in weak:
                    weak.append(d)
        return weak

    def _weak_from_history(self, history: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Weak *measurable* defect classes from the previous iteration's eval.

        Closes the research loop on measurement even without user feedback.
        Classes with too little support on the frozen eval are skipped: they are
        a coverage gap, not a fixable weakness.
        """
        if not history:
            return []
        entry = history[-1] or {}
        slices = entry.get("slices") or {}
        metrics = entry.get("metrics") or {}
        weak: List[str] = []
        for d in C.DEFECTS:
            f1 = (slices.get(d) or {}).get("f1")
            if f1 is None:
                continue
            support = metrics.get(f"support_{d}")
            if support is not None and float(support) < C.MIN_CLASS_SUPPORT:
                continue
            if float(f1) < C.DEFECT_MACROF1_FLOOR:
                weak.append(d)
        return weak

    def constraint_names(self) -> List[str]:
        return ["rom_mb_fp16", "dual_output"]

    def check_constraints(self, spec, metrics: Dict[str, Any],
                          slices: Optional[Dict[str, Any]] = None) -> List[str]:
        """Measurable hard constraints for the analyze/decide loop.

        ROM is a hard constraint (not a metric target), so it is checked here and
        its failures feed the same failure list the decide node acts on.
        """
        problems: List[str] = []
        rom_mb = metrics.get("rom_mb")
        if rom_mb is not None and float(rom_mb) > C.ROM_BUDGET_MB:
            problems.append(f"ROM {float(rom_mb):.1f} MB exceeds the {C.ROM_BUDGET_MB:.0f} MB "
                            "fp16 deployment budget")
        if metrics.get("rom_ok") == 0.0 and rom_mb is None:
            problems.append("ROM hard constraint not met on the exported model")
        if not C.quality_only() and not any(str(k).startswith("f1_") for k in metrics):
            problems.append("dual output incomplete: no per-defect F1 (defect head missing)")
        if getattr(spec, "auto_process", False) and any(
                c.name == "dual_output" and c.blocking for c in spec.hard_constraints):
            if metrics.get("dual_output_complete") != 1:
                problems.append("dual output schema was not independently verified")
        return problems

    def select_best(self, protocol, candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Feasibility-first model choice across all IQA acceptance gates."""
        if not candidates:
            return None

        primary = str(getattr(protocol, "primary_metric", "") or "defect_macro_f1")

        def rank(c: Dict[str, Any]):
            m = c.get("metrics") or {}
            feasible = 1 if (c.get("passed") and m.get("rom_ok", 1.0) == 1.0) else 0
            macro = m.get("defect_macro_f1_cov", m.get("defect_macro_f1", 0.0)) or 0.0
            hard = m.get("hardcase_macro_f1_cov", m.get("hardcase_macro_f1", 0.0)) or 0.0
            plcc = m.get("plcc", 0.0) or 0.0
            srcc = m.get("srcc", 0.0) or 0.0
            focus = {"plcc": float(plcc), "srcc": float(srcc)}.get(primary, float(macro))
            # the primary metric decides; the rest only break ties so the chosen
            # "best" model matches the gated metric of the run (guide 3).
            return (feasible, focus, float(macro), float(hard), float(plcc), float(srcc))

        return max(candidates, key=rank).get("model_id")

    def _render_report(self, spec, candidates, iteration: int, feedback, weak, *,
                       status: Optional[Dict[str, Any]] = None,
                       optimization_plan: Optional[List[Dict[str, Any]]] = None,
                       data_plan: Optional[Dict[str, Any]] = None) -> str:
        lines = [
            f"# IQA research & design (iteration {iteration})",
            "",
            "> OFFLINE MODE: no live literature search or cloud model was used. Candidates and "
            "thresholds come from the local IQA workspace + the requirement report; all results "
            "are labelled with their real provenance (frozen pseudo-MOS benchmark).",
            "",
            "## Business objective",
            f"{spec.business or '(not stated)'}",
            "",
            "## Hard constraints",
            f"- ROM < {C.ROM_BUDGET_MB:.0f} MB fp16 (deployment drops the text tower)",
            "- dual output: 0-100 quality score + 5 defect multi-labels (blur/over_exp/under_exp/flare/white_balance)",
            "",
            "## Acceptance floors",
            _primary_floor_line(spec),
            _secondary_floor_line(spec),
            "",
            "## Frozen evaluation reality check",
            *_reality_lines(C.quality_only()),
            "",
            "## Candidates",
        ]
        for c in candidates:
            lines.append(f"- **{c.method}** ({c.family}): {c.description} "
                         f"(expected gain: {c.expected_gain}; cost: {c.cost_estimate})")
        if feedback:
            lines += ["", "## Feedback driving this iteration"]
            for f in feedback:
                lines.append(f"- {f.get('feedback_id', f.get('scenario', '?'))}: "
                             f"{f.get('actual', '')} (expected: {f.get('expected', '')})")
        if weak:
            lines += ["", "## Weak classes flagged this iteration", f"- {', '.join(weak)}"]
        if status:
            from . import plans as P
            lines += ["", *P.render_precision_status(status)]
            if optimization_plan:
                lines += P.render_optimization_plan(optimization_plan)
            if data_plan:
                lines += P.render_data_plan(data_plan)
        return "\n".join(lines)

    def parameter_space(self, spec, methods: List[str]) -> Dict[str, Any]:
        space: Dict[str, Any] = {}
        for m in methods:
            space[m] = dict(_VARIANT_CONFIG.get(m, {}))
        return space

    # ------------------------------------------------------------------ #
    # assets / coding
    # ------------------------------------------------------------------ #
    def prepare_dataset(self, tp, spec, plan, version: str) -> Dict[str, Any]:
        seed = int(getattr(plan.data_recipe, "seed", 7) or 7)
        return D.export_benchmark(tp, version, seed=seed)

    def inspect_development_data(self, spec) -> Dict[str, Any]:
        return D.development_source_summary()

    def method_entry(self, root, method: str, spec) -> Optional[str]:
        return self.materialise_method(Path(root), method)

    @staticmethod
    def materialise_method(root: Path, method: str) -> Optional[str]:
        fname = _METHOD_ALIASES.get(method) or METHOD_FILES.get(method)
        if not fname:
            return None
        src = Path(__file__).resolve().parent / "methods" / fname
        dst_rel = f"model_pool/{method}/method.py"
        dst = Path(root) / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        NL = "\n"
        body = (
            "# -*- coding: utf-8 -*-" + NL
            + '"""Auto-materialised entry for IQA method ' + repr(method) + " (plugin " + fname + ')."""' + NL
            + "from algorithm_plugins.iqa.methods import " + fname[:-3] + " as _impl" + NL + NL
            + "METHOD = " + repr(method) + NL + NL + NL
            + "def run(ctx, config):" + NL
            + "    return _impl.run(ctx, config)" + NL
            + NL + "def predict(ctx, config, weights_path):" + NL
            + "    return _impl.predict(ctx, config, weights_path)" + NL
        )
        dst.write_text(body, encoding="utf-8")
        (dst.parent / "config.yaml").write_text(
            json.dumps({"method": method, "impl": fname,
                        "config": _VARIANT_CONFIG.get(method, {})}, indent=2), encoding="utf-8")
        (dst.parent / "METHOD_SOURCE").write_text(str(src), encoding="utf-8")
        return dst_rel

    def coding_context_files(self, spec) -> List[str]:
        return ["algorithm_plugins/iqa/plugin.py", "algorithm_plugins/iqa/metrics.py",
                "algorithm_plugins/iqa/config.py", "algorithm_plugins/iqa/data.py",
                "algorithm_plugins/iqa/methods/common.py",
                "algorithm_plugins/iqa/methods/mobileclip2_v7_score.py",
                "algorithm_plugins/iqa/methods/score_moe.py",
                "algorithm_plugins/iqa/methods/multitask_adapter.py"]

    # ------------------------------------------------------------------ #
    # experiments / evaluation
    # ------------------------------------------------------------------ #
    def experiment_env(self, spec) -> Dict[str, Any]:
        return {"python": "local", "backend": "torch", "device": self._device()}

    def experiment_split(self, spec) -> str:
        return "test"

    def experiment_synthetic(self, spec) -> bool:
        return False

    def experiment_resources(self, spec) -> Dict[str, Any]:
        return {"device": self._device(), "synthetic_demo": False,
                "gpu": self._device() == "cuda", "rom_budget_mb": C.ROM_BUDGET_MB}

    @staticmethod
    def _device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def evaluate(self, tp, run_id: str, protocol, model_id: str, *, spec=None,
                 resources: Optional[Dict[str, Any]] = None,
                 synthetic: bool = False):
        from algo_agent.schemas import contracts as CT
        from algo_agent.paths import write_json

        version = getattr(protocol, "dataset_version", "") or "v1"
        run_dir = Path(tp.runs) / run_id
        pred_path = run_dir / "pred_test.json"
        if not pred_path.exists():
            legacy = run_dir / "pred_test.json"
            raise FileNotFoundError(f"predictions missing for run {run_id}: {legacy}")

        labels_path = D.eval_labels_path(tp, version)
        if labels_path is None:
            raise FileNotFoundError(f"frozen eval labels missing for version {version}")
        labels = D.load_rows(labels_path)
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        if getattr(protocol, "stage", "legacy") != "legacy":
            import math
            if len(preds) != len(labels) or len({p.get("path") for p in preds}) != len(preds):
                raise ValueError("prediction coverage/uniqueness mismatch")
            if {p.get("path") for p in preds} != {r.get("path") for r in labels}:
                raise ValueError("prediction identities do not match evaluation inputs")
            if any(p.get("score") is None or not math.isfinite(float(p["score"])) for p in preds):
                raise ValueError("nonfinite or missing predicted score")

        quality_only = C.quality_only()
        if quality_only:
            # MOS track: the primary gate is the task's own PLCC threshold; the
            # companion SRCC floor sits at the same value.  Defect gates do not
            # apply (the benchmark has no annotations).
            floors = {"plcc": float(getattr(protocol, "threshold", None) or C.QUALITY_PLCC_FLOOR),
                      "srcc": C.QUALITY_SRCC_FLOOR,
                      "defect_macro_f1": 0.0, "hardcase_macro_f1": 0.0, "grade_agreement": 0.0}
            report = M.evaluate_predictions(labels, preds, floors=floors, quality_only=True)
        else:
            floors = {"plcc": C.SCORE_PLCC_FLOOR, "srcc": C.SCORE_SRCC_FLOOR,
                      "defect_macro_f1": C.DEFECT_MACROF1_FLOOR,
                      "hardcase_macro_f1": C.HARDCASE_F1_FLOOR,
                      "grade_agreement": C.GRADE_AGREEMENT_FLOOR}
            report = M.evaluate_predictions(labels, preds, floors=floors)

        run_result = {}
        rp = run_dir / "result.json"
        if rp.exists():
            try:
                run_result = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                run_result = {}
        params_m = float(run_result.get("rom_params_millions", C.DEPLOY_PARAMS_MILLIONS) or C.DEPLOY_PARAMS_MILLIONS)
        rom = C.rom_estimate(params_m)
        report["rom"] = rom
        report["rom_ok"] = bool(rom["ok"])
        report["rom_mb"] = float(rom["rom_mb"])

        metrics = M.flatten_report(report)
        metrics["dual_output_complete"] = float(bool(preds) and all(
            p.get("score") is not None and set(p.get("defects", {})) == set(C.DEFECTS)
            and all(isinstance(p["defects"][d], dict)
                    and p["defects"][d].get("present") in (0, 1)
                    and p["defects"][d].get("severity") in (0, 1, 2, 3)
                    for d in C.DEFECTS) for p in preds))
        metrics["rom_mb"] = float(rom["rom_mb"])
        metrics["rom_ok"] = 1.0 if rom["ok"] else 0.0
        slices = M.slices_from_report(report)

        threshold = getattr(protocol, "threshold", None)
        value = metrics.get(getattr(protocol, "primary_metric", "defect_macro_f1"))
        passed = None if threshold is None else bool(value is not None and value >= threshold)
        eval_id = f"eval_{run_id}"
        out_dir = Path(tp.evaluation) / "results" / eval_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "iqa_report.json", report)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "slices.json", slices)

        failures = []
        for name, gate in (report.get("gates") or {}).items():
            if not gate:
                failures.append(name)
        result = CT.EvalResult(
            eval_id=eval_id, task_id=tp.slug, model_id=model_id, protocol_id=protocol.protocol_id,
            dataset_version=version, metrics=metrics, slices=slices,
            resources={**(resources or {}), "rom": rom, "device": run_result.get("device", "unknown")},
            failures=failures, passed=passed, synthetic=False,
            stage=getattr(protocol, "stage", "legacy"),
            notes=("frozen IQA protocol; independent metric code (parity-checked with "
                   "iqa_agent.tools.iqa_model.evaluate); " + report.get("coverage_note", "")))
        write_json(out_dir / "eval_result.json", result.model_dump())
        return result

    def model_manifest(self, spec, tp, run_id: str, protocol) -> Dict[str, Any]:
        run_dir = Path(tp.runs) / run_id
        info: Dict[str, Any] = {"env": "local_gpu", "weights_ref": f"runs/{run_id}/pred_test.json",
                               "export": {"format": "predictions+model_card", "text_tower": "dropped"}}
        res = run_dir / "result.json"
        if res.exists():
            try:
                r = json.loads(res.read_text(encoding="utf-8"))
                info["export"]["rom"] = r.get("rom", {})
                info["env"] = "local_gpu" if r.get("device") == "cuda" else "local_cpu"
            except Exception:
                pass
        return info

    # ------------------------------------------------------------------ #
    # analysis / diagnosis  (feedback -> actionable next step)
    # ------------------------------------------------------------------ #
    def diagnose(self, request: Dict[str, Any]) -> Dict[str, Any]:
        metrics = request.get("metrics") or {}
        slices = request.get("slices") or {}
        recs: List[str] = []
        categories: List[str] = []

        support = {d: metrics.get(f"support_{d}") for d in C.DEFECTS}
        unmeasurable = [d for d, s in support.items() if s is not None and float(s) < C.MIN_CLASS_SUPPORT]
        if unmeasurable:
            categories.append("data_coverage_gap")
            recs.append(f"extend the frozen eval with >= {C.MIN_CLASS_SUPPORT} labelled positives for "
                        f"{unmeasurable}; until then their F1 is scored 0 and the all-class Macro-F1 is capped")

        f1 = {d: (slices.get(d, {}) or {}).get("f1") for d in C.DEFECTS}
        measured = {d: v for d, v in f1.items() if v is not None}
        if measured:
            weakest = sorted(measured.items(), key=lambda kv: kv[1])[:2]
            categories.append("weak_defect_head")
            recs.append("weakest defect classes " + ", ".join(f"{d}={v:.3f}" for d, v in weakest)
                        + " -> class-balanced BCE (pos_weight) + per-class threshold tuning")

        plcc = metrics.get("plcc")
        if plcc is not None and plcc < C.SCORE_PLCC_FLOOR:
            categories.append("score_head_underfit")
            recs.append(f"PLCC={plcc:.3f} < {C.SCORE_PLCC_FLOOR}: keep the score head + add ranking loss, "
                        "and note the pseudo-MOS ceiling (score is quantised by severity)")

        hc = metrics.get("hardcase_macro_f1")
        if hc is not None and hc < C.HARDCASE_F1_FLOOR:
            categories.append("hardcase_gap")
            recs.append(f"hardcase Macro-F1={hc:.3f} < {C.HARDCASE_F1_FLOOR}: add hardcase-focused sampling/augmentation")

        macro = metrics.get("defect_macro_f1")
        if macro is not None and macro < C.DEFECT_MACROF1_FLOOR:
            cov = metrics.get("defect_macro_f1_cov")
            recs.append(f"all-class Macro-F1={macro:.3f} (< {C.DEFECT_MACROF1_FLOOR}); "
                        f"supported-class Macro-F1={cov}")

        if not categories:
            categories.append("no_action")
        return {
            "status": "ok", "offline": True, "reproducible": True,
            "category": categories[0], "categories": categories,
            "reason": "; ".join(recs) if recs else "all tracked gates met",
            "recommendations": recs,
            "unmeasurable_defects": unmeasurable,
            "weak_defects": [d for d, _ in sorted(measured.items(), key=lambda kv: kv[1])[:2]] if measured else [],
        }

    # ------------------------------------------------------------------ #
    def assets(self) -> Dict[str, Any]:
        return C.assets_available()


PLUGIN = IqaPlugin()

__all__ = ["IqaPlugin", "PLUGIN"]
