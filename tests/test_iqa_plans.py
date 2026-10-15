# -*- coding: utf-8 -*-
"""IQA design-report tests: precision status, optimisation plan, data plan.

These pin the three artefacts the task owner asked for so a future refactor
cannot silently drop them again.
"""

from __future__ import annotations

from algorithm_plugins.iqa import config as IC
from algorithm_plugins.iqa import plans as P
from algorithm_plugins.iqa.plugin import PLUGIN

IQA_REPORT = ("轻量双输出 IQA 模型研发：ROM < 60MB fp16，输出 0-100 质量分与 "
              "blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签，"
              "缺陷 Macro-F1 >= 0.85，PLCC/SRCC >= 0.90，分档一致率 >= 0.90")


def _spec():
    return PLUGIN.parse_requirement(IQA_REPORT, "t_plan", "t_plan")


def test_precision_status_uses_frozen_baseline_before_any_run():
    status = P.precision_status(history=[], iteration=1)
    assert status["source"].startswith("frozen registry baseline")
    metrics = {r["metric"]: r for r in status["rows"]}
    assert metrics["defect_macro_f1"]["floor"] == IC.DEFECT_MACROF1_FLOOR
    assert metrics["plcc"]["floor"] == IC.SCORE_PLCC_FLOOR
    # the frozen baseline really fails the floors -> it must show up as failing
    assert "plcc" in status["failing"]
    assert any("white_balance" in n for n in status["notes"])


def test_precision_status_reads_metrics_from_history_and_default_judges_floors():
    history = [{"metrics": {"defect_macro_f1": 0.9, "defect_macro_f1_cov": 0.92,
                            "plcc": 0.95, "srcc": 0.94, "hardcase_macro_f1": 0.9,
                            "grade_agreement": 0.93, "rom_mb": 22.9}}]
    status = P.precision_status(history=history, iteration=2)
    assert status["failing"] == []
    assert status["best_model"] is None
    for row in status["rows"]:
        assert row["verdict"] == "pass"


def test_precision_status_surfaces_best_model_from_history():
    history = [{"best_model": "m_adapter", "metrics": {"plcc": 0.4}}]
    status = P.precision_status(history=history, iteration=2)
    assert status["best_model"] == "m_adapter"
    assert status["source"] == "agent run"


def test_precision_status_fails_rom_budget_when_too_large():
    history = [{"metrics": {"rom_mb": IC.ROM_BUDGET_MB + 10}}]
    status = P.precision_status(history=history, iteration=2)
    assert "rom_mb" in status["failing"]


def test_eval_support_matches_frozen_benchmark():
    support = P.eval_support()
    if not IC.eval_jsonl().exists():
        return
    assert support["blur"] == 150
    assert support["flare"] == 9
    assert support["white_balance"] == 0


def test_optimization_plan_covers_model_loss_and_training_strategy():
    status = P.precision_status(history=[], iteration=1)
    items = P.optimization_plan(["flare", "under_exp"], status, 1)
    tracks = " ".join(i["track"] for i in items)
    assert "loss" in tracks
    assert "training strategy" in tracks
    ids = {i["id"] for i in items}
    assert {"opt-score-rank-loss", "opt-defect-asymmetric-loss",
            "opt-train-strategy"} <= ids
    for item in items:
        for key in ("change", "why_now", "expected_effect", "risk"):
            assert item[key], f"{item['id']} missing {key}"


def test_data_plan_has_annotation_and_augmentation_and_names_local_model():
    status = P.precision_status(history=[], iteration=1)
    plan = P.data_plan(["flare", "white_balance"], status)
    assert plan["annotation"]["local_model"]["model_id"] == "qwen3.8-27b"
    assert plan["annotation"]["local_model"]["path"] == "local_largemodel/qwen3.8-27b"
    assert plan["annotation"]["review_protocol"]["rare_classes"]
    # white_balance has 0 positives -> non-zero gap and a concrete target
    wb = next(t for t in plan["targets"] if t["defect"] == "white_balance")
    assert wb["current_positives"] == 0
    assert wb["target_positives"] >= IC.MIN_CLASS_SUPPORT
    assert wb["gap"] == wb["target_positives"]
    aug = plan["augmentation"]["per_class"]
    assert "white_balance" in aug and "flare" in aug
    assert any("train split" in r for r in plan["augmentation"]["rules"])


def test_design_report_renders_all_three_sections():
    res = PLUGIN.design(_spec(), iteration=1)
    md = res["report_markdown"]
    assert "## 1. 当前产出精度状态" in md
    assert "## 2. 模型 & 损失函数 & 训练策略优化计划" in md
    assert "## 3. 数据标注调优计划 & 数据扩增计划" in md
    assert "qwen3.8-27b" in md
    assert "数据扩增计划" in md
    assert res["precision_status"]["rows"]
    assert res["optimization_plan"]
    assert res["data_plan"]["targets"]
