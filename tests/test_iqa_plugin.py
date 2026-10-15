# -*- coding: utf-8 -*-
"""IQA business-plugin tests: domain routing, metrics, dataset isolation, evaluator.

Deliberately torch-free so the plugin contract is verifiable without a GPU: the
heavy method implementations (MobileCLIP2 adapter) are exercised by the real
research loop, not by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algo_agent.config import load_settings
from algo_agent.paths import TaskPaths, ensure_task_dirs
from algo_agent.schemas import contracts as CT

from algorithm_plugins.iqa import config as IC
from algorithm_plugins.iqa import data as ID
from algorithm_plugins.iqa import metrics as IM
from algorithm_plugins.iqa.plugin import PLUGIN

IQA_REPORT = ("轻量双输出 IQA 模型研发：ROM < 60MB fp16，输出 0-100 质量分与 "
              "blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签，"
              "缺陷 Macro-F1 >= 0.85，PLCC/SRCC >= 0.90")


# --------------------------------------------------------------------------- #
# domain routing
# --------------------------------------------------------------------------- #
def test_detect_domain_claims_iqa_text():
    from algorithm_plugins.registry import detect_domain

    assert detect_domain(IQA_REPORT) == "iqa"
    assert detect_domain("复现 IQA v7 基线并改进 flare 检测") == "iqa"


def test_detect_domain_leaves_generic_quality_task_to_demo():
    from algorithm_plugins.registry import detect_domain

    generic = "在移动端对图像做质量分预测，RMSE 不超过 0.3，包大小不超过 20MB"
    assert detect_domain(generic) == "demo_tabular"


def test_registry_loads_iqa_plugin():
    from algorithm_plugins.registry import load_plugin

    plugin = load_plugin("iqa")
    assert plugin is not None
    assert plugin.domain == "iqa"


# --------------------------------------------------------------------------- #
# requirement parsing / validation
# --------------------------------------------------------------------------- #
def test_parse_requirement_carries_report_floors():
    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_iqa", "iqa_demo")
    assert spec.domain == "iqa"
    assert spec.primary_metric.name == "defect_macro_f1"
    assert spec.primary_metric.direction == "max"
    assert spec.primary_metric.threshold == pytest.approx(IC.DEFECT_MACROF1_FLOOR)
    floors = {c.name for c in spec.hard_constraints}
    assert "rom_mb_fp16" in floors and "dual_output" in floors
    rom = next(c for c in spec.hard_constraints if c.name == "rom_mb_fp16")
    assert rom.value == pytest.approx(IC.ROM_BUDGET_MB) and rom.blocking
    assert spec.demo is False and spec.synthetic is False
    assert any("white_balance" in u for u in spec.unknowns)


def test_validate_requirement_flags_missing_rom():
    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_iqa", "iqa_demo")
    assert PLUGIN.validate_requirement(spec.model_dump()) == []
    broken = spec.model_dump()
    broken["hard_constraints"] = []
    assert PLUGIN.validate_requirement(broken)


def test_validate_dataset_requires_eval_split_and_checksums():
    ok = {"splits": {"eval": 610}, "checksums": {"a": "b"}}
    assert PLUGIN.validate_dataset(ok) == []
    assert PLUGIN.validate_dataset({"splits": {"eval": 5}, "checksums": {}})


def test_deployment_scope_states_text_tower_dropped():
    scope = PLUGIN.deployment_scope()
    assert scope["rom_budget_mb"] == IC.ROM_BUDGET_MB
    assert "dropped" in scope["text_tower"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _labeled(paths, defects_present, scores, hardcase=()):
    rows = []
    for i, path in enumerate(paths):
        rows.append({
            "path": path, "image": Path(path).name, "score": scores[i],
            "hardcase": path in hardcase,
            "defects": {d: {"present": int(d in defects_present[i]), "severity": 1}
                        for d in IC.DEFECTS},
        })
    return rows


def _preds(labeled, score_shift=0.0, drop=()):
    out = []
    for r in labeled:
        out.append({
            "path": r["path"], "image": r["image"], "score": r["score"] + score_shift,
            "defects": {d: {"present": int(bool(r["defects"][d]["present"]) and d not in drop)}
                        for d in IC.DEFECTS},
        })
    return out


def test_metrics_perfect_predictions_show_structural_macro_ceiling():
    # 20 rows, >=5 positives for every measurable class; white_balance has 0
    # positives, so the all-class macro-F1 is structurally capped at 0.8 even
    # with perfect per-class predictions (this is the honest frozen-benchmark
    # reality the report calls out).
    cycle = ["blur", "over_exp", "under_exp", "flare"]
    defects = [[cycle[i % 4]] if i % 5 != 4 else [] for i in range(40)]
    scores = [34.0 + (i % 7) * 9 for i in range(40)]
    paths = [f"img/{i}.jpg" for i in range(40)]
    labeled = _labeled(paths, defects, scores, hardcase={"img/1.jpg", "img/2.jpg"})
    report = IM.evaluate_predictions(labeled, _preds(labeled))
    assert all(report["per_defect_f1"][d] == pytest.approx(1.0)
               for d in ("blur", "over_exp", "under_exp", "flare"))
    assert report["class_support"]["white_balance"] == 0
    assert report["defect_macro_f1"] == pytest.approx(0.8)
    assert report["defect_macro_f1_cov"] == pytest.approx(1.0)
    assert report["plcc"] == pytest.approx(1.0)
    assert report["srcc"] == pytest.approx(1.0)
    assert report["grade_agreement"] == pytest.approx(1.0)
    # all-class gate cannot be met on this split; the covered-class gate can
    assert report["gates"]["defect_ok"] is False
    assert report["gates"]["defect_ok_covered"] is True
    assert report["pass"] is False


def test_metrics_missing_prediction_counts_as_skipped():
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    labeled = _labeled(paths, [["blur"], [], ["over_exp"]], [88.0, 46.0, 66.0])
    report = IM.evaluate_predictions(labeled, _preds(labeled)[:2])
    assert report["n_matched"] == 2


def test_metrics_unmeasurable_class_caps_all_class_macro():
    paths = [f"p{i}.jpg" for i in range(10)]
    defects = [["blur"] if i % 2 else [] for i in range(10)]
    scores = [float(30 + i * 5) for i in range(10)]
    labeled = _labeled(paths, defects, scores)
    report = IM.evaluate_predictions(labeled, _preds(labeled))
    assert report["class_support"]["white_balance"] == 0
    assert report["class_support"]["flare"] == 0
    assert report["defect_macro_f1"] < report["defect_macro_f1_cov"]
    assert "not measurable" in report["coverage_note"]


def test_flatten_and_slices_shapes():
    paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    defects = [["blur"], ["blur"], [], ["flare"], []]
    labeled = _labeled(paths, defects, [88.0, 66.0, 46.0, 28.0, 88.0])
    report = IM.evaluate_predictions(labeled, _preds(labeled))
    flat = IM.flatten_report(report)
    assert "defect_macro_f1" in flat and "f1_blur" in flat and "support_blur" in flat
    slices = IM.slices_from_report(report)
    assert set(slices) == set(IC.DEFECTS)
    assert {"f1", "precision", "recall", "gt_pos"} <= set(slices["blur"])


def test_rom_estimate_boundary():
    assert IC.rom_estimate(11.46)["ok"] is True
    assert IC.rom_estimate(30.0)["ok"] is False


# --------------------------------------------------------------------------- #
# dataset export / isolation
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tp(tmp_path) -> TaskPaths:
    settings = load_settings(str(tmp_path))
    paths = TaskPaths(settings=settings, slug="iqa_unit")
    ensure_task_dirs(paths)
    return paths


@pytest.mark.skipif(not IC.dev_jsonl().exists(), reason="frozen IQA benchmark not present")
def test_export_benchmark_isolates_eval_labels(tp):
    manifest = ID.export_benchmark(tp, "v1", seed=7)
    release = tp.release_dir("v1")
    assert (release / "dev.jsonl").exists()
    assert (release / "eval_inputs.jsonl").exists()
    assert (release / "dev_split.json").exists()

    inputs = ID.load_rows(release / "eval_inputs.jsonl")
    assert inputs and all(set(r) <= {"path", "image"} for r in inputs), "eval inputs must be label-free"

    labels_path = ID.eval_labels_path(tp, "v1")
    assert labels_path is not None and labels_path.exists()
    assert "protocols" in str(labels_path), "labels must live outside the method data dir"

    assert manifest["splits"]["dev"] == 410
    assert manifest["splits"]["eval"] == 610
    assert manifest["synthetic"] is False
    # plugin info shape carries checksum_files; the graph hashes them into a
    # DatasetManifest.checksums dict that validate_dataset inspects.
    assert manifest["checksum_files"]
    assert PLUGIN.validate_dataset({"splits": {"eval": 610}, "checksums": {"dev.jsonl": "abc"}}) == []


# --------------------------------------------------------------------------- #
# coding hook: materialised method entry + evaluator contract
# --------------------------------------------------------------------------- #
def test_materialise_method_writes_runnable_entry(tp):
    rel = PLUGIN.materialise_method(tp.root, "multitask_adapter")
    assert rel == "model_pool/multitask_adapter/method.py"
    entry = tp.root / rel
    assert entry.exists()
    from algorithm_plugins.common.plugin import load_method
    module = load_method(str(entry))
    assert callable(module.run)


def test_materialise_method_handles_variants(tp):
    for method in ("multitask_adapter_balanced", "multitask_adapter_long", "measure_threshold"):
        rel = PLUGIN.materialise_method(tp.root, method)
        assert rel and (tp.root / rel).exists()


def test_materialise_unknown_method_returns_none(tp):
    assert PLUGIN.materialise_method(tp.root, "does_not_exist") is None


def _tiny_release(tp, version="vT"):
    release = tp.release_dir(version)
    release.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(6):
        labels.append({
            "path": f"eval/{i}.jpg", "image": f"{i}.jpg", "score": 88.0 - i * 10,
            "hardcase": i < 2,
            "defects": {d: {"present": int(d == "blur" and i % 2 == 0)} for d in IC.DEFECTS},
        })
    (release / "eval_inputs.jsonl").write_text(
        "\n".join(json.dumps({"path": r["path"], "image": r["image"]}) for r in labels),
        encoding="utf-8")
    proto = tp.evaluation / "protocols" / version
    proto.mkdir(parents=True, exist_ok=True)
    (proto / "eval_labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in labels), encoding="utf-8")
    return labels


def test_evaluate_emits_frozen_metrics_and_rom(tp):
    version = "vT"
    labels = _tiny_release(tp, version)
    run_id = "run_vT_probe"
    run_dir = tp.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preds = [{"path": r["path"], "image": r["image"], "score": r["score"],
              "defects": {d: {"present": r["defects"][d]["present"]} for d in IC.DEFECTS}}
             for r in labels]
    (run_dir / "pred_test.json").write_text(json.dumps(preds), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"rom_params_millions": 11.46}), encoding="utf-8")

    protocol = CT.EvalProtocol(protocol_id="iqa_frozen_vT", primary_metric="defect_macro_f1",
                               direction="max", threshold=0.85, frozen=True, dataset_version=version)
    result = PLUGIN.evaluate(tp, run_id, protocol, "m_probe", spec=None,
                             resources={"device": "cpu"}, synthetic=False)
    assert isinstance(result, CT.EvalResult)
    # only blur has positives here, so the all-class macro is 1/5 by convention
    assert result.metrics["f1_blur"] == pytest.approx(1.0)
    assert result.metrics["defect_macro_f1"] == pytest.approx(0.2)
    assert result.metrics["rom_mb"] == pytest.approx(22.92)
    assert result.metrics["rom_ok"] == 1.0
    assert set(result.slices) == set(IC.DEFECTS)
    assert result.passed is False
    assert (tp.evaluation / "results" / result.eval_id / "iqa_report.json").exists()


def test_evaluate_missing_predictions_raises(tp):
    _tiny_release(tp, "vM")
    protocol = CT.EvalProtocol(protocol_id="iqa_frozen_vM", primary_metric="defect_macro_f1",
                               direction="max", threshold=0.85, dataset_version="vM")
    with pytest.raises(FileNotFoundError):
        PLUGIN.evaluate(tp, "run_missing", protocol, "m", spec=None, resources={}, synthetic=False)


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #
def test_diagnose_surfaces_unmeasurable_and_weak_classes():
    out = PLUGIN.diagnose({"metrics": {"plcc": 0.5, "hardcase_macro_f1": 0.4,
                                       "defect_macro_f1": 0.42, "defect_macro_f1_cov": 0.53,
                                       "support_white_balance": 0.0, "support_flare": 9.0},
                           "slices": {"blur": {"f1": 0.2}, "over_exp": {"f1": 0.9},
                                      "under_exp": {"f1": 0.6}, "flare": {"f1": 0.0},
                                      "white_balance": {"f1": 0.0}}})
    assert "data_coverage_gap" in out["categories"]
    assert "white_balance" in out["unmeasurable_defects"]
    assert set(out["weak_defects"]) == {"flare", "white_balance"}
    assert "hardcase_gap" in out["categories"]
    assert out["recommendations"]


# --------------------------------------------------------------------------- #
# contract hooks are actually wired into the graph
# --------------------------------------------------------------------------- #
def test_intake_node_runs_plugin_requirement_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from algo_agent.graph.nodes.intake import intake_node
    from algo_agent.storage.repository import Repository

    settings = load_settings(str(tmp_path))
    repo = Repository(settings.db_path)
    repo.create_task(PLUGIN.parse_requirement(IQA_REPORT, "t_val", "t_val"))
    ctx = SimpleNamespace(repo=repo, settings=settings)
    intake_node({"task_id": "t_val", "task_slug": "t_val", "raw_text": IQA_REPORT}, ctx)

    events = repo.read_events("t_val")
    parsed = next(e for e in events if "requirement parsed" in e["message"])
    assert parsed["data"]["validation_problems"] == []

    monkeypatch.setattr(PLUGIN, "validate_requirement", lambda spec: ["rom hard constraint missing"])
    repo.create_task(PLUGIN.parse_requirement(IQA_REPORT, "t_val2", "t_val2"))
    intake_node({"task_id": "t_val2", "task_slug": "t_val2", "raw_text": IQA_REPORT},
                SimpleNamespace(repo=repo, settings=settings))
    events2 = repo.read_events("t_val2")
    assert any(e["level"] == "warning" and e["data"].get("problems") == ["rom hard constraint missing"]
               for e in events2)


def test_validate_release_records_dataset_warnings(tmp_path):
    from types import SimpleNamespace

    from algo_agent.graph.nodes.assets import _validate_release

    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_ds", "t_ds")
    manifest = CT.DatasetManifest(dataset_id="iqa_dual_output", version="v1", source="x",
                                  splits={"eval": 5}, checksums={})
    captured = []
    ctx = SimpleNamespace(repo=SimpleNamespace(
        append_event=lambda *a, **k: captured.append((a, k))))
    problems = _validate_release(ctx, "t_ds", spec, manifest)
    assert problems, "undersized/checksum-less release must be flagged"
    assert captured and captured[0][1]["level"] == "warning"


def test_validate_release_passes_a_good_manifest(tmp_path):
    from types import SimpleNamespace

    from algo_agent.graph.nodes.assets import _validate_release

    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_ds", "t_ds")
    manifest = CT.DatasetManifest(dataset_id="iqa_dual_output", version="v1",
                                  splits={"eval": 610}, checksums={"eval_labels.jsonl": "abc"})
    captured = []
    ctx = SimpleNamespace(repo=SimpleNamespace(
        append_event=lambda *a, **k: captured.append((a, k))))
    assert _validate_release(ctx, "t_ds", spec, manifest) == []
    assert captured == []


# --------------------------------------------------------------------------- #
# closed-loop hooks: constraints, model selection, history-driven feedback
# --------------------------------------------------------------------------- #
def test_check_constraints_flags_rom_and_missing_defect_head():
    assert PLUGIN.check_constraints(None, {"rom_mb": 22.9, "rom_ok": 1.0, "f1_blur": 0.5}) == []
    bad = PLUGIN.check_constraints(None, {"rom_mb": 80.0, "rom_ok": 0.0})
    assert any("ROM" in b for b in bad)
    assert any("dual output" in b for b in bad)


def test_select_best_prefers_feasible_then_covered_macro():
    cands = [
        {"model_id": "m_high_but_infeasible", "passed": False,
         "metrics": {"defect_macro_f1_cov": 0.95, "rom_ok": 1.0}},
        {"model_id": "m_feasible", "passed": True,
         "metrics": {"defect_macro_f1_cov": 0.7, "rom_ok": 1.0}},
        {"model_id": "m_feasible_best", "passed": True,
         "metrics": {"defect_macro_f1_cov": 0.82, "rom_ok": 1.0, "plcc": 0.6}},
    ]
    assert PLUGIN.select_best(None, cands) == "m_feasible_best"
    assert PLUGIN.select_best(None, []) is None


def test_weak_from_history_ignores_unmeasurable_classes():
    history = [{
        "metrics": {"support_blur": 90.0, "support_flare": 9.0, "support_white_balance": 0.0},
        "slices": {"blur": {"f1": 0.55}, "over_exp": {"f1": 0.9}, "under_exp": {"f1": 0.7},
                   "flare": {"f1": 0.4}, "white_balance": {"f1": 0.0}},
    }]
    weak = PLUGIN._weak_from_history(history)
    assert "blur" in weak and "flare" in weak
    assert "white_balance" not in weak  # 0 support -> coverage gap, not a weakness
    assert "over_exp" not in weak
    assert PLUGIN._weak_from_history([]) == []


def test_design_uses_history_when_no_user_feedback():
    history = [{
        "metrics": {"support_blur": 90.0, "support_flare": 9.0, "support_white_balance": 0.0},
        "slices": {"blur": {"f1": 0.55}, "over_exp": {"f1": 0.9}, "under_exp": {"f1": 0.7},
                   "flare": {"f1": 0.4}, "white_balance": {"f1": 0.0}},
    }]
    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_h", "t_h")
    res = PLUGIN.design(spec, iteration=2, feedback=[], history=history)
    assert [c.method for c in res["candidates"]] == ["multitask_adapter_balanced"]


def test_design_escalates_instead_of_repeating_tried_variants():
    spec = PLUGIN.parse_requirement(IQA_REPORT, "t_esc", "t_esc")
    weak = {"support_flare": 9.0, "support_white_balance": 0.0}
    slices = {"blur": {"f1": 0.5}, "flare": {"f1": 0.3}}
    hist1 = [{"metrics": weak, "slices": slices, "methods": ["measure_threshold", "multitask_adapter"]}]
    first = [c.method for c in PLUGIN.design(spec, iteration=2, feedback=[], history=hist1)["candidates"]]
    assert first == ["multitask_adapter_balanced"]

    hist2 = hist1 + [{"metrics": weak, "slices": slices, "methods": ["multitask_adapter_balanced"]}]
    second = [c.method for c in PLUGIN.design(spec, iteration=3, feedback=[], history=hist2)["candidates"]]
    assert second == ["multitask_adapter_long"]


def test_history_with_metrics_carries_methods_and_metrics():
    from types import SimpleNamespace

    from algo_agent.graph.nodes.research import _history_with_metrics

    plan = SimpleNamespace(candidates=[CT.Candidate(candidate_id="c", method="multitask_adapter")],
                           data_recipe=CT.DataRecipe(recipe_id="r"))
    ev = CT.EvalResult(eval_id="e1", task_id="t", model_id="m_x", protocol_id="p",
                       dataset_version="v1", metrics={"defect_macro_f1": 0.5},
                       slices={"blur": {"f1": 0.4}}, failures=[], passed=False)
    repo = SimpleNamespace(list_evals=lambda t: [ev],
                           list_plans=lambda t: [{"plan_id": "p001", "version": "v1"}],
                           get_plan=lambda pid: plan, list_jobs=lambda t: [])
    spec = SimpleNamespace(primary_metric=SimpleNamespace(name="defect_macro_f1", direction="max"),
                           task_slug="t")
    from algo_agent.config import Settings
    history = _history_with_metrics(SimpleNamespace(repo=repo, settings=Settings(root=Path("/tmp"))), "t", spec)
    assert history[0]["methods"] == ["multitask_adapter"]
    assert history[0]["metrics"]["defect_macro_f1"] == 0.5
    assert history[0]["slices"]["blur"]["f1"] == 0.4


def test_history_picks_best_eval_on_ties_using_secondary_gates():
    """Two runs can share the primary metric; the status must show the better one."""
    from types import SimpleNamespace

    from algo_agent.graph.nodes.research import _history_with_metrics

    def _ev(mid, cov, plcc):
        return CT.EvalResult(eval_id=f"e_{mid}", task_id="t", model_id=mid, protocol_id="p",
                             dataset_version="v1",
                             metrics={"defect_macro_f1": 0.424, "defect_macro_f1_cov": cov,
                                      "plcc": plcc},
                             slices={}, failures=[], passed=False)

    evals = [_ev("rule", 0.53, 0.4195), _ev("adapter", 0.53, 0.4383)]
    repo = SimpleNamespace(list_evals=lambda t: evals,
                           list_plans=lambda t: [{"plan_id": "p001", "version": "v1"}],
                           get_plan=lambda pid: None, list_jobs=lambda t: [])
    spec = SimpleNamespace(primary_metric=SimpleNamespace(name="defect_macro_f1", direction="max"),
                           task_slug="t")
    from algo_agent.config import Settings
    history = _history_with_metrics(SimpleNamespace(repo=repo, settings=Settings(root=Path("/tmp"))), "t", spec)
    assert history[0]["best_model"] == "adapter"
    assert history[0]["metrics"]["plcc"] == 0.4383
