#!/usr/bin/env python3
"""Play an operator through the real dialogue API; never supply model outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtimes/agent_runtime/src"))
sys.path.insert(0, str(ROOT))

from algo_agent.cli.client import ControlClient
from algo_agent.config import load_settings
from algo_agent.paths import TaskPaths, read_json, register_task_member, write_json
from algo_agent.providers import readiness_report


REQUIREMENT = (
    "轻量双输出 IQA 自主研发验证。使用当前本地 IQA pseudo-MOS 基准，"
    "输出 0-100 质量分和 blur/over_exp/under_exp/flare/white_balance 五类缺陷。"
    "主指标 PLCC >= 0.85，SRCC >= 0.85，defect_macro_f1 >= 0.85，"
    "grade_agreement >= 0.90，hardcase_macro_f1 >= 0.85，ROM < 60 MB。"
    "自主研究模型结构、训练策略、loss 和数据方案，产生可执行的代码与数据实验，"
    "依据开发集结果再研究、改进并迭代，不等待审批。"
    "不能通过降低指标、替换测试集、删除困难样本或修改测试真值取得达标。"
    "原始样本及历史产物只读，训练集增强和少量本地弱标注可以实验。"
    "历史测试集已暴露，不能宣称本次复测是新的独立业务验收。"
    "即使某些门槛受标签覆盖限制，仍在预算内测量代码和数据干预对可测指标的影响，"
    "最终明确哪些达标、哪些未达标、哪些不可测。历史模型结果不代表理论上限。"
)


class DialogueValidation:
    def __init__(self, args):
        self.args = args
        self.settings = load_settings(str(ROOT))
        self.settings.env.update({
            "ALGO_STATE_DIR": str(Path(args.state_dir).expanduser().resolve()),
            "ALGO_OFFLINE": "0",
            "ALGO_CODEX_ALLOW_REAL": "1",
            "ALGO_SEARCH_BACKEND": args.search_backend,
        })
        self.settings.workflow["memory"] = {"share_lessons": False}
        register_task_member(self.settings, "iqa", args.task_id, args.track)
        self.tp = TaskPaths(self.settings, args.task_id)
        if self.tp.root.exists() and any(self.tp.root.iterdir()):
            raise ValueError("validation track already contains artifacts; choose a new track")
        self.client = ControlClient(self.settings, force_mode="embedded", extras={})
        self.ctrl = self.client._controller()
        self.turns = []
        self.checks = {}

    def say(self, text):
        started = datetime.now(timezone.utc).isoformat()
        print(f"[user] {text}", flush=True)
        try:
            result = self.client.message(self.args.task_id, text, conversation_id="validation_operator")
        except Exception as exc:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)[:1000]}}
        turn = {"at": started, "user": text, "agent": result}
        self.turns.append(turn)
        write_json(self.tp.docs / "validation" / "dialogue.json", self.turns)
        print("[agent] " + json.dumps(result, ensure_ascii=False, default=str)[:5500], flush=True)
        return result

    def create(self):
        budget = {
            "max_research_calls": self.args.iterations, "max_iterations": self.args.iterations,
            "max_research_steps": 12, "max_candidates": 2,
            "max_coding_calls": 8, "max_fix_attempts": 2,
            "max_experiments": 9, "max_epochs": 40,
            "max_wall_clock_s": self.args.wall_seconds, "max_cloud_tokens": 400000,
            "max_no_improve": self.args.iterations,
            "max_data_operations": 8, "max_added_samples": 240,
            "max_annotation_samples": 4, "max_output_file_mb": 2048,
        }
        spec = {
            "task_id": self.args.task_id, "task_slug": self.args.task_id, "domain": "iqa",
            "business": REQUIREMENT, "inputs": ["rgb_image_256"],
            "outputs": ["quality_score_0_100", "defect_multilabel_5"],
            "target_hardware": "mobile", "precision": "fp16", "auto_process": True,
            "primary_metric": {"name": "plcc", "direction": "max", "threshold": .85},
            "secondary_metrics": [
                {"name": "srcc", "direction": "max", "threshold": .85},
                {"name": "defect_macro_f1", "direction": "max", "threshold": .85},
                {"name": "grade_agreement", "direction": "max", "threshold": .90},
                {"name": "hardcase_macro_f1", "direction": "max", "threshold": .85},
                {"name": "rom_mb", "direction": "min", "threshold": 60, "strict": True},
            ],
            "hard_constraints": [
                {"name": "rom_mb_fp16", "value": 60, "unit": "MB"},
                {"name": "dual_output", "value": "score+defects"},
            ],
            "research_queries": ["image quality assessment"],
            "unknowns": [
                "Historical test benchmark is already exposed; this run is a workflow/replication validation.",
                "Pseudo-MOS is not independently collected human MOS.",
                "On-device latency and packaged ROM are not measured; ROM is an fp16 parameter estimate.",
            ],
            "budget": budget, "data_sources": [],
        }
        result = self.client.create_task(REQUIREMENT, self.args.task_id, task_spec=spec,
                                         auto_process=True, run=False)
        self.turns.append({"at": datetime.now(timezone.utc).isoformat(),
                           "user": REQUIREMENT, "transport": "create_task with explicit TaskSpec",
                           "agent": result})
        write_json(self.tp.docs / "validation" / "setup.json", {
            "task_spec": self.ctrl.repo.get_spec(self.args.task_id).model_dump(),
            "readiness": readiness_report(self.settings), "state_dir": str(self.settings.state_dir),
            "mode": "live providers; simulated operator; no fabricated agent responses",
            "holdout_fresh": False, "global_credentials_modified": False,
        })

    def run(self):
        self.create()
        self.say("请确认这个任务的完整验收条件。模型、训练策略、loss、数据各方面如何展开研究？"
                 "当前尚未运行实验，请不要引用历史成绩冒充本轮结果。")
        before = self.ctrl.repo.get_state(self.args.task_id)
        self.say("如何暂停任务？")
        after = self.ctrl.repo.get_state(self.args.task_id)
        self.checks["question_did_not_pause_or_start"] = before.phase == after.phase and not after.paused
        self.say("请研究并比较模型结构、训练方案、loss 和训练数据增强的干预。"
                 "首轮请包含一个可复现基线与一个需要实际编码、保存权重的新方法，"
                 "并实施一个小规模训练数据增强实验。后续用实测失败项驱动下一轮研究，"
                 "有价值时检验排序损失或类别平衡策略，并保留消融；不降低验收标准。")
        self.say("暂停任务")
        self.checks["explicit_pause"] = self.ctrl.repo.get_state(self.args.task_id).paused
        self.say("恢复任务")
        self.checks["explicit_resume"] = not self.ctrl.repo.get_state(self.args.task_id).paused
        seen_events, discussed_plans, discussed_iterations = 0, set(), set()
        started = time.monotonic()
        last_progress = 0
        while True:
            state = self.ctrl.state(self.args.task_id)
            events = self.ctrl.events(self.args.task_id, after=seen_events)
            for event in events:
                seen_events = max(seen_events, event["event_seq"])
                print(f"[event {seen_events}] {event['kind']}: {event['message']}", flush=True)
            if state["phase"] in {"accepted", "stopped", "failed", "awaiting_input"}:
                break
            plan_id = state.get("active_plan_id")
            if plan_id and plan_id not in discussed_plans:
                discussed_plans.add(plan_id)
                self.say("请解释当前计划的模型结构、训练策略、loss、数据操作、对照实验和失败条件。"
                         "请区分已运行结果与预期收益。")
            iteration = int(state.get("iteration") or 0)
            if iteration >= 2 and iteration not in discussed_iterations:
                discussed_iterations.add(iteration)
                self.say("上一轮有哪些实测改进和失败？这些结果如何影响本轮研究、模型、loss、训练及数据计划？")
            if time.monotonic() - started > self.args.wall_seconds + 30:
                self.ctrl.cancel(self.args.task_id)
                break
            if time.monotonic() - last_progress > 30:
                print(f"[progress] phase={state['phase']} iteration={iteration} plan={plan_id}", flush=True)
                last_progress = time.monotonic()
            time.sleep(1)
        self.ctrl.wait(self.args.task_id)
        self.say("本轮最终是否满足全部 IQA 验收条件？请逐项给出本轮实测值、门槛、差距与证据，"
                 "说明模型、训练、loss、数据方案分别有哪些实际执行，以及尚未验证的环节。")
        self.client.export(self.args.task_id)
        return self.report()

    def report(self):
        tid = self.args.task_id
        spec, state = self.ctrl.repo.get_spec(tid), self.ctrl.state(tid)
        evaluations = self.ctrl.repo.list_evals(tid)
        coding = [read_json(p) for p in self.tp.coding_records.glob("*/result.json")]
        plans = [self.ctrl.repo.get_plan(p["plan_id"]) for p in self.ctrl.repo.list_plans(tid)]
        jobs = self.ctrl.repo.list_jobs(tid)
        research = [read_json(p) for p in sorted((self.tp.docs / "research").glob("research_iter*.json"))]
        events = self.ctrl.events(tid, after=0)
        research_events = [e for e in events if e["kind"] == "agent.research_design"]
        models = self.ctrl.repo.list_models(tid)
        chosen = [e for e in evaluations if e.model_id == state.get("best_model_id") and e.stage != "final"]
        development = [e for e in evaluations if e.stage == "development"]
        if not chosen and development:
            chosen = [max(development, key=lambda e: e.metrics.get("plcc", -1))]
        best = chosen[-1] if chosen else None
        checks = {
            **self.checks,
            "live_cloud_design_completed": any(r.get("status") == "completed" and not r.get("offline")
                                               for r in research),
            "live_search_returned_refs": any(
                t.get("tool") == "search_sources" and t.get("used_search_service") and t.get("refs")
                for r in [*research, *(e["data"] for e in research_events)]
                for t in r.get("tool_calls", [])),
            "multiple_research_iterations": len(research) >= 2,
            "live_code_applied": any(r.get("tool") in {"codex_cli", "codex_ssh"} and r.get("status") == "completed"
                                     and r.get("changed_files") for r in coding),
            "data_operations_executed": any(
                read_json(p).get("provenance") for p in (self.tp.data / "releases").glob("v*/manifest.json")),
            "experiments_succeeded": any(j.status == "succeeded" for j in jobs),
            "independent_development_metrics": any(e.stage == "development" for e in evaluations),
            "saved_weights": any(m.weights_hash for m in models),
            "final_evaluation_executed": any(e.stage == "final" for e in evaluations),
        }
        targets = [{
            "name": t.name, "threshold": t.threshold, "direction": t.direction, "strict": t.strict,
            "development_value": best.metrics.get(t.name) if best else None,
            "development_meets": t.meets(best.metrics[t.name]) if best and t.name in best.metrics else None,
        } for t in [spec.primary_metric, *spec.secondary_metrics]]
        summary = {
            "task_id": tid, "track": str(self.tp.root), "state": state, "checks": checks,
            "acceptance": {"agent_phase": state["phase"], "fresh_holdout": False,
                           "new_independent_business_acceptance": False,
                           "reason": "Historical benchmark already exposed; no new holdout supplied."},
            "targets": targets, "budget": self.ctrl.budget(tid),
            "plans": [p.model_dump() for p in plans], "coding": coding, "events": events,
            "jobs": [j.model_dump() for j in jobs], "evaluations": [e.model_dump() for e in evaluations],
        }
        directory = self.tp.docs / "validation"
        write_json(directory / "summary.json", summary)
        transcript = ["# IQA Live Agent Dialogue", "", "Operator utterances are scripted. Agent responses are real."]
        for turn in self.turns:
            transcript.extend(["", f"## User ({turn['at']})", turn["user"], "", "### Agent",
                               turn["agent"].get("answer") or
                               "````json\n" + json.dumps(turn["agent"], ensure_ascii=False, indent=2) + "\n````"])
        (directory / "dialogue.md").write_text("\n\n".join(transcript) + "\n", encoding="utf-8")
        report = [
            "# IQA Dialogue Validation",
            f"Task: `{tid}`; phase: `{state['phase']}`; stop reason: {state.get('stop_reason')}",
            "This is a live-provider workflow/replication run on an already exposed benchmark, not fresh business acceptance.",
            "## Observed Workflow Checks",
            *[f"- {key}: `{bool(value)}`" for key, value in checks.items()],
            "## Measured Development Gates",
            "````json", json.dumps(targets, ensure_ascii=False, indent=2), "````",
            "## Evidence",
            "- [Actual conversation](dialogue.md)",
            "- [Machine-readable results and plans](summary.json)",
            "- [Initial frozen contract and readiness](setup.json)",
            "- [Research knowledge](../knowledge/README.md)",
            "No flag above proves research quality, causal attribution or production readiness. "
            "Inspect the real research, code diff, training logs and evaluation records.",
        ]
        (directory / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        print("[summary] " + json.dumps({"phase": state["phase"], "checks": checks,
                                         "report": str(directory / "report.md")}, ensure_ascii=False), flush=True)
        return 0 if state["phase"] == "accepted" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--live", action="store_true", help="allow real cloud/Codex calls for this bounded run")
    parser.add_argument("--task-id", default=f"iqa_dialogue_{stamp}")
    parser.add_argument("--track", default=f"dialogue_{stamp}")
    parser.add_argument("--state-dir", default=f"/root/.algo_agent/validations/iqa_dialogue_{stamp}")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--wall-seconds", type=int, default=2700)
    parser.add_argument("--search-backend", choices=["arxiv", "disabled"], default="arxiv")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required: this script intentionally calls real research/coding providers")
    os.environ.update(QAA_ROOT=str(ROOT.parent), IQA_BENCHMARK="pseudo", CUDA_VISIBLE_DEVICES="0",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                      PYTHONDONTWRITEBYTECODE="1")
    validation = DialogueValidation(args)
    try:
        return validation.run()
    except KeyboardInterrupt:
        validation.ctrl.cancel(args.task_id)
        validation.ctrl.wait(args.task_id, timeout=20)
        validation.report()
        return 130
    except Exception:
        if validation.ctrl.repo.task_exists(args.task_id):
            if validation.ctrl.repo.get_state(args.task_id).phase not in {"accepted", "stopped", "failed"}:
                validation.ctrl.cancel(args.task_id)
            validation.ctrl.wait(args.task_id)
            validation.report()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
