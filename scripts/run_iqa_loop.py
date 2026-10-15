#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive the real IQA R&D loop end to end (offline research + local GPU runs).

The agent's graph is domain-agnostic; the IQA behaviour comes entirely from
``algorithm_plugins/iqa``. This script only plays the role of the human
operator: it approves the plan and continues across iteration boundaries.

Usage:
    # --root is the *platform* root (guide 4); the task assets land in
    # <platform_root>/<task_id>_experiment/, e.g. ./iqa_experiment/.
    python3 scripts/run_iqa_loop.py --root . --task-id iqa --max-iterations 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runtimes" / "agent_runtime" / "src"))
sys.path.insert(0, str(REPO))

IQA_REPORT = (
    "轻量双输出 IQA 模型研发：ROM < 60MB fp16，输出 0-100 质量分与 "
    "blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签；"
    "缺陷 Macro-F1 >= 0.85，PLCC/SRCC >= 0.90，分档一致率 >= 0.90，hardcase Macro-F1 >= 0.85，"
    "参考 iqa_agent 的冻结基准与 MobileCLIP2-S0 方案。"
)


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO),
                    help="platform root; task assets are written to <root>/<task_id>_experiment/ "
                         "(guide 4). Default: the repo root.")
    ap.add_argument("--state-dir", default="")
    ap.add_argument("--task-id", default="iqa",
                    help="task name; the task directory becomes <task_id>_experiment/")
    ap.add_argument("--parent-task", default="", help="owner task slug for a related track")
    ap.add_argument("--track", default="", help="directory name under the owner's tracks/")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--report", default="")
    ap.add_argument("--fresh", action="store_true",
                    help="require a new empty task; refuses to erase existing results or shared state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--methods-hint", default="")
    args = ap.parse_args()

    import os
    if args.state_dir:
        os.environ["ALGO_STATE_DIR"] = args.state_dir

    from algo_agent import bootstrap
    from algo_agent.config import load_settings
    from algo_agent.service.controller import Controller

    root = Path(args.root)
    if root.name.endswith("_experiment"):
        # Backwards compatible with the old (buggy) invocation that passed the
        # task directory as --root and produced <root>/<slug>_experiment/.
        print(f"[layout] --root '{root}' looks like a task directory; using its parent "
              f"'{root.parent}' as the platform root (guide 4).", flush=True)
        root = root.parent
    settings = load_settings(str(root))
    from algo_agent.paths import TaskPaths, register_task_member, slugify as _slugify

    task_slug = _slugify(args.task_id)
    if args.parent_task:
        register_task_member(settings, args.parent_task, task_slug, args.track or task_slug)
    task_dir = TaskPaths(settings, task_slug).root
    if args.fresh and task_dir.exists() and any(task_dir.iterdir()):
        ap.error("--fresh refuses to erase existing task files; use a new task/track ID")
    bootstrap.init_platform(settings)

    hint = [m.strip() for m in args.methods_hint.split(",") if m.strip()] or None
    ctl = Controller(settings, inline=True,
                     extras={"pause_between_iterations": True, "methods_hint": hint})
    if args.fresh and ctl.repo.task_exists(args.task_id):
        ap.error("--fresh refuses to erase shared task state; use a new task/track ID")

    task_id = args.task_id
    if args.resume:
        pending = ctl.pending_interrupt(task_id)
        kind = (pending or {}).get("type")
        print(f"[resume] task {task_id} paused at {kind}", flush=True)
        if kind == "iteration_boundary":
            boundary = ctl.continue_task(task_id)
        elif kind == "approval_request":
            boundary = ctl.approve(task_id, pending["plan_id"], pending["plan_hash"])
        elif ctl.at_end(task_id) and ctl.repo.list_models(task_id):
            # The graph already ran to completion (no pending interrupt) but the
            # task has finished experiments. Re-entering from `intake` would
            # silently rewind to iteration 1 and re-plan the whole run, so stop
            # instead and say so explicitly.
            state = ctl.state(task_id)
            iterations = int(state.get("iteration", 0) or 0)
            print(f"[resume] task {task_id} already finished "
                  f"(phase={state.get('phase')}, best={state.get('best_model_id')}, "
                  f"iterations={iterations}); nothing to resume. "
                  f"Use a new --task-id, or drop --resume with --fresh to restart.", flush=True)
            boundary = {"type": "finished", "done": True, "interrupt": {},
                        "iterations": iterations,
                        "trace": _trace_from_repo(ctl, task_id)}
        else:
            boundary = ctl.run(task_id)
    else:
        spec = ctl.create_task(args.report or IQA_REPORT, task_id=task_id, domain="iqa")
        print(f"[task] {spec.task_id} domain={spec.domain} primary={spec.primary_metric.name} "
              f"threshold={spec.primary_metric.threshold}", flush=True)
        print(f"[task] assets={json.dumps(_plugin_assets(), ensure_ascii=False)}", flush=True)
        boundary = ctl.run(task_id)
    iterations = 0
    trace = []
    while True:
        kind = boundary.get("type")
        if boundary.get("done"):
            iterations = max(iterations, int(boundary.get("iterations", 0) or 0))
            if boundary.get("trace"):
                trace = boundary["trace"]
            print(f"[end] task finished: {kind}", flush=True)
            break
        itv = boundary.get("interrupt") or {}
        if kind == "approval_request":
            plan_id, plan_hash = itv.get("plan_id"), itv.get("plan_hash")
            print(f"\n=== approval requested: {plan_id} candidates="
                  f"{[c.get('method') if isinstance(c, dict) else c for c in itv.get('candidates', [])]} ===",
                  flush=True)
            for c in itv.get("candidates", []):
                if isinstance(c, dict):
                    print(f"    - {c.get('method')} [{c.get('family')}] {c.get('expected_gain')}", flush=True)
            boundary = ctl.approve(task_id, plan_id, plan_hash)
            continue
        if kind == "iteration_boundary":
            iterations = max(iterations, int(itv.get("iteration", iterations + 1)))
            snap = ctl.state(task_id)
            trace.append(_iteration_snapshot(ctl, task_id, iterations))
            print(f"\n=== iteration {iterations} done: best={itv.get('best_model')} "
                  f"phase={snap.get('phase')} ===", flush=True)
            print(_dumps(itv.get("metrics")), flush=True)
            if iterations >= args.max_iterations:
                print(f"[stop] reached --max-iterations={args.max_iterations}", flush=True)
                ctl.continue_task(task_id, stop=True)
                break
            boundary = ctl.continue_task(task_id)
            continue
        print(f"[interrupt] unhandled {kind}: {_dumps(itv)[:400]}", flush=True)
        break

    snap = ctl.state(task_id)
    best_id = snap.get("best_model_id")
    best_eval = None
    for ev in (ctl.repo.list_evals(task_id) if best_id else []):
        if ev.model_id == best_id:
            best_eval = {"eval_id": ev.eval_id, "passed": ev.passed,
                         "metrics": ev.metrics, "slices": ev.slices, "failures": ev.failures}
    summary = {
        "task_id": task_id, "root": str(root), "task_dir": str(task_dir),
        "task_slug": _slugify(task_id),
        "phase": snap.get("phase"),
        "stop_reason": snap.get("stop_reason"), "best_model_id": snap.get("best_model_id"),
        "best_eval": best_eval,
        "iterations": iterations, "trace": trace,
        "events": [{"kind": e["kind"], "level": e["level"], "message": e["message"]}
                   for e in ctl.events(task_id)],
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    out = task_dir / "iqa_run_summary.json"
    out.write_text(_dumps(summary), encoding="utf-8")
    print(f"\n[summary] written to {out}", flush=True)
    print(_dumps({k: summary[k] for k in ("phase", "stop_reason", "best_model_id", "iterations")}))
    return 0


def _plugin_assets():
    try:
        from algorithm_plugins.iqa import config as IC
        from algorithm_plugins.iqa.plugin import PLUGIN
        return {"available": PLUGIN.assets(), "benchmark_dir": str(IC.benchmark_dir())}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def _trace_from_repo(ctl, task_id: str):
    """Rebuild the per-iteration trace from persisted models/evals.

    Used when the graph has already run to completion, so a read-only inspect
    still reports what each iteration produced instead of an empty trace.
    """
    import re

    groups = {}
    for m in ctl.repo.list_models(task_id):
        match = re.search(r"run_v(\d+)_", m.run_id or "")
        iteration = int(match.group(1)) if match else 1
        evals = ctl.repo.list_evals(task_id, model_id=m.model_id)
        ev = evals[-1] if evals else None
        groups.setdefault(iteration, []).append({
            "model_id": m.model_id, "run_id": m.run_id, "env": m.env,
            "metrics": (ev.metrics if ev else {}),
            "passed": (ev.passed if ev else None),
            "failures": (ev.failures if ev else []),
        })
    return [{"iteration": it, "runs": runs} for it, runs in sorted(groups.items())]


def _iteration_snapshot(ctl, task_id: str, iteration: int):
    runs = []
    for m in ctl.repo.list_models(task_id):
        evs = ctl.repo.list_evals(task_id, model_id=m.model_id)
        ev = evs[-1] if evs else None
        runs.append({"model_id": m.model_id, "run_id": m.run_id, "env": m.env,
                     "metrics": (ev.metrics if ev else {}),
                     "passed": (ev.passed if ev else None),
                     "failures": (ev.failures if ev else [])})
    return {"iteration": iteration, "runs": runs}


if __name__ == "__main__":
    raise SystemExit(main())
