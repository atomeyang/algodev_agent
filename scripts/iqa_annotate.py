#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-label IQA images with the local Qwen3.8-27B and measure agreement.

This is the *data-construction* tool: it uses the local auxiliary VLM to propose
defect labels (the annotation bottleneck flagged in the run report), then reports
its agreement against the existing ground truth so a human can judge how much of
the labelling can be delegated.

Outputs are **candidates**, written with `review_required: true`; nothing here
may enter a frozen dataset version without human confirmation.

Usage:
    python3 scripts/iqa_annotate.py --split eval --limit 20
    python3 scripts/iqa_annotate.py --task iqa --version v2 --images a.jpg b.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runtimes" / "agent_runtime" / "src"))
sys.path.insert(0, str(REPO))

from algorithm_plugins.iqa import annotation as A  # noqa: E402
from algorithm_plugins.iqa import config as C  # noqa: E402
from algo_agent.config import load_settings  # noqa: E402
from algo_agent.paths import TaskPaths, ensure_task_dirs, safe_join, write_json  # noqa: E402


def output_paths(tp: TaskPaths, version: str, run_id: str, out: str = ""):
    run = tp.annotation_run(version, run_id)
    labels = tp.output_path(out) if out else run / "prelabels.jsonl"
    if not labels.is_relative_to(tp.annotations) or labels.suffix != ".jsonl":
        raise ValueError("--out must be a .jsonl inside this task's data/annotations/")
    report = labels.with_suffix(".agreement.json")
    scheme = safe_join(tp.annotations, "plans", version, f"{run_id}.json")
    if any(p.exists() for p in (labels, report, scheme)):
        raise ValueError("annotation output already exists; choose a new run ID")
    return labels, report, scheme


def load_rows(split: str):
    path = C.dev_jsonl() if split == "dev" else C.eval_jsonl()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agreement(rows, preds):
    """Prelabel-vs-GT agreement, computed over rows that actually have GT."""
    by_path = {r["path"]: r for r in rows}
    per_class = {d: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for d in C.DEFECTS}
    sev_err, sev_n, wb_ok, wb_n, paired = [], 0, 0, 0, 0
    for p in preds:
        if not p.get("ok"):
            continue
        gt = by_path.get(p["path"])
        if not gt:
            continue
        paired += 1
        for d in C.DEFECTS:
            g = int(bool((gt.get("defects") or {}).get(d, {}).get("present", 0)))
            q = int(p["defects"][d]["present"])
            cell = per_class[d]
            cell["tp" if (g and q) else "fp" if q else "fn" if g else "tn"] += 1
            if d == "white_balance" and g and q:
                wb_n += 1
                gdir = str((gt.get("defects") or {}).get(d, {}).get("direction", "")).lower()
                if gdir and gdir == p["defects"][d].get("direction"):
                    wb_ok += 1
            if g and q:
                sev_n += 1
                sev_err.append(abs(float((gt["defects"][d]).get("severity", 0))
                                   - float(p["defects"][d]["severity"])))
    out = {"n_paired": paired, "per_class": {}}
    for d, c in per_class.items():
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out["per_class"][d] = {"precision": round(prec, 4), "recall": round(rec, 4),
                               "f1": round(f1, 4), "support": c["tp"] + c["fn"], **c}
    if sev_err:
        out["severity_mae"] = round(sum(sev_err) / len(sev_err), 3)
        out["severity_n"] = sev_n
    out["wb_direction_acc"] = round(wb_ok / wb_n, 4) if wb_n else None
    out["wb_direction_n"] = wb_n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO), help="platform root, not an output directory")
    ap.add_argument("--task", default="iqa", help="registered task slug")
    ap.add_argument("--version", default="v1", help="annotation scheme version")
    ap.add_argument("--run-id", default="", help="unique annotation attempt ID")
    ap.add_argument("--split", default="eval", choices=["dev", "eval"])
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="", help="optional task-relative path inside data/annotations/")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    tp = TaskPaths(load_settings(args.root), args.task)
    run_id = args.run_id or f"manual_{uuid.uuid4().hex[:12]}"
    try:
        out, report, scheme = output_paths(tp, args.version, run_id, args.out)
    except ValueError as exc:
        ap.error(str(exc))

    h = A.health()
    print(f"[annotate] endpoint {h['base_url']} available={h['available']} model={h['model_id']}",
          flush=True)
    if not h["available"]:
        print(f"[annotate] local model unreachable ({h['detail']}) -- aborting, nothing written.",
              flush=True)
        return 2

    rows = load_rows(args.split) if not args.images else []
    paths = args.images or [r["path"] for r in rows]
    ensure_task_dirs(tp)
    write_json(scheme, {"task_slug": tp.slug, "version": args.version, "run_id": run_id,
                        "split": args.split, "limit": args.limit,
                        "prompt_version": A.PROMPT_VERSION, "prompt": A._PROMPT,
                        "model_id": h["model_id"], "review_required": True})
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = out.open("x", encoding="utf-8")
    print(f"[annotate] pre-labelling {min(len(paths), args.limit)} / {len(paths)} images "
          f"(prompt={A.PROMPT_VERSION})", flush=True)

    def progress(i, n, row):
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        state = "ok" if row.get("ok") else f"FAIL({row.get('error')})"
        print(f"  [{i}/{n}] {Path(row['path']).name} {state} "
              f"{row.get('elapsed_s', 0)}s", flush=True)

    try:
        preds = A.annotate_many(paths, limit=args.limit, timeout=args.timeout, on_progress=progress)
    finally:
        stream.close()
    ok = sum(1 for p in preds if p.get("ok"))
    print(f"\n[annotate] wrote {len(preds)} rows ({ok} ok) -> {out}", flush=True)

    rep = agreement(rows, preds) if rows else {"n_paired": 0}
    print("[annotate] prelabel vs GT (candidate labels only; human review still required):")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    write_json(report, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
