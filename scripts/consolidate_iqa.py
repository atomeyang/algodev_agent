#!/usr/bin/env python3
"""Consolidate existing IQA outputs; dry-run unless --apply is supplied."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "runtimes" / "agent_runtime" / "src"))
sys.path.insert(0, str(REPO))

from algo_agent.config import Settings  # noqa: E402
from algo_agent.layout_migration import consolidate_task  # noqa: E402

MOVES = {
    "iqa_annotations": "data/annotations/runs/legacy/pilot",
    "iqa_mos_experiment": "tracks/mos",
    "iqa_rd/iqa_rd_experiment": "history/legacy_rd",
    "iqa_rd/iqa_run_summary.json": "history/legacy_rd/iqa_run_summary.json",
    "iqa_rd": "history/legacy_rd/legacy_platform",
    "docs/iqa_run_report.md": "docs/iqa_run_report.md",
}
MEMBERS = {"iqa_mos": "tracks/mos", "iqa_rd": "history/legacy_rd"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = consolidate_task(Settings(Path(args.root)), "iqa", MOVES, MEMBERS, apply=args.apply)
    print(json.dumps({"status": result["status"], "moves": result["moves"],
                      "verified_files": sum(len(r["files"]) for r in result["records"].values())},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
