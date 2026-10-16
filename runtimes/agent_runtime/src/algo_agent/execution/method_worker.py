"""Subprocess entry that runs one method job in isolation.

Run as ``python -m algo_agent.execution.method_worker <job.json>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_job(job_path: Path) -> dict:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    from ..config import Settings
    from ..paths import LAYOUT_FILE, TaskPaths
    for parent in job_path.resolve().parents:
        if parent.name.endswith("_experiment") and (parent / LAYOUT_FILE).is_file():
            tp = TaskPaths(Settings(parent.parent), job["task_slug"])
            for key in ("task_root", "release_dir", "out_dir", "method_entry", "inference_weights"):
                if job.get(key):
                    job[key] = str(tp.resolve_reference(job[key]))
            for key in ("task_root", "release_dir", "out_dir", "method_entry"):
                tp.output_path(job[key])
            break
    return job


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: method_worker <job.json>", file=sys.stderr)
        return 2
    job_path = Path(argv[1])
    job = load_job(job_path)
    limits = job.get("resource_limits") or {}
    if limits.get("memory_mb") or limits.get("output_file_mb"):
        import resource
        for key, kind in (("memory_mb", resource.RLIMIT_AS), ("output_file_mb", resource.RLIMIT_FSIZE)):
            if limits.get(key):
                cap = int(limits[key]) * 1024 * 1024
                resource.setrlimit(kind, (cap, cap))
    plugins = job_path.parent / "code" / "plugins"
    if plugins.exists():
        sys.path.insert(0, str(plugins))
    from algorithm_plugins.common.plugin import MethodContext, load_method

    out_dir = Path(job["out_dir"])
    ctx = MethodContext(
        task_root=Path(job["task_root"]),
        data_dir=Path(job["release_dir"]),
        out_dir=out_dir,
        method_dir=Path(job["method_entry"]).parent,
        seed=int(job.get("seed", 0)),
        split=job.get("split", "val"),
    )
    module = load_method(job["method_entry"])
    try:
        if job.get("stage") == "final":
            weights = Path(job["inference_weights"])
            result = module.predict(ctx, job.get("config", {}), weights)
        else:
            result = module.run(ctx, job.get("config", {}))
    except Exception as exc:  # surface a structured failure, not a bare traceback only
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps({"status": "failed", "error": repr(exc)}), encoding="utf-8")
        raise
    result.setdefault("status", "succeeded")
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"method={result.get('method')} metrics={result.get('metrics')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
