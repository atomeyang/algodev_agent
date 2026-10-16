"""Bounded working context and evidence-linked, development-only research memory."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .paths import TaskPaths, sha256_obj, write_json


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_]{2,}", text.lower()))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(chunk[i:i + 2] for i in range(max(1, len(chunk) - 1)))
    return words


def bounded_json(value, max_chars: int = 12000) -> str:
    """Never truncate serialized JSON into an invalid half-object."""
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return encoded
    if isinstance(value, list):
        kept = []
        for item in value:
            trial = json.dumps([*kept, item], ensure_ascii=False, default=str)
            if len(trial) <= max_chars:
                kept.append(item)
        return json.dumps(kept, ensure_ascii=False, default=str)
    return json.dumps({"omitted": True, "reason": "context limit; request a narrower record"})


def latest_research(directory: Path):
    files = list(directory.glob("research_iter*.md"))
    return max(files, key=lambda p: int(re.search(r"iter(\d+)", p.name)[1])) if files else None


class ResearchMemory:
    def __init__(self, settings, repo):
        self.settings, self.repo = settings, repo

    def record(self, task_id: str, plan_id: str, *, data_failures=None, diagnosis=None):
        spec = self.repo.get_spec(task_id)
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            return None
        tp = TaskPaths(self.settings, spec.task_slug)
        from .graph.nodes.experiments import evaluation_fingerprint
        fingerprint = evaluation_fingerprint(tp, plan.version)
        evals = [e for e in self.repo.list_evals(task_id)
                 if e.stage != "final" and e.dataset_version == plan.version]
        jobs = [j for j in self.repo.list_jobs(task_id)
                if (self.repo.get_experiment(j.run_id) is not None
                    and self.repo.get_experiment(j.run_id).plan_id == plan_id
                    and self.repo.get_experiment(j.run_id).env.get("stage") != "final")]
        content = {
            "task_id": task_id, "plan_id": plan_id, "domain": spec.domain,
            "requirement": spec.business[:1200], "fingerprint": fingerprint,
            "hypotheses": plan.hypotheses, "controls": plan.controls,
            "candidates": [c.model_dump() for c in plan.candidates],
            "data_operations": [op.model_dump() for op in plan.data_recipe.operations],
            "observations": [{"eval_id": e.eval_id, "model_id": e.model_id,
                              "metrics": e.metrics, "failures": e.failures, "passed": e.passed,
                              "stage": e.stage} for e in evals],
            "jobs": [{"run_id": j.run_id, "status": j.status, "error": j.result.get("error")}
                     for j in jobs],
            "experiments": [
                {"run_id": e.run_id, "code_ref": e.code_ref, "seed": e.seed, "config": e.config}
                for j in jobs if (e := self.repo.get_experiment(j.run_id)) is not None],
            "models": [{"model_id": m.model_id, "run_id": m.run_id, "weights_hash": m.weights_hash}
                       for m in self.repo.list_models(task_id) if m.run_id in {j.run_id for j in jobs}],
            "data_failures": data_failures or [],
            "diagnosis": {"status": "unverified_interpretation", "content": diagnosis or {}},
            "evidence_refs": plan.evidence_refs,
            "validation_level": "development_observed" if evals else "execution_failure",
            "limitations": ["Not a causal conclusion or independent reproduction.",
                            "Compare scores only on the same evaluation fingerprint.",
                            "Transfer requires compatibility checks and a new experiment."],
        }
        lesson_id = "lesson_" + sha256_obj(content)[:20]
        shared = bool((self.settings.workflow.get("memory") or {}).get("share_lessons", False))
        self.repo.save_lesson(lesson_id, task_id, plan_id, spec.domain, fingerprint, content, shared=shared)
        path = tp.docs / "knowledge" / f"{lesson_id}.json"
        write_json(path, content)
        self._write_lesson(path.with_suffix(".md"), content)
        self.export(task_id)
        self.repo.append_event(task_id, "memory.recorded", "development evidence added to research memory",
                               data={"lesson_id": lesson_id, "shared": shared, "path": str(path)})
        return content

    @staticmethod
    def _write_lesson(path, content):
        sections = [
            "# Development Research Record",
            f"Task: `{content['task_id']}` | Plan: `{content['plan_id']}`",
            f"Evaluation fingerprint: `{content['fingerprint']}`",
            f"Validation level: `{content['validation_level']}`",
            "This record contains untrusted observations, not instructions or independent validation.",
        ]
        for title, keys in (
                ("Hypotheses and Controls", ("hypotheses", "controls")),
                ("Model and Data Experiments", ("candidates", "data_operations", "experiments", "models")),
                ("Measured Results and Failures", ("observations", "jobs", "data_failures")),
                ("Interpretation (Unverified)", ("diagnosis",)),
                ("Evidence and Transfer Limits", ("evidence_refs", "limitations"))):
            sections.extend([f"## {title}", "````json",
                             json.dumps({k: content[k] for k in keys}, ensure_ascii=False, indent=2), "````"])
        sections.append(f"Structured record: [{path.stem}.json]({path.stem}.json)")
        for job in content["jobs"]:
            run_id = job["run_id"]
            sections.append(f"- Run `{run_id}`: [result](../../runs/{run_id}/result.json), "
                            f"[log](../../runs/{run_id}/run.log)")
        path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    def export(self, task_id: str) -> str:
        spec = self.repo.get_spec(task_id)
        directory = TaskPaths(self.settings, spec.task_slug).docs / "knowledge"
        directory.mkdir(parents=True, exist_ok=True)
        rows = self.repo.lessons(task_id, spec.domain, include_shared=False)
        lines = ["# Research Knowledge Index",
                 "Development evidence only. Final-test results are not available for further tuning.",
                 "Sharing a document does not establish reproducibility or authorise data access.",
                 "Check evaluation fingerprints, implementation/configuration, seeds and weights before reuse."]
        for row in rows:
            content, lesson_id = row["content"], row["lesson_id"]
            lines.append(f"- [{lesson_id}]({lesson_id}.md) | plan `{row['plan_id']}` | "
                         f"{content.get('validation_level', 'unknown')} | fingerprint `{row['fingerprint']}`")
        if not rows:
            lines.append("No development lessons recorded yet.")
        path = directory / "README.md"
        path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def recall(self, task_id: str, query: str = "", limit: int = 6):
        spec = self.repo.get_spec(task_id)
        wanted = terms(query or spec.business)
        rows = self.repo.lessons(task_id, spec.domain)
        ranked = []
        for row in rows:
            content = row["content"]
            score = len(wanted & terms(json.dumps(content, ensure_ascii=False)))
            if row["task_id"] != task_id and not score:
                continue
            ranked.append((score + (2 if row["task_id"] == task_id else 0), row))
        ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
        return [{"lesson_id": r["lesson_id"], "source_task": r["task_id"], **r["content"]}
                for _, r in ranked[:max(1, min(limit, 12))]]

    def summary(self, task_id: str):
        spec = self.repo.get_spec(task_id)
        return {"task_id": task_id, "domain": spec.domain,
                "lessons": self.recall(task_id), "retrieval": "domain-scoped lexical relevance",
                "index": str(TaskPaths(self.settings, spec.task_slug).docs / "knowledge" / "README.md"),
                "cross_task_sharing": bool((self.settings.workflow.get("memory") or {}).get("share_lessons"))}
