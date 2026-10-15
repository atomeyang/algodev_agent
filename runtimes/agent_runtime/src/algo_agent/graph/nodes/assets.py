"""prepare_assets node: build an immutable dataset release with provenance.

A domain plugin may own the release layout via the optional ``prepare_dataset``
hook. Without a hook the built-in synthetic-tabular generator is used, so the
demo task keeps its original dataset contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from algorithm_plugins.registry import hook, resolve_plugin

from ...paths import TaskPaths, sha256_file, write_json
from ...schemas import contracts as C
from ...schemas.enums import TaskPhase
from ..state import TaskState


def _validate_release(ctx, task_id: str, spec: C.TaskSpec,
                      manifest: C.DatasetManifest) -> List[str]:
    """Run the plugin's required dataset-validation hook on the built release.

    Problems are recorded as warnings (the release is still usable) so the
    demo/synthetic path keeps its exact semantics when a plugin is absent.
    """
    plugin = resolve_plugin(spec)
    validator = hook(plugin, "validate_dataset")
    if validator is None:
        return []
    try:
        problems = list(validator(manifest.model_dump()) or [])
    except Exception as exc:
        problems = [f"dataset validator raised: {exc}"]
    if problems:
        ctx.repo.append_event(task_id, "stage.prepare_assets",
                              "dataset validation flagged problems", level="warning",
                              data={"problems": problems, "dataset": manifest.dataset_id})
    return problems


def _prepare_via_plugin(state: TaskState, ctx, tp: TaskPaths, spec: C.TaskSpec, plan: C.PlanSpec,
                        version: str, release: Path,
                        prepare: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
    """Build the release with the plugin hook and record identical provenance."""
    task_id = state["task_id"]
    info = dict(prepare(tp, spec, plan, version) or {})
    raw_dir = tp.data / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = info.get("dataset_id", spec.domain)
    raw_manifest = info.pop("raw_manifest", None) or {"dataset_id": dataset_id}
    raw_rel = str((raw_dir / f"{dataset_id}_source.json").relative_to(tp.root))
    write_json(raw_dir / f"{dataset_id}_source.json", raw_manifest)
    write_json(tp.data / "recipes" / f"recipe_{version}.json", plan.data_recipe.model_dump())

    checksum_files: List[str] = list(info.pop("checksum_files", []) or [])
    checksums = {}
    for rel in checksum_files:
        path = Path(rel)
        if not path.is_absolute():
            path = tp.root / rel
        if path.exists():
            checksums[Path(rel).name] = sha256_file(path)

    manifest = C.DatasetManifest(
        dataset_id=dataset_id, version=version,
        source=info.get("source", "task_local"),
        raw_refs=info.get("raw_refs", [raw_rel]),
        annotations=info.get("annotations", ""), splits=info.get("splits", {}),
        recipe_id=info.get("recipe_id", plan.data_recipe.recipe_id),
        code_ref=info.get("code_ref", f"algorithm_plugins/{spec.domain}/methods.py"),
        seed=plan.data_recipe.seed, checksums=checksums,
        license=info.get("license", "internal"), upload_policy=info.get("upload_policy", "restricted"),
        synthetic=bool(info.get("synthetic", False)))
    ctx.repo.save_dataset(manifest)
    write_json(release / "manifest.json", manifest.model_dump())
    ctx.repo.append_event(task_id, "stage.prepare_assets",
                          f"dataset {manifest.dataset_id}:{version} built ({manifest.splits})",
                          data={"checksums": checksums, "synthetic": manifest.synthetic,
                                "domain": spec.domain, "note": info.get("note", "")})
    problems = _validate_release(ctx, task_id, spec, manifest)
    ctx.repo.update_state(task_id, phase=TaskPhase.EXECUTING.value)
    return {"stage": "prepare_assets", "dataset_version": version,
            "dataset_problems": problems}


def prepare_assets_node(state: TaskState, ctx) -> Dict[str, Any]:
    task_id = state["task_id"]
    tp = TaskPaths(ctx.settings, state["task_slug"])
    spec = ctx.repo.get_spec(task_id)
    plan = ctx.repo.get_plan(state["plan_id"])
    version = f"v{int(state.get('iteration', 1)) or 1}"
    release = tp.release_dir(version)
    release.mkdir(parents=True, exist_ok=True)
    if spec.auto_process:
        from ...execution.data_pipeline import prepare_development
        try:
            manifest = prepare_development(tp, spec, plan, version, ctx.repo)
        except (ValueError, FileNotFoundError) as exc:
            write_json(tp.plan_dir(plan.version) / "data_failure.json", {"error": str(exc)})
            ctx.repo.append_event(task_id, "data.experiment_failed", str(exc), level="warning")
            return {"stage": "prepare_assets", "dataset_version": version, "dataset_problems": [str(exc)],
                    "decision": "data_failed", "model_ids": [], "eval_ids": [], "best_feasible_model": None}
        ctx.repo.append_event(task_id, "stage.prepare_assets", "versioned development dataset prepared",
                              data={"version": version, "splits": manifest.splits,
                                    "operations": len(manifest.provenance)})
        return {"stage": "prepare_assets", "dataset_version": version, "dataset_problems": []}

    plugin = resolve_plugin(spec)
    prepare = hook(plugin, "prepare_dataset")
    if prepare is not None:
        return _prepare_via_plugin(state, ctx, tp, spec, plan, version, release, prepare)

    if not spec.synthetic and not spec.demo:
        ctx.repo.append_event(task_id, "stage.prepare_assets",
                              "no local raw data and no business dataset configured", level="warning")
        return {"stage": "prepare_assets", "decision": "needs_input",
                "stop_reason": "dataset source missing; import user materials or attach data"}

    from algorithm_plugins.demo_tabular import build_dataset
    info = build_dataset(release, seed=plan.data_recipe.seed, n=600)
    # raw snapshot + recipe (raw is conceptually read-only for the executor)
    raw_dir = tp.data / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "demo_tabular_source.json", {"generator": "demo_tabular", **info})
    write_json(tp.data / "recipes" / f"recipe_{version}.json", plan.data_recipe.model_dump())
    checksums = {f.name: sha256_file(f) for f in sorted(release.glob("*.npz"))}

    manifest = C.DatasetManifest(
        dataset_id="demo_tabular", version=version, source="synthetic_demo",
        raw_refs=[str((raw_dir / "demo_tabular_source.json").relative_to(tp.root))],
        annotations="low_light slice: x0 < 0", splits=info["counts"],
        recipe_id=plan.data_recipe.recipe_id, code_ref="algorithm_plugins/demo_tabular/methods.py",
        seed=plan.data_recipe.seed, checksums=checksums, license="synthetic",
        upload_policy="restricted", synthetic=True)
    ctx.repo.save_dataset(manifest)
    write_json(tp.data / "releases" / version / "manifest.json", manifest.model_dump())
    ctx.repo.append_event(task_id, "stage.prepare_assets",
                          f"dataset {manifest.dataset_id}:{version} built ({manifest.splits})",
                          data={"checksums": checksums, "synthetic": True})
    problems = _validate_release(ctx, task_id, spec, manifest)
    ctx.repo.update_state(task_id, phase=TaskPhase.EXECUTING.value)
    return {"stage": "prepare_assets", "dataset_version": version,
            "dataset_problems": problems}


__all__ = ["prepare_assets_node"]
