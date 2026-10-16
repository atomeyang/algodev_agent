"""Deterministic task boundaries shared by manual and autonomous execution."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .paths import sha256_text


def task_key(task_id: str) -> str:
    return sha256_text(task_id)[:12]


def method_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,79}", value):
        raise ValueError(f"invalid method identifier: {value!r}")
    return value


class TaskHalted(RuntimeError):
    pass


def remaining_seconds(spec) -> float:
    return spec.budget.max_wall_clock_s - (
        datetime.now(timezone.utc) - datetime.fromisoformat(spec.created_at)).total_seconds()


def check_running(repo, task_id: str) -> None:
    state = repo.get_state(task_id)
    if state.paused or state.phase in {"stopped", "failed", "accepted"}:
        raise TaskHalted(f"task is {state.phase}")
    from .storage.repository import BudgetExceeded
    if remaining_seconds(repo.get_spec(task_id)) <= 0:
        raise BudgetExceeded("task wall-clock budget exhausted")


def acceptance_failures(spec, evaluation) -> list[str]:
    if evaluation is None:
        return ["no successful independent evaluation"]
    failures = list(evaluation.failures)
    if evaluation.passed is not True:
        failures.append("independent evaluator did not pass")
    for target in [spec.primary_metric, *spec.secondary_metrics]:
        value = evaluation.metrics.get(target.name)
        if target.threshold is None:
            failures.append(f"{target.name}: acceptance threshold missing")
        elif value is None or not math.isfinite(float(value)):
            failures.append(f"{target.name}: finite measurement missing")
        elif target.meets(value) is not True:
            failures.append(f"{target.name}={value} misses {target.threshold}")
    from algorithm_plugins.registry import hook, resolve_plugin
    plugin = resolve_plugin(spec)
    checker = hook(plugin, "check_constraints")
    names_hook = hook(plugin, "constraint_names")
    supported = set(names_hook()) if names_hook else set()
    if checker:
        failures.extend(checker(spec, evaluation.metrics, evaluation.slices) or [])
    for constraint in spec.hard_constraints:
        if not constraint.blocking:
            continue
        if constraint.name in {"rom_mb", "rom_mb_fp16", "package_size_mb"}:
            actual = evaluation.metrics.get("rom_mb")
            limit = constraint.value
            if constraint.unit.lower() == "gb":
                limit = float(limit) * 1024
            elif constraint.unit.lower() == "kb":
                limit = float(limit) / 1024
            if actual is None or not math.isfinite(float(actual)):
                failures.append(f"{constraint.name}: not measured")
            elif float(actual) > float(limit):
                failures.append(f"{constraint.name}: exceeds {limit} MB")
        elif checker is None or constraint.name not in supported:
            failures.append(f"{constraint.name}: no constraint validator")
    return list(dict.fromkeys(failures))


def data_budget_problems(spec, plan, used=None, reserved=None) -> list[str]:
    """Reject known recipe costs before materialising any part of the recipe."""
    used, reserved = used or {}, reserved or {}
    demand = {"data_operations": len(plan.data_recipe.operations),
              "added_samples": 0, "annotation_samples": 0}
    for op in plan.data_recipe.operations:
        field = "count" if op.kind in {"augment", "resample"} else "limit" if op.kind == "annotate" else ""
        if not field or field not in op.params:
            continue
        amount = op.params[field]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            return [f"{op.operation_id}: {field} must be a positive integer"]
        demand["annotation_samples" if op.kind == "annotate" else "added_samples"] += amount
    return [f"recipe {key}={amount} exceeds remaining budget "
            f"{getattr(spec.budget, 'max_' + key) - used.get(key, 0) - reserved.get(key, 0)}"
            for key, amount in demand.items()
            if amount + used.get(key, 0) + reserved.get(key, 0) > getattr(spec.budget, "max_" + key)]


def plan_problems(spec, plan, previous=None) -> list[str]:
    problems = []
    if not plan.candidates:
        problems.append("no executable candidates")
    if len(plan.candidates) > spec.budget.max_candidates:
        problems.append("candidate budget exceeded")
    try:
        names = [method_name(c.method) for c in plan.candidates]
    except ValueError as exc:
        return [str(exc)]
    if len(set(names)) != len(names):
        problems.append("duplicate candidate methods")
    p, target = plan.eval_protocol, spec.primary_metric
    if (p.primary_metric, p.direction, p.threshold) != (target.name, target.direction, target.threshold):
        problems.append("plan changed the frozen acceptance target")
    if plan.change_scope.write_paths != ["model_pool"]:
        problems.append("plan changed the permitted code paths")
    if plan.change_scope.resource_budget != spec.budget:
        problems.append("plan changed the task budget")
    sources = {s.source_id: s for s in spec.data_sources}
    operations = plan.data_recipe.operations
    if len({op.operation_id for op in operations}) != len(operations):
        problems.append("duplicate data operation IDs")
    for op in operations:
        if op.source_id and op.source_id not in sources:
            problems.append(f"{op.operation_id}: data source not authorised")
        if op.split == "final_test":
            source = sources.get(op.source_id)
            if (op.kind != "extend" or source is None or not source.independent
                    or source.label_kind not in {"human", "verified"} or not source.checksum):
                problems.append("final test extensions require an independently labelled, pinned source")
        if op.kind in {"filter", "resample", "annotate", "relabel"} and op.split != "train":
            problems.append(f"{op.kind} is training-only; do not tune the test labels/distribution")
    for candidate in plan.candidates:
        if candidate.action != "reuse" and not candidate.implementation_goal:
            problems.append(f"{candidate.method}: implementation_goal missing")
        for key in ("epochs", "max_epochs", "mlp_max_iter"):
            value = candidate.params.get(key)
            if value is not None and (not isinstance(value, int) or not 0 < value <= spec.budget.max_epochs):
                problems.append(f"{candidate.method}: {key} outside epoch budget")
    if previous:
        if plan.data_recipe.seed != previous.data_recipe.seed:
            problems.append("base split seed is frozen; use operation seeds for augmentation")
        if plan.change_scope.allowed_data_sources != previous.change_scope.allowed_data_sources:
            problems.append("plan changed the authorised data sources")
        if plan.change_scope.upload_policy != previous.change_scope.upload_policy:
            problems.append("plan changed the upload policy")
    if spec.auto_process:
        problems.extend(data_budget_problems(spec, plan))
    return problems
