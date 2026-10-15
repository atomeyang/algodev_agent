"""intake node: parse a natural-language requirement into a TaskSpec.

Deterministic keyword/regex parsing (no cloud call). Missing key items are left
as explicit unknowns and surfaced to the user (guide 14: new requirement).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ...paths import slugify
from ...schemas import contracts as C
from ...schemas.enums import TaskPhase
from ..state import TaskState

_METRIC_SYNONYMS = {
    "rmse": ("rmse", "均方根误差", "回归"),
    "mae": ("mae", "平均绝对误差"),
    "psnr": ("psnr", "峰值信噪比"),
    "ssim": ("ssim", "结构相似"),
    "accuracy": ("accuracy", "准确率", "acc"),
    "f1": ("f1", "f1-score"),
    "latency_ms": ("延迟", "latency", "耗时"),
}
_NUM = r"([0-9]+(?:\.[0-9]+)?)"


def parse_requirement(text: str, task_id: str, task_slug: str = "", demo: bool = False,
                      synthetic: bool = False, domain: str = "") -> C.TaskSpec:
    slug = task_slug or slugify(task_id)

    # A domain plugin may own requirement parsing (its own metrics, hard
    # constraints and unknowns). Without a hook we keep the deterministic
    # keyword parser below so existing tasks behave exactly as before.
    from algorithm_plugins.registry import detect_domain, load_plugin
    dom = domain or detect_domain(text)
    plugin = load_plugin(dom)
    parser = getattr(plugin, "parse_requirement", None) if plugin is not None else None
    if callable(parser):
        spec = parser(text, task_id, slug, demo=demo, synthetic=synthetic)
        spec.domain = dom
        return spec

    lower = text.lower()
    unknowns: List[str] = []

    metric_name = None
    for name, keys in _METRIC_SYNONYMS.items():
        if any(k.lower() in lower for k in keys):
            metric_name = name
            break
    if metric_name is None:
        metric_name = "rmse"
        unknowns.append("primary metric not stated; defaulted to rmse")

    threshold = None
    for pat in (rf"(?:不超过|小于|低于|<=|<|达到|至少|不低于|>=)\s*{_NUM}",
                rf"{_NUM}\s*(?:以下|以内|之内)"):
        m = re.search(pat, text)
        if m:
            threshold = float(m.group(1))
            break
    if threshold is None:
        unknowns.append("acceptance threshold not stated")

    direction = "min" if metric_name in {"rmse", "mae", "latency_ms"} else "max"

    hardware = None
    for key, hw in (("cpu", "cpu"), ("gpu", "gpu"), ("移动端", "mobile"), ("手机", "mobile"),
                    ("边缘", "edge"), ("服务器", "server")):
        if key in lower:
            hardware = hw
            break
    if hardware is None:
        unknowns.append("target hardware unspecified")

    constraints: List[C.HardConstraint] = []
    ms = re.search(rf"(?:部署包|包大小|模型大小|rom)\s*(?:不超过|<=|<|小于)?\s*{_NUM}\s*(mb|m|kb|gb)", lower)
    if ms:
        constraints.append(C.HardConstraint(name="package_size_mb", value=float(ms.group(1)),
                                            unit=ms.group(2), measurement_scope="deployment package",
                                            blocking=True))
    if "rgb" in lower:
        constraints.append(C.HardConstraint(name="input_color", value="RGB",
                                            measurement_scope="input tensor", blocking=False))

    inputs = ["rgb_image"] if "rgb" in lower or "图像" in text else []
    outputs = ["quality_score"] if "质量" in text or "quality" in lower else []
    if not inputs:
        unknowns.append("input format unspecified")
    if not outputs:
        unknowns.append("output format unspecified")

    return C.TaskSpec(
        task_id=task_id, task_slug=slug, business=text.strip(),
        inputs=inputs, outputs=outputs, target_hardware=hardware,
        hard_constraints=constraints,
        primary_metric=C.MetricTarget(name=metric_name, direction=direction, threshold=threshold,
                                      unit="", unknown=threshold is None),
        budget=C.Budget(), unknowns=unknowns, demo=demo, synthetic=synthetic, domain=dom,
    )


def intake_node(state: TaskState, ctx) -> Dict[str, Any]:
    from ...paths import TaskPaths, ensure_task_dirs
    task_id = state["task_id"]
    spec = ctx.repo.get_spec(task_id)
    if not ctx.repo.task_exists(task_id):
        spec = parse_requirement(state.get("raw_text", ""), task_id, state["task_slug"])
        ctx.repo.create_task(spec)
    tp = TaskPaths(ctx.settings, state["task_slug"])
    ensure_task_dirs(tp)
    # The business plugin owns a *required* requirement-validation hook; run it
    # here so contract violations surface as warnings during intake instead of
    # silently propagating into the plan.
    from algorithm_plugins.registry import resolve_plugin
    plugin = resolve_plugin(spec)
    req_problems: List[str] = []
    validator = getattr(plugin, "validate_requirement", None)
    if callable(validator):
        try:
            req_problems = list(validator(spec.model_dump()) or [])
        except Exception as exc:  # a broken validator must not abort intake
            req_problems = [f"requirement validator raised: {exc}"]
    if req_problems:
        ctx.repo.append_event(task_id, "stage.intake",
                              "requirement validation flagged problems",
                              level="warning", data={"problems": req_problems})
    ctx.repo.append_event(task_id, "stage.intake",
                          f"requirement parsed; unknowns={len(spec.unknowns)}",
                          data={"unknowns": spec.unknowns, "metric": spec.primary_metric.model_dump(),
                                "validation_problems": req_problems})
    phase = TaskPhase.AWAITING_INPUT.value if spec.primary_metric.threshold is None else TaskPhase.RESEARCHING.value
    ctx.repo.update_state(task_id, phase=phase)
    if spec.auto_process and (spec.primary_metric.threshold is None or req_problems):
        reason = "requirements incomplete: " + "; ".join(req_problems or ["acceptance threshold missing"])
        ctx.repo.update_state(task_id, phase="awaiting_input", stop_reason=reason)
        return {"stage": "intake", "decision": "needs_input", "stop_reason": reason}
    if spec.auto_process and spec.budget.max_iterations == 0:
        ctx.repo.update_state(task_id, phase="stopped", stop_reason="iteration budget is zero")
        return {"stage": "intake", "decision": "needs_input", "stop_reason": "iteration budget is zero"}
    return {"stage": "intake", "unknown_items": spec.unknowns, "iteration": 1}


__all__ = ["intake_node", "parse_requirement"]
