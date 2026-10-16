"""Bind metric names to local comparators, values and units before execution."""

from __future__ import annotations

import re
from typing import List, Tuple

from .schemas import contracts as C


ALIASES = {
    "rmse": ("rmse", "均方根误差"),
    "mae": ("mae", "平均绝对误差"),
    "psnr": ("psnr", "峰值信噪比"),
    "ssim": ("ssim", "结构相似"),
    "accuracy": ("accuracy", "准确率", "acc"),
    "hardcase_macro_f1": ("hardcase_macro_f1", "hardcase macro-f1"),
    "defect_macro_f1": ("defect_macro_f1", "defect macro-f1", "缺陷macro-f1"),
    "defect_macro_f1_cov": ("defect_macro_f1_cov",),
    "f1": ("f1-score", "f1"),
    "plcc": ("plcc", "pearson"),
    "srcc": ("srcc", "spearman"),
    "grade_agreement": ("grade_agreement", "分档一致率"),
    "latency_ms": ("latency_ms", "latency", "延迟", "耗时"),
    "rom_mb": ("rom_mb", "rom", "部署包", "包大小", "模型大小"),
}
_NAMES = {alias: name for name, aliases in ALIASES.items() for alias in aliases}
_METRIC = re.compile(
    "|".join(
        rf"(?<![a-zA-Z0-9_]){re.escape(a)}(?![a-zA-Z0-9_])" if a.isascii() else re.escape(a)
        for a in sorted(_NAMES, key=len, reverse=True)), re.I)
_BOUND = re.compile(
    r"^\s*(?:应当|应该|要求|需|为|是|:|：|=)?\s*"
    r"(?P<op><=|>=|≤|≥|<|>|不超过|不低于|不高于|至少|至多|小于|大于|低于|高于|达到|"
    r"at\s+least|at\s+most)?\s*"
    r"(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>%|％|milliseconds?|ms|seconds?|秒|毫秒|s|gb|mb|kb)?"
    r"\s*(?P<suffix>以下|以内|以上)?", re.I)
_MINIMUM = {"rmse", "mae", "latency_ms", "rom_mb"}
_RATES = {"accuracy", "f1", "defect_macro_f1", "defect_macro_f1_cov", "hardcase_macro_f1", "ssim",
          "grade_agreement"}
_LOWER = {">", ">=", "≥", "不低于", "至少", "大于", "高于", "at least"}
_UPPER = {"<", "<=", "≤", "不超过", "不高于", "至多", "小于", "低于", "at most"}


def metric_targets(text: str) -> Tuple[List[C.MetricTarget], List[str]]:
    """Missing/contradictory bounds stay unknown; no number crosses a metric."""
    matches = list(_METRIC.finditer(text))
    targets, problems = {}, []
    primary = None
    for i, match in enumerate(matches):
        name = _NAMES[match.group().lower()]
        if re.search(r"(主指标|首要指标|primary\s+metric)\s*[:：]?\s*$",
                     text[max(0, match.start() - 24):match.start()], re.I):
            primary = name
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fragment = text[match.end():end]
        j = i
        # PLCC/SRCC >= x: a single bound explicitly applies to the joined names.
        while re.fullmatch(r"[\s/、和与&]+", fragment) and j + 1 < len(matches):
            j += 1
            end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
            fragment = text[matches[j].end():end]
        bound = _BOUND.match(fragment)
        direction = "min" if name in _MINIMUM else "max"
        target = C.MetricTarget(name=name, direction=direction, unknown=True)
        if bound:
            value = float(bound["value"])
            unit = (bound["unit"] or "").lower()
            op = (bound["op"] or "").lower()
            suffix = bound["suffix"]
            if unit in {"%", "％"}:
                value /= 100
            if name == "latency_ms":
                if unit in {"s", "seconds", "second", "秒"}:
                    value *= 1000
                unit = "ms"
            elif name == "rom_mb":
                value *= {"gb": 1024, "kb": 1 / 1024}.get(unit, 1)
                unit = "MB"
            wrong_direction = ((direction == "min" and (op in _LOWER or suffix == "以上"))
                               or (direction == "max" and (op in _UPPER or suffix in {"以下", "以内"})))
            valid_range = (0 <= value <= 1 if name in _RATES else
                           -1 <= value <= 1 if name in {"plcc", "srcc"} else value >= 0)
            if wrong_direction or not valid_range:
                problems.append(f"{name}: conflicting comparator or out-of-range threshold")
            else:
                target = C.MetricTarget(name=name, direction=direction, threshold=value,
                                       strict=op in {"<", ">", "小于", "大于", "低于", "高于"},
                                       unit=unit if unit not in {"%", "％"} else "")
        else:
            problems.append(f"{name}: acceptance threshold not stated")
        previous = targets.get(name)
        if previous and previous.threshold != target.threshold:
            target.threshold, target.unknown = None, True
            problems.append(f"{name}: conflicting repeated targets")
        elif previous:
            target.strict = target.strict or previous.strict
        targets[name] = target
    ordered = list(targets.values())
    if primary in targets:
        ordered = [targets[primary]] + [t for t in ordered if t.name != primary]
    elif ordered:
        # Prefer a predictive quality metric over latency/size unless explicitly primary.
        ordered.sort(key=lambda t: t.name in {"latency_ms", "rom_mb"})
    return ordered, list(dict.fromkeys(problems))


def requirement_problems(spec: C.TaskSpec) -> List[str]:
    problems = []
    names = set()
    for target in [spec.primary_metric, *spec.secondary_metrics]:
        if target.name in names:
            problems.append(f"duplicate target: {target.name}")
        names.add(target.name)
        if target.threshold is None or target.unknown:
            problems.append(f"{target.name}: explicit acceptance threshold required")
        elif target.name in _RATES and not 0 <= target.threshold <= 1:
            problems.append(f"{target.name}: threshold must be in [0, 1]")
        elif target.name in {"plcc", "srcc"} and not -1 <= target.threshold <= 1:
            problems.append(f"{target.name}: threshold must be in [-1, 1]")
    return problems
