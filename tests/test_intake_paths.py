"""Requirement parsing and path-traversal safety (guide 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from algo_agent.graph.nodes.intake import parse_requirement
from algo_agent.paths import safe_join, slugify


def test_parse_requirement_extracts_structured_fields():
    text = "在移动端对图像做质量分预测，RMSE 不超过 0.3，包大小不超过 20MB，输入为 RGB 图像"
    spec = parse_requirement(text, "t1")
    assert spec.primary_metric.name == "rmse"
    assert spec.primary_metric.direction == "min"
    assert spec.primary_metric.threshold == pytest.approx(0.3)
    assert spec.target_hardware == "mobile"
    assert spec.inputs == ["rgb_image"]
    assert spec.outputs == ["quality_score"]
    assert any(c.name == "package_size_mb" and c.value == 20 for c in spec.hard_constraints)
    assert spec.unknowns == []


def test_parse_requirement_records_unknowns_without_faking_values():
    spec = parse_requirement("improve the model a bit", "t2")
    assert spec.primary_metric.threshold is None
    assert spec.primary_metric.unknown is True
    assert spec.unknowns, "unstated fields must be recorded as unknowns"
    assert spec.target_hardware is None


def test_slugify_normalises_and_prefixes_digits():
    assert slugify("My Task / v2!") == "my_task_v2"
    assert slugify("123 numbers").startswith("t_")


def test_safe_join_rejects_traversal_and_absolute_paths(tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    assert safe_join(base, "a", "b.txt") == (base / "a" / "b.txt").resolve()
    with pytest.raises(ValueError):
        safe_join(base, "..", "escape.txt")
    with pytest.raises(ValueError):
        safe_join(base, str(tmp_path / "elsewhere"))


def test_task_dir_follows_the_task_name_experiment_convention(tmp_path: Path):
    """Guide 4: the task assets live in <task_name>_experiment/ under the platform root."""
    from algo_agent.config import load_settings
    from algo_agent.paths import TaskPaths

    settings = load_settings(str(tmp_path))
    assert TaskPaths(settings, "iqa").root == tmp_path / "iqa_experiment"
    assert TaskPaths(settings, "iqa_rd").root == tmp_path / "iqa_rd_experiment"


def test_iqa_driver_defaults_to_platform_root_and_task_name():
    """Regression guard: --root is the platform root, so the task dir is not nested."""
    import re
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "scripts" / "run_iqa_loop.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--root", default=str(REPO)' in src
    assert 'ap.add_argument("--task-id", default="iqa"' in src
    assert re.search(r'if root\.name\.endswith\("_experiment"\)', src)
