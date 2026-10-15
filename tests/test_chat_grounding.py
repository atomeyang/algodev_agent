"""Chat REPL grounding: free-text questions must answer, not silently record feedback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from algo_agent.cli.chat import _extract_sections, _pick_sections


SAMPLE = """# report

## 1. 当前产出精度状态 (iteration 1)
plcc table here

## 2. 模型 & 损失函数 & 训练策略优化计划
opt-mos-frozen-head and friends

## 3. 数据标注调优计划 & 数据扩增计划
annotation and augmentation
"""


def test_extract_sections_splits_on_h2():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.md"
        p.write_text(SAMPLE, encoding="utf-8")
        secs = _extract_sections(str(p))
    titles = [s["title"] for s in secs]
    assert len(secs) == 3
    assert "当前产出精度状态" in titles[0]
    assert "数据扩增" in titles[2]
    assert "opt-mos-frozen-head" in secs[1]["text"]


def test_pick_sections_routes_by_intent():
    secs = [{"title": s["title"], "text": s["text"]} for s in _extract_sections(_write(SAMPLE))]
    assert [t for t, _ in _pick_sections("下一步改进计划是什么", secs)][0].startswith("2.")
    assert [t for t, _ in _pick_sections("现在什么情况了", secs)][0].startswith("1.")
    assert [t for t, _ in _pick_sections("数据标注和扩增怎么做", secs)][0].startswith("3.")


def test_pick_sections_returns_empty_for_unrelated():
    secs = [{"title": s["title"], "text": s["text"]} for s in _extract_sections(_write(SAMPLE))]
    assert _pick_sections("hello there", secs) == []


def _write(text: str) -> str:
    import tempfile as _t
    d = _t.mkdtemp()
    p = Path(d) / "r.md"
    p.write_text(text, encoding="utf-8")
    return str(p)
