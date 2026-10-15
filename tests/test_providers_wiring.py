"""Provider selection + offline semantics + DeepSeek-flash research wiring."""

from __future__ import annotations

from pathlib import Path

from algo_agent.config import Settings
from algo_agent.providers import build_providers
from algo_agent.providers.base import ResearchDesignResult
from algo_agent.providers.deepseek import DEFAULT_MODEL, DeepSeekResearchProvider
from algo_agent.providers.offline import OfflineLocalModel, OfflineResearchProvider
from algo_agent.providers.qwen_local import QwenLocalProvider
from algo_agent.schemas import contracts as C


def _settings(env):
    return Settings(root=Path("."), workflow={}, env=dict(env))


def test_offline_when_nothing_configured():
    s = _settings({})
    assert s.offline() is True
    p = build_providers(s)
    assert isinstance(p["research"], OfflineResearchProvider)
    assert p["chat"] is None
    assert p["offline"] is True


def test_algo_offline_env_forces_offline_even_with_credentials():
    s = _settings({"ALGO_OFFLINE": "1", "ALGO_CLOUD_API_KEY": "k"})
    assert s.offline() is True
    assert build_providers(s)["offline"] is True


def test_cloud_key_selects_deepseek_flash_research_and_chat():
    s = _settings({"ALGO_CLOUD_API_KEY": "k"})
    assert s.offline() is False
    p = build_providers(s)
    assert isinstance(p["research"], DeepSeekResearchProvider)
    assert p["research"].model == DEFAULT_MODEL == "deepseek-flash"
    assert p["chat"] is p["research"]              # chat prefers the cloud model
    assert isinstance(p["local"], OfflineLocalModel)  # no local endpoint configured
    assert p["offline"] is False


def test_local_only_is_online_but_research_stays_deepseek_only():
    s = _settings({"ALGO_LOCAL_BASE_URL": "http://127.0.0.1:8084/v1",
                   "ALGO_LOCAL_MODEL": "qwen3.8-27b"})
    assert s.offline() is False
    p = build_providers(s)
    assert isinstance(p["research"], OfflineResearchProvider)  # research = DeepSeek when keyed
    assert isinstance(p["local"], QwenLocalProvider)
    assert p["chat"] is p["local"]                 # local model answers the chat


def test_cloud_model_override():
    s = _settings({"ALGO_CLOUD_API_KEY": "k", "ALGO_CLOUD_MODEL": "deepseek-v4-pro"})
    assert build_providers(s)["research"].model == "deepseek-v4-pro"


def test_parse_json_handles_fences_and_embedded_objects():
    D = DeepSeekResearchProvider
    assert D._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert D._parse_json('prefix {"a": 2} suffix') == {"a": 2}
    assert D._parse_json("") is None


def test_answer_returns_trimmed_content(monkeypatch):
    prov = DeepSeekResearchProvider(_settings({"ALGO_CLOUD_API_KEY": "k"}))
    monkeypatch.setattr(prov, "_chat", lambda *a, **k: {"content": "  the answer  ", "usage": {}})
    assert prov.answer("q", "ctx") == "the answer"


def test_design_constrains_candidates_to_plugin_catalog(monkeypatch):
    prov = DeepSeekResearchProvider(_settings({"ALGO_CLOUD_API_KEY": "k"}))
    base = ResearchDesignResult(
        report_markdown="## 1. 当前产出精度状态\nmeasured numbers\n\n"
                        "## 2. 模型 & 损失函数 & 训练策略优化计划\nplan",
        candidates=[C.Candidate(candidate_id="mobileclip2_mos_head", method="mobileclip2_mos_head",
                                family="transfer+ridge-head", description="baseline",
                                expected_gain="g", cost_estimate="c")],
    )
    monkeypatch.setattr(prov, "_domain_grounding", lambda request: base)
    monkeypatch.setattr(prov, "_chat", lambda *a, **k: {
        "content": '{"report_markdown": "### analysis\\nswitch to lora",'
                   ' "candidates": [{"method": "mobileclip2_mos_head", "description": "better"},'
                   ' {"method": "invented_method"}]}',
        "usage": {"prompt_tokens": 5}})
    res = prov.design({})
    assert [c.method for c in res.candidates] == ["mobileclip2_mos_head"]
    assert res.candidates[0].description == "better"
    assert res.offline is False
    assert "LLM research analysis" in res.report_markdown
    assert "## 1. 当前产出精度状态" in res.report_markdown  # structured sections preserved


def test_design_degrades_to_domain_template_on_cloud_failure(monkeypatch):
    prov = DeepSeekResearchProvider(_settings({"ALGO_CLOUD_API_KEY": "k"}))
    base = ResearchDesignResult(report_markdown="## 1. 当前产出精度状态\nx",
                                candidates=[C.Candidate(candidate_id="m", method="m")])
    monkeypatch.setattr(prov, "_domain_grounding", lambda request: base)

    def _boom(*a, **k):
        raise RuntimeError("HTTP 500 upsteam")

    monkeypatch.setattr(prov, "_chat", _boom)
    res = prov.design({})
    assert res.offline is False
    assert [c.method for c in res.candidates] == ["m"]
    assert any("degraded to the domain template" in u for u in res.unknowns)
