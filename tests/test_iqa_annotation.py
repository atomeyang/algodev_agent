# -*- coding: utf-8 -*-
"""Local-VLM annotation helper tests (torch-free, no network)."""

from __future__ import annotations

from algorithm_plugins.iqa import annotation as A


def test_extract_json_from_fenced_block():
    text = 'sure!\n```json\n{"a": 1}\n```\nthere you go'
    assert A._extract_json(text) == {"a": 1}


def test_extract_json_from_balanced_object_with_chatter():
    text = 'analysis ... {"defects": {"blur": {"present": 1, "severity": 2}}} done'
    out = A._extract_json(text)
    assert out["defects"]["blur"]["severity"] == 2


def test_extract_json_returns_none_for_nothing_parseable():
    assert A._extract_json("") is None
    assert A._extract_json("no json here") is None


def test_normalise_clamps_severity_and_coerces_presence():
    payload = {
        "defects": {
            "blur": {"present": 1, "severity": 9},
            "over_exp": {"present": 0, "severity": 3},
            "white_balance": {"present": 1, "severity": 2, "direction": "WARM"},
        },
        "quality_score_0_100": 150,
        "confidence_0_1": 0.7,
    }
    out = A._normalise(payload)
    assert out["defects"]["blur"] == {"present": 1, "severity": 3}
    # present and severity must be consistent (prompt invariant); a stray
    # severity>0 with present=0 resolves to present=1 rather than dropping signal
    assert out["defects"]["over_exp"] == {"present": 1, "severity": 3}
    assert out["defects"]["white_balance"]["direction"] == "warm"
    assert out["quality_score_0_100"] == 100.0
    assert set(out["defects"]) == set(A.C.DEFECTS)


def test_normalise_defaults_direction_to_none_when_invalid():
    out = A._normalise({"defects": {"white_balance": {"present": 1, "severity": 1,
                                                      "direction": "purple"}}})
    assert out["defects"]["white_balance"]["direction"] == "none"


def test_normalise_severity_implies_present_when_flag_missing():
    out = A._normalise({"defects": {"flare": {"severity": 2}}})
    assert out["defects"]["flare"]["present"] == 1


def test_endpoint_config_points_at_the_local_served_model():
    cfg = A.endpoint_config()
    assert cfg["model_id"] == A.DEFAULT_MODEL_ID
    assert cfg["base_url"].startswith("http")
    assert cfg["max_tokens"] >= 2048  # reasoning models need headroom


def test_model_dir_is_under_local_largemodel():
    d = A.model_dir()
    assert d.parent.name == "local_largemodel"
    assert d.name == "qwen3.8-27b"
