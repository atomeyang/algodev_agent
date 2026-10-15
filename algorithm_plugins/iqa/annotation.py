# -*- coding: utf-8 -*-
"""Local-LLM assisted defect pre-labelling (data-construction stage).

The *annotation* helper for the light dual-output IQA task: it asks the local
auxiliary vision-language model (``local_largemodel/qwen3.8-27b``, served by
llama.cpp over an OpenAI-compatible endpoint) to propose a defect vector for a
candidate image.  The proposal is a **candidate label, never ground truth** —
it is written out for human review before it may enter a dataset version.

If the endpoint is unreachable the caller gets a structured error and the
pipeline falls back to the honest offline path; nothing is invented.
"""

from __future__ import annotations

import base64
import json
import re
import time
import hashlib
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C

DEFAULT_MODEL_ID = "qwen3.8-27b"
DEFAULT_BASE_URL = "http://127.0.0.1:8084/v1"
PROMPT_VERSION = "iqa-prelabel-v1"

_PROMPT = (
    "你是图像质量标注员，只做判断、不解释。对给定图片判断下列五类缺陷是否存在，"
    "并给出严重度：blur(模糊)、over_exp(过曝)、under_exp(欠曝)、flare(炫光/光晕)、"
    "white_balance(白平衡异常，含 cool 偏冷/warm 偏暖/green 偏绿/magenta 偏品红)。\n"
    "severity：0=无、1=轻、2=中、3=重；present 与 severity>0 必须一致。\n"
    "严格只输出一行 JSON，不要 markdown 代码块，不要多余文字：\n"
    '{"defects":{"blur":{"present":0,"severity":0},"over_exp":{"present":0,"severity":0},'
    '"under_exp":{"present":0,"severity":0},"flare":{"present":0,"severity":0},'
    '"white_balance":{"present":0,"severity":0,"direction":"none"}},'
    '"quality_score_0_100":0,"confidence_0_1":0.0}'
)


# --------------------------------------------------------------------------- #
# endpoint resolution
# --------------------------------------------------------------------------- #
def model_dir() -> Path:
    return C.REPO_ROOT / "local_largemodel" / DEFAULT_MODEL_ID


def endpoint_config() -> Dict[str, Any]:
    """Resolve endpoint from repo config, then env overrides, then defaults."""
    import os

    cfg: Dict[str, Any] = {"model_id": DEFAULT_MODEL_ID, "base_url": DEFAULT_BASE_URL,
                           "max_tokens": 2048, "temperature": 0.0}
    path = model_dir() / "configs" / "endpoint.yaml"
    if path.exists():
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            cfg["model_id"] = data.get("model_id", cfg["model_id"])
            cfg["base_url"] = data.get("base_url", cfg["base_url"])
            sampling = data.get("sampling") or {}
            cfg["max_tokens"] = int(sampling.get("max_tokens", cfg["max_tokens"]))
            cfg["temperature"] = float(sampling.get("temperature", cfg["temperature"]))
        except Exception:
            pass
    cfg["base_url"] = os.environ.get("ALGO_LOCAL_BASE_URL", cfg["base_url"]).rstrip("/")
    cfg["model_id"] = os.environ.get("ALGO_LOCAL_MODEL", cfg["model_id"])
    return cfg


def health(timeout: float = 5.0) -> Dict[str, Any]:
    """Probe the local endpoint. Returns {available, model_id, base_url, detail}."""
    cfg = endpoint_config()
    url = f"{cfg['base_url'].rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        ids = [m.get("id") for m in payload.get("data", [])] or \
              [m.get("name") for m in payload.get("models", [])]
        return {"available": True, "base_url": cfg["base_url"], "model_id": cfg["model_id"],
                "served": ids, "detail": "ok"}
    except Exception as exc:  # pragma: no cover - depends on the machine
        return {"available": False, "base_url": cfg["base_url"], "model_id": cfg["model_id"],
                "served": [], "detail": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced JSON object out of a (possibly chatty) answer."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalise(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a model answer into the frozen defect schema."""
    raw = payload.get("defects") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for d in C.DEFECTS:
        item = raw.get(d) or {}
        if not isinstance(item, dict):
            item = {}
        try:
            sev = int(round(float(item.get("severity", 0) or 0)))
        except (TypeError, ValueError):
            sev = 0
        sev = max(0, min(3, sev))
        present = item.get("present")
        if present is None:
            present = 1 if sev > 0 else 0
        present = 1 if int(bool(present)) or sev > 0 else 0
        entry: Dict[str, Any] = {"present": present, "severity": sev if present else 0}
        if d == "white_balance":
            direction = str(item.get("direction", "none") or "none").lower()
            entry["direction"] = direction if direction in C.WB_DIRECTIONS else "none"
        out[d] = entry
    score = payload.get("quality_score_0_100")
    try:
        score = None if score is None else max(0.0, min(100.0, float(score)))
    except (TypeError, ValueError):
        score = None
    return {"defects": out, "quality_score_0_100": score,
            "confidence_0_1": payload.get("confidence_0_1")}


def annotate(path: str, *, cfg: Optional[Dict[str, Any]] = None,
             timeout: float = 300.0, retries: int = 1, rubric: str = "") -> Dict[str, Any]:
    """Pre-label one image with the local VLM. Never raises on model failure."""
    cfg = cfg or endpoint_config()
    prompt = _PROMPT + ("\nAdditional annotation rubric (keep the JSON schema unchanged):\n" + rubric
                        if rubric else "")
    prompt_version = PROMPT_VERSION + ("-" + hashlib.sha256(prompt.encode()).hexdigest()[:12] if rubric else "")
    img = Path(path)
    started = time.time()
    if not img.exists():
        return {"path": path, "ok": False, "error": "image_not_found",
                "model_id": cfg["model_id"], "prompt_version": prompt_version}
    try:
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    except Exception as exc:
        return {"path": path, "ok": False, "error": f"read_failed: {exc}",
                "model_id": cfg["model_id"], "prompt_version": prompt_version}

    suffix = img.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    body = {"model": cfg["model_id"], "temperature": cfg.get("temperature", 0.0),
            "max_tokens": cfg.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]}
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"

    last_error = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
            message = (payload.get("choices") or [{}])[0].get("message") or {}
            text = message.get("content") or ""
            parsed = _extract_json(text)
            if parsed is None and message.get("reasoning_content"):
                parsed = _extract_json(message["reasoning_content"])
            if parsed is None:
                last_error = "no_json_in_answer"
                continue
            out = _normalise(parsed)
            try:
                schema_valid = (all(d in parsed.get("defects", {}) for d in C.DEFECTS)
                                and 0 <= float(parsed["quality_score_0_100"]) <= 100
                                and 0 <= float(parsed["confidence_0_1"]) <= 1)
                for d in C.DEFECTS:
                    item = parsed["defects"][d]
                    schema_valid &= (item["present"] in (0, 1) and item["severity"] in (0, 1, 2, 3)
                                     and bool(item["present"]) == (item["severity"] > 0))
            except (KeyError, TypeError, ValueError):
                schema_valid = False
            out.update({"path": path, "ok": True, "model_id": cfg["model_id"],
                        "prompt_version": prompt_version, "schema_valid": bool(schema_valid),
                        "elapsed_s": round(time.time() - started, 2),
                        "usage": payload.get("usage"),
                        "review_required": True,
                        "provenance": f"prelabel={cfg['model_id']}, prompt={prompt_version}"})
            return out
        except urllib.error.URLError as exc:
            last_error = f"endpoint_unreachable: {exc}"
        except Exception as exc:  # pragma: no cover
            last_error = f"{type(exc).__name__}: {exc}"
    return {"path": path, "ok": False, "error": last_error or "unknown",
            "model_id": cfg["model_id"], "prompt_version": prompt_version,
            "elapsed_s": round(time.time() - started, 2)}


def annotate_many(paths: List[str], *, limit: int = 0, cfg: Optional[Dict[str, Any]] = None,
                  timeout: float = 300.0, on_progress=None) -> List[Dict[str, Any]]:
    """Sequential pre-labelling (the local server is single-model; keep it serial)."""
    cfg = cfg or endpoint_config()
    todo = paths[:limit] if limit else list(paths)
    results: List[Dict[str, Any]] = []
    for i, p in enumerate(todo, 1):
        row = annotate(p, cfg=cfg, timeout=timeout)
        results.append(row)
        if on_progress:
            on_progress(i, len(todo), row)
    return results
