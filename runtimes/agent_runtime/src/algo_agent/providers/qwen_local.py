"""Local auxiliary model adapter (OpenAI-compatible endpoint)."""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import contracts as C
from .base import LocalModelProvider, ProviderStatus


class QwenLocalProvider(LocalModelProvider):
    name = "qwen_local"

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.env_get("ALGO_LOCAL_BASE_URL")
        self.model = settings.env_get("ALGO_LOCAL_MODEL") or "qwen3.8-27b"

    def status(self) -> ProviderStatus:
        if not self.base_url:
            return ProviderStatus(self.name, False, "local", "ALGO_LOCAL_BASE_URL not set")
        import httpx
        try:
            r = httpx.get(f"{self.base_url.rstrip('/')}/models", timeout=5.0)
            ok = r.status_code < 500
            return ProviderStatus(self.name, ok, "local", f"model={self.model}",
                                  capabilities={"endpoint_status": r.status_code})
        except Exception as exc:
            return ProviderStatus(self.name, False, "local", f"endpoint unreachable: {exc}")

    def diagnose(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.base_url:
            return {"status": "blocked", "reason": "local endpoint not configured"}
        import httpx
        prompt = ("Classify this algorithm regression issue into one of "
                  "[model_capacity, data_coverage, preprocessing, quantisation, runtime_config, metric_protocol]. "
                  f"Evidence: {request.get('summary', '')}. Answer as JSON with keys category, reason, reproducible.")
        try:
            r = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions",
                           json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
                           timeout=60.0)
            r.raise_for_status()
            import json
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            return {"status": "failed", "reason": str(exc)}

    def answer(self, question: str, context: str) -> str:
        """Conversational reply grounded in the live task context (chat /ask)."""
        if not self.base_url:
            return ""
        import httpx
        system = ("You are the algorithm R&D control agent. Answer strictly from the supplied live "
                  "task context; never invent numbers. Reply in the question's language, concise markdown.")
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content":
                                 f"# Live task context\n{context}\n\n# Question\n{question}"}],
                   "temperature": 0.2, "max_tokens": 2048}
        try:
            r = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload, timeout=240.0)
            r.raise_for_status()
            message = r.json()["choices"][0]["message"]
            return (message.get("content") or "").strip()
        except Exception:
            return ""
