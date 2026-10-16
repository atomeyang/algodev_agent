"""Provider factory: pick cloud/local/coding adapters from configuration."""

from __future__ import annotations

from typing import Any, Dict

from .base import (  # noqa: F401
    CodingExecutor,
    CodingOutcome,
    LocalModelProvider,
    ProviderStatus,
    ResearchDesignResult,
    ResearchProvider,
    SearchResult,
)
from .codex_cli import CodexCliExecutor
from .codex_ssh import SshCodexExecutor
from .deepseek import DeepSeekResearchProvider
from .offline import OfflineCodingExecutor, OfflineLocalModel, OfflineResearchProvider
from .qwen_local import QwenLocalProvider


def build_providers(settings, force_offline: bool | None = None) -> Dict[str, Any]:
    """Select research / local / coding adapters from configuration.

    Research/design is cloud-first (DeepSeek-V4-Flash) because that is where the
    reasoning quality matters; a local Qwen endpoint is the fallback for the
    *conversational* reply when no cloud credential is configured. When neither
    is available the deterministic offline path is used, so nothing is ever
    fabricated (``offline`` stays True in the report).
    """
    offline = settings.offline() if force_offline is None else force_offline
    if offline:
        cloud_ready = local_ready = False
    else:
        cloud_ready = settings.cloud_ready()
        local_ready = settings.local_ready()

    if offline:
        research = OfflineResearchProvider(settings)
        local = OfflineLocalModel()
        chat = None
    else:
        research = DeepSeekResearchProvider(settings) if cloud_ready else OfflineResearchProvider(settings)
        local = QwenLocalProvider(settings) if local_ready else OfflineLocalModel()
        # Conversational replies prefer the cloud model; otherwise the local endpoint.
        chat = research if cloud_ready else (local if local_ready else None)
        if cloud_ready and settings.env_get("ALGO_CHAT_MODEL"):
            chat = DeepSeekResearchProvider(settings, role="chat")

    transport = settings.env_get("ALGO_CODING_TRANSPORT") or "local"
    if transport not in {"local", "ssh"}:
        raise ValueError("ALGO_CODING_TRANSPORT must be local or ssh")
    codex = SshCodexExecutor(settings) if transport == "ssh" else CodexCliExecutor(settings)
    # pre-select coding executor; controller may still fall back per-task
    coding = codex
    intent = DeepSeekResearchProvider(settings, role="intent") if cloud_ready and settings.env_get("ALGO_INTENT_MODEL") else chat
    return {"research": research, "local": local, "coding": coding,
            "coding_fallback": OfflineCodingExecutor(settings), "offline": offline,
            "chat": chat, "cloud_ready": cloud_ready, "local_ready": local_ready,
            "codex": codex, "intent": intent}


def provider_report(settings) -> list:
    providers = build_providers(settings)
    report = [providers["research"].status(), providers["local"].status(), providers["codex"].status()]
    return report


def readiness_report(settings, *, probe: bool = False) -> Dict[str, Any]:
    """Configuration readiness is distinct from live model/auth verification."""
    import shutil
    from .search import search_backend
    providers = build_providers(settings)
    coding = providers["coding"]
    checks = {
        "research_configured": not providers["offline"] and providers["cloud_ready"],
        "search_configured": search_backend(settings) == "arxiv" or (
            search_backend(settings) == "http" and bool(settings.env_get("ALGO_SEARCH_URL"))),
        "coding_enabled": coding.allow_real,
        "coding_binary_found": shutil.which(coding.bin) is not None,
        "local_configured": providers["local_ready"],
    }
    blockers = [name for name in ("research_configured", "search_configured", "coding_enabled",
                                  "coding_binary_found") if not checks[name]]
    configured = coding.status()
    coding_probe = None
    if isinstance(coding, SshCodexExecutor):
        checks["coding_transport_configured"] = configured.available
        if not configured.available:
            blockers.append("coding_transport_configured")
        if probe:
            coding_probe = coding.probe()
            checks["coding_remote_ready"] = coding_probe.available
            if not coding_probe.available:
                blockers.append("coding_remote_ready")
    return {"checks": checks, "blockers": blockers, "configuration_ready": not blockers,
            "live_verified": False,
            "models": {role: getattr(providers.get(role), "model", None)
                       for role in ("chat", "intent", "research", "local", "coding")},
            "local_probe": providers["local"].status().__dict__ if probe else None,
            "coding_transport": configured.__dict__,
            "coding_probe": coding_probe.__dict__ if coding_probe else None,
            "warnings": ["Credentials/authentication and model generation require a separate smoke test.",
                         "Local experiment processes are not a security sandbox.",
                         "Codex internal token usage is not a hard-bounded provider budget."]}


__all__ = [
    "build_providers", "provider_report", "ResearchProvider", "LocalModelProvider", "CodingExecutor",
    "ResearchDesignResult", "CodingOutcome", "ProviderStatus", "SearchResult", "CodexCliExecutor", "SshCodexExecutor",
    "DeepSeekResearchProvider", "QwenLocalProvider", "OfflineResearchProvider", "OfflineLocalModel",
    "OfflineCodingExecutor",
]
