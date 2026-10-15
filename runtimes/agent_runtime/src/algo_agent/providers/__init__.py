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

    codex = CodexCliExecutor(settings)
    # pre-select coding executor; controller may still fall back per-task
    coding = codex
    return {"research": research, "local": local, "coding": coding,
            "coding_fallback": OfflineCodingExecutor(settings), "offline": offline,
            "chat": chat, "cloud_ready": cloud_ready, "local_ready": local_ready,
            "codex": codex}


def provider_report(settings) -> list:
    providers = build_providers(settings)
    report = [providers["research"].status(), providers["local"].status(), providers["codex"].status()]
    return report


__all__ = [
    "build_providers", "provider_report", "ResearchProvider", "LocalModelProvider", "CodingExecutor",
    "ResearchDesignResult", "CodingOutcome", "ProviderStatus", "SearchResult", "CodexCliExecutor",
    "DeepSeekResearchProvider", "QwenLocalProvider", "OfflineResearchProvider", "OfflineLocalModel",
    "OfflineCodingExecutor",
]
