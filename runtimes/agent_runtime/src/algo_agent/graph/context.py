"""Dependency-injection context passed to every graph node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..agents import CodexCodingAgent, ResearchDesignAgent
from ..config import Settings
from ..evidence import DeltaBuilder, DependencyCache, EvidenceReader, EvidenceStore, PacketBuilder
from ..execution import Runner, Worker
from ..storage.repository import Repository


@dataclass
class GraphContext:
    settings: Settings
    repo: Repository
    providers: Dict[str, Any]
    research_agent: ResearchDesignAgent
    coding_agent: CodexCodingAgent
    runner: Runner
    worker: Worker
    evidence: EvidenceStore
    packets: PacketBuilder
    reader: EvidenceReader
    delta: DeltaBuilder
    cache: DependencyCache
    inline: bool = True          # run jobs synchronously (offline demo / tests)
    docs: Optional[Any] = None   # docs_export module handle
    extras: Dict[str, Any] = field(default_factory=dict)


__all__ = ["GraphContext"]
