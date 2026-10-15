"""Assemble the full runtime (storage, providers, agents, graph) in one place.

Both the control service and the offline demo/tests construct the runtime
through :func:`build_runtime`, so there is a single canonical wiring and no
divergence between the service path and the CLI path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .. import docs_export
from ..agents import CodexCodingAgent, ResearchDesignAgent
from ..config import Settings, ensure_dirs
from ..evidence import DeltaBuilder, DependencyCache, EvidenceReader, EvidenceStore, PacketBuilder
from ..execution import Runner, Worker
from ..graph import GraphContext, build_graph, make_checkpointer
from ..providers import build_providers
from ..storage.repository import Repository
from .events import EventBus


@dataclass
class Runtime:
    settings: Settings
    repo: Repository
    bus: EventBus
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
    ctx: GraphContext
    graph: Any

    def start_workers(self) -> None:
        self.worker.start()

    def stop_workers(self) -> None:
        self.worker.stop()


def build_runtime(settings: Settings, *, inline: bool = True,
                  extras: Optional[Dict[str, Any]] = None) -> Runtime:
    ensure_dirs(settings)
    repo = Repository(settings.db_path)
    bus = EventBus(repo)
    providers = build_providers(settings)

    evidence = EvidenceStore(settings, repo)
    packets = PacketBuilder(settings, repo)
    reader = EvidenceReader(settings, repo)
    delta = DeltaBuilder(repo)
    cache = DependencyCache(repo)

    runner = Runner(settings, repo)
    worker = Worker(settings, repo, runner=runner)

    research_agent = ResearchDesignAgent(settings, repo, providers["research"], evidence, packets)
    coding_agent = CodexCodingAgent(settings, repo, providers["coding"], providers["coding_fallback"])

    ctx = GraphContext(
        settings=settings, repo=repo, providers=providers, research_agent=research_agent,
        coding_agent=coding_agent, runner=runner, worker=worker, evidence=evidence,
        packets=packets, reader=reader, delta=delta, cache=cache, inline=inline,
        docs=docs_export, extras=dict(extras or {}),
    )
    checkpointer = make_checkpointer(settings)
    graph = build_graph(ctx, checkpointer=checkpointer)
    return Runtime(settings=settings, repo=repo, bus=bus, providers=providers,
                   research_agent=research_agent, coding_agent=coding_agent, runner=runner,
                   worker=worker, evidence=evidence, packets=packets, reader=reader,
                   delta=delta, cache=cache, ctx=ctx, graph=graph)


__all__ = ["Runtime", "build_runtime"]
