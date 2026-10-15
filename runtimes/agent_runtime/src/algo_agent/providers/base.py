"""Provider interfaces: research/design, local auxiliary model, coding executor.

Adapters are swappable (guide 5, 13). Every adapter reports whether it is
``available`` and why not, so a missing credential or endpoint becomes a
diagnosable blocker rather than a fabricated success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schemas import contracts as C


@dataclass
class ProviderStatus:
    name: str
    available: bool
    kind: str = ""
    detail: str = ""
    version: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    refs: List[C.EvidenceRef] = field(default_factory=list)
    note: str = ""
    used_search_service: bool = False


@dataclass
class ResearchDesignResult:
    report_markdown: str = ""
    candidates: List[C.Candidate] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    data_recipe: Optional[C.DataRecipe] = None
    eval_protocol: Optional[C.EvalProtocol] = None
    cost: Dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    change_requests: List[C.ChangeRequest] = field(default_factory=list)
    offline: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # Domain plugins may return extra structured sections (e.g. the IQA
    # precision-status / optimisation-plan / data-plan bundle). They are kept
    # verbatim so they survive into research_iter{N}.json instead of being
    # silently dropped by the dataclass field filter.
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodingOutcome:
    status: str = "completed"
    changed_files: List[str] = field(default_factory=list)
    diff: str = ""
    verifications: List[C.Verification] = field(default_factory=list)
    tool: str = "offline"
    model: str = "offline-template"
    open_issues: List[str] = field(default_factory=list)
    change_requests: List[C.ChangeRequest] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class ResearchProvider:
    name = "base"

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, False, "research", "not implemented")

    def search_sources(self, query: str, budget: C.Budget) -> SearchResult:
        return SearchResult(note="search not implemented")

    def design(self, request: Dict[str, Any]) -> ResearchDesignResult:
        raise NotImplementedError


class LocalModelProvider:
    name = "base"

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, False, "local", "not implemented")

    def diagnose(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "blocked", "reason": "local model not implemented"}


class CodingExecutor:
    name = "base"

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, False, "coding", "not implemented")

    def capabilities(self) -> Dict[str, Any]:
        return {}

    def execute(self, task: C.CodingTaskSpec, context: Dict[str, Any]) -> CodingOutcome:
        raise NotImplementedError
