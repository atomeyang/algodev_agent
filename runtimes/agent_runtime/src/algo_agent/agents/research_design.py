"""ResearchDesignAgent: autonomous research/design unit (guide 10).

Runs a bounded internal loop (search -> register evidence -> pack -> design) and
returns a structured ``ResearchDesignResult``. It never fabricates research: in
offline mode the provider marks results as offline material.
"""

from __future__ import annotations

import uuid
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..evidence import EvidenceStore, PacketBuilder, EvidenceReader
from ..paths import DIRECTORY_RULES, TaskPaths, safe_join
from ..policy import check_running, remaining_seconds
from ..providers import ResearchDesignResult, ResearchProvider
from ..schemas import contracts as C
from ..schemas.enums import AgentRole, ResultStatus
from ..storage.repository import Repository, BudgetExceeded


@dataclass
class ResearchDesignRequest:
    task_spec: C.TaskSpec
    task_slug: str
    iteration: int = 1
    parent_plan_id: str = ""
    budget: C.Budget = field(default_factory=C.Budget)
    slices: List[str] = field(default_factory=list)
    methods_hint: List[str] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    invocation_id: str = ""


class ResearchDesignAgent:
    role = AgentRole.RESEARCH_DESIGN.value

    def __init__(self, settings: Settings, repo: Repository, provider: ResearchProvider,
                 evidence: Optional[EvidenceStore] = None, packet_builder: Optional[PacketBuilder] = None):
        self.settings = settings
        self.repo = repo
        self.provider = provider
        self.evidence = evidence or EvidenceStore(settings, repo)
        self.packets = packet_builder or PacketBuilder(settings, repo)

    def run(self, request: ResearchDesignRequest) -> ResearchDesignResult:
        check_running(self.repo, request.task_spec.task_id)
        self.repo.claim_budget(request.task_spec.task_id, "research_calls", 1,
                               request.budget.max_research_calls)
        invocation_id = request.invocation_id or f"inv_{uuid.uuid4().hex[:10]}"
        baseline = request.parent_plan_id or "base_v1"
        session_id = self.repo.find_session(request.task_spec.task_id, self.role, request.parent_plan_id, baseline) \
            or f"sess_{uuid.uuid4().hex[:10]}"
        self.repo.upsert_session(session_id, request.task_spec.task_id, self.role,
                                 request.parent_plan_id, baseline)
        self.repo.save_invocation(C.AgentInvocation(
            invocation_id=invocation_id, task_id=request.task_spec.task_id, role=self.role,
            plan_id=request.parent_plan_id, session_id=session_id, budget=request.budget))

        # --- internal loop 1: search (bounded, single call per invocation) ---
        tool_calls: List[Dict[str, Any]] = []
        search = self.provider.search_sources(request.task_spec.business or request.task_spec.task_slug,
                                              request.budget)
        registered: List[str] = []
        for ref in search.refs:
            stored = self.evidence.register_source(
                request.task_spec.task_id, ref.uri, source_type=ref.source_type, locator=ref.locator,
                version=ref.version, upload_policy=ref.upload_policy, trust=ref.trust)
            registered.append(stored.evidence_id)
        tool_calls.append({"tool": "search_sources", "refs": len(registered),
                           "used_search_service": search.used_search_service, "note": search.note})

        # --- internal loop 2: build evidence packet ---
        packet = self.packets.build_packet(
            task_id=request.task_spec.task_id, requirement=request.task_spec.business,
            plan_id=request.parent_plan_id, refs=registered,
            unknowns=list(request.task_spec.unknowns) + ([search.note] if search.note else []))
        tool_calls.append({"tool": "build_packet", "refs": len(packet.refs)})

        # --- internal loop 3: design ---
        provider_request = {
            "directory_contract": DIRECTORY_RULES,
            "task_root": str(TaskPaths(self.settings, request.task_slug).root),
            "task_spec": request.task_spec.model_dump(),
            "evidence_packet": packet.model_dump(),
            "iteration": request.iteration,
            "feedback": request.feedback,
            "history": request.history,
            "slices": request.slices,
            "methods_hint": request.methods_hint,
            "constraints": [hc.model_dump() for hc in request.task_spec.hard_constraints],
        }
        provider_request["data_operation_schema"] = C.DataOperation.model_json_schema()
        from ..execution.data_pipeline import operation_capabilities
        provider_request["data_operation_capabilities"] = operation_capabilities(request.task_spec.domain)
        provider_request["data_sources"] = [
            s.model_dump(exclude={"path"}) for s in request.task_spec.data_sources]
        provider_request["available_tools"] = {
            "search_sources": {"query": "string"},
            "read_evidence": {"evidence_id": "registered task evidence ID", "range": "0:100"},
            "get_code_excerpt": {"path": "model_pool/... or algorithm_plugins/...", "start": 0, "lines": 100},
            "get_run_summary": {"run_id": "development run belonging to this task"},
            "inspect_data": {},
            "search_assets": {"asset_type": "optional asset type"},
        }
        provider_request["observations"] = []
        if hasattr(self.provider, "_chat"):
            provider_request["_chat"] = lambda *a, **kw: self._budgeted_chat(request, *a, **kw)
        remaining = request.budget.max_research_steps if request.task_spec.auto_process else 1
        result = ResearchDesignResult(status="blocked", unknowns=["research step budget exhausted"])
        while remaining > 0:
            check_running(self.repo, request.task_spec.task_id)
            remaining -= 1
            provider_request["remaining_steps"] = remaining
            result = self.provider.design(provider_request)
            if result.status == "failed" and remaining > 0 and request.task_spec.auto_process:
                provider_request["observations"].append({
                    "tool": "schema_validation", "errors": result.unknowns,
                    "instruction": "Repair the structured plan; do not claim success without candidates."})
                continue
            if result.status != "needs_tools":
                break
            requests = result.extras.get("tool_requests", [])
            if not isinstance(requests, list) or not requests:
                result.status = "failed"
                result.unknowns.append("invalid research tool request")
                break
            for call in requests:
                if remaining <= 0:
                    break
                remaining -= 1
                check_running(self.repo, request.task_spec.task_id)
                try:
                    observation = self._tool(request, call["tool"], call.get("arguments") or {}, invocation_id)
                except Exception as exc:
                    observation = {"status": "failed", "error": str(exc)[:1000]}
                # Bound individual observations, without truncating JSON into invalid syntax.
                encoded = json.dumps(observation, ensure_ascii=False)
                if len(encoded) > 18000:
                    observation = {"status": "too_large", "message": "narrow the requested range"}
                provider_request["observations"].append({"call": call, "result": observation})
                provider_request["observations"] = provider_request["observations"][-8:]
                tool_calls.append({"tool": call.get("tool"), "result": observation})
        if result.status == "needs_tools":
            result.status = "blocked"
            result.unknowns.append("research tool/model step budget exhausted before a plan")
        result.tool_calls = tool_calls + list(result.tool_calls)
        result.evidence_refs = list(dict.fromkeys(list(result.evidence_refs) + registered))

        # --- structured-output validation (bounded repair) ---
        try:
            result.candidates = [C.Candidate(**c.model_dump()) if isinstance(c, C.Candidate) else C.Candidate(**c)
                                 for c in result.candidates]
        except Exception as exc:
            result.status = ResultStatus.FAILED.value
            result.unknowns.append(f"candidate schema invalid: {exc}")
        if not result.candidates and result.status == ResultStatus.COMPLETED.value:
            result.status = ResultStatus.BLOCKED.value
            result.unknowns.append("no viable candidates produced")

        self.repo.append_event(request.task_spec.task_id, "agent.research_design",
                               f"research design iteration {request.iteration}: {len(result.candidates)} candidates",
                               data={"invocation_id": invocation_id, "session_id": session_id,
                                     "offline": result.offline, "status": result.status,
                                     "tool_calls": result.tool_calls})
        return result

    def _budgeted_chat(self, request, system, user, **kwargs):
        check_running(self.repo, request.task_spec.task_id)
        limit = request.budget.max_cloud_tokens
        estimate = len((system + user).encode("utf-8")) + int(kwargs.get("max_tokens", 8192))
        self.repo.reserve_budget(request.task_spec.task_id, "cloud_tokens", estimate, limit)
        usage = {}
        try:
            kwargs["timeout"] = max(1, min(240, remaining_seconds(request.task_spec)))
            response = self.provider._chat(system, user, **kwargs)
            usage = response.get("usage") or {}
            return response
        finally:
            total = usage.get("total_tokens")
            if total is None and all(k in usage for k in ("prompt_tokens", "completion_tokens")):
                total = usage["prompt_tokens"] + usage["completion_tokens"]
            self.repo.settle_budget(request.task_spec.task_id, "cloud_tokens",
                                    used=estimate if total is None else max(0, total), reserved=estimate)
            self.repo.add_ledger_entry(
                request.task_spec.task_id, self.role, "research", usage.get("prompt_tokens"),
                usage.get("completion_tokens"), None, "USD", "unknown",
                {**usage, "estimated_charge": estimate if total is None else None})

    def _tool(self, request, name, arguments, invocation_id):
        task_id = request.task_spec.task_id
        tp = TaskPaths(self.settings, request.task_slug)
        reader = EvidenceReader(self.settings, self.repo)
        if name == "search_sources":
            result = self.provider.search_sources(str(arguments["query"])[:2000], request.budget)
            refs = [self.evidence.register_source(
                task_id, r.uri, source_type=r.source_type, locator=r.locator).model_dump()
                    for r in result.refs[:8]]
            return {"refs": refs, "note": result.note}
        if name == "read_evidence":
            ref = next((r for r in self.evidence.list(task_id)
                        if r.evidence_id == arguments["evidence_id"]), None)
            if ref is None:
                raise ValueError("evidence does not belong to this task")
            if ref.uri.startswith(("https://", "http://")):
                return reader.read_remote(ref, invocation_id)
            return reader.read_evidence(ref, tp.slug, self.role, arguments.get("range", "full"), invocation_id)
        if name == "get_code_excerpt":
            path = str(arguments["path"])
            if path.startswith("model_pool/"):
                actual = safe_join(tp.root, path)
            elif path.startswith(f"algorithm_plugins/{request.task_spec.domain}/"):
                import algorithm_plugins
                actual = safe_join(Path(algorithm_plugins.__file__).parent.parent, path)
            else:
                raise ValueError("only task methods and the selected plugin's source may be inspected")
            if actual.suffix != ".py" or actual.stat().st_size > 300000:
                raise ValueError("bounded Python source only")
            start = max(0, int(arguments.get("start", 0)))
            lines = max(1, min(150, int(arguments.get("lines", 100))))
            return {"path": path, "content": "\n".join(actual.read_text().splitlines()[start:start + lines])}
        if name == "get_run_summary":
            run_id = arguments["run_id"]
            experiment = self.repo.get_experiment(run_id)
            if experiment is None or experiment.task_id != task_id or experiment.env.get("stage") == "final":
                raise ValueError("not a development run for this task")
            job = next((j for j in self.repo.list_jobs(task_id) if j.run_id == run_id), None)
            return {"experiment": experiment.model_dump(), "status": job.status if job else "missing",
                    "result": job.result if job else {}}
        if name == "inspect_data":
            from ..execution.data_pipeline import development_summary
            return development_summary(tp)
        if name == "search_assets":
            return {"assets": [a.model_dump() for a in self.repo.search_assets(
                asset_type=arguments.get("asset_type"), status="published")[:12]]}
        raise ValueError(f"unsupported research tool: {name}")


__all__ = ["ResearchDesignAgent", "ResearchDesignRequest"]
