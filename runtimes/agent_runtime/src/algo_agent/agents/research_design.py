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
from ..paths import DIRECTORY_RULES, TaskPaths, safe_join, sha256_obj
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
        call_key = f"research:{request.task_spec.task_id}:{request.iteration}"
        request_hash = sha256_obj({"task": request.task_spec.model_dump(), "parent": request.parent_plan_id,
                                  "history": request.history, "feedback": request.feedback,
                                  "slices": request.slices, "methods_hint": request.methods_hint,
                                  "provider": getattr(self.provider, "name", type(self.provider).__name__),
                                  "model": getattr(self.provider, "model", "")})
        completed = self.repo.research_step(call_key + ":completed", request_hash)
        if completed:
            return ResearchDesignResult.from_payload(completed)
        if not self.repo.research_step(call_key, request_hash):
            self.repo.claim_budget(request.task_spec.task_id, "research_calls", 1,
                                   request.budget.max_research_calls)
            self.repo.save_research_step(call_key, request.task_spec.task_id, request_hash, {"started": True})
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
        query = (request.task_spec.research_queries or [request.task_spec.domain.replace("_", " ")])[0]
        search = self._search(request, query)
        registered = [r["evidence_id"] for r in search["refs"]]
        tool_calls.append({"tool": "search_sources", "refs": len(registered),
                           "used_search_service": search["used_search_service"], "note": search["note"]})
        if request.task_spec.require_live_research and not (
                search["used_search_service"] and registered):
            return ResearchDesignResult(status="blocked", unknowns=[
                "live research required but source discovery returned no evidence", search["note"]])

        # --- internal loop 2: build evidence packet ---
        packet = self.packets.build_packet(
            task_id=request.task_spec.task_id, requirement=request.task_spec.business,
            plan_id=request.parent_plan_id, refs=registered,
            unknowns=list(request.task_spec.unknowns) + ([search["note"]] if search["note"] else []))
        tool_calls.append({"tool": "build_packet", "refs": len(packet.refs)})

        # --- internal loop 3: design ---
        provider_request = {
            "directory_contract": DIRECTORY_RULES,
            "task_root": str(TaskPaths(self.settings, request.task_slug).root),
            "task_spec": request.task_spec.model_dump(),
            "evidence_packet": packet.model_dump(),
            "iteration": request.iteration,
            "feedback": request.feedback[-16:],
            "history": request.history[-8:],
            "slices": request.slices,
            "methods_hint": request.methods_hint,
            "constraints": [hc.model_dump() for hc in request.task_spec.hard_constraints],
        }
        from ..memory import ResearchMemory, bounded_json
        provider_request["research_memory"] = json.loads(bounded_json(
            ResearchMemory(self.settings, self.repo).recall(request.task_spec.task_id), 16000))
        provider_request["search_documents"] = search["documents"]
        provider_request["data_operation_schema"] = C.DataOperation.model_json_schema()
        from ..execution.data_pipeline import operation_capabilities
        provider_request["data_operation_capabilities"] = operation_capabilities(request.task_spec.domain)
        from algorithm_plugins.registry import hook, resolve_plugin
        context_files = hook(resolve_plugin(request.task_spec), "coding_context_files")
        provider_request["reference_code_files"] = list(context_files(request.task_spec)) if context_files else []
        provider_request["data_sources"] = [
            s.model_dump(exclude={"path"}) for s in request.task_spec.data_sources]
        provider_request["available_tools"] = {
            "search_sources": {"query": "string"},
            "read_evidence": {"evidence_id": "registered task evidence ID", "range": "0:100"},
            "get_code_excerpt": {"path": "model_pool/... or algorithm_plugins/...", "start": 0, "lines": 100},
            "get_run_summary": {"run_id": "development run belonging to this task"},
            "inspect_data": {},
            "search_assets": {"asset_type": "optional asset type", "query": "optional search terms"},
            "read_asset": {"asset_id": "registered asset", "version": "version", "filename": "body file"},
            "recall_lessons": {"query": "hypothesis or bottleneck to investigate"},
        }
        provider_request["observations"] = []
        provider_request["budget_ledger"] = self.repo.get_budget(request.task_spec.task_id)
        # Freeze the initial frame so checkpoint replay has stable evidence IDs,
        # packet timestamps and memory instead of issuing a subtly different query.
        frame_key = call_key + ":frame"
        frame = self.repo.research_step(frame_key, request_hash)
        if frame is None:
            self.repo.save_research_step(frame_key, request.task_spec.task_id, request_hash, provider_request)
        else:
            provider_request = frame
        if hasattr(self.provider, "_chat"):
            provider_request["_chat"] = lambda *a, **kw: self._budgeted_chat(request, *a, **kw)
        remaining = request.budget.max_research_steps if request.task_spec.auto_process else 1
        result = ResearchDesignResult(status="blocked", unknowns=["research step budget exhausted"])
        while remaining > 0:
            check_running(self.repo, request.task_spec.task_id)
            if remaining <= 2:
                provider_request["available_tools"] = {}
                provider_request["finalize_now"] = (
                    "Tool budget is closed. Return the final executable candidates now, or explicitly "
                    "state why blocked. At most one format-repair call remains; do not request more tools.")
            remaining -= 1
            provider_request["remaining_steps"] = remaining
            result = self.provider.design(provider_request)
            if result.status == "completed" and request.task_spec.auto_process:
                problems = self._plan_problems(request, result)
                if problems:
                    tool_calls.append({"tool": "plan_validation", "errors": problems})
                    provider_request["observations"].append({
                        "tool": "plan_validation", "errors": problems,
                        "draft": json.loads(bounded_json({
                            "candidates": [c.model_dump() for c in result.candidates],
                            "data_operations": result.extras.get("data_operations", [])}, 12000)),
                        "instruction": "Repair the plan within the original task contract and budgets."})
                    result.status = "failed"
                    result.unknowns.extend(problems)
            if result.status == "failed" and remaining > 0 and request.task_spec.auto_process:
                provider_request["observations"].append({
                    "tool": "schema_validation", "errors": result.unknowns,
                    "instruction": "Repair the structured plan; do not claim success without candidates."})
                continue
            if result.status != "needs_tools":
                break
            requests = result.extras.get("tool_requests", [])
            if (not isinstance(requests, list) or not requests
                    or any(not isinstance(c, dict) or not isinstance(c.get("tool"), str)
                           or not isinstance(c.get("arguments", {}), dict) for c in requests)):
                result.status = "failed"
                result.unknowns.append("invalid research tool request")
                if remaining > 0 and request.task_spec.auto_process:
                    provider_request["observations"].append({
                        "tool": "schema_validation",
                        "errors": ["tool_requests must be a list of {tool: string, arguments: object}"]})
                    continue
                break
            if remaining <= 1:
                provider_request["observations"].append({
                    "tool": "budget", "error": "No further tools allowed. Finalise the plan in the next response."})
                continue
            for call in requests:
                # Keep one synthesis call and one schema-repair call; tool batches
                # must not consume the only opportunity to return a plan.
                if remaining <= 2:
                    provider_request["observations"].append({
                        "tool": "budget", "skipped_tools": True,
                        "instruction": "Remaining tool requests skipped to reserve final plan synthesis."})
                    break
                remaining -= 1
                check_running(self.repo, request.task_spec.task_id)
                try:
                    identity = sha256_obj({"request": request_hash, "call": call})
                    key = f"{call_key}:tool:{identity}"
                    observation = self.repo.research_step(key, identity)
                    if observation is None:
                        observation = self._tool(request, call["tool"], call.get("arguments") or {}, invocation_id)
                        if len(json.dumps(observation, ensure_ascii=False)) <= 18000:
                            self.repo.save_research_step(key, request.task_spec.task_id, identity, observation)
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
        registered = [r.evidence_id for r in self.evidence.list(request.task_spec.task_id)]
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
        if result.status == "completed":
            self.repo.save_research_step(call_key + ":completed", request.task_spec.task_id,
                                         request_hash, result.payload())
        return result

    def _plan_problems(self, request, result):
        from ..graph.nodes.design_plan import build_plan_spec
        from ..policy import plan_problems, data_budget_problems
        try:
            plan = build_plan_spec(request.task_spec, result.payload(), request.iteration,
                                   request.parent_plan_id, self.settings)
            previous = self.repo.get_plan(request.parent_plan_id) if request.parent_plan_id else None
            ledger = self.repo.get_budget(request.task_spec.task_id)
            return list(dict.fromkeys(
                plan_problems(request.task_spec, plan, previous) +
                data_budget_problems(request.task_spec, plan, ledger["used"], ledger["reserved"])))
        except (ValueError, TypeError, KeyError) as exc:
            return [f"invalid plan schema: {str(exc)[:1500]}"]

    def _search(self, request, query):
        from ..providers.search import search_backend
        query = str(query).strip()[:2000]
        identity = sha256_obj({"query": query, "backend": search_backend(self.settings),
                              "url": self.settings.env_get("ALGO_SEARCH_URL"),
                              "provider": getattr(self.provider, "name", type(self.provider).__name__)})
        key = f"search:{request.task_spec.task_id}:{request.iteration}:{identity}"
        cached = self.repo.research_step(key, identity)
        if cached is not None:
            return cached
        result = self.provider.search_sources(query, request.budget)
        refs = [self.evidence.register_source(
            request.task_spec.task_id, r.uri, source_type=r.source_type, locator=r.locator,
            version=r.version, upload_policy=r.upload_policy, trust=r.trust).model_dump()
                for r in result.refs[:8]]
        ids = {r["uri"]: r["evidence_id"] for r in refs}
        documents = [{**d, "evidence_id": ids.get(d.get("url"))} for d in result.documents[:8]]
        payload = {"refs": refs, "documents": documents, "query": query,
                   "used_search_service": result.used_search_service, "note": result.note}
        self.repo.save_research_step(key, request.task_spec.task_id, identity, payload)
        return payload

    def _budgeted_chat(self, request, system, user, **kwargs):
        check_running(self.repo, request.task_spec.task_id)
        identity = sha256_obj({"system": system, "user": user, "kwargs": kwargs,
                              "provider": getattr(self.provider, "name", type(self.provider).__name__),
                              "model": getattr(self.provider, "model", "")})
        key = f"model:{request.task_spec.task_id}:{request.iteration}:{identity}"
        cached = self.repo.research_step(key, identity)
        if cached is not None:
            return cached
        limit = request.budget.max_cloud_tokens
        estimate = len((system + user).encode("utf-8")) + int(kwargs.get("max_tokens", 8192))
        self.repo.reserve_budget(request.task_spec.task_id, "cloud_tokens", estimate, limit)
        usage = {}
        try:
            kwargs["timeout"] = max(1, min(240, remaining_seconds(request.task_spec)))
            response = self.provider._chat(system, user, **kwargs)
            usage = response.get("usage") or {}
            self.repo.save_research_step(key, request.task_spec.task_id, identity,
                                         {k: v for k, v in response.items()
                                          if k not in {"raw_response", "reasoning"}})
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
            return self._search(request, arguments["query"])
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
            summary = development_summary(tp)
            if not summary["releases"]:
                from algorithm_plugins.registry import hook, resolve_plugin
                preview = hook(resolve_plugin(request.task_spec), "inspect_development_data")
                if preview:
                    summary["source_preview"] = preview(request.task_spec)
            return summary
        if name == "search_assets":
            from ..reuse.compat import ReuseRequest, compatibility_for
            from ..memory import terms
            wanted = terms(str(arguments.get("query", request.task_spec.business)))
            assets = [a for a in self.repo.search_assets(asset_type=arguments.get("asset_type"))
                      if a.validation_status in {"researched", "reproduced", "business_validated", "reusable"}]
            assets.sort(key=lambda a: len(wanted & terms(a.asset_id + " " + a.source_task)), reverse=True)
            reuse = ReuseRequest(inputs=request.task_spec.inputs, outputs=request.task_spec.outputs,
                                 device=request.task_spec.target_hardware or "")
            return {"assets": [{**a.model_dump(), "compatibility": compatibility_for(
                self.repo, a.asset_id, a.asset_version, reuse).model_dump()} for a in assets[:12]]}
        if name == "read_asset":
            from ..storage.assets import AssetRegistry
            from ..paths import sha256_file
            registry = AssetRegistry(self.settings, self.repo)
            asset = registry.get(arguments["asset_id"], arguments["version"])
            if asset.validation_status == "deprecated" or asset.upload_policy != "open":
                raise ValueError("asset is deprecated or its body is not authorised for research upload")
            filename = arguments["filename"]
            path = safe_join(registry.asset_dir(asset.asset_id, asset.asset_version, asset.asset_type), filename)
            if filename not in asset.checksums or sha256_file(path) != asset.checksums[filename]:
                raise ValueError("asset body not registered or changed")
            if path.suffix not in {".md", ".txt", ".json", ".yaml", ".py"} or path.stat().st_size > 16000:
                raise ValueError("bounded text asset required")
            return {"asset_id": asset.asset_id, "content": path.read_text(), "verified_hash": asset.checksums[filename]}
        if name == "recall_lessons":
            from ..memory import ResearchMemory, bounded_json
            return {"lessons": json.loads(bounded_json(ResearchMemory(self.settings, self.repo).recall(
                task_id, str(arguments.get("query", ""))), 16000))}
        raise ValueError(f"unsupported research tool: {name}")


__all__ = ["ResearchDesignAgent", "ResearchDesignRequest"]
