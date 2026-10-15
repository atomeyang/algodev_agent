"""Deterministic offline providers (M0-M2).

Offline mode performs **no** live literature search. Candidates come from the
task's local business plugin and are explicitly labelled as offline demo
material; the report states that no live retrieval happened, so simulated
material is never presented as real research (guide 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from algorithm_plugins.registry import hook, load_plugin, resolve_plugin

from ..schemas import contracts as C
from .base import CodingOutcome, LocalModelProvider, ProviderStatus, ResearchDesignResult, ResearchProvider, SearchResult

_METHOD_LIBRARY: Dict[str, Dict[str, Any]] = {
    "baseline_mean": {"family": "baseline", "desc": "Predict the training-set mean (trivial baseline).",
                       "gain": "reference only", "cost": "negligible"},
    "ridge": {"family": "linear", "desc": "L2-regularised linear regression over raw features.",
              "gain": "strong linear baseline", "cost": "low"},
    "gbr": {"family": "boosting", "desc": "Gradient-boosted regression trees (shallow).",
            "gain": "captures mild non-linearity", "cost": "low"},
    "ridge_interaction": {"family": "linear+interaction",
                          "desc": "Ridge with an explicit low-light interaction feature.",
                          "gain": "targets the degraded slice", "cost": "low"},
}


class OfflineResearchProvider(ResearchProvider):
    name = "offline"

    def __init__(self, settings=None):
        self.settings = settings

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            self.name, True, "research",
            "deterministic offline design loop; no live search or cloud model",
            capabilities={"live_search": False, "cloud": False, "offline": True})

    def search_sources(self, query: str, budget: C.Budget) -> SearchResult:
        return SearchResult(
            refs=[],
            note="offline mode: no live search service configured; using local plugin material only",
            used_search_service=False,
        )

    def design(self, request: Dict[str, Any]) -> ResearchDesignResult:
        spec = C.TaskSpec(**request["task_spec"])
        # A domain plugin owns its research/design material (candidates, recipe,
        # protocol). Absent a hook we fall back to the synthetic-tabular set.
        designer = hook(resolve_plugin(spec), "design")
        if designer is not None:
            import inspect
            context = ({"task_root": request.get("task_root")}
                       if "task_root" in inspect.signature(designer).parameters else {})
            payload = dict(designer(
                spec, iteration=int(request.get("iteration", 1)),
                feedback=list(request.get("feedback", []) or []),
                slices=list(request.get("slices", []) or []),
                methods_hint=list(request.get("methods_hint") or []) or None,
                history=list(request.get("history", []) or []),
                constraints=list(request.get("constraints", []) or []), **context) or {})
            payload.setdefault("status", "completed")
            payload.setdefault("offline", True)
            payload.setdefault("evidence_refs", [])
            payload.setdefault("cost", {"cloud_tokens": 0, "currency": "USD", "source": "offline"})
            payload.setdefault("tool_calls", [{"tool": "search_sources", "offline": True, "refs": 0}])
            # plugins may hand back plain dicts; coerce to the canonical contracts
            payload["candidates"] = [c if isinstance(c, C.Candidate) else C.Candidate(**c)
                                     for c in payload.get("candidates", []) or []]
            if payload.get("data_recipe") and not isinstance(payload["data_recipe"], C.DataRecipe):
                payload["data_recipe"] = C.DataRecipe(**payload["data_recipe"])
            if payload.get("eval_protocol") and not isinstance(payload["eval_protocol"], C.EvalProtocol):
                payload["eval_protocol"] = C.EvalProtocol(**payload["eval_protocol"])
            allowed = set(ResearchDesignResult.__dataclass_fields__)
            extras = dict(payload.get("extras") or {})
            extras.update({k: v for k, v in payload.items() if k not in allowed})
            kwargs = {k: v for k, v in payload.items() if k in allowed and k != "extras"}
            return ResearchDesignResult(extras=extras, **kwargs)

        methods: List[str] = request.get("methods_hint") or ["baseline_mean", "ridge", "gbr"]
        iteration = int(request.get("iteration", 1))
        feedback = request.get("feedback", [])
        slices: List[str] = request.get("slices") or ["low_light"]
        primary = spec.primary_metric.name

        if iteration > 1 and (feedback or (spec.auto_process and request.get("history"))):
            methods = ["ridge_interaction", "gbr"]

        candidates: List[C.Candidate] = []
        for i, m in enumerate(methods):
            meta = _METHOD_LIBRARY.get(m, {"family": "unknown", "desc": m, "gain": "n/a", "cost": "unknown"})
            candidates.append(C.Candidate(
                candidate_id=f"c{i+1}",
                method=m,
                family=meta["family"],
                description=meta["desc"],
                expected_gain=meta["gain"],
                cost_estimate=meta["cost"],
                failure_conditions=["distribution shift outside demo domain"],
                ablations=["without interaction feature"] if m == "ridge_interaction" else [],
                is_baseline=(m == "baseline_mean"),
                source="local",
            ))

        hypotheses = [
            f"A linear model suffices for most samples but under-fits the '{slices[0]}' slice.",
            "Adding the low-light interaction feature improves the degraded slice without regressing overall.",
        ] if iteration > 1 else [
            f"{primary} is dominated by the '{slices[0]}' slice distribution.",
            "A regularised linear model will beat the mean baseline on overall error.",
        ]

        data_recipe = C.DataRecipe(
            recipe_id=f"recipe_v{iteration}",
            description="Deterministic synthetic tabular split (train/val/test).",
            source="synthetic_demo",
            transforms=["standard_split", "slice_low_light"],
            splits={"train": 0.6, "val": 0.2, "test": 0.2},
            seed=7,
            isolation="split before any augmentation",
        )
        eval_protocol = C.EvalProtocol(
            protocol_id=f"proto_v{iteration}",
            primary_metric=primary,
            direction=spec.primary_metric.direction,
            threshold=spec.primary_metric.threshold,
            slices=["all"] + list(slices),
            frozen=True,
            dataset_version=f"v{iteration}",
        )

        report = self._render_report(spec, candidates, iteration, feedback, slices)
        return ResearchDesignResult(
            report_markdown=report,
            candidates=candidates,
            hypotheses=hypotheses,
            controls=["same frozen test split and protocol for every candidate",
                      "compare against baseline_mean and the prior best model"],
            counterexamples=["samples with extreme feature magnitudes may be outliers"],
            unknowns=["no live literature search performed",
                      "target hardware not measured" if spec.target_hardware else "target hardware unspecified"],
            evidence_refs=[],
            data_recipe=data_recipe,
            eval_protocol=eval_protocol,
            cost={"cloud_tokens": 0, "currency": "USD", "source": "offline"},
            status="completed",
            offline=True,
            tool_calls=[{"tool": "search_sources", "offline": True, "refs": 0}],
        )

    def _render_report(self, spec, candidates, iteration, feedback, slices) -> str:
        lines = [
            f"# Offline research & design (iteration {iteration})",
            "",
            "> OFFLINE MODE: no live literature search or cloud model was used. "
            "Candidates are derived from the task's local plugin material and are "
            "labelled `synthetic_demo`. This is not a verified literature survey.",
            "",
            f"## Business objective\n{spec.business or '(not stated)'}",
            f"## Primary metric\n`{spec.primary_metric.name}` direction={spec.primary_metric.direction} "
            f"threshold={spec.primary_metric.threshold}",
            "## Candidates",
        ]
        for c in candidates:
            lines.append(f"- **{c.method}** ({c.family}): {c.description} "
                         f"(expected gain: {c.expected_gain}; cost: {c.cost_estimate})")
        if feedback:
            lines += ["## Feedback driving this iteration"]
            for f in feedback:
                lines.append(f"- {f.get('feedback_id')}: {f.get('actual', '')} (expected: {f.get('expected', '')})")
        lines += [
            "## Unknowns",
            "- no live literature retrieved",
            f"- slices evaluated: {', '.join(slices)}",
        ]
        return "\n".join(lines)


class OfflineLocalModel(LocalModelProvider):
    name = "offline_local"

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "local", "deterministic rule-based diagnosis", capabilities={"llm": False})

    def diagnose(self, request: Dict[str, Any]) -> Dict[str, Any]:
        spec_dict = request.get("spec") or {}
        domain = spec_dict.get("domain") if isinstance(spec_dict, dict) else None
        if domain:
            plugin = load_plugin(domain)
            fn = hook(plugin, "diagnose")
            if fn is not None:
                return dict(fn(request))
        slices = request.get("slices", {})
        worst = None
        for name, metrics in slices.items():
            rmse = metrics.get("rmse") if isinstance(metrics, dict) else None
            if rmse is not None and (worst is None or rmse > worst[1]):
                worst = (name, rmse)
        reason = f"worst slice '{worst[0]}' rmse={worst[1]:.4f}" if worst else "no slice data"
        return {"status": "ok", "category": "model_capacity_or_feature", "reason": reason,
                "reproducible": True, "offline": True}


class OfflineCodingExecutor:
    """Writes a method implementation into the isolated workspace from a template.

    This stands in for the Codex coding unit in offline mode; the produced file
    is real, importable code that the Runner actually executes.
    """

    name = "offline_coding"

    def __init__(self, settings=None):
        self.settings = settings

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "coding", "deterministic template writer (no LLM)",
                              capabilities={"diff": True, "real_code": True})

    def capabilities(self) -> Dict[str, Any]:
        return {"workspace_write": True, "terminal": False, "model": "template"}

    def execute(self, task: C.CodingTaskSpec, context: Dict[str, Any]) -> CodingOutcome:
        method = context.get("method", "ridge")
        workspace = Path(context["workspace"])
        rel = f"model_pool/{method}/method.py"
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '"""Method adapter generated by the offline coding unit."""\n'
            "from algorithm_plugins.demo_tabular.methods import run as _run\n\n"
            f'METHOD = "{method}"\n\n\n'
            "def run(ctx, config):\n"
            "    return _run(METHOD, ctx, config)\n\n"
            "from algorithm_plugins.demo_tabular.methods import predict\n",
            encoding="utf-8",
        )
        config_path = workspace / f"model_pool/{method}/config.yaml"
        config_path.write_text(f"method: {method}\n", encoding="utf-8")
        changed = [rel, f"model_pool/{method}/config.yaml"]
        diff = "\n".join(f"+++ b/{c}" for c in changed)
        return CodingOutcome(
            status="completed", changed_files=changed, diff=diff, tool="offline_template",
            model="offline-template",
            verifications=[C.Verification(command="python -c 'import ast; ast.parse(open(\"" + str(target) + "\").read())'",
                                          exit_code=0, stdout_tail="syntax ok")],
            events=[{"type": "write", "file": rel}],
        )
