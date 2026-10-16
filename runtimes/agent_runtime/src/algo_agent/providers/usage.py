"""Account for auxiliary calls without trusting missing provider usage as zero."""

from __future__ import annotations

from ..schemas import contracts as C


def tracked_call(repo, spec, role: str, text: str, max_output: int, call):
    estimate = len(text.encode("utf-8")) + max_output
    repo.reserve_budget(spec.task_id, "cloud_tokens", estimate, spec.budget.max_cloud_tokens)
    usage = {}
    try:
        result = call()
        if isinstance(result, dict):
            usage = result.get("usage") or {}
        return result
    finally:
        total = usage.get("total_tokens")
        if total is None and all(k in usage for k in ("prompt_tokens", "completion_tokens")):
            total = usage["prompt_tokens"] + usage["completion_tokens"]
        used = estimate if not isinstance(total, (int, float)) or total < 0 else total
        repo.settle_budget(spec.task_id, "cloud_tokens", used=used, reserved=estimate)
        repo.add_ledger_entry(spec.task_id, role, "model_call", usage.get("prompt_tokens"),
                              usage.get("completion_tokens"), None, "USD", "unknown",
                              {**usage, "estimated_tokens": estimate if used == estimate else None})
