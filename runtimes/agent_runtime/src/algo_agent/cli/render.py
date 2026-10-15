"""Rich rendering helpers shared by the CLI commands and the chat REPL.

Every function accepts plain dicts (the JSON contracts) so the same renderer
works for both the embedded controller and the HTTP service.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)

_LEVEL_STYLE = {"info": "dim", "notice": "cyan", "warning": "yellow", "error": "bold red"}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def banner(text: str) -> None:
    console.print(Panel(Text(text, style="bold cyan"), expand=False))


def info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def warn(msg: str) -> None:
    err_console.print(f"[yellow]{msg}[/yellow]")


def error(msg: str) -> None:
    err_console.print(f"[bold red]{msg}[/bold red]")


def state_panel(state: Dict[str, Any]) -> None:
    task = state.get("task", {}) or {}
    kv = Table.grid(padding=(0, 2))
    kv.add_column(style="bold")
    kv.add_column()
    kv.add_row("task", f"{task.get('task_id', state.get('task_id', '?'))}  (slug={task.get('task_slug', '?')})")
    kv.add_row("phase", _fmt(state.get("phase")))
    kv.add_row("iteration", _fmt(state.get("iteration")))
    kv.add_row("active plan", _fmt(state.get("active_plan_id")))
    kv.add_row("best model", _fmt(state.get("best_model_id")))
    kv.add_row("paused", _fmt(state.get("paused")))
    kv.add_row("state version", _fmt(state.get("state_version")))
    if state.get("stop_reason"):
        kv.add_row("stop reason", _fmt(state.get("stop_reason")))
    budget = state.get("budget_used", {}) or {}
    kv.add_row("budget used", _fmt(budget.get("used")))
    if task.get("primary_metric"):
        m = task["primary_metric"]
        kv.add_row("primary metric", f"{m.get('name')} {m.get('direction')} {m.get('threshold')}")
    console.print(Panel(kv, title="task state", expand=False))
    pending = state.get("pending_approval")
    if pending:
        approval_panel(pending)


def approval_panel(pending: Dict[str, Any]) -> None:
    kind = pending.get("type", "interrupt")
    body = Table.grid(padding=(0, 2))
    body.add_column(style="bold")
    body.add_column()
    body.add_row("type", kind)
    if pending.get("plan_id"):
        body.add_row("plan", f"{pending.get('plan_id')} ({pending.get('plan_version', '?')})")
    if pending.get("plan_hash"):
        body.add_row("plan hash", pending["plan_hash"])
    if pending.get("iteration") is not None:
        body.add_row("iteration", str(pending["iteration"]))
    if pending.get("best_model"):
        body.add_row("best model", str(pending["best_model"]))
    if pending.get("metrics"):
        body.add_row("metrics", ", ".join(f"{k}={_fmt(v)}" for k, v in pending["metrics"].items()))
    if pending.get("candidates"):
        body.add_row("candidates", ", ".join(pending["candidates"]))
    if pending.get("message"):
        body.add_row("message", pending["message"])
    console.print(Panel(body, title=f"waiting at {kind}", border_style="yellow", expand=False))
    if kind == "approval_request":
        info("approve with: algo-agent approve --task <id> --plan %s --hash %s"
             % (pending.get("plan_id", "?"), (pending.get("plan_hash") or "")[:16]))
    elif kind == "iteration_boundary":
        info("submit /feedback to iterate, or /cancel to stop")


def plan_panel(view: Dict[str, Any]) -> None:
    plan = view.get("plan", {}) or {}
    console.print(Panel(Text(plan.get("summary") or "(no summary)", style="white"),
                        title=f"plan {plan.get('plan_id')} ({plan.get('version')})", expand=False))
    if plan.get("candidates"):
        t = Table("method", "hypothesis", "params")
        for c in plan["candidates"]:
            t.add_row(c.get("method", "?"), (c.get("hypothesis") or "")[:60],
                      str((plan.get("change_scope", {}) or {}).get("parameter_space", {}).get(c.get("method"), {})))
        console.print(t)
    if plan.get("eval_protocol"):
        p = plan["eval_protocol"]
        info(f"protocol {p.get('protocol_id')}: {p.get('primary_metric')} {p.get('direction')} <= {p.get('threshold')}")
    info(f"plan hash: {view.get('plan_hash')}")


def metrics_table(rows: Iterable[Dict[str, Any]], title: str = "evaluations") -> None:
    rows = list(rows)
    if not rows:
        info("(no evaluations yet)")
        return
    metric_keys: List[str] = []
    for r in rows:
        for k in (r.get("metrics") or {}):
            if k not in metric_keys:
                metric_keys.append(k)
    t = Table(title=title)
    t.add_column("model")
    t.add_column("passed")
    for k in metric_keys[:4]:
        t.add_column(k)
    for r in rows:
        style = "green" if r.get("passed") else "red"
        t.add_row(r.get("model_id", "?"), Text(str(r.get("passed")), style=style),
                  *[_fmt((r.get("metrics") or {}).get(k)) for k in metric_keys[:4]])
    console.print(t)


def jobs_table(jobs: List[Dict[str, Any]]) -> None:
    if not jobs:
        info("(no jobs)")
        return
    t = Table("job", "kind", "status", "run", "result")
    for j in jobs:
        style = {"succeeded": "green", "failed": "red", "running": "yellow"}.get(j.get("status"), "white")
        t.add_row(j.get("job_id", "?"), j.get("kind", ""), Text(j.get("status", ""), style=style),
                  j.get("run_id", ""), str(j.get("result", {}))[:50])
    console.print(t)


def events_table(events: List[Dict[str, Any]]) -> None:
    if not events:
        info("(no events)")
        return
    t = Table("seq", "kind", "level", "message")
    for e in events:
        t.add_row(str(e.get("event_seq", "")), e.get("kind", ""),
                  Text(e.get("level", "info"), style=_LEVEL_STYLE.get(e.get("level", "info"), "white")),
                  (e.get("message") or "")[:80])
    console.print(t)


def budget_panel(budget: Dict[str, Any]) -> None:
    limits = budget.get("limits", {}) or {}
    used = budget.get("used", {}) or {}
    t = Table("budget", "limit", "used")
    for k, v in limits.items():
        t.add_row(k, _fmt(v), _fmt(used.get(k)))
    console.print(Panel(t, title="budget", expand=False))
    ledger = budget.get("ledger", []) or []
    if ledger:
        lt = Table("role", "kind", "cost", "at")
        for row in ledger[-8:]:
            lt.add_row(row.get("role", ""), row.get("kind", ""), _fmt(row.get("cost")), row.get("created_at", "")[:19])
        console.print(lt)


def tasks_table(tasks: List[Dict[str, Any]]) -> None:
    if not tasks:
        info("(no tasks)")
        return
    t = Table("task", "phase", "iteration", "best model", "updated")
    for task in tasks:
        t.add_row(task.get("task_id", "?"), task.get("phase", ""), str(task.get("iteration", "")),
                  str(task.get("best_model_id") or "-"), str(task.get("updated_at", ""))[:19])
    console.print(t)


def doctor_report(report: Dict[str, Any]) -> None:
    console.print(Panel(f"root: {report.get('root')}\noffline: {report.get('offline')}  ok: {report.get('ok')}",
                        title="doctor", expand=False))
    t = Table("check", "ok", "level", "detail")
    for c in report.get("checks", []):
        style = "green" if c.get("ok") else {"error": "red", "warning": "yellow"}.get(c.get("level"), "yellow")
        t.add_row(c.get("name", ""), Text(str(c.get("ok")), style=style), c.get("level", ""), c.get("detail", ""))
    console.print(t)


def assets_table(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        info("(no assets)")
        return
    t = Table("asset", "version", "type", "status", "source", "license")
    for a in rows:
        style = {"reusable": "green", "business_validated": "cyan",
                 "deprecated": "red"}.get(a.get("validation_status"), "white")
        t.add_row(a.get("asset_id", "?"), a.get("asset_version", ""), a.get("asset_type", ""),
                  Text(str(a.get("validation_status", "")), style=style),
                  a.get("source_task", "") or "-", a.get("license", "") or "-")
    console.print(t)


def compat_panel(result: Dict[str, Any]) -> None:
    status = result.get("status", "unknown")
    style = {"compatible": "green", "adaptation_required": "yellow",
             "unknown": "cyan", "incompatible": "red"}.get(status, "white")
    console.print(Panel(Text(status, style=style), title="compatibility", expand=False))
    for key, label in (("reasons", "reason"), ("missing", "missing"),
                       ("adaptation", "adaptation"), ("verification_actions", "action")):
        for line in result.get(key) or []:
            info(f"{label}: {line}")


_KEY_METRICS = ("plcc", "srcc", "grade_agreement", "defect_macro_f1",
                "hardcase_macro_f1", "rom_mb", "n_scored")


def answer_panel(state: Dict[str, Any], evals: Iterable[Dict[str, Any]],
                 plan: Optional[Dict[str, Any]], best_eval: Optional[Dict[str, Any]],
                 *, question: str = "", design_doc: str = "") -> None:
    """Grounded, offline reply to a free-text question about the active task.

    The answer is assembled deterministically from the live task state, the
    stored evaluations and the active plan -- no language model is called. This
    keeps the REPL honest in offline mode: it reports what actually happened.
    """
    if question:
        console.print(Panel(Text(question, style="cyan"), title="you", expand=False))
    st = state or {}
    st_lines = [
        f"phase={st.get('phase')}  iteration={st.get('iteration')}  "
        f"paused={st.get('paused')}  plan={st.get('active_plan_id') or '-'}",
        f"best_model={st.get('best_model_id') or '-'}  budget_used={st.get('budget_used')}",
    ]
    if st.get("stop_reason"):
        st_lines.append(f"stop_reason={st.get('stop_reason')}")
    console.print(Panel(Text("\n".join(st_lines)), title="task status (live)", expand=False))

    def _ordered(m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        m = m or {}
        keys = [k for k in _KEY_METRICS if k in m] + [k for k in m if k not in _KEY_METRICS]
        return {k: m[k] for k in keys}

    rows = [{"model_id": e.get("model_id"), "passed": e.get("passed"),
             "metrics": _ordered(e.get("metrics"))} for e in (evals or [])]
    if rows:
        metrics_table(rows, title="evaluations")
    if best_eval:
        m = best_eval.get("metrics") or {}
        head = f"model={best_eval.get('model_id')}  passed={best_eval.get('passed')}"
        body = "\n".join(f"{k} = {_fmt(m[k])}" for k in _KEY_METRICS if k in m)
        console.print(Panel(Text(head + "\n" + body), title="best model", expand=False))

    if plan:
        plan_panel(plan)

    hints = ["OFFLINE answer: assembled from the task state / evaluations / plan "
             "(no cloud LLM called)."]
    if design_doc:
        hints.append("设计报告(1 当前精度状态 / 2 模型&损失&训练计划 / 3 数据标注&扩增计划): "
                     + design_doc)
    hints.append("提交真实反馈用 /feedback key=value；完整查看用 /report。")
    console.print(Panel(Text("\n".join(hints), style="dim"), title="notes", expand=False))


def answer_text(question: str, body: str, *, source: str = "", model: str = "") -> None:
    """Focused answer: echo the question, then render the matched content."""
    if question:
        console.print(Panel(Text(question, style="cyan"), title="you", expand=False))
    console.print(Markdown(body or "(no content)"))
    if source and source not in ("offline",):
        tag = source + (f":{model}" if model else "")
        console.print(f"[dim]answered by {tag} (live)[/dim]")


def markdown(text: str) -> None:
    console.print(Markdown(text or ""))


def print_json(obj: Any) -> None:
    import json
    console.print_json(json.dumps(obj, ensure_ascii=False, default=str))


__all__ = [
    "console", "err_console", "banner", "info", "warn", "error", "state_panel",
    "approval_panel", "plan_panel", "metrics_table", "jobs_table", "events_table",
    "budget_panel", "tasks_table", "doctor_report", "markdown", "print_json",
    "assets_table", "compat_panel",
]
