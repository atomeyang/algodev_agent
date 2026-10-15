"""Interactive chat REPL (prompt_toolkit) that drives the same controller.

Slash commands reuse the exact service contract; free text is treated as a new
requirement when there is no active task, otherwise as feedback for the active
task. Ctrl+C clears the current input and never kills a running job.
"""

from __future__ import annotations

from pathlib import Path

import shlex
from typing import Any, Dict, List, Optional

from ..config import Settings
from . import render
from .client import ControlClient, ClientError

HELP = """\
Available commands:
  /new [requirement]         create a new task (optionally with its requirement text)
  /use <task_id>             switch the active task
  /tasks                     list tasks
  /status                    show task phase / plan / best model
  /plan [plan_id]            show the active (or given) plan
  /approve [plan_id] [hash]  approve the pending plan (hash optional if unchanged)
  /feedback key=value ...    submit feedback (scenario, actual, expected, model_id, severity)
  /ask <question>            answer via the live chat model (grounded; state-only fallback)
  /improve <goal>            drive the research+coding sub-agents toward a further improvement
  /report [--full]           status + evaluations + full design report
  /pause /resume /cancel     boundary pause, resume, and cancel (never deletes assets)
  /budget                    show budget limits / usage / ledger
  /events [n]                show the last n events
  /export                    export overview + iteration docs
  /attach <path>             register a local file as an evidence reference (embedded mode)
  /help                      show this help
  /quit                      exit (Ctrl+D also works)
"""


class ChatSession:
    def __init__(self, settings: Settings, client: ControlClient, *, task_id: Optional[str] = None):
        self.settings = settings
        self.client = client
        self.task_id = task_id
        self._history_file = settings.state_dir / "chat_history"

    # ---- entry point --------------------------------------------------- #
    def run(self) -> int:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        session = PromptSession(history=FileHistory(str(self._history_file)))
        render.banner("algo-agent chat  (type /help for commands; Ctrl+C clears input, Ctrl+D quits)")
        render.info(f"root={self.settings.root}  mode={self.client.mode}  offline={self.settings.offline()}")
        if self.task_id:
            try:
                self._show_status()
            except ClientError as exc:
                render.warn(f"task {self.task_id} not found: {exc.message}")

        while True:
            prompt = f"[{self.task_id}]> " if self.task_id else "algo-agent> "
            try:
                line = session.prompt(prompt)
            except KeyboardInterrupt:  # Ctrl+C clears input, keeps the session alive
                render.info("(input cleared — jobs keep running)")
                continue
            except EOFError:  # Ctrl+D
                render.info("bye")
                return 0
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    if self._dispatch(line) == "quit":
                        render.info("bye")
                        return 0
                else:
                    self._free_text(line)
            except ClientError as exc:
                render.error(f"error: {exc.message}")
                if exc.recovery:
                    render.info(f"hint: {exc.recovery}")
            except Exception as exc:  # pragma: no cover - defensive
                render.error(f"unexpected error: {exc!r}")

    # ---- helpers ------------------------------------------------------- #
    def _need_task(self) -> str:
        if not self.task_id:
            raise ClientError("no active task", recovery="create one with /new <requirement>")
        return self.task_id

    def _show_status(self) -> Dict[str, Any]:
        st = self.client.state(self._need_task())
        render.state_panel(st)
        return st

    def _boundary(self, boundary: Dict[str, Any]) -> None:
        if boundary.get("type") == "end":
            render.info("task reached the end of the graph")
            self._show_status()
        else:
            pending = boundary.get("interrupt") or {}
            render.approval_panel(pending)

    def _free_text(self, text: str) -> None:
        if not self.task_id:
            self._do_new(text)
            return
        tokens = text.split()
        # `key=value ...` is an explicit feedback payload (kept for back-compat).
        if "=" in text and all("=" in tok for tok in tokens):
            payload: Dict[str, Any] = {}
            for tok in tokens:
                key, _, value = tok.partition("=")
                payload[key] = value
            res = self.client.submit_feedback(self.task_id, payload)
            render.info(f"feedback {res.get('feedback_id')} recorded")
            if res.get("continued"):
                self._boundary(res.get("boundary", {}))
            else:
                self._maybe_run()
            return
        # Plain text is a question: answer it from the live, grounded state.
        self._answer(text)

    def _grounded(self):
        task_id = self._need_task()
        st = self.client.state(task_id)
        try:
            evals = self.client.evals(task_id)
        except ClientError:
            evals = []
        best_id = st.get("best_model_id")
        best = next((e for e in evals if e.get("model_id") == best_id), None)
        if best is None and evals:
            best = evals[-1]
        plan = None
        plan_id = st.get("active_plan_id") or (st.get("pending_approval") or {}).get("plan_id")
        if plan_id:
            try:
                plan = self.client.plan_view(task_id, plan_id)
            except ClientError:
                plan = None
        return st, evals, plan, best

    def _design_doc(self) -> str:
        try:
            from ..paths import TaskPaths, slugify
            research = TaskPaths(self.settings, slugify(self.task_id)).root / "docs" / "research"
            docs = sorted(research.glob("research_iter*.md")) if research.is_dir() else []
            if docs:
                return str(docs[-1])
        except Exception:  # pragma: no cover - defensive
            pass
        return ""

    def _measured_block(self) -> str:
        """Live Evaluator numbers, shown before the design-time baseline table."""
        st, evals, plan, best = self._grounded()
        m = (best or {}).get("metrics") or {}
        if not m:
            return ""
        return (
            "**实测(本轮 Evaluator,冻结官方真实 MOS 测试集)**\n\n"
            f"- best model: `{st.get('best_model_id')}`  passed: `{(best or {}).get('passed')}`"
            f"  n_scored: {m.get('n_scored')}\n"
            f"- plcc = {m.get('plcc')} (floor 0.85)  srcc = {m.get('srcc')} (floor 0.85)\n"
            f"- grade_agreement = {m.get('grade_agreement')}  rom_mb = {m.get('rom_mb')}\n\n"
            "*下表为设计阶段(迭代 1 开始前)的参考基线快照,仅作起点对照:*\n\n"
        )

    def _llm_answer(self, question: str) -> bool:
        """Ask the configured chat model (DeepSeek/local); True when it replied."""
        if not self.task_id:
            return False
        try:
            res = self.client.answer(self.task_id, question)
        except ClientError as exc:
            render.warn(f"chat model unavailable ({exc.message}); answering from grounded state")
            return False
        if res and res.get("answer"):
            render.answer_text(question, res["answer"],
                               source=res.get("source", ""), model=res.get("model", ""))
            return True
        if res and res.get("source") == "error":
            render.warn(f"chat model error: {res.get('error')}; answering from grounded state")
        return False

    def _answer(self, question: str) -> None:
        """Answer a free-text question, preferring the live chat model."""
        if self._llm_answer(question):
            return
        doc = self._design_doc()
        sections = _extract_sections(doc) if doc else []
        picked = _pick_sections(question, sections)
        if picked:
            body = "\n\n".join(f"### {title}\n\n{text}".strip() for title, text in picked)
            if any("当前产出精度状态" in title for title, _ in picked):
                body = self._measured_block() + body
        else:
            st, evals, plan, best = self._grounded()
            m = (best or {}).get("metrics") or {}
            body = (
                f"- task: `{st.get('task_id')}`  phase: `{st.get('phase')}`  "
                f"plan: `{st.get('active_plan_id')}`\n"
                f"- best model: `{st.get('best_model_id')}`\n"
                f"- metrics: plcc={m.get('plcc')} srcc={m.get('srcc')} "
                f"grade_agreement={m.get('grade_agreement')} rom_mb={m.get('rom_mb')}\n"
            )
            if doc:
                body += f"- design report: `{doc}`"
        render.answer_text(question, body)

    def _report(self, *, full: bool = False) -> None:
        st, evals, plan, best = self._grounded()
        doc = self._design_doc()
        render.answer_panel(st, evals, plan, best, question="", design_doc=doc)
        if not full:
            return
        # The full three-part design report lives on disk (plan.summary is only a
        # short abstract), so render the file itself when asked.
        if doc:
            try:
                render.info(f"design report: {doc}")
                render.markdown(Path(doc).read_text(encoding="utf-8"))
                return
            except OSError as exc:  # pragma: no cover - defensive
                render.warn(f"could not read design report: {exc}")
        summary = (plan.get("plan") or {}).get("summary") if plan else ""
        if summary:
            render.info("design report (plan summary)")
            render.markdown(summary)

    def _improve(self, goal: str) -> None:
        """Drive the research/coding sub-agents toward a further improvement.

        On a live (non-terminal) task this submits feedback, which wakes the
        diagnose -> research -> design loop. On a terminal task it starts a new
        task that references the current best model, so the loop can iterate
        again without resuming a finished graph.
        """
        if not self.task_id:
            self._do_new(goal)
            return
        st = self.client.state(self.task_id)
        best = st.get("best_model_id") or ""
        if st.get("phase") in ControlClient.TERMINAL_PHASES:
            request = (f"IQA 继续改进:在 {self.task_id} 的最佳模型 {best} 基础上,{goal} "
                       f"主指标 PLCC/SRCC,冻结官方真实 MOS 测试集,不降低 ROM 与缺陷约束。")
            render.info(f"task {self.task_id} is terminal ({st.get('phase')}); starting a new task")
            self._do_new(request)
            return
        payload: Dict[str, Any] = {"scenario": "improvement",
                                   "actual": f"current best {best}", "expected": goal}
        if best:
            payload["model_id"] = best
        try:
            res = self.client.submit_feedback(self.task_id, payload)
        except ClientError as exc:
            render.warn(f"feedback rejected ({exc.message}); advancing the graph instead")
            self._maybe_run()
            return
        render.info(f"feedback {res.get('feedback_id')} recorded (drives diagnose -> research -> design)")
        if res.get("continued"):
            self._boundary(res.get("boundary", {}))
        else:
            self._maybe_run()

    def _maybe_run(self) -> None:
        st = self.client.state(self.task_id)
        if st.get("pending_approval"):
            render.state_panel(st)
            return
        if st.get("phase") not in ControlClient.TERMINAL_PHASES and not st.get("paused"):
            boundary = self.client.run(self.task_id)
            self._boundary(boundary)

    def _do_new(self, text: str, *, demo: bool = False) -> None:
        created = self.client.create_task(text, demo=demo, synthetic=demo)
        self.task_id = created["task_id"]
        render.info(f"created task {self.task_id} (slug={created['task_slug']})")
        boundary = self.client.run(self.task_id, raw_text=text)
        self._boundary(boundary)

    # ---- commands ------------------------------------------------------ #
    def _dispatch(self, line: str) -> Optional[str]:
        parts = shlex.split(line)
        cmd, args = parts[0], parts[1:]
        if cmd in ("/quit", "/exit"):
            return "quit"
        if cmd == "/help":
            render.console.print(HELP)
        elif cmd == "/new":
            text = " ".join(args) if args else ""
            if not text:
                from prompt_toolkit import prompt
                text = prompt("describe the requirement: ").strip()
            if text:
                self._do_new(text)
        elif cmd == "/use":
            self.task_id = args[0]
            render.info(f"active task: {self.task_id}")
            self._show_status()
        elif cmd == "/tasks":
            render.tasks_table(self.client.list_tasks())
        elif cmd == "/status":
            self._show_status()
        elif cmd == "/ask":
            question = " ".join(args).strip()
            if question:
                self._answer(question)
            else:
                render.warn("usage: /ask <question>")
        elif cmd == "/improve":
            goal = " ".join(args).strip()
            if goal:
                self._improve(goal)
            else:
                render.warn("usage: /improve <goal>  e.g. /improve 用 LoRA+rank loss 把 PLCC 提到 0.90")
        elif cmd == "/report":
            self._report(full="--full" in args)
        elif cmd == "/plan":
            st = self.client.state(self._need_task())
            plan_id = args[0] if args else (st.get("active_plan_id") or (st.get("pending_approval") or {}).get("plan_id"))
            if not plan_id:
                raise ClientError("no plan yet", recovery="run /status to see the current phase")
            render.plan_panel(self.client.plan_view(self.task_id, plan_id))
        elif cmd == "/approve":
            self._approve(args)
        elif cmd == "/feedback":
            self._feedback(args)
        elif cmd in ("/pause", "/resume", "/cancel"):
            res = self.client.control(self._need_task(), cmd[1:])
            render.info(f"{cmd}: {res.get('note', 'ok')}")
            if res.get("boundary"):
                self._boundary(res["boundary"])
        elif cmd == "/budget":
            render.budget_panel(self.client.budget(self._need_task()))
        elif cmd == "/events":
            n = int(args[0]) if args and args[0].isdigit() else 20
            evs = self.client.events(self._need_task())
            render.events_table(evs[-n:])
        elif cmd == "/export":
            res = self.client.export(self._need_task())
            for key, path in res.items():
                render.info(f"{key}: {path}")
        elif cmd == "/attach":
            self._attach(args)
        else:
            render.warn(f"unknown command {cmd}; try /help")
        return None

    def _approve(self, args: List[str]) -> None:
        task_id = self._need_task()
        st = self.client.state(task_id)
        pending = st.get("pending_approval") or {}
        plan_id = args[0] if args and not args[0].startswith("--") else pending.get("plan_id")
        plan_hash = pending.get("plan_hash", "")
        if "--hash" in args:
            plan_hash = args[args.index("--hash") + 1]
        elif len(args) >= 2:
            plan_hash = args[1]
        if not plan_id or not plan_hash:
            raise ClientError("nothing to approve",
                              recovery="check /status; a plan must be awaiting approval")
        boundary = self.client.approve(task_id, plan_id, plan_hash,
                                       expected_state_version=st.get("state_version"))
        render.info(f"approved plan {plan_id}")
        self._boundary(boundary)

    def _feedback(self, args: List[str]) -> None:
        task_id = self._need_task()
        payload: Dict[str, Any] = {}
        for token in args:
            if "=" in token:
                key, _, value = token.partition("=")
                payload[key] = value
        if not payload:
            from prompt_toolkit import prompt
            payload["scenario"] = prompt("scenario: ").strip()
            payload["actual"] = prompt("actual behaviour: ").strip()
            payload["expected"] = prompt("expected behaviour: ").strip()
        res = self.client.submit_feedback(task_id, payload)
        render.info(f"feedback {res.get('feedback_id')} recorded"
                    + (" (continuing)" if res.get("continued") else ""))
        if res.get("needs_model_confirmation"):
            render.warn(f"model not specified; candidates: {res.get('model_candidates')}")
        if res.get("continued"):
            self._boundary(res.get("boundary", {}))
        else:
            self._maybe_run()

    def _attach(self, args: List[str]) -> None:
        if not args:
            raise ClientError("usage: /attach <path>")
        if self.client.mode != "embedded":
            render.warn("/attach registers local files only in embedded mode")
            return
        ctrl = self.client._controller()
        ref = ctrl.rt.evidence.register_source(self._need_task(), uri=args[0], source_type="local")
        render.info(f"registered {ref.evidence_id} -> {ref.uri} (scope={ref.scope}, policy={ref.upload_policy})")



_INTENT: List[tuple] = [
    (("精度", "指标", "状态", "情况", "达标", "plcc", "srcc", "结果", "分数", "score"),
     ("当前产出精度状态",)),
    (("计划", "改进", "优化", "下一步", "next", "提升", "模型", "损失", "训练", "策略", "方案", "loss"),
     ("模型", "损失", "训练策略优化")),
    (("数据", "标注", "扩增", "augment", "样本", "qwen", "本地", "人工", "复核"),
     ("数据标注", "数据扩增")),
]


def _extract_sections(path: str) -> List[Dict[str, str]]:
    """Split a design report markdown file into its level-2 sections."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    sections: List[Dict[str, str]] = []
    title, buf = None, []
    for line in lines:
        if line.startswith("## "):
            if title is not None:
                sections.append({"title": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        sections.append({"title": title, "text": "\n".join(buf).strip()})
    return sections


def _pick_sections(question: str, sections: List[Dict[str, str]]):
    """Match a question to sections by lightweight keyword intent."""
    q = (question or "").lower()
    picked, seen = [], set()
    for keys, needles in _INTENT:
        if not any(k in q for k in keys):
            continue
        for sec in sections:
            if sec["title"] in seen:
                continue
            if any(n in sec["title"] for n in needles):
                picked.append((sec["title"], sec["text"]))
                seen.add(sec["title"])
    return picked


def run_chat(settings: Settings, client: ControlClient, task_id: Optional[str] = None) -> int:
    return ChatSession(settings, client, task_id=task_id).run()


__all__ = ["ChatSession", "run_chat", "HELP"]
