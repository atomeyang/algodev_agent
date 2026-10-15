"""``algo-agent`` command line entry point (Typer + Rich).

Global options come before the subcommand, e.g.::

    algo-agent --root ./agentic_algorithm init
    algo-agent --root ./agentic_algorithm doctor
    algo-agent --root ./agentic_algorithm service start
    algo-agent --root ./agentic_algorithm demo --offline
    algo-agent --root ./agentic_algorithm chat --task demo_task

``--json`` prints machine-readable JSON on stdout and keeps human progress on
stderr. Distinct exit codes let scripts react without parsing text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from .. import bootstrap
from ..config import load_settings
from . import render
from .client import ControlClient, ClientError, default_service_url
from .service import start_service, stop_service, service_status
from ..reuse import (ReuseRequest, compatibility_for, export_reproduction,
                     export_task_bundle, import_task_bundle, migrate_state,
                     validate_reproduction)
from ..schemas import contracts as C
from ..storage import AssetError, AssetRegistry
from ..storage.repository import Repository

app = typer.Typer(add_completion=False, help="Algorithm self-research & optimization Agent")
service_app = typer.Typer(help="Manage the local control service")
app.add_typer(service_app, name="service")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_FOUND = 3
EXIT_STALE = 4
EXIT_SERVICE = 5
EXIT_TIMEOUT = 6


# ------------------------------------------------------------------------- #
# context helpers
# ------------------------------------------------------------------------- #
def _exit_for(exc: ClientError) -> int:
    if exc.status == 404:
        return EXIT_NOT_FOUND
    if exc.status == 409:
        return EXIT_STALE
    if exc.status == 503:
        return EXIT_SERVICE
    if exc.status == 504:
        return EXIT_TIMEOUT
    return EXIT_ERROR


def _fail(exc: ClientError) -> None:
    render.error(exc.message)
    if exc.recovery:
        render.info(f"hint: {exc.recovery}")
    raise typer.Exit(_exit_for(exc))


class Ctx:
    def __init__(self, root: Optional[str], json_mode: bool, service_url: Optional[str],
                 embedded: bool):
        self.settings = load_settings(root)
        self.json_mode = json_mode
        self.service_url = service_url
        self.embedded = embedded

    def client(self, *, extras: Optional[Dict[str, Any]] = None) -> ControlClient:
        mode = "embedded" if self.embedded else None
        return ControlClient(self.settings, service_url=self.service_url,
                             force_mode=mode, extras=extras)

    def repo(self) -> Repository:
        """Open the local state database directly (embedded-only commands)."""
        return Repository(self.settings.db_path)

    def emit(self, obj: Any, human=None) -> None:
        if self.json_mode:
            typer.echo(json.dumps(obj, ensure_ascii=False, default=str))
        elif human is not None:
            human(obj)
        else:
            render.print_json(obj)


@app.callback()
def _main_callback(
    ctx: typer.Context,
    root: Optional[str] = typer.Option(None, "--root", help="platform root (defaults to CWD/marker)"),
    json_output: bool = typer.Option(False, "--json", help="machine-readable JSON on stdout"),
    service_url: Optional[str] = typer.Option(None, "--service-url", help="control service URL"),
    embedded: bool = typer.Option(False, "--embedded", help="drive the controller in-process"),
) -> None:
    ctx.obj = Ctx(root, json_output, service_url, embedded)


# ------------------------------------------------------------------------- #
# bootstrap
# ------------------------------------------------------------------------- #
@app.command()
def init(ctx: typer.Context) -> None:
    """Create platform dirs, default config, marker and database (idempotent)."""
    c: Ctx = ctx.obj
    report = bootstrap.init_platform(c.settings)
    report["documents"] = _ensure_status_documents(c)
    c.emit(report, lambda r: _init_human(r))


@app.command("auto-process")
def auto_process(
    ctx: typer.Context,
    requirement: str = typer.Argument("", help="requirement with an explicit acceptance threshold"),
    task: Optional[str] = typer.Option(None, "--task"),
    spec_file: Optional[Path] = typer.Option(None, "--spec", help="structured TaskSpec JSON"),
    parent_task: str = typer.Option("", "--parent-task", help="owner task ID for a related track"),
    track: str = typer.Option("", "--track", help="directory name under the owner's tracks/"),
    domain: str = typer.Option("", "--domain"),
    demo: bool = typer.Option(False, "--demo"),
) -> None:
    """Research, implement, run data/model experiments and evaluate without approval waits."""
    c: Ctx = ctx.obj
    payload = json.loads(spec_file.read_text(encoding="utf-8")) if spec_file else None
    if not requirement and not payload and not task:
        raise typer.BadParameter("provide a requirement, --spec, or an existing --task")
    client = c.client()
    try:
        result = client.create_task(requirement, task, auto_process=True, demo=demo, synthetic=demo,
                                    domain=domain, task_spec=payload, run=True,
                                    parent_task=parent_task, track=track)
        boundary = result.get("boundary")
        if boundary is None:
            timeout = float((payload or {}).get("budget", {}).get("max_wall_clock_s", 3600)) + 60
            boundary = client.wait_boundary(result["task_id"], timeout=timeout)
        c.emit(boundary, _boundary_human)
        if boundary["state"]["phase"] != "accepted":
            raise typer.Exit(2)
    except ClientError as exc:
        _fail(exc)


def _ensure_status_documents(c: "Ctx") -> List[str]:
    """Generate the M5 status/version DOCX if absent (never clobbers an existing one)."""
    from ..docs_export import export_system_status, export_version_notes
    out: List[str] = []
    try:
        repo = c.repo()
        for name, maker in (("agent_status.docx", export_system_status),
                            ("version_notes_v1.0.docx", export_version_notes)):
            target = c.settings.docs_dir / name
            out.append(str(target if target.exists() else maker(c.settings, repo)))
    except Exception as exc:  # doc generation must never break init
        render.warn(f"could not generate status documents: {exc}")
    return out


def _init_human(r: Dict[str, Any]) -> None:
    render.banner(f"initialised {r['root']}")
    render.info(f"database: {r['db']} (schema {r['schema_version']})")
    for path in r.get("created", []):
        render.info(f"created: {path}")
    for path in r.get("kept", []):
        render.info(f"kept   : {path}")
    for path in r.get("documents", []):
        render.info(f"document: {path}")


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Check the environment, providers and directories."""
    c: Ctx = ctx.obj
    report = bootstrap.doctor(c.settings)
    c.emit(report, render.doctor_report)
    if not report.get("ok"):
        raise typer.Exit(EXIT_ERROR)


# ------------------------------------------------------------------------- #
# service
# ------------------------------------------------------------------------- #
@service_app.command("start")
def service_start(
    ctx: typer.Context,
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    foreground: bool = typer.Option(False, "--foreground", help="run in the foreground"),
) -> None:
    """Start the control service (background by default)."""
    c: Ctx = ctx.obj
    st = c.settings
    host = host or st.env_get("ALGO_SERVICE_HOST") or "127.0.0.1"
    port = port or int(st.env_get("ALGO_SERVICE_PORT") or 8765)
    result = start_service(st, host=host, port=port, foreground=foreground)
    c.emit(result, lambda r: render.info(
        f"service {'started' if r.get('started') else 'not started'} at {r.get('url')}"
        + (f" (pid {r.get('pid')})" if r.get("pid") else "")
        + (f" log={r.get('log')}" if r.get("log") else "")))
    if result.get("error"):
        render.error(result["error"])
        raise typer.Exit(EXIT_ERROR)


@service_app.command("status")
def service_status_cmd(ctx: typer.Context) -> None:
    """Show whether the control service is running and reachable."""
    c: Ctx = ctx.obj
    url = c.service_url or default_service_url(c.settings)
    result = service_status(c.settings, url)
    c.emit(result, lambda r: render.info(
        f"running={r['running']} reachable={r['reachable']} pid={r['pid']} url={r['url']}"))
    if not result["running"] and not result["reachable"]:
        raise typer.Exit(EXIT_SERVICE)


@service_app.command("stop")
def service_stop(ctx: typer.Context) -> None:
    """Stop a background control service."""
    c: Ctx = ctx.obj
    result = stop_service(c.settings)
    c.emit(result, lambda r: render.info(
        f"stopped={r.get('stopped')} pid={r.get('pid')} {r.get('note', '')}"))


# ------------------------------------------------------------------------- #
# demo / chat
# ------------------------------------------------------------------------- #
@app.command()
def demo(
    ctx: typer.Context,
    offline: bool = typer.Option(False, "--offline", help="force offline synthetic mode"),
    task_id: str = typer.Option("demo_task", "--task", help="task id to create/use"),
    root_clean: bool = typer.Option(False, "--fresh", help="recreate the demo task from scratch"),
) -> None:
    """Run the full offline demo: plan -> approve -> train -> feedback -> accept."""
    from ..demo import run_demo
    c: Ctx = ctx.obj
    try:
        report = run_demo(c.settings, client=c.client(), task_id=task_id,
                          offline=offline, fresh=root_clean, json_mode=c.json_mode)
    except ClientError as exc:
        _fail(exc)
    if c.json_mode:
        typer.echo(json.dumps(report, ensure_ascii=False, default=str))


@app.command()
def chat(
    ctx: typer.Context,
    task: Optional[str] = typer.Option(None, "--task", help="task id to attach to"),
) -> None:
    """Open the interactive chat REPL."""
    from .chat import run_chat
    c: Ctx = ctx.obj
    client = c.client()
    raise typer.Exit(run_chat(c.settings, client, task_id=task))


# ------------------------------------------------------------------------- #
# task operations
# ------------------------------------------------------------------------- #
def _require_task(c: Ctx, task: Optional[str]) -> str:
    if not task:
        render.error("--task is required")
        raise typer.Exit(EXIT_NOT_FOUND)
    return task


@app.command()
def status(ctx: typer.Context, task: Optional[str] = typer.Option(None, "--task")) -> None:
    """Show task phase, active plan and best model."""
    c: Ctx = ctx.obj
    try:
        st = c.client().state(_require_task(c, task))
    except ClientError as exc:
        _fail(exc)
    c.emit(st, render.state_panel)


@app.command("tasks")
def tasks_cmd(ctx: typer.Context) -> None:
    """List all known tasks."""
    c: Ctx = ctx.obj
    try:
        items = c.client().list_tasks()
    except ClientError as exc:
        _fail(exc)
    c.emit(items, render.tasks_table)


@app.command()
def approve(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    plan: Optional[str] = typer.Option(None, "--plan", help="plan id (defaults to pending)"),
    plan_hash: Optional[str] = typer.Option(None, "--hash", help="plan hash to confirm"),
    state_version: Optional[int] = typer.Option(None, "--state-version"),
) -> None:
    """Approve a pending plan after verifying its hash and state version."""
    c: Ctx = ctx.obj
    client = c.client()
    try:
        st = client.state(task)
        pending = st.get("pending_approval") or {}
        plan = plan or pending.get("plan_id")
        plan_hash = plan_hash or pending.get("plan_hash")
        if not plan or not plan_hash:
            raise ClientError("no plan is awaiting approval",
                              recovery="check: algo-agent status --task " + task)
        boundary = client.approve(task, plan, plan_hash,
                                  expected_state_version=state_version or st.get("state_version"))
    except ClientError as exc:
        _fail(exc)
    c.emit(boundary, lambda b: _boundary_human(b))


def _boundary_human(boundary: Dict[str, Any]) -> None:
    if boundary.get("type") == "end" or boundary.get("done"):
        render.info("task reached the end of the graph")
        render.state_panel(boundary.get("state", {}))
    else:
        render.approval_panel(boundary.get("interrupt") or {})


@app.command()
def feedback(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    file: Optional[str] = typer.Option(None, "--file", help="feedback.json with the full record"),
    model: Optional[str] = typer.Option(None, "--model", help="model version the feedback refers to"),
    scenario: str = typer.Option("", "--scenario"),
    actual: str = typer.Option("", "--actual"),
    expected: str = typer.Option("", "--expected"),
    severity: str = typer.Option("medium", "--severity"),
    source: str = typer.Option("user", "--source"),
    repro: str = typer.Option("", "--repro"),
) -> None:
    """Record external problem evidence and enter the diagnosis/iteration loop."""
    c: Ctx = ctx.obj
    payload: Dict[str, Any] = {}
    if file:
        payload.update(json.loads(Path(file).read_text(encoding="utf-8")))
    payload.setdefault("model_id", model)
    for key, value in (("scenario", scenario), ("actual", actual), ("expected", expected),
                       ("severity", severity), ("source", source), ("repro", repro)):
        if value:
            payload[key] = value
    try:
        res = c.client().submit_feedback(task, payload)
    except ClientError as exc:
        _fail(exc)
    c.emit(res, lambda r: (
        render.info(f"feedback {r['feedback_id']} recorded (model={r.get('model_id')})"),
        render.warn(f"confirm model: candidates={r['model_candidates']}")
        if r.get("needs_model_confirmation") else None,
        _boundary_human(r["boundary"]) if r.get("boundary") else None))


@app.command()
def pause(ctx: typer.Context, task: str = typer.Option(..., "--task")) -> None:
    """Request a boundary pause (current jobs are not killed)."""
    _control(ctx, task, "pause")


@app.command()
def resume(ctx: typer.Context, task: str = typer.Option(..., "--task")) -> None:
    """Resume a paused task."""
    _control(ctx, task, "resume")


@app.command()
def cancel(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    job: Optional[str] = typer.Option(None, "--job"),
) -> None:
    """Cancel queued/running jobs (saved assets are never deleted)."""
    _control(ctx, task, "cancel", job)


def _control(ctx: typer.Context, task: str, action: str, job: Optional[str] = None) -> None:
    c: Ctx = ctx.obj
    try:
        res = c.client().control(task, action, job_id=job)
    except ClientError as exc:
        _fail(exc)
    c.emit(res, lambda r: render.info(f"{action}: {r}"))


@app.command()
def export(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    iteration: Optional[int] = typer.Option(None, "--iteration"),
) -> None:
    """Export the task overview and iteration documents."""
    c: Ctx = ctx.obj
    try:
        res = c.client().export(task, iteration=iteration)
    except ClientError as exc:
        _fail(exc)
    c.emit(res, lambda r: [render.info(f"{k}: {v}") for k, v in r.items()])


@app.command()
def events(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    after: int = typer.Option(0, "--after"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Show task events after a given sequence number."""
    c: Ctx = ctx.obj
    try:
        evs = c.client().events(task, after=after, limit=limit)
    except ClientError as exc:
        _fail(exc)
    c.emit(evs, render.events_table)


@app.command()
def budget(ctx: typer.Context, task: str = typer.Option(..., "--task")) -> None:
    """Show budget limits, usage and ledger."""
    c: Ctx = ctx.obj
    try:
        b = c.client().budget(task)
    except ClientError as exc:
        _fail(exc)
    if "limits" in b:
        c.emit(b, render.budget_panel)
    else:
        c.emit({"budget_used": b.get("budget_used", {})}, lambda r: render.print_json(r))


@app.command()
def jobs(
    ctx: typer.Context,
    task: Optional[str] = typer.Option(None, "--task"),
) -> None:
    """List experiment jobs."""
    c: Ctx = ctx.obj
    try:
        items = c.client().jobs(task)
    except ClientError as exc:
        _fail(exc)
    c.emit(items, render.jobs_table)


plan_app = typer.Typer(help="Inspect plans")
app.add_typer(plan_app, name="plan")


@plan_app.command("show")
def plan_show(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    plan: Optional[str] = typer.Option(None, "--plan"),
) -> None:
    """Show a plan (defaults to the active/pending one)."""
    c: Ctx = ctx.obj
    client = c.client()
    try:
        st = client.state(task)
        plan_id = plan or st.get("active_plan_id") or (st.get("pending_approval") or {}).get("plan_id")
        if not plan_id:
            raise ClientError("no plan for this task", recovery="run a task first", status=404)
        view = client.plan_view(task, plan_id)
    except ClientError as exc:
        _fail(exc)
    c.emit(view, render.plan_panel)


# ------------------------------------------------------------------------- #
# reuse assets / reproduction / bundles / migration / docs (embedded-local)
# ------------------------------------------------------------------------- #
assets_app = typer.Typer(help="Inspect and manage versioned, immutable reuse assets")
reuse_app = typer.Typer(help="Compatibility checks and reproduction packages")
compat_app = typer.Typer(help="Check whether a reusable asset fits this task")
bundle_app = typer.Typer(help="Export/import portable task bundles")
docs_app = typer.Typer(help="Generate DOCX status and version records")
app.add_typer(assets_app, name="assets")
app.add_typer(reuse_app, name="reuse")
app.add_typer(compat_app, name="compat")
app.add_typer(bundle_app, name="bundle")
app.add_typer(docs_app, name="docs")


def _fail_asset(exc: AssetError) -> None:
    """Registry errors are not ClientError; convert so exit codes stay uniform."""
    _fail(ClientError(exc.message, recovery=exc.recovery, status=exc.status))


def _registry(c: Ctx) -> AssetRegistry:
    return AssetRegistry(c.settings, c.repo())


def _split_csv(raw: str) -> List[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _load_json_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


@assets_app.command("list")
def assets_list(
    ctx: typer.Context,
    asset_type: Optional[str] = typer.Option(None, "--type", help="recipe|dataset|model|evaluation|knowledge|component"),
    status: Optional[str] = typer.Option(None, "--status", help="researched|reproduced|business_validated|reusable|deprecated"),
    query: str = typer.Option("", "--query", help="substring match on id/type/source task"),
) -> None:
    """List registered asset versions (sorted by validation status)."""
    c: Ctx = ctx.obj
    try:
        items = _registry(c).search(asset_type=asset_type, status=status, query=query)
    except AssetError as exc:
        _fail_asset(exc)
    c.emit([m.model_dump() for m in items], render.assets_table)


@assets_app.command("show")
def assets_show(
    ctx: typer.Context,
    asset_id: str = typer.Argument(...),
    version: str = typer.Option("v1", "--version"),
) -> None:
    """Show one asset version and its dependency closure."""
    c: Ctx = ctx.obj
    try:
        reg = _registry(c)
        manifest = reg.get(asset_id, version)
        deps = reg.resolve_dependencies(asset_id, version)
    except AssetError as exc:
        _fail_asset(exc)
    c.emit({"asset": manifest.model_dump(), "dependencies": deps})


@assets_app.command("register")
def assets_register(
    ctx: typer.Context,
    file: str = typer.Option(..., "--file", help="AssetManifest JSON"),
    body: Optional[List[str]] = typer.Option(None, "--body", help="body file to copy into the version dir (repeatable)"),
    body_ref: str = typer.Option("", "--body-ref", help="reference recorded in the index"),
) -> None:
    """Register an immutable asset version (idempotent for identical content)."""
    c: Ctx = ctx.obj
    try:
        manifest = C.AssetManifest(**_load_json_file(file))
        stored = _registry(c).register(
            manifest, body_files=[Path(b) for b in (body or [])], body_ref=body_ref)
    except AssetError as exc:
        _fail_asset(exc)
    except (OSError, ValueError, TypeError) as exc:
        _fail(ClientError(f"cannot read asset manifest {file}: {exc}", status=400))
    c.emit(stored.model_dump(), lambda m: render.info(
        f"registered {m['asset_id']}:{m['asset_version']} ({m['validation_status']})"))


@assets_app.command("publish")
def assets_publish(
    ctx: typer.Context,
    asset_id: str = typer.Argument(...),
    version: str = typer.Option("v1", "--version"),
    validation: Optional[str] = typer.Option(None, "--validation", help="validation record JSON"),
    status: Optional[str] = typer.Option(None, "--status", help="advance to this exact status"),
) -> None:
    """Advance a version's validation status (requires a real validation record)."""
    c: Ctx = ctx.obj
    try:
        manifest = _registry(c).publish(
            asset_id, version, validation=_load_json_file(validation), new_status=status)
    except AssetError as exc:
        _fail_asset(exc)
    except (OSError, ValueError) as exc:
        _fail(ClientError(f"cannot read validation record: {exc}", status=400))
    c.emit(manifest.model_dump(), lambda m: render.info(
        f"{m['asset_id']}:{m['asset_version']} -> {m['validation_status']}"))


@assets_app.command("deprecate")
def assets_deprecate(
    ctx: typer.Context,
    asset_id: str = typer.Argument(...),
    version: str = typer.Option("v1", "--version"),
    reason: str = typer.Option("", "--reason"),
) -> None:
    """Mark a version deprecated (immutable; kept for audit)."""
    c: Ctx = ctx.obj
    try:
        manifest = _registry(c).deprecate(asset_id, version, reason=reason)
    except AssetError as exc:
        _fail_asset(exc)
    c.emit(manifest.model_dump(), lambda m: render.info(
        f"deprecated {m['asset_id']}:{m['asset_version']}"))


@reuse_app.command("export")
def reuse_export(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    out: Optional[str] = typer.Option(None, "--out", help="output dir (default: <task>/deliverables/reproduction)"),
) -> None:
    """Freeze an asset/data/weights closure into a reproduction package."""
    c: Ctx = ctx.obj
    try:
        result = export_reproduction(c.settings, c.repo(), task,
                                     out_dir=Path(out) if out else None)
    except (KeyError, FileNotFoundError) as exc:
        _fail(ClientError(f"task not found: {task} ({exc})", status=404,
                          recovery="check: algo-agent tasks"))
    c.emit(result, lambda r: (
        render.info(f"reproduction package: {r['dir']}"),
        render.info(f"files: {', '.join(r['files'])}"),
        render.info(f"external refs (supply separately): {len(r['external_refs'])}")))


@reuse_app.command("validate")
def reuse_validate(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    package: Optional[str] = typer.Option(None, "--package", help="package dir (default: deliverables/reproduction)"),
    actual: Optional[str] = typer.Option(None, "--actual-metrics", help="JSON of re-run metrics to compare"),
) -> None:
    """Check a reproduction package resolves; status validated|mismatch|blocked|failed."""
    c: Ctx = ctx.obj
    try:
        result = validate_reproduction(
            c.settings, c.repo(), task,
            package_dir=Path(package) if package else None,
            actual_metrics=_load_json_file(actual))
    except (KeyError, FileNotFoundError) as exc:
        _fail(ClientError(f"task not found: {task} ({exc})", status=404))
    except (OSError, ValueError) as exc:
        _fail(ClientError(f"cannot read actual metrics: {exc}", status=400))
    c.emit(result, lambda r: (
        render.info(f"status: {r['status']}"),
        [render.warn(f"blocker: {b}") for b in r.get("blockers", [])],
        [render.warn(f"mismatch: {m}") for m in r.get("mismatches", [])]))
    if result.get("status") != "validated":
        raise typer.Exit(EXIT_ERROR)


@compat_app.command("check")
def compat_check(
    ctx: typer.Context,
    asset: str = typer.Option(..., "--asset"),
    version: str = typer.Option("v1", "--version"),
    inputs: str = typer.Option("", "--inputs", help="comma-separated input names"),
    outputs: str = typer.Option("", "--outputs", help="comma-separated output names"),
    device: str = typer.Option("", "--device"),
    license: str = typer.Option("", "--license"),
    distribution: str = typer.Option("", "--distribution", help="business distribution; blank = unknown"),
    hard_constraints: str = typer.Option("", "--hard-constraints", help="comma-separated"),
) -> None:
    """Check explicit compatibility; anything unverifiable is reported unknown."""
    c: Ctx = ctx.obj
    request = ReuseRequest(inputs=_split_csv(inputs), outputs=_split_csv(outputs),
                           device=device, license=license, distribution=distribution,
                           hard_constraints=_split_csv(hard_constraints))
    result = compatibility_for(c.repo(), asset, version, request)
    c.emit(result.model_dump(), render.compat_panel)


@bundle_app.command("export")
def bundle_export(
    ctx: typer.Context,
    task: str = typer.Option(..., "--task"),
    out: Optional[str] = typer.Option(None, "--out", help="output dir (default: <slug>_bundle)"),
) -> None:
    """Export a hash-verified, secret-filtered task bundle."""
    c: Ctx = ctx.obj
    try:
        result = export_task_bundle(c.settings, c.repo(), task,
                                    out_dir=Path(out) if out else None)
    except (KeyError, FileNotFoundError) as exc:
        _fail(ClientError(f"task not found: {task} ({exc})", status=404))
    c.emit(result, lambda r: (
        render.info(f"bundle: {r.get('dir')}"),
        render.info(f"{r.get('files')} file(s), {r.get('excluded')} excluded, "
                    f"{len(r.get('external_refs', []))} external reference(s)")))


@bundle_app.command("import")
def bundle_import(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="bundle dir or bundle_manifest.json"),
    as_task: Optional[str] = typer.Option(None, "--as", help="target task id (default: source id)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Import a bundle; never overwrites an existing same-named task."""
    c: Ctx = ctx.obj
    result = import_task_bundle(c.settings, c.repo(), Path(path),
                                target_task_id=as_task, dry_run=dry_run)
    c.emit(result.model_dump(), lambda r: (
        render.info(f"import: {r['status']} task={r['task_id']} source={r.get('source_task')}"),
        [render.info(f"  {e}") for e in r.get("events", [])]))
    if result.status != "ready":
        raise typer.Exit(EXIT_ERROR)


@app.command()
def migrate(
    ctx: typer.Context,
    no_backup: bool = typer.Option(False, "--no-backup"),
) -> None:
    """Back up and migrate the state DB to the current schema/graph version."""
    c: Ctx = ctx.obj
    result = migrate_state(c.settings, backup=not no_backup)
    c.emit(result, lambda r: render.info(
        f"migrate: {r.get('status')} {r.get('from_version')} -> {r.get('to_version')}"))
    if result.get("status") not in ("ready", "ok"):
        raise typer.Exit(EXIT_ERROR)


@docs_app.command("status")
def docs_status(ctx: typer.Context) -> None:
    """Regenerate docs/agent_status.docx (milestones, tasks)."""
    from ..docs_export import export_system_status
    c: Ctx = ctx.obj
    path = export_system_status(c.settings, c.repo())
    c.emit({"path": str(path)}, lambda r: render.info(f"wrote {r['path']}"))


@docs_app.command("version-notes")
def docs_version_notes(
    ctx: typer.Context,
    version: str = typer.Option("v1.0", "--version"),
) -> None:
    """Regenerate docs/version_notes_<version>.docx."""
    from ..docs_export import export_version_notes
    c: Ctx = ctx.obj
    path = export_version_notes(c.settings, c.repo(), version)
    c.emit({"path": str(path)}, lambda r: render.info(f"wrote {r['path']}"))


def main() -> None:  # pragma: no cover - console_script shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
