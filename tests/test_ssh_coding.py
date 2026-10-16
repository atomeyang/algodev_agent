"""SSH transport contracts and subprocess integration using a fake Codex CLI.

No network, paid generation, or real sandbox availability is claimed here.
"""

import base64
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

import pytest

from algo_agent.config import Settings
from algo_agent.providers import build_providers, readiness_report, ssh_worker
from algo_agent.providers.base import ProviderStatus
from algo_agent.providers.codex_ssh import RemoteWorkerError, SshCodexExecutor
from algo_agent.schemas import contracts as C


def encoded(text):
    return base64.b64encode(text.encode()).decode()


def settings(tmp_path, **values):
    return Settings(tmp_path, env={
        "ALGO_OFFLINE": "1", "ALGO_CODING_TRANSPORT": "ssh", "ALGO_CODEX_ALLOW_REAL": "1",
        "ALGO_CODING_SSH_HOST": "worker-alias", "ALGO_CODING_SSH_ROOT": str(tmp_path / "remote"),
        "ALGO_CODING_MODEL": "requested-test-model", **values})


def task(**values):
    return C.CodingTaskSpec(coding_id="c", task_id="t", plan_id="p", objective="implement",
                            write_paths=["model_pool/new"], entry="model_pool/new/method.py",
                            **values)


@pytest.fixture
def fake_codex(tmp_path):
    binary = tmp_path / "fake-codex"
    binary.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys, time
args = sys.argv[1:]
if args == ["--version"]:
    print("fake-codex test fixture")
elif args == ["login", "status"]:
    print("test authentication stub")
elif args[:2] == ["sandbox", "linux"]:
    pass  # Stub only: does not claim a real kernel sandbox.
elif args[0] == "exec":
    assert args[args.index("--sandbox") + 1] == "workspace-write"
    assert 'approval_policy="never"' in args
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    p = json.load(sys.stdin)
    goal = p["task"]["objective"]
    if goal == "sleep":
        time.sleep(15)
    path = pathlib.Path("model_pool/new/method.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    if goal == "outside":
        pathlib.Path("coding_contract.json").write_text("tampered")
    if goal == "symlink":
        path.symlink_to("/etc/passwd")
    else:
        path.write_text("broken (" if goal == "bad_syntax" else "def run(ctx, config):\n    return {}\n")
    print(json.dumps({"type": "turn.completed", "model": "reported-test-model",
                      "usage": {"input_tokens": 4, "output_tokens": 3}}))
else:
    raise SystemExit(1)
''')
    binary.chmod(0o700)
    return binary


@pytest.fixture
def loopback_executor(tmp_path, monkeypatch, fake_codex):
    executor = SshCodexExecutor(settings(tmp_path, ALGO_CODING_SSH_CODEX=str(fake_codex)))
    source = Path(ssh_worker.__file__).read_text()
    # Exercise the real stdlib worker and JSON pipe, replacing SSH only.
    monkeypatch.setattr(executor, "_argv", lambda: [sys.executable, "-c", source])
    return executor


def workspace(tmp_path):
    root = tmp_path / "local"
    root.mkdir()
    (root / "coding_contract.json").write_text("{}")
    return root


def test_factory_selects_remote_and_unknown_transport_fails_closed(tmp_path):
    assert isinstance(build_providers(settings(tmp_path))["coding"], SshCodexExecutor)
    with pytest.raises(ValueError, match="TRANSPORT"):
        build_providers(settings(tmp_path, ALGO_CODING_TRANSPORT="typo"))


def test_configured_is_not_probed_and_no_host_never_connects(tmp_path, monkeypatch):
    monkeypatch.setattr(SshCodexExecutor, "_request", lambda *a, **k: pytest.fail("unexpected SSH"))
    report = readiness_report(settings(tmp_path))
    assert report["checks"]["coding_transport_configured"]
    assert not report["live_verified"] and report["coding_probe"] is None
    missing = SshCodexExecutor(settings(tmp_path, ALGO_CODING_SSH_HOST=""))
    assert not missing.probe().available
    disabled = SshCodexExecutor(settings(tmp_path, ALGO_CODEX_ALLOW_REAL="0"))
    assert not disabled.probe().available


@pytest.mark.parametrize("host", ["-oProxyCommand=bad", "host;bad", "host with spaces", "x\nbad"])
def test_ssh_destination_rejects_option_or_shell_injection(tmp_path, host):
    assert not SshCodexExecutor(settings(tmp_path, ALGO_CODING_SSH_HOST=host)).status().available


def test_ssh_argv_requires_host_verification_and_never_forwards_credentials(tmp_path):
    executor = SshCodexExecutor(settings(tmp_path, ALGO_CODING_SSH_PYTHON="/home/user/venv with space/python"))
    argv = executor._argv()
    assert argv[-2] == "worker-alias"
    assert "StrictHostKeyChecking=yes" in argv
    assert "BatchMode=yes" in argv and "ForwardAgent=no" in argv
    assert "ClearAllForwardings=yes" in argv and "PermitLocalCommand=no" in argv
    remote = shlex.split(argv[-1])
    assert remote[:2] == ["/home/user/venv with space/python", "-c"]
    assert remote[2] == Path(ssh_worker.__file__).read_text()


@pytest.mark.parametrize("name", [
    "../outside.py", "/tmp/out.py", "model_pool/new/../../outside.py",
    "model_pool//new/a.py", "model_pool/new/.env", "model_pool/new/auth.json",
    "model_pool/new/a.pem", "model_pool/new/weights.pt", "data/train.csv",
    "model_pool/new/link\\bad.py",
])
def test_wire_rejects_unsafe_or_non_source_paths(name):
    with pytest.raises(ValueError):
        ssh_worker.decode_files({name: encoded("test")})


def test_source_transfer_rejects_binary_links_and_oversize(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    (root / "model_pool/new").mkdir(parents=True)
    method = root / "model_pool/new/a.py"
    method.symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        ssh_worker.snapshot(root)
    method.unlink()
    method.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        ssh_worker.snapshot(root)
    method.write_text("source")
    monkeypatch.setattr(ssh_worker, "MAX_BUNDLE_BYTES", 3)
    with pytest.raises(ValueError, match="limit"):
        ssh_worker.snapshot(root)


def test_parent_rejects_entire_return_before_applying_good_file(tmp_path):
    root = workspace(tmp_path)
    baseline = ssh_worker.snapshot(root)
    response = {"files": {"model_pool/new/method.py": encoded("good"),
                          "coding_contract.json": encoded("tampered")}}
    with pytest.raises(ValueError, match="scope"):
        SshCodexExecutor._apply(root, baseline, response, ["model_pool/new"])
    assert (root / "coding_contract.json").read_text() == "{}"
    assert not (root / "model_pool").exists()


def test_parent_rejects_prefix_escape_and_stale_workspace(tmp_path):
    root = workspace(tmp_path)
    baseline = ssh_worker.snapshot(root)
    response = {"files": {"model_pool/new_evil/method.py": encoded("bad")}}
    with pytest.raises(ValueError, match="scope"):
        SshCodexExecutor._apply(root, baseline, response, ["model_pool/new"])
    (root / "coding_contract.json").write_text("changed")
    with pytest.raises(ValueError, match="changed during"):
        SshCodexExecutor._apply(root, baseline, {"files": {}}, ["model_pool/new"])


def test_remote_probe_has_no_generation_and_cleans_temporary_directory(loopback_executor):
    result = loopback_executor.probe()
    assert result.available
    assert result.capabilities["probe_checks"]["sandbox"]
    assert not result.capabilities["model_generation_verified"]
    assert not list(Path(loopback_executor.root).iterdir())


def test_worker_roundtrip_returns_real_source_diff_and_model_usage(loopback_executor, tmp_path):
    root = workspace(tmp_path)
    result = loopback_executor.execute(task(), {"workspace": str(root)})
    assert result.status == "completed"
    assert result.tool == "codex_ssh"
    assert result.model == "reported-test-model"
    assert result.cost["requested_model"] == "requested-test-model"
    assert result.cost["usage"]["input_tokens"] == 4
    assert result.cost["transport"]["sandbox"] == "workspace-write"
    assert (root / "model_pool/new/method.py").read_text().startswith("def run")
    assert not list(Path(loopback_executor.root).iterdir())


@pytest.mark.parametrize("goal", ["outside", "symlink"])
def test_worker_rejects_out_of_scope_or_symlink_outputs(loopback_executor, tmp_path, goal):
    root = workspace(tmp_path)
    request = task().model_copy(update={"objective": goal})
    result = loopback_executor.execute(request, {"workspace": str(root)})
    assert result.status == "blocked"
    assert (root / "coding_contract.json").read_text() == "{}"
    assert not (root / "model_pool").exists()
    assert not list(Path(loopback_executor.root).iterdir())


def test_remote_coding_still_requires_local_independent_acceptance(tmp_path, loopback_executor):
    from algo_agent.service.controller import Controller
    from algo_agent.paths import TaskPaths
    ctrl = Controller(Settings(tmp_path / "platform", env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "remote", task_spec={
        "business": "remote transport regression", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}, "budget": {"max_fix_attempts": 0}})
    ctrl.rt.coding_agent.executor = loopback_executor
    coding = task(require_real=True, budget=spec.budget).model_copy(update={
        "task_id": spec.task_id, "objective": "bad_syntax",
        "acceptance_commands": [
            "python -c \"import ast,pathlib; ast.parse(pathlib.Path('model_pool/new/method.py').read_text())\""]})
    result = ctrl.rt.coding_agent.submit(coding, {"task_slug": spec.task_slug})
    assert result.status == "failed"
    assert not result.changed_files
    assert not (TaskPaths(ctrl.settings, spec.task_slug).model_pool / "new/method.py").exists()


def test_cancelled_controller_requests_remote_cleanup_and_propagates(tmp_path, monkeypatch):
    from algo_agent.policy import TaskHalted
    executor = SshCodexExecutor(settings(tmp_path))
    operations = []
    def request(payload, **kwargs):
        operations.append(payload["operation"])
        if payload["operation"] == "execute":
            raise TaskHalted("cancelled")
        return {"protocol": 1, "cancel_requested": True}
    monkeypatch.setattr(executor, "_request", request)
    with pytest.raises(TaskHalted):
        executor.execute(task(), {"workspace": str(workspace(tmp_path))})
    assert operations == ["execute", "cancel"]


def test_network_failure_is_blocked_not_silent_local_or_offline_fallback(tmp_path, monkeypatch):
    executor = SshCodexExecutor(settings(tmp_path))
    monkeypatch.setattr(executor, "_request", lambda *a, **kw: (_ for _ in ()).throw(
        RemoteWorkerError("connection refused")))
    result = executor.execute(task(), {"workspace": str(workspace(tmp_path))})
    assert result.status == "blocked" and result.tool == "codex_ssh"
    assert "connection refused" in result.open_issues[0]


def worker_request(root, codex, goal="sleep", timeout=5):
    return {"protocol": 1, "operation": "execute", "root": str(root), "codex": str(codex),
            "job_id": "job_" + uuid.uuid4().hex, "files": {"coding_contract.json": encoded("{}")},
            "write_paths": ["model_pool/new"], "prompt": json.dumps({"task": {"objective": goal}}),
            "timeout_s": timeout, "expires_at": time.time() + timeout}


def test_remote_deadline_stops_child_even_without_controller(tmp_path, fake_codex):
    request = worker_request(tmp_path / "remote", fake_codex, timeout=.6)
    result = ssh_worker.run(request)
    assert result["status"] == "failed"
    assert "wall-clock" in result["reason"]
    assert not list((tmp_path / "remote").iterdir())


@pytest.mark.parametrize("interrupt", ["cancel", "sighup"])
def test_remote_cancellation_and_ssh_hangup_reap_codex(tmp_path, fake_codex, interrupt):
    root = tmp_path / "remote"
    request = worker_request(root, fake_codex, timeout=10)
    child = subprocess.Popen([sys.executable, "-c", Path(ssh_worker.__file__).read_text()],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    try:
        child.stdin.write(json.dumps(request))
        child.stdin.close()
        child.stdin = None
        record = root / request["job_id"] / "process.json"
        deadline = time.monotonic() + 5
        while not record.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        info = json.loads(record.read_text())
        if interrupt == "cancel":
            assert ssh_worker.run({**request, "operation": "cancel"})["cancel_requested"]
        else:
            child.send_signal(signal.SIGHUP)
        stdout, stderr = child.communicate(timeout=5)
        assert json.loads(stdout)["status"] in {"failed", "blocked"}
        assert ssh_worker._identity(info["pid"]) != info["start"]
        assert not list(root.iterdir())
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_readiness_surfaces_remote_sandbox_failure_without_claiming_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(SshCodexExecutor, "probe", lambda self: ProviderStatus(
        self.name, False, "coding", "remote sandbox: namespace denied"))
    report = readiness_report(settings(tmp_path), probe=True)
    assert "coding_remote_ready" in report["blockers"]
    assert "namespace denied" in report["coding_probe"]["detail"]
    assert not report["live_verified"]


def test_intake_probes_remote_before_any_research(tmp_path, monkeypatch):
    from algo_agent.graph.nodes.intake import intake_node
    from algo_agent.service.controller import Controller
    calls = []
    def probe(self):
        calls.append("probe")
        return ProviderStatus(self.name, False, "coding", "remote sandbox unavailable")
    monkeypatch.setattr(SshCodexExecutor, "probe", probe)
    ctrl = Controller(settings(tmp_path, ALGO_OFFLINE="0", ALGO_CLOUD_API_KEY="fake",
                               ALGO_SEARCH_BACKEND="arxiv"))
    spec = ctrl.create_task("test", "preflight", auto_process=True, task_spec={
        "domain": "demo_tabular", "business": "test", "require_live_research": True,
        "primary_metric": {"name": "rmse", "threshold": .5}})
    result = intake_node({"task_id": spec.task_id, "task_slug": spec.task_slug}, ctrl.rt.ctx)
    assert calls == ["probe"]
    assert result["decision"] == "needs_input"
    assert "remote sandbox unavailable" in result["stop_reason"]
    assert not ctrl.repo.get_budget(spec.task_id)["used"].get("research_calls")


def test_delayed_delivery_never_starts_an_expired_remote_job(tmp_path, fake_codex):
    request = worker_request(tmp_path / "remote", fake_codex, goal="implement")
    request["expires_at"] = time.time() - 1
    result = ssh_worker.run(request)
    assert result["status"] == "blocked" and "expired" in result["reason"]
    assert not list((tmp_path / "remote").iterdir())


def test_auth_failure_stops_before_sandbox_or_generation(tmp_path, monkeypatch):
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        failed = argv[1:3] == ["login", "status"]
        return subprocess.CompletedProcess(argv, int(failed), "private account data", "")
    monkeypatch.setattr(ssh_worker.subprocess, "run", run)
    result = ssh_worker.probe("codex", tmp_path)
    assert not result["available"]
    assert "private account data" not in result["detail"]
    assert len(commands) == 2


def test_source_credentials_block_upload_before_network(tmp_path, monkeypatch):
    executor = SshCodexExecutor(settings(tmp_path))
    root = workspace(tmp_path)
    (root / ".env").write_text("not-a-real-secret")
    monkeypatch.setattr(executor, "_request", lambda *a, **k: pytest.fail("credential upload attempted"))
    result = executor.execute(task(), {"workspace": str(root)})
    assert result.status == "blocked"


def test_sandbox_failure_event_stays_blocked_even_if_worker_reports_failed(tmp_path, monkeypatch):
    executor = SshCodexExecutor(settings(tmp_path))
    events = json.dumps({"type": "item.completed", "item": {
        "type": "command_execution", "exit_code": 1,
        "aggregated_output": "bwrap: No permissions to create new namespace"}})
    monkeypatch.setattr(executor, "_request", lambda *a, **k: {
        "protocol": 1, "status": "failed", "returncode": 1, "stdout": events})
    result = executor.execute(task(), {"workspace": str(workspace(tmp_path))})
    assert result.status == "blocked"
    assert any("sandbox unavailable" in issue for issue in result.open_issues)


def test_cli_preflight_and_http_route_forward_probe_flag(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from fastapi.testclient import TestClient
    from algo_agent.cli.main import app
    from algo_agent.cli.client import ControlClient
    from algo_agent.service import app as service
    from algo_agent.service.controller import Controller
    seen = []
    def readiness(self, *, probe_coding=False):
        seen.append(probe_coding)
        return {"blockers": ["coding_remote_ready"], "live_verified": False}
    monkeypatch.setattr(ControlClient, "readiness", readiness)
    result = CliRunner().invoke(app, ["--root", str(tmp_path), "--json",
                                     "preflight", "--probe-coding", "--strict"])
    assert result.exit_code == 2 and seen == [True]
    monkeypatch.setattr(Controller, "readiness", readiness)
    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    monkeypatch.setattr(service, "_CONTROLLER", ctrl)
    with TestClient(service.create_app(ctrl.settings)) as client:
        assert client.get("/readiness?probe_coding=true").status_code == 200
    assert seen == [True, True]


def test_remote_stop_kills_group_descendant_ignoring_term():
    descendant = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"
    parent = (
        "import subprocess,sys,time\n"
        f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}],stdout=subprocess.PIPE,text=True)\n"
        "child.stdout.readline()\n"
        "print(child.pid,flush=True)\n"
        "time.sleep(30)\n")
    process = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        pid = int(process.stdout.readline())
        ssh_worker._stop(process)
        deadline = time.monotonic() + 2
        state = ""
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail(f"remote descendant still running: {pid} state={state}")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@pytest.mark.parametrize("tool,expected", [("codex_cli", True), ("codex_ssh", True),
                                          ("offline_template", False)])
def test_iqa_validation_report_recognizes_remote_real_coding(tmp_path, tool, expected):
    from argparse import Namespace
    from scripts.validate_iqa_dialogue import DialogueValidation
    from algo_agent.paths import TaskPaths, read_json, write_json
    from algo_agent.service.controller import Controller
    ctrl = Controller(Settings(tmp_path, env={"ALGO_OFFLINE": "1"}))
    spec = ctrl.create_task("", "report", task_spec={
        "business": "report unit test", "demo": True, "synthetic": True,
        "primary_metric": {"name": "rmse", "threshold": .5}})
    validation = object.__new__(DialogueValidation)
    validation.args = Namespace(task_id=spec.task_id)
    validation.ctrl = ctrl
    validation.tp = TaskPaths(ctrl.settings, spec.task_slug)
    validation.turns, validation.checks = [], {}
    write_json(validation.tp.coding_records / "fixture/result.json", {
        "tool": tool, "status": "completed", "changed_files": ["model_pool/new/method.py"]})
    validation.report()
    summary = read_json(validation.tp.docs / "validation/summary.json")
    assert summary["checks"]["live_code_applied"] is expected
