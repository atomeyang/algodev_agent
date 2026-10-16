"""Standalone stdlib-only worker, sent over SSH without installing this package.

Only trusted transport configuration is executable. Task contents remain Codex
input, and workspace-write sandboxing is mandatory on the remote machine.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

PROTOCOL = 1
MAX_FILES = 2000
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
SOURCE_SUFFIXES = {".py", ".pyi", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
                   ".cfg", ".ini", ".sh", ".c", ".cpp", ".h", ".hpp", ".cu", ".cmake"}
SENSITIVE_NAMES = {"auth.json", "credentials.json", "id_rsa", "id_ed25519", "known_hosts"}


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("invalid source path")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError("source path must be a normalized relative path")
    if any(part.startswith(".") or part.lower() in SENSITIVE_NAMES for part in p.parts):
        raise ValueError("hidden or credential paths cannot be transferred")
    if name != "coding_contract.json":
        if len(p.parts) < 2 or p.parts[0] not in {"model_pool", "algorithm_plugins"}:
            raise ValueError("only method sources and plugin context may be transferred")
        if p.suffix.lower() not in SOURCE_SUFFIXES and p.name not in {"METHOD_SOURCE", "CMakeLists.txt"}:
            raise ValueError("only source files may be transferred, not data or weights")
    return p


def in_scope(name, scopes):
    path = safe_relative(name)
    if path.parts[0] != "model_pool":
        return False
    return any(path == PurePosixPath(scope) or PurePosixPath(scope) in path.parents for scope in scopes)


def validate_scopes(scopes):
    if not isinstance(scopes, list) or not scopes:
        raise ValueError("remote write scopes must be method directories")
    for scope in scopes:
        if not isinstance(scope, str):
            raise ValueError("remote write scopes must be strings")
        p = safe_relative(scope + "/scope.py")
        if p.parts[0] != "model_pool" or len(p.parts) < 3:
            raise ValueError("remote write scopes must be inside a method directory")


def decode_files(files):
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("source file count limit exceeded")
    decoded, size = {}, 0
    for name, encoded in files.items():
        safe_relative(name)
        if not isinstance(encoded, str) or len(encoded) > 2 * MAX_BUNDLE_BYTES:
            raise ValueError("invalid source encoding or size")
        data = base64.b64decode(encoded, validate=True)
        data.decode("utf-8")
        size += len(data)
        if size > MAX_BUNDLE_BYTES:
            raise ValueError("source transfer exceeds 16 MiB")
        decoded[name] = data
    for name in decoded:
        if any(str(parent) in decoded for parent in PurePosixPath(name).parents):
            raise ValueError("source file/directory collision")
    return decoded


def snapshot(workspace):
    files, size = {}, 0
    for path in sorted(Path(workspace).rglob("*")):
        rel = path.relative_to(workspace)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("source symlinks are not transferable")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("only regular, non-hardlinked source files may be transferred")
        name = rel.as_posix()
        safe_relative(name)
        size += info.st_size
        if size > MAX_BUNDLE_BYTES or len(files) >= MAX_FILES:
            raise ValueError("source bundle exceeds transfer limits")
        data = path.read_bytes()
        data.decode("utf-8")
        files[name] = base64.b64encode(data).decode("ascii")
    return files


def write_sources(workspace, decoded):
    for name, data in decoded.items():
        path = Path(workspace) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _root(value):
    root = Path(value)
    if not root.is_absolute() or len(root.parts) < 3 or ".." in root.parts:
        raise ValueError("SSH worker root must be a dedicated absolute directory")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("SSH worker root must not contain symlinks")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("SSH worker root must be owned by this user with mode 0700")
    return root


def _identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _stop(child):
    _kill_group(child.pid)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(.05)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=5)


def probe(codex, workspace, timeout=20):
    checks = {"python": sys.version_info >= (3, 11), "linux": sys.platform == "linux", "codex": False,
              "authentication": False, "sandbox": False}
    detail, version = "", ""
    deadline = time.monotonic() + timeout
    for key, argv in (
        ("codex", [codex, "--version"]),
        ("authentication", [codex, "login", "status"]),
        ("sandbox", [codex, "sandbox", "linux", "/bin/true"]),
    ):
        try:
            proc = subprocess.run(argv, cwd=workspace, capture_output=True, text=True,
                                  timeout=max(.1, deadline - time.monotonic()))
            checks[key] = proc.returncode == 0
            if key == "codex":
                version = proc.stdout.strip()[:200]
            if not checks[key]:
                # Authentication output can contain account details; do not export it.
                detail = f"remote {key} probe failed"
                if key == "sandbox":
                    detail += ": " + (proc.stderr or proc.stdout)[-1500:]
                break
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = f"remote {key} probe failed: {type(exc).__name__}"
            break
    return {"available": all(checks.values()), "checks": checks, "version": version,
            "detail": detail or "remote CLI, login status and sandbox probes passed; no model generation tested"}


def _cancel(root, job_id):
    job = root / job_id
    if job.is_symlink() or not job.is_dir():
        return {"cancel_requested": False}
    (job / "cancel").touch(mode=0o600)
    try:
        record = json.loads((job / "process.json").read_text())
        pid = int(record["pid"])
        if record.get("start") and _identity(pid) == record["start"]:
            _kill_group(pid)
    except (OSError, ValueError, KeyError):
        pass
    return {"cancel_requested": True}


def run(request):
    if request.get("protocol") != PROTOCOL:
        raise ValueError("SSH worker protocol mismatch")
    root = _root(request["root"])
    operation = request.get("operation")
    if operation == "probe":
        with tempfile.TemporaryDirectory(prefix="probe_", dir=root) as workspace:
            return {"protocol": PROTOCOL, "probe": probe(request["codex"], workspace)}
    job_id = request.get("job_id", "")
    if not re.fullmatch(r"job_[0-9a-f]{32}", job_id):
        raise ValueError("invalid remote job identity")
    if operation == "cancel":
        return {"protocol": PROTOCOL, **_cancel(root, job_id)}
    if operation != "execute":
        raise ValueError("unsupported SSH worker operation")
    decoded = decode_files(request["files"])
    scopes = request.get("write_paths", [])
    validate_scopes(scopes)
    seconds = float(request["timeout_s"])
    if not 0 < seconds <= 86400:
        raise ValueError("remote coding timeout must be in (0, 86400]")
    expires_at = float(request["expires_at"])
    if not math.isfinite(expires_at):
        raise ValueError("invalid remote coding expiry")
    seconds = min(seconds, expires_at - time.time())
    if seconds <= 0:
        return {"protocol": PROTOCOL, "status": "blocked", "reason": "remote coding request expired"}
    deadline = time.monotonic() + seconds
    job = root / job_id
    job.mkdir(mode=0o700, exist_ok=False)
    child = None
    try:
        workspace, temporary = job / "workspace", job / "tmp"
        workspace.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        write_sources(workspace, decoded)
        checked = probe(request["codex"], workspace, min(20, seconds))
        if not checked["available"]:
            return {"protocol": PROTOCOL, "status": "blocked", "reason": checked["detail"], "probe": checked}
        if (job / "cancel").exists() or time.monotonic() >= deadline:
            return {"protocol": PROTOCOL, "status": "blocked", "reason": "remote coding cancelled or timed out"}
        argv = [request["codex"], "exec", "--cd", str(workspace), "--sandbox", "workspace-write",
                "-c", 'approval_policy="never"', "--skip-git-repo-check", "--json"]
        if request.get("model"):
            argv += ["--model", request["model"]]
        if request.get("reasoning_effort"):
            argv += ["-c", "model_reasoning_effort=" + json.dumps(request["reasoning_effort"])]
        argv.append("-")
        env = dict(os.environ)
        env.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                   PYTHONDONTWRITEBYTECODE="1")
        env.pop("SSH_AUTH_SOCK", None)
        # Prompt and logs stay outside the source tree and are not returned as code.
        prompt_path = job / "prompt.json"
        prompt_path.write_text(request["prompt"], encoding="utf-8")
        with prompt_path.open("rb") as stdin, (job / "stdout").open("wb") as out, (job / "stderr").open("wb") as err:
            child = subprocess.Popen(argv, cwd=workspace, stdin=stdin, stdout=out, stderr=err,
                                     env=env, start_new_session=True)
            process_record = job / "process.tmp"
            process_record.write_text(json.dumps({"pid": child.pid, "start": _identity(child.pid)}))
            process_record.replace(job / "process.json")
            reason = ""
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "remote coding wall-clock budget exhausted"
                elif (job / "cancel").exists():
                    reason = "remote coding cancelled"
                elif sum((job / n).stat().st_size for n in ("stdout", "stderr")) > MAX_LOG_BYTES:
                    reason = "remote coding log limit exceeded"
                if reason:
                    _stop(child)
                    break
                time.sleep(.1)
        if (job / "cancel").exists():
            reason = "remote coding cancelled"
        elif not reason and child.returncode != 0:
            reason = "remote codex exec returned non-zero"
        stdout = (job / "stdout").read_bytes()[:MAX_LOG_BYTES].decode("utf-8", errors="replace")
        stderr = (job / "stderr").read_bytes()[:MAX_LOG_BYTES].decode("utf-8", errors="replace")
        result = {"protocol": PROTOCOL, "status": "failed" if reason else "completed",
                  "reason": reason, "returncode": child.returncode, "stdout": stdout, "stderr": stderr}
        if sum((job / n).stat().st_size for n in ("stdout", "stderr")) > MAX_LOG_BYTES:
            return {**result, "status": "failed", "reason": "remote coding log limit exceeded"}
        if not reason and child.returncode == 0:
            after = snapshot(workspace)
            changed = {name: data for name, data in after.items() if request["files"].get(name) != data}
            deleted = sorted(set(request["files"]) - set(after))
            if any(not in_scope(name, scopes) for name in [*changed, *deleted]):
                return {**result, "status": "blocked", "reason": "remote write scope violation"}
            result.update(files=changed, deleted=deleted)
        return result
    finally:
        if child is not None:
            _stop(child)
        shutil.rmtree(job)


def main():
    os.umask(0o077)
    def interrupted(signum, frame):
        raise RuntimeError("SSH worker interrupted")

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        wire = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(wire) > MAX_REQUEST_BYTES:
            raise ValueError("SSH worker request too large")
        request = json.loads(wire)
        result = run(request)
    except Exception as exc:
        result = {"protocol": PROTOCOL, "status": "blocked",
                  "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
