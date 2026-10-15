"""Evidence service: source registry, packet builder, scoped reader, delta and
dependency cache (guide 12).

All reads pass through the reader, which enforces scope, suffix and size limits
in code (not just in prompts) so raw data and frozen final-test assets never
leak to a cloud role.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import Settings
from ..paths import TaskPaths, read_json, sha256_text, safe_join
from ..schemas import contracts as C
from ..storage.repository import Repository


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class EvidenceStore:
    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo

    def register_source(self, task_id: str, uri: str, source_type: str = "local",
                        locator: str = "", version: str = "", scope: str = "task",
                        upload_policy: str = "restricted", trust: str = "medium",
                        checksum: str = "") -> C.EvidenceRef:
        ref = C.EvidenceRef(evidence_id=_id("ev"), uri=uri, locator=locator, version=version,
                            checksum=checksum or sha256_text(uri), scope=scope,
                            upload_policy=upload_policy, source_type=source_type, trust=trust)
        self.repo.register_evidence(ref, task_id)
        return ref

    def get(self, evidence_id: str) -> Optional[C.EvidenceRef]:
        return self.repo.get_evidence(evidence_id)

    def list(self, task_id: str) -> List[C.EvidenceRef]:
        return self.repo.list_evidence(task_id)


class PacketBuilder:
    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo

    def build_packet(self, task_id: str, requirement: str, plan_id: str = "", baseline_id: str = "",
                     metric_delta: Optional[Dict[str, float]] = None,
                     slices: Optional[Dict[str, Any]] = None, failures: Optional[List[str]] = None,
                     counterexamples: Optional[List[str]] = None,
                     unknowns: Optional[List[str]] = None, refs: Optional[List[str]] = None) -> C.EvidencePacket:
        ref_objs: List[C.EvidenceRef] = []
        for rid in refs or []:
            ref = self.repo.get_evidence(rid)
            if ref:
                ref_objs.append(ref)
        return C.EvidencePacket(
            packet_id=_id("pkt"), task_id=task_id, plan_id=plan_id, baseline_id=baseline_id,
            requirement=requirement, metric_delta=metric_delta or {}, slices=slices or {},
            failures=failures or [], counterexamples=counterexamples or [], unknowns=unknowns or [],
            refs=ref_objs,
        )


class EvidenceReader:
    """Reads evidence within scope, enforcing the upload boundary."""

    FORBIDDEN_MARKERS = ("data/raw", "data/releases", "test.npz", "final_test",
                         "evaluation/protocols", "pred_test", ".env")

    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo
        self.policy = settings.upload_policy()
        self.max_bytes = int(self.policy.get("max_read_bytes", 262144))
        self.allowed = set(self.policy.get("allowed_suffixes", [".md", ".txt", ".json", ".py", ".yaml"]))

    def read_remote(self, ref: C.EvidenceRef, invocation_id: str = "") -> Dict[str, Any]:
        """Fetch bounded public text; redirects and private destinations are rejected."""
        import ipaddress
        import socket
        import ssl
        import http.client
        from urllib.parse import urlsplit
        audit = C.ReadAudit(audit_id=_id("ra"), role="research_design",
                            invocation_id=invocation_id, refs=[ref.evidence_id])
        try:
            url = urlsplit(ref.uri)
            if url.scheme not in {"http", "https"} or url.username or url.password:
                return self._deny(audit, "not a public HTTP URL")
            if url.port not in {None, 80, 443}:
                return self._deny(audit, "nonstandard remote port")
            port = url.port or (443 if url.scheme == "https" else 80)
            addresses = socket.getaddrinfo(url.hostname, port, type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                return self._deny(audit, "private/network-local destinations are forbidden")
            # Connect to the checked address, not a second DNS lookup. TLS still
            # verifies the original hostname, so DNS rebinding cannot reach LAN hosts.
            connection = http.client.HTTPConnection(url.hostname, port, timeout=20)
            try:
                connection.sock = socket.create_connection((addresses[0][4][0], port), timeout=20)
                if url.scheme == "https":
                    connection.sock = ssl.create_default_context().wrap_socket(
                        connection.sock, server_hostname=url.hostname)
                connection.request("GET", (url.path or "/") + ("?" + url.query if url.query else ""),
                                   headers={"Accept-Encoding": "identity"})
                response = connection.getresponse()
                if not 200 <= response.status < 300:
                    return self._deny(audit, f"remote HTTP {response.status}; redirects are not followed")
                if not any(t in response.getheader("Content-Type", "") for t in
                           ("text/", "application/json")):
                    return self._deny(audit, "only text sources supported; import extracted PDF text")
                content = response.read(self.max_bytes + 1)
                if len(content) > self.max_bytes:
                    return self._deny(audit, "remote source exceeds read limit")
            finally:
                connection.close()
            audit.upload_bytes = len(content)
            self.repo.add_read_audit(audit)
            return {"status": "ok", "content": content.decode("utf-8", errors="replace"),
                    "source": ref.uri}
        except Exception as exc:
            return self._deny(audit, str(exc))

    def _resolve(self, ref: C.EvidenceRef, task_slug: str) -> Optional[Path]:
        uri = ref.uri
        if uri.startswith(("http://", "https://")):
            return None  # remote fetch handled by provider layer
        tp = TaskPaths(self.settings, task_slug)
        base = tp.root
        if uri.startswith("root://"):
            path = safe_join(self.settings.root, uri[len("root://"):])
            path = tp.resolve_reference(str(path))
            shared = self.settings.root / "shared_assets"
            if path != base and base not in path.parents and shared not in path.parents:
                raise ValueError("root evidence is outside this task and shared assets")
            return path
        rel = uri[len("task://"):] if uri.startswith("task://") else uri
        return safe_join(base, rel)

    def read_evidence(self, ref: C.EvidenceRef, task_slug: str, role: str,
                      range_spec: str = "full", invocation_id: str = "") -> Dict[str, Any]:
        requested = range_spec
        audit = C.ReadAudit(audit_id=_id("ra"), role=role, invocation_id=invocation_id,
                            refs=[ref.evidence_id], requested_range=requested)
        # scope check
        if ref.scope not in {"task", "shared"}:
            return self._deny(audit, "evidence not in current task or shared scope")

        try:
            path = self._resolve(ref, task_slug)
        except ValueError as exc:
            return self._deny(audit, str(exc))
        if path is None:
            # remote: rely on provider tool; return the ref, no upload yet
            audit.status = "deferred_remote"; audit.upload_bytes = 0
            self.repo.add_read_audit(audit)
            return {"status": "deferred_remote", "ref": ref.model_dump()}

        if any(marker in str(path).replace("\\", "/") for marker in self.FORBIDDEN_MARKERS):
            return self._deny(audit, "raw data / frozen final-test assets are not uploadable")
        if not path.exists():
            return self._deny(audit, f"locator not resolvable: {path}")
        if path.suffix and path.suffix.lower() not in self.allowed:
            return self._deny(audit, f"suffix {path.suffix} not allowed for upload")
        size = path.stat().st_size
        if size > self.max_bytes:
            return self._deny(audit, f"file exceeds max_read_bytes ({size}>{self.max_bytes})")

        text = path.read_text(encoding="utf-8", errors="replace")
        snippet = self._apply_range(text, range_spec)
        audit.status = "ok"
        audit.actual_range = range_spec
        audit.upload_bytes = len(snippet.encode("utf-8"))
        self.repo.add_read_audit(audit)
        return {"status": "ok", "ref": ref.model_dump(), "content": snippet,
                "upload_bytes": audit.upload_bytes}

    def _apply_range(self, text: str, range_spec: str) -> str:
        if not range_spec or range_spec == "full":
            return text
        if ":" in range_spec:
            try:
                a, b = (int(x) if x else None for x in range_spec.split(":", 1))
                lines = text.splitlines()
                return "\n".join(lines[a:b])
            except ValueError:
                raise ValueError("invalid evidence line range")
        raise ValueError("invalid evidence line range")

    def _deny(self, audit: C.ReadAudit, reason: str) -> Dict[str, Any]:
        audit.status = "denied"
        audit.actual_range = ""
        self.repo.add_read_audit(audit)
        return {"status": "denied", "reason": reason}


class DeltaBuilder:
    def __init__(self, repo: Repository):
        self.repo = repo

    def compare_runs(self, task_id: str, baseline_model: str, candidate_model: str) -> Dict[str, Any]:
        def metrics(mid):
            evals = self.repo.list_evals(task_id, mid)
            return evals[-1].metrics if evals else {}
        b, c = metrics(baseline_model), metrics(candidate_model)
        delta = {k: round(c[k] - b[k], 6) for k in set(b) & set(c)}
        return {"baseline": baseline_model, "candidate": candidate_model, "delta": delta,
                "baseline_metrics": b, "candidate_metrics": c}


class DependencyCache:
    def __init__(self, repo: Repository):
        self.repo = repo

    @staticmethod
    def key(requirement: str, evidence: Iterable[str], code: str, data: str,
            evaluation: str, model: str, prompt_version: str) -> str:
        parts = [requirement, ",".join(sorted(evidence)), code, data, evaluation, model, prompt_version]
        return sha256_text("|".join(parts))

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.repo.cache_get(key)

    def put(self, key: str, value: Any, dependencies: List[str]) -> None:
        self.repo.cache_put(key, value, dependencies)

    def invalidate_by_dependency(self, dependency: str) -> int:
        return self.repo.cache_invalidate_by_dependency(dependency)


__all__ = ["EvidenceStore", "PacketBuilder", "EvidenceReader", "DeltaBuilder", "DependencyCache"]
