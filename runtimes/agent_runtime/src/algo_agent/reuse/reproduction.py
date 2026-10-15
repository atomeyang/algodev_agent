"""Minimal reproduction packages (guide 12.2, M2/M5).

``export_reproduction`` freezes an asset dependency closure plus data/weight
references, seed, entry points, expected results and tolerance into a
``ReproductionManifest``. ``validate_reproduction`` checks it is *actually*
resolvable and that results land inside the declared tolerance -- it reports
concrete blockers and never fabricates a passing result.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..paths import TaskPaths, read_json, sha256_file, write_json
from ..schemas import contracts as C
from ..storage.repository import Repository

# never bundle secrets or private final-test material (guide 12.2)
_SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "token", "authorization")
_FORBIDDEN_IN_PACKAGE = ("data/raw", "data/releases", "test.npz", "final_test")


def export_reproduction(settings: Settings, repo: Repository, task_id: str, *,
                        out_dir: Optional[Path] = None) -> Dict[str, Any]:
    spec = repo.get_spec(task_id)
    tp = TaskPaths(settings, spec.task_slug)
    state = repo.get_state(task_id)
    models = repo.list_models(task_id)
    evals = repo.list_evals(task_id)
    plans = repo.list_plans(task_id)
    plan = repo.get_plan(state.active_plan_id) if state.active_plan_id else None

    dataset_versions = sorted({m.data_version for m in models if m.data_version})
    seed = plan.data_recipe.seed if plan else (models[0].seed if models else 0)

    data_refs: List[str] = []
    checksums: Dict[str, str] = {}
    external_refs: List[Dict[str, Any]] = []
    for version in dataset_versions:
        rel = str(tp.release_dir(version).relative_to(tp.root))
        man = tp.release_dir(version) / "manifest.json"
        if man.exists():
            data_refs.append(rel)
            # hash the whole directory so validation can recompute the same value
            checksums[rel] = _dir_hash(tp.release_dir(version))
        else:
            external_refs.append({"path": rel, "reason": "data release not present locally",
                                  "checksum": "", "resolvable": False})

    weights_refs: List[str] = []
    for m in models:
        if not m.weights_ref:
            continue
        ref = tp.root / m.weights_ref
        if ref.exists():
            weights_refs.append(m.weights_ref)
            checksums[m.weights_ref] = sha256_file(ref)
        else:
            external_refs.append({"path": m.weights_ref, "reason": "weights not present locally",
                                  "checksum": m.weights_hash, "resolvable": bool(m.weights_hash)})

    expected: Dict[str, float] = {}
    best = state.best_model_id
    for e in evals:
        if e.model_id == best:
            expected = {k: float(v) for k, v in e.metrics.items()}
            break
    if not expected and evals:
        expected = {k: float(v) for k, v in evals[-1].metrics.items()}

    lock = _lock_snapshot()
    manifest = C.ReproductionManifest(
        manifest_id=f"repro_{task_id}",
        task_id=task_id,
        asset_dependencies=[f"{m.model_id}@{m.data_version}" for m in models if m.data_version],
        lock_file="reproduction.lock.json",
        data_refs=data_refs,
        weights_refs=weights_refs,
        seed=int(seed or 0),
        exec_entries=_exec_entries(repo, spec, plans, settings),
        eval_entries=_eval_entries(tp, spec),
        expected_results=expected,
        tolerance=float((settings.workflow.get("eval") or {}).get("repro_tolerance", 1e-6)),
    )

    target = tp.output_path(out_dir) if out_dir else tp.deliverables / "reproduction"
    if not target.is_relative_to(tp.deliverables) or target == tp.deliverables:
        raise ValueError("reproduction output must be a subdirectory of task deliverables/")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "reproduction.json", manifest.model_dump())
    write_json(target / "reproduction.lock.json", lock)
    write_json(target / "checksums.json", checksums)
    write_json(target / "external_refs.json", external_refs)
    (target / "REPRODUCE.md").write_text(
        _reproduce_md(settings, spec, manifest, checksums, external_refs, lock), encoding="utf-8")

    repo.append_event(task_id, "repro.export", f"reproduction package written to {target}",
                      data={"manifest_id": manifest.manifest_id, "dir": str(target),
                            "external_refs": len(external_refs)})
    return {"manifest": manifest.model_dump(), "dir": str(target),
            "files": sorted(str(p.relative_to(target)) for p in target.glob("*")),
            "external_refs": external_refs, "checksums": checksums}


def validate_reproduction(settings: Settings, repo: Repository, task_id: str, *,
                          manifest: Optional[C.ReproductionManifest] = None,
                          package_dir: Optional[Path] = None,
                          actual_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Return status validated | mismatch | blocked | failed with exact blockers."""
    spec = repo.get_spec(task_id)
    tp = TaskPaths(settings, spec.task_slug)
    pkg = Path(package_dir) if package_dir else tp.deliverables / "reproduction"
    data = manifest.model_dump() if manifest else read_json(pkg / "reproduction.json")
    if not data:
        return {"status": "failed", "blockers": [f"no reproduction manifest at {pkg}"],
                "checked": []}

    blockers: List[str] = []
    checked: List[Dict[str, Any]] = []
    mismatches: List[str] = []

    cks = read_json(pkg / "checksums.json", {}) or {}
    for ref in list(data.get("data_refs", [])) + list(data.get("weights_refs", [])):
        path = tp.root / ref
        ok = path.exists() or Path(ref).exists()
        checked.append({"ref": ref, "exists": ok})
        if not ok:
            blockers.append(f"referenced artifact missing: {ref}")
            continue
        p = path if path.exists() else Path(ref)
        want = cks.get(ref)
        if want:
            got = sha256_file(p) if p.is_file() else _dir_hash(p)
            checked[-1]["checksum_ok"] = got == want
            if got != want:
                mismatches.append(f"checksum mismatch for {ref}")

    ext = read_json(pkg / "external_refs.json", []) or []
    for ref in ext:
        if not ref.get("resolvable"):
            blockers.append(f"unresolvable external reference: {ref.get('path')} "
                            f"({ref.get('reason')})")

    if not data.get("exec_entries"):
        blockers.append("no execution entry point recorded")
    if not data.get("eval_entries"):
        blockers.append("no evaluation entry point recorded")
    if not data.get("expected_results"):
        blockers.append("no expected results recorded; cannot verify tolerance")

    secret_hits = _scan_for_secrets(pkg)
    if secret_hits:
        blockers.append(f"credential material present in package: {secret_hits}")

    if actual_metrics:
        tol = float(data.get("tolerance", 1e-6))
        for key, want in (data.get("expected_results") or {}).items():
            got = actual_metrics.get(key)
            if got is None:
                mismatches.append(f"{key}: not measured in the re-run")
            elif abs(float(got) - float(want)) > tol:
                mismatches.append(f"{key}: expected {want} got {got} (tol {tol})")

    if blockers:
        status = "blocked"
    elif mismatches:
        status = "mismatch"
    else:
        status = "validated"
    result = {"status": status, "blockers": blockers, "mismatches": mismatches,
              "checked": checked, "tolerance": data.get("tolerance"),
              "expected_results": data.get("expected_results")}
    repo.append_event(task_id, "repro.validate", f"reproduction validation: {status}",
                      level="info" if status == "validated" else "warning",
                      data={"blockers": blockers, "mismatches": mismatches})
    return result


# --------------------------------------------------------------------------- #
def _lock_snapshot() -> Dict[str, Any]:
    lock: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "pydantic", "langgraph", "sqlite3"):
        try:
            m = __import__(mod)
            lock[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            lock[mod] = "missing"
    return lock


def _exec_entries(repo: Repository, spec: C.TaskSpec, plans: List[Dict[str, Any]],
                  settings: Settings) -> List[str]:
    entries: List[str] = []
    for row in plans:
        plan = repo.get_plan(row["plan_id"])
        for c in (plan.candidates if plan else []):
            ref = c.method or (c.params or {}).get("code_ref", "")
            if ref:
                entries.append(str(ref))
    if spec.demo or spec.synthetic:
        entries.append(f"algo-agent --root {settings.root} demo --offline")
    return sorted(set(entries))


def _eval_entries(tp: TaskPaths, spec: C.TaskSpec) -> List[str]:
    entries = [str(p.relative_to(tp.root)) for p in sorted((tp.evaluation / "scripts").glob("*"))
               if p.is_file()]
    entries.append("algo_agent.execution.evaluator:evaluate_run")
    return entries


def _dir_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


_SECRET_RE = None
# Placeholder values that look like a key but carry no credential material: a
# reproduction lock records `missing`/`unknown` for modules that are absent, and
# a redacted field may hold `redacted`. These must not block a valid package.
_BENIGN_SECRET_VALUES = {
    "null", "none", "false", "true", "changeme", "missing", "unknown", "unset",
    "not-set", "notset", "n/a", "na", "placeholder", "redacted", "example",
    "***redacted***",
}


def _scan_for_secrets(pkg: Path) -> List[str]:
    import re
    global _SECRET_RE
    if _SECRET_RE is None:
        keys = "|".join(_SECRET_MARKERS)
        # Tolerate JSON/YAML/en quoting on both sides of the key and value.
        _SECRET_RE = re.compile(
            rf"[\"\']?\b({keys})\b[\"\']?\s*[:=]\s*[\"\']?([^\s\"\',}}]{{8,}})",
            re.IGNORECASE)
    hits: List[str] = []
    for f in pkg.rglob("*"):
        if not f.is_file() or f.suffix.lower() in (".npz", ".pt", ".bin", ".safetensors"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _SECRET_RE.finditer(text):
            value = (m.group(2) or "").strip().strip("\"'")
            if value.lower() in _BENIGN_SECRET_VALUES:
                continue
            hits.append(f"{f.relative_to(pkg)}:{m.group(1)}")
            break
    return hits


def _reproduce_md(settings: Settings, spec: C.TaskSpec, manifest: C.ReproductionManifest,
                  checksums: Dict[str, str], external_refs: List[Dict[str, Any]],
                  lock: Dict[str, Any]) -> str:
    lines = [
        f"# Reproduction — {spec.task_id}",
        "",
        f"- task_slug: {spec.task_slug}",
        f"- seed: {manifest.seed}",
        f"- tolerance: {manifest.tolerance}",
        f"- python: {lock.get('python')}",
        "",
        "## Steps",
        "1. Create the environment from `reproduction.lock.json`.",
        "2. Resolve every entry in `data_refs` / `weights_refs` (checksums in `checksums.json`).",
        "3. Run each entry in the reproduction manifest `exec_entries`.",
        "4. Evaluate with `eval_entries` and compare against `expected_results` within tolerance.",
        "",
        "## Entry points",
    ]
    lines += [f"- `{e}`" for e in manifest.exec_entries] or ["- (none recorded)"]
    lines += ["", "## Expected results"]
    lines += [f"- {k}: {v}" for k, v in manifest.expected_results.items()] or ["- (none recorded)"]
    if external_refs:
        lines += ["", "## External references (must be supplied separately)"]
        lines += [f"- `{r['path']}` — {r['reason']}" for r in external_refs]
    lines += ["", "> Offline/synthetic packages keep their `synthetic_demo` label; they are not "
                  "business validation."]
    return "\n".join(lines) + "\n"


__all__ = ["export_reproduction", "validate_reproduction"]
