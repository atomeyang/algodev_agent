"""Versioned data experiments with a fixed development split and one-shot holdout.

Raw sources are never edited. Recipes are complete recipes applied to the same
base, not incremental patches to the previous round. The directory boundary
prevents accidental leakage; untrusted code still needs an OS/container sandbox.
"""

from __future__ import annotations

import copy
import json
import uuid
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import read_json, sha256_file, sha256_obj, write_json
from ..policy import check_running, remaining_seconds, task_key
from ..schemas import contracts as C


def operation_capabilities(domain: str) -> dict:
    shared = {
        "resample": {"split": "train", "params": ["count", "seed", "weight"],
                     "semantics": "replaces training set with count draws; count charged as added_samples"},
        "extend": {"splits": ["train", "development_test", "final_test"],
                   "source": "registered source_id; test labels must be trusted"},
    }
    if domain == "iqa":
        return {**shared,
                "resample": {"split": "train", "params": ["count", "seed", "defect", "weight"],
                             "semantics": "replaces training set with count draws; count charged as added_samples"},
                "augment": {"splits": ["train", "development_test"],
                            "params": {"transform": "hflip", "label_preserving": True, "count": "positive integer"}},
                "filter": {"split": "train", "params": ["score_min", "score_max"]},
                "annotate": {"split": "train", "params": ["limit", "min_confidence", "rubric"],
                             "labels": "weak only; local VLM; fixed defect JSON schema"},
                "relabel": {"split": "train", "source": "registered JSONL with path, score, defects"}}
    if domain == "demo_tabular":
        return {**shared,
                "augment": {"split": "train", "params": ["count", "noise_std (0..1)", "seed"]},
                "filter": {"split": "train", "params": ["feature", "minimum", "maximum"]},
                "resample": {"split": "train", "params": ["count", "seed", "slice=low_light", "weight"]}}
    return {"status": "domain data operations adapter missing"}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"}


def verify_release(tp, version: str) -> C.DatasetManifest:
    release = tp.release_dir(version)
    manifest = C.DatasetManifest(**read_json(release / "manifest.json"))
    actual = _hashes(release)
    for name, digest in manifest.checksums.items():
        path = release / name
        if name.startswith("protocol/"):
            path = tp.evaluation / "protocols" / version / name.removeprefix("protocol/")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"immutable dataset changed: {version}/{name}")
    expected = {k: v for k, v in manifest.checksums.items() if not k.startswith("protocol/")}
    if actual != expected:
        raise ValueError(f"immutable dataset files changed: {version}")
    for path, digest in manifest.external_checksums.items():
        if not Path(path).is_file() or sha256_file(Path(path)) != digest:
            raise ValueError(f"source image changed after data release: {path}")
    return manifest


def _source(spec, op) -> tuple[C.DataSource, Path]:
    source = next((s for s in spec.data_sources if s.source_id == op.source_id), None)
    if source is None:
        raise ValueError(f"unregistered source: {op.source_id}")
    path = Path(source.path)
    if not source.checksum or sha256_file(path) != source.checksum:
        raise ValueError(f"source changed since task intake: {source.source_id}")
    return source, path


def _claim(repo, spec, key: str, amount: int) -> None:
    check_running(repo, spec.task_id)
    repo.claim_budget(spec.task_id, key, amount, getattr(spec.budget, f"max_{key}"))


def _identity(row: dict) -> str:
    return str(Path(row["path"]).resolve())


def _disjoint(groups: list[list[dict]]) -> None:
    seen_paths, seen_subjects, seen_content = set(), set(), set()
    for group in groups:
        paths = {_identity(r) for r in group}
        subjects = {str(r["subject"]) for r in group if r.get("subject") is not None}
        content = {sha256_file(Path(p)) for p in paths}
        if paths & seen_paths or subjects & seen_subjects or content & seen_content:
            raise ValueError("train/development/final sample or subject overlap")
        seen_paths.update(paths)
        seen_subjects.update(subjects)
        seen_content.update(content)


def _image_hashes(release: Path) -> dict[str, str]:
    paths = set()
    for name in ("dev.jsonl", "eval_inputs.jsonl"):
        if (release / name).exists():
            paths.update(_identity(r) for r in _rows(release / name))
    return {p: sha256_file(Path(p)) for p in sorted(paths)}


def _validate_rows(rows: list[dict], *, labels: bool = True) -> None:
    if not rows:
        raise ValueError("empty dataset split")
    for row in rows:
        if not Path(row.get("path", "")).is_file():
            raise ValueError(f"image missing: {row.get('path')}")
        if labels and (row.get("score") is None or not math.isfinite(float(row["score"]))):
            raise ValueError("finite score required; run annotation after importing unlabelled images")


def _freeze_base(tp, spec, plan, repo) -> Path:
    base_version = "_auto_base"
    base = tp.release_dir(base_version)
    if (base / "manifest.json").exists():
        verify_release(tp, base_version)
        return base
    # A partial build can be retried; it has never been published as a release.
    from algorithm_plugins.registry import hook, resolve_plugin
    prepare = hook(resolve_plugin(spec), "prepare_dataset")
    base.mkdir(parents=True, exist_ok=True)
    if prepare:
        prepare(tp, spec, plan, base_version)
    elif spec.demo or spec.synthetic:
        from algorithm_plugins.demo_tabular import build_dataset
        build_dataset(base, seed=plan.data_recipe.seed, n=600)
    else:
        raise ValueError("domain plugin must supply prepare_dataset for real data")
    checksums = _hashes(base)
    proto = tp.evaluation / "protocols" / base_version
    if proto.exists():
        checksums.update({f"protocol/{k}": v for k, v in _hashes(proto).items()})
    manifest = C.DatasetManifest(
        dataset_id=f"{task_key(spec.task_id)}_base", task_id=spec.task_id, version=base_version,
        source=spec.domain, seed=plan.data_recipe.seed, checksums=checksums, stage="base")
    if spec.domain == "iqa":
        manifest.external_checksums = _image_hashes(base)
    write_json(base / "manifest.json", manifest.model_dump())
    return base


def _array(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError("invalid tabular data shapes")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("tabular source contains nonfinite values")
        return {"X": X.copy(), "y": y.copy()}


def _tabular(tp, spec, plan, base, release, repo) -> tuple[dict, list]:
    data = {"train": _array(base / "train.npz"),
            "development_test": _array(base / "val.npz"),
            "final_test": _array(base / "test.npz")}
    provenance = []
    for op in plan.data_recipe.operations:
        _claim(repo, spec, "data_operations", 1)
        d = data[op.split]
        rng = np.random.default_rng(int(op.params.get("seed", plan.data_recipe.seed)))
        before = len(d["y"])
        source_hash = ""
        if op.kind == "resample":
            count = int(op.params.get("count", before))
            if count <= 0:
                raise ValueError("resample count must be positive")
            _claim(repo, spec, "added_samples", count)
            weights = np.ones(before)
            if op.params.get("slice") == "low_light":
                weights[d["X"][:, 0] < 0] = float(op.params.get("weight", 2))
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("invalid resampling weights")
            indices = rng.choice(before, count, replace=True, p=weights / weights.sum())
            data[op.split] = {k: v[indices] for k, v in d.items()}
        elif op.kind == "filter":
            feature = int(op.params.get("feature", 0))
            mask = np.ones(before, dtype=bool)
            if "minimum" in op.params:
                mask &= d["X"][:, feature] >= float(op.params["minimum"])
            if "maximum" in op.params:
                mask &= d["X"][:, feature] <= float(op.params["maximum"])
            if not mask.any():
                raise ValueError("filter removed every training row")
            data[op.split] = {k: v[mask] for k, v in d.items()}
        elif op.kind == "augment":
            if op.split != "train":
                raise ValueError("tabular noise augmentation is training-only")
            count = int(op.params.get("count", before))
            if count <= 0:
                raise ValueError("augmentation count must be positive")
            _claim(repo, spec, "added_samples", count)
            idx = rng.choice(before, count, replace=True)
            sigma = float(op.params.get("noise_std", 0.01))
            if not math.isfinite(sigma) or not 0 <= sigma <= 1:
                raise ValueError("noise_std must be in [0,1]")
            data[op.split] = {"X": np.concatenate([d["X"], d["X"][idx] + rng.normal(0, sigma, d["X"][idx].shape)]),
                             "y": np.concatenate([d["y"], d["y"][idx]])}
        elif op.kind == "extend":
            source, path = _source(spec, op)
            if op.split != "train" and source.label_kind not in {"human", "verified"}:
                raise ValueError("test extension labels must be trusted at task intake")
            extra = _array(path)
            _claim(repo, spec, "added_samples", len(extra["y"]))
            data[op.split] = {k: np.concatenate([d[k], extra[k]]) for k in d}
            source_hash = source.checksum
        else:
            raise ValueError(f"tabular operation not supported: {op.kind}")
        provenance.append({**op.model_dump(), "before": before,
                           "after": len(data[op.split]["y"]), "source_hash": source_hash})
    signatures = [{sha256_obj(x.tolist()) for x in data[s]["X"]} for s in data]
    if any(signatures[i] & signatures[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("tabular split overlap")
    for split in ("train", "val"):
        np.savez(release / f"{split}.npz", **data["train"])
    dev = data["development_test"]
    np.savez(release / "test.npz", X=dev["X"], y=np.zeros_like(dev["y"]))
    proto = tp.evaluation / "protocols" / release.name
    proto.mkdir(parents=True, exist_ok=True)
    np.savez(proto / "development_labels.npz", **dev)
    # The selected plan's final extension is frozen but never exposed to research.
    np.savez(proto / "final_labels.npz", **data["final_test"])
    shutil.copy2(base / "split.json", release / "split.json")
    return {k: len(v["y"]) for k, v in data.items()}, provenance


def _iqa(tp, spec, plan, base, release, repo) -> tuple[dict, list]:
    from algorithm_plugins.iqa.data import _split_dev
    from algorithm_plugins.iqa import annotation
    dev = _rows(base / "dev.jsonl")
    split = read_json(base / "dev_split.json")
    groups = {"train": [dev[i] for i in split["train"]],
              "development_test": [dev[i] for i in split["val"]],
              "final_test": _rows(tp.evaluation / "protocols" / base.name / "eval_labels.jsonl")}
    _disjoint(list(groups.values()))
    provenance = []
    for op in plan.data_recipe.operations:
        _claim(repo, spec, "data_operations", 1)
        rows = groups[op.split]
        before, source_hash, details = len(rows), "", []
        rng = np.random.default_rng(int(op.params.get("seed", plan.data_recipe.seed)))
        if op.kind == "resample":
            count = int(op.params.get("count", before))
            if count <= 0:
                raise ValueError("resample count must be positive")
            _claim(repo, spec, "added_samples", count)
            weights = np.ones(before)
            defect = op.params.get("defect")
            for i, row in enumerate(rows):
                if (row.get("defects") or {}).get(defect, {}).get("present"):
                    weights[i] = float(op.params.get("weight", 2))
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("invalid resampling weights")
            idx = rng.choice(before, count, replace=True, p=weights / weights.sum())
            groups[op.split] = [copy.deepcopy(rows[i]) for i in idx]
        elif op.kind == "filter":
            low, high = float(op.params.get("score_min", 0)), float(op.params.get("score_max", 100))
            groups[op.split] = [r for r in rows if low <= float(r["score"]) <= high]
        elif op.kind in {"extend", "relabel"}:
            source, path = _source(spec, op)
            imported = _rows(path)
            if op.split != "train" and source.label_kind not in {"human", "verified"}:
                raise ValueError("test extension requires trusted labels")
            source_hash = source.checksum
            for row in imported:
                row["label_provenance"] = {"source_id": source.source_id, "label_kind": source.label_kind,
                                            "source_hash": source.checksum}
            if op.kind == "extend":
                _claim(repo, spec, "added_samples", len(imported))
                rows.extend(imported)
            else:
                _claim(repo, spec, "annotation_samples", len(imported))
                by_path = {_identity(r): r for r in imported}
                if not set(by_path) <= {_identity(r) for r in rows}:
                    raise ValueError("relabel source contains non-training samples")
                groups[op.split] = [{**r, **by_path.get(_identity(r), {})} for r in rows]
        elif op.kind == "annotate":
            # Only a loopback VLM may receive image bytes in automatic mode.
            import ipaddress
            from urllib.parse import urlsplit
            cfg = annotation.endpoint_config()
            hostname = urlsplit(cfg["base_url"]).hostname or ""
            if hostname != "localhost" and not ipaddress.ip_address(hostname).is_loopback:
                raise ValueError("automatic annotation requires a local endpoint")
            limit = int(op.params.get("limit", len(rows)))
            if limit <= 0:
                raise ValueError("annotation limit must be positive")
            confidence = float(op.params.get("min_confidence", 0.8))
            if not 0 <= confidence <= 1:
                raise ValueError("invalid annotation confidence")
            attempt = f"{op.operation_id}_{uuid.uuid4().hex[:12]}"
            annotation_dir = tp.annotation_run(release.name, attempt)
            annotation_dir.mkdir(parents=True, exist_ok=False)
            write_json(tp.annotations / "plans" / release.name / f"{attempt}.json",
                       {"plan_id": plan.plan_id, "operation": op.model_dump(),
                        "prompt_version": annotation.PROMPT_VERSION, "prompt": annotation._PROMPT})
            for row in rows[:limit]:
                _claim(repo, spec, "annotation_samples", 1)
                try:
                    answer = annotation.annotate(row["path"], cfg=cfg, retries=0,
                                               timeout=max(1, min(300, remaining_seconds(spec))),
                                               rubric=str(op.params.get("rubric", ""))[:8000])
                except Exception as exc:
                    write_json(annotation_dir / "failure.json", {"path": row["path"], "error": str(exc)})
                    raise
                score, conf = answer.get("quality_score_0_100"), answer.get("confidence_0_1")
                valid = (answer.get("ok") and answer.get("schema_valid") is True
                         and score is not None and conf is not None
                         and math.isfinite(float(score)) and math.isfinite(float(conf))
                         and float(conf) >= confidence)
                with (annotation_dir / "responses.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"response": answer, "accepted_as_weak_label": bool(valid)},
                                            ensure_ascii=False) + "\n")
                details.append({k: answer.get(k) for k in
                                ("path", "ok", "error", "model_id", "prompt_version", "usage")})
                if valid:
                    row.update(score=float(score), defects=answer["defects"],
                               label_provenance={"label_kind": "weak", "model_id": answer.get("model_id"),
                                                 "prompt_version": answer.get("prompt_version"),
                                                 "confidence": float(conf), "review_required": True})
            write_json(annotation_dir / "summary.json",
                       {"processed": len(details), "operation_id": op.operation_id,
                        "review_required": True})
            if not any(r.get("label_provenance", {}).get("label_kind") == "weak" for r in rows[:limit]):
                raise ValueError("annotation produced no usable weak labels")
        elif op.kind == "augment":
            from PIL import Image, ImageOps
            if op.params.get("transform", "hflip") != "hflip" or op.params.get("label_preserving") is not True:
                raise ValueError("supported image augmentation: hflip with explicit label_preserving=true")
            count = int(op.params.get("count", before))
            if not 0 < count <= before:
                raise ValueError("image augmentation count must be in [1, split size]")
            _claim(repo, spec, "added_samples", count)
            image_dir = release / "images" / op.operation_id
            image_dir.mkdir(parents=True, exist_ok=True)
            for i in rng.choice(before, count, replace=False):
                check_running(repo, spec.task_id)
                row = copy.deepcopy(rows[i])
                path = image_dir / f"{i}.png"
                with Image.open(row["path"]) as image:
                    ImageOps.mirror(image).save(path)
                row.update(path=str(path), image=f"{op.operation_id}_{i}.png",
                           parent_path=rows[i]["path"])
                rows.append(row)
        _validate_rows(groups[op.split])
        provenance.append({**op.model_dump(), "before": before, "after": len(groups[op.split]),
                           "source_hash": source_hash, "annotation_results": details})
    for rows in groups.values():
        _validate_rows(rows)
    _disjoint(list(groups.values()))
    train, dev_test = groups["train"], groups["development_test"]
    _write_rows(release / "dev.jsonl", train)
    write_json(release / "dev_split.json", _split_dev(train, seed=plan.data_recipe.seed))
    _write_rows(release / "eval_inputs.jsonl",
                [{"path": r["path"], "image": r.get("image")} for r in dev_test])
    proto = tp.evaluation / "protocols" / release.name
    _write_rows(proto / "eval_labels.jsonl", dev_test)
    _write_rows(proto / "final_labels.jsonl", groups["final_test"])
    return {k: len(v) for k, v in groups.items()}, provenance


def prepare_development(tp, spec, plan, version: str, repo) -> C.DatasetManifest:
    from ..policy import plan_problems
    problems = plan_problems(spec, plan)
    if problems:
        raise ValueError("; ".join(problems))
    release = tp.release_dir(version)
    if (release / "manifest.json").exists():
        manifest = verify_release(tp, version)
        if manifest.recipe_id != sha256_obj(plan.data_recipe.model_dump()):
            raise ValueError("immutable dataset recipe changed")
        return manifest
    base = _freeze_base(tp, spec, plan, repo)
    release.mkdir(parents=True, exist_ok=True)
    if spec.domain == "iqa":
        counts, provenance = _iqa(tp, spec, plan, base, release, repo)
    elif spec.domain == "demo_tabular":
        counts, provenance = _tabular(tp, spec, plan, base, release, repo)
    else:
        raise ValueError("domain needs an autonomous development/holdout data adapter")
    write_json(release / "recipe.json", plan.data_recipe.model_dump())
    proto = tp.evaluation / "protocols" / version
    checksums = {**_hashes(release), **{f"protocol/{k}": v for k, v in _hashes(proto).items()}}
    manifest = C.DatasetManifest(
        dataset_id=f"{task_key(spec.task_id)}_{spec.domain}", task_id=spec.task_id, version=version,
        source=spec.domain, splits=counts, recipe_id=sha256_obj(plan.data_recipe.model_dump()),
        seed=plan.data_recipe.seed, checksums=checksums, provenance=provenance,
        stage="development", synthetic=spec.synthetic)
    if spec.domain == "iqa":
        manifest.external_checksums = _image_hashes(release)
        manifest.external_checksums.update({_identity(r): sha256_file(Path(r["path"]))
                                            for r in _rows(proto / "final_labels.jsonl")})
    write_json(release / "manifest.json", manifest.model_dump())
    repo.save_dataset(manifest)
    return manifest


def prepare_final(tp, spec, version: str) -> str:
    """Build label-free final inputs only after the selected model is frozen."""
    verify_release(tp, version)
    final_version = f"{version}_final"
    release = tp.release_dir(final_version)
    if (release / "manifest.json").exists():
        verify_release(tp, final_version)
        return final_version
    release.mkdir(parents=True, exist_ok=True)
    proto = tp.evaluation / "protocols" / final_version
    proto.mkdir(parents=True, exist_ok=True)
    origin = tp.evaluation / "protocols" / version
    if spec.domain == "iqa":
        labels = _rows(origin / "final_labels.jsonl")
        _write_rows(release / "eval_inputs.jsonl",
                    [{"path": r["path"], "image": r.get("image")} for r in labels])
        shutil.copy2(origin / "final_labels.jsonl", proto / "eval_labels.jsonl")
    else:
        labels = _array(origin / "final_labels.npz")
        np.savez(release / "test.npz", X=labels["X"], y=np.zeros_like(labels["y"]))
        shutil.copy2(origin / "final_labels.npz", proto / "final_labels.npz")
        shutil.copy2(tp.release_dir(version) / "split.json", release / "split.json")
    checksums = {**_hashes(release), **{f"protocol/{k}": v for k, v in _hashes(proto).items()}}
    manifest = C.DatasetManifest(dataset_id=f"{task_key(spec.task_id)}_final",
                                 task_id=spec.task_id, version=final_version,
                                 stage="final", checksums=checksums)
    if spec.domain == "iqa":
        manifest.external_checksums = _image_hashes(release)
    write_json(release / "manifest.json", manifest.model_dump())
    return final_version


def development_summary(tp) -> dict[str, Any]:
    """Only aggregate development statistics may enter cloud research context."""
    manifests = sorted(tp.data.glob("releases/v*/manifest.json"))
    output = []
    for path in manifests[-6:]:
        obj = read_json(path)
        if obj.get("stage") != "development":
            continue
        output.append({"version": obj["version"],
                       "splits": {k: v for k, v in obj["splits"].items() if k != "final_test"},
                       "operations": [{k: p.get(k) for k in
                                       ("operation_id", "kind", "split", "hypothesis", "before", "after")}
                                      for p in obj.get("provenance", []) if p.get("split") != "final_test"]})
    return {"releases": output}
