# -*- coding: utf-8 -*-
"""Dataset export for the IQA plugin.

Turns the *frozen* IQA benchmark produced by the ``iqa_agent`` research engine
into an immutable task release, with a deliberate train/eval isolation:

release (``data/releases/<v>/``), handed to the method subprocess
    ``dev.jsonl``          -- 410 labelled rows (train/val; pseudo-MOS + defects)
    ``dev_split.json``     -- deterministic subject-group-disjoint train/val index
    ``eval_inputs.jsonl``  -- 610 rows: ``{path, image}`` only, NO labels
    ``benchmark_meta.json``-- counts, per-defect support, source hashes

evaluation protocol dir (``evaluation/protocols/<v>/``), NOT given to methods
    ``eval_labels.jsonl``  -- 610 labelled rows, read only by the frozen evaluator

Defect labels / pseudo-MOS come from the iqa_agent benchmark; the eval split is
subject-group-disjoint from dev, so no image/subject leaks into training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as C


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def defect_support(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    support = {d: 0 for d in C.DEFECTS}
    for r in rows:
        for d in C.DEFECTS:
            if int(bool((r.get("defects") or {}).get(d, {}).get("present", 0))):
                support[d] += 1
    return support


def _strip_labels(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"path": row.get("path"), "image": row.get("image")}


def _split_dev(rows: List[Dict[str, Any]], seed: int = 7, val_frac: float = 0.25) -> Dict[str, Any]:
    """Subject-group-disjoint train/val index (deterministic)."""
    subjects = sorted({str(r.get("subject") or r.get("image") or i) for i, r in enumerate(rows)})
    import random
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac)))
    val_subjects = set(subjects[:n_val])
    train_idx, val_idx = [], []
    for i, r in enumerate(rows):
        subj = str(r.get("subject") or r.get("image") or i)
        (val_idx if subj in val_subjects else train_idx).append(i)
    return {"seed": seed, "val_frac": val_frac, "train": train_idx, "val": val_idx,
            "n_train": len(train_idx), "n_val": len(val_idx), "group": "subject"}


def _negative_defects() -> Dict[str, Dict[str, Any]]:
    out = {d: {"present": 0, "severity": 0} for d in C.DEFECTS}
    out["white_balance"]["direction"] = "neutral"
    return out


def _koniq_rows() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read the real-MOS KonIQ-10k CSV into (dev_rows, eval_rows).

    ``dev`` = official *training* + *validation* images (labels visible to the
    method); ``eval`` = the official *test* images (labels frozen).  KonIQ has
    no defect taxonomy, so every row carries an all-absent 5-defect map; the
    real human MOS is the score.
    """
    import csv
    csv_path, img_dir = C.koniq_sets_csv(), C.koniq_image_dir()
    if not csv_path.exists():
        raise FileNotFoundError(f"KonIQ metadata not found at {csv_path}")
    dev: List[Dict[str, Any]] = []
    ev: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            name = (r.get("image_name") or "").strip()
            if not name:
                continue
            ip = Path(img_dir) / name
            if not ip.exists():
                continue
            row = {
                "path": str(ip), "image": name, "subject": name,
                "score": float(r["MOS"]), "defects": _negative_defects(),
                "benchmark": "koniq10k", "set": r.get("set", ""),
            }
            (dev if r.get("set") in ("training", "validation") else ev).append(row)
    return dev, ev


def export_benchmark(tp, version: str, *, seed: int = 7) -> Dict[str, Any]:
    """Materialise the release + frozen evaluation protocol for *version*."""
    if C.benchmark_source() == "koniq":
        return _export_koniq(tp, version, seed=seed)
    release = tp.release_dir(version)
    release.mkdir(parents=True, exist_ok=True)

    dev_src, eval_src = C.dev_jsonl(), C.eval_jsonl()
    if not dev_src.exists() or not eval_src.exists():
        raise FileNotFoundError(
            f"IQA benchmark not found under {C.benchmark_dir()}; set QAA_ROOT to the workspace")

    dev_rows = load_rows(dev_src)
    eval_rows = load_rows(eval_src)
    splits = _split_dev(dev_rows, seed=seed)

    (release / "dev.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in dev_rows) + "\n", encoding="utf-8")
    (release / "eval_inputs.jsonl").write_text(
        "\n".join(json.dumps(_strip_labels(r), ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8")
    (release / "dev_split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # frozen labels live outside the method's data_dir
    proto_dir = tp.evaluation / "protocols" / version
    proto_dir.mkdir(parents=True, exist_ok=True)
    (proto_dir / "eval_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8")
    (proto_dir / "protocol.json").write_text(json.dumps({
        "protocol_id": f"iqa_frozen_{version}", "dataset_version": version,
        "primary_metric": "defect_macro_f1", "direction": "max",
        "defects": C.DEFECTS, "min_class_support": C.MIN_CLASS_SUPPORT,
        "floors": {"plcc": C.SCORE_PLCC_FLOOR, "srcc": C.SCORE_SRCC_FLOOR,
                   "defect_macro_f1": C.DEFECT_MACROF1_FLOOR,
                   "hardcase_macro_f1": C.HARDCASE_F1_FLOOR,
                   "grade_agreement": C.GRADE_AGREEMENT_FLOOR},
        "rom_budget_mb": C.ROM_BUDGET_MB, "frozen": True,
    }, indent=2), encoding="utf-8")

    dev_support = defect_support(dev_rows)
    eval_support = defect_support(eval_rows)
    meta = {
        "dataset_id": "iqa_dual_output", "version": version,
        "source": "qaa_iqa_agent_frozen_benchmark",
        "benchmark_dir": str(C.benchmark_dir()),
        "splits": {"dev": len(dev_rows), "eval": len(eval_rows)},
        "dev_defect_support": dev_support, "eval_defect_support": eval_support,
        "dev_subjects": len({r.get("subject") for r in dev_rows}),
        "eval_subjects": len({r.get("subject") for r in eval_rows}),
        "label_semantics": "pseudo-MOS (88/66/46/28 by severity) + 5-defect multi-label; "
                           "subject-group-disjoint dev/eval",
        "synthetic": False,
        "copied_files": ["dev.jsonl", "eval_inputs.jsonl", "dev_split.json"],
    }
    (release / "benchmark_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")

    checksum_files = [
        f"data/releases/{version}/dev.jsonl",
        f"data/releases/{version}/eval_inputs.jsonl",
        f"evaluation/protocols/{version}/eval_labels.jsonl",
    ]
    return {
        "dataset_id": "iqa_dual_output",
        "source": "qaa_iqa_agent_frozen_benchmark",
        "splits": {"dev": len(dev_rows), "eval": len(eval_rows)},
        "annotations": meta["label_semantics"],
        "recipe_id": f"iqa_recipe_{version}",
        "code_ref": "algorithm_plugins/iqa/methods/multitask_adapter.py",
        "license": "internal",
        "upload_policy": "restricted",
        "synthetic": False,
        "checksum_files": checksum_files,
        "raw_manifest": meta,
        "note": f"frozen benchmark exported to {release}",
    }


def _export_koniq(tp, version: str, *, seed: int = 7) -> Dict[str, Any]:
    """Export the real-MOS KonIQ-10k benchmark (quality-only track).

    Protocol mirror of the pseudo path: dev keeps labels, the eval release holds
    ``{path, image}`` only, and the frozen MOS labels live under
    ``evaluation/protocols/<version>/`` where the method subprocess cannot see
    them.  No defect annotations exist on this benchmark, so defect support is
    all-zero and the evaluator runs in ``quality_only`` mode.
    """
    release = tp.release_dir(version)
    release.mkdir(parents=True, exist_ok=True)

    dev_rows, eval_rows = _koniq_rows()
    if not dev_rows or not eval_rows:
        raise FileNotFoundError(
            f"KonIQ images missing under {C.koniq_image_dir()} (csv {C.koniq_sets_csv()})")

    train_idx = [i for i, r in enumerate(dev_rows) if r.get("set") == "training"]
    val_idx = [i for i, r in enumerate(dev_rows) if r.get("set") == "validation"]
    splits = {"seed": seed, "val_frac": (len(val_idx) / len(dev_rows)) if dev_rows else 0.0,
              "train": train_idx, "val": val_idx, "n_train": len(train_idx), "n_val": len(val_idx),
              "group": "official_koniq_split"}

    (release / "dev.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in dev_rows) + "\n", encoding="utf-8")
    (release / "eval_inputs.jsonl").write_text(
        "\n".join(json.dumps(_strip_labels(r), ensure_ascii=False) for r in eval_rows) + "\n",
        encoding="utf-8")
    (release / "dev_split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    proto_dir = tp.evaluation / "protocols" / version
    proto_dir.mkdir(parents=True, exist_ok=True)
    (proto_dir / "eval_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8")
    (proto_dir / "protocol.json").write_text(json.dumps({
        "protocol_id": f"iqa_koniq_{version}", "dataset_version": version,
        "primary_metric": "plcc", "direction": "max", "quality_only": True,
        "defects": C.DEFECTS, "min_class_support": C.MIN_CLASS_SUPPORT,
        "floors": {"plcc": C.QUALITY_PLCC_FLOOR, "srcc": C.QUALITY_SRCC_FLOOR},
        "rom_budget_mb": C.ROM_BUDGET_MB, "frozen": True,
        "label_semantics": "real human MOS (KonIQ-10k), official train/val/test split",
        "scores_source": "koniq10k_mos",
    }, indent=2), encoding="utf-8")

    meta = {
        "dataset_id": "iqa_ms_koniq10k", "version": version,
        "source": "koniq10k_real_mos",
        "koniq_dir": str(C.koniq_dir()), "koniq_csv": str(C.koniq_sets_csv()),
        "image_dir": str(C.koniq_image_dir()),
        "splits": {"dev": len(dev_rows), "eval": len(eval_rows),
                   "dev_train": len(train_idx), "dev_val": len(val_idx)},
        "dev_defect_support": {d: 0 for d in C.DEFECTS},
        "eval_defect_support": {d: 0 for d in C.DEFECTS},
        "dev_subjects": len({r.get("subject") for r in dev_rows}),
        "eval_subjects": len({r.get("subject") for r in eval_rows}),
        "score_range_eval": [min(r["score"] for r in eval_rows), max(r["score"] for r in eval_rows)],
        "label_semantics": "real human MOS (KonIQ-10k); no defect annotations (quality-only)",
        "synthetic": False, "quality_only": True,
        "copied_files": ["dev.jsonl", "eval_inputs.jsonl", "dev_split.json"],
    }
    (release / "benchmark_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")

    return {
        "dataset_id": "iqa_ms_koniq10k",
        "source": "koniq10k_real_mos",
        "splits": {"dev": len(dev_rows), "eval": len(eval_rows)},
        "annotations": meta["label_semantics"],
        "recipe_id": f"iqa_ms_recipe_{version}",
        "code_ref": "algorithm_plugins/iqa/methods/score_mos_transfer.py",
        "license": "research", "upload_policy": "restricted", "synthetic": False,
        "quality_only": True,
        "checksum_files": [f"data/releases/{version}/dev.jsonl",
                           f"data/releases/{version}/eval_inputs.jsonl",
                           f"evaluation/protocols/{version}/eval_labels.jsonl"],
        "raw_manifest": meta,
        "note": f"real-MOS KonIQ-10k benchmark exported to {release}",
    }


def eval_labels_path(tp, version: str) -> Optional[Path]:
    p = tp.evaluation / "protocols" / version / "eval_labels.jsonl"
    if p.exists():
        return p
    p2 = tp.release_dir(version) / "eval_labels.jsonl"
    return p2 if p2.exists() else None


def release_inputs_path(tp, version: str) -> Optional[Path]:
    p = tp.release_dir(version) / "eval_inputs.jsonl"
    if p.exists():
        return p
    p2 = tp.release_dir(version) / "eval.jsonl"
    return p2 if p2.exists() else None


__all__ = ["load_rows", "defect_support", "export_benchmark", "eval_labels_path",
           "release_inputs_path"]
