# -*- coding: utf-8 -*-
"""Deterministic image measurements (self-contained port of iqa_agent measure).

Kept dependency-light (numpy + OpenCV) so a method subprocess needs no import of
the parent workspace package. Values are identical to ``iqa_agent/tools/measure``.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _load(path: str, max_side: int = 768):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _region(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def sharpness(img) -> Dict[str, float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    mask = np.zeros_like(mag, bool)
    mask[cy - r:cy + r, cx - r:cx + r] = True
    hf_ratio = float(mag[~mask].sum() / (mag.sum() + 1e-9))
    return {"laplacian_var": lap_var, "tenengrad": tenengrad, "hf_ratio": hf_ratio}


def exposure(img) -> Dict[str, float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    over = float((g >= 250).mean())
    under = float((g <= 5).mean())
    return {"mean_lum": float(g.mean()), "std_lum": float(g.std()),
            "over_ratio": over, "under_ratio": under,
            "p99": float(np.percentile(g, 99)), "p1": float(np.percentile(g, 1))}


def white_balance(img) -> Dict[str, Any]:
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    means = np.array([r.mean(), g.mean(), b.mean()])
    gray_world_dev = float(np.abs(means - means.mean()).mean() / (means.mean() + 1e-6))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_m, b_m = float(lab[:, :, 1].mean() - 128), float(lab[:, :, 2].mean() - 128)
    if abs(a_m) < 4 and abs(b_m) < 4:
        direction = "neutral"
    elif abs(a_m) >= abs(b_m):
        direction = "green" if a_m < 0 else "magenta"
    else:
        direction = "cool" if b_m < 0 else "warm"
    return {"gray_world_dev": gray_world_dev, "lab_a_mean": a_m, "lab_b_mean": b_m,
            "direction": direction, "cast_mag": float(np.hypot(a_m, b_m))}


def glare(img) -> Dict[str, float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    bright = ((v >= 245) & (s <= 40)).astype(np.uint8)
    area = float(bright.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    big = 0.0
    if n > 1:
        areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
        big = float(areas[0] / bright.size)
    return {"bright_ratio": area, "largest_blob_ratio": big}


def measure(path: str) -> Dict[str, Any]:
    img = _load(path)
    center = _region(img, (0.25, 0.25, 0.75, 0.75))
    return {
        "path": path,
        "global": {**sharpness(img), **exposure(img), **white_balance(img), **glare(img)},
        "center": {"sharpness": sharpness(center), "exposure": exposure(center)},
    }


__all__ = ["measure", "sharpness", "exposure", "white_balance", "glare"]


# --------------------------------------------------------------------------- #
# threshold-rule defect head with an on-disk feature cache
# --------------------------------------------------------------------------- #
def defects_from_thresholds(m: Dict[str, float], thresholds: Dict[str, Dict[str, Any]],
                           defects) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in defects:
        spec = thresholds[d]
        feat, direction, t = spec["feature"], spec["direction"], float(spec["threshold"])
        v = float(m[feat])
        present = int(v < t) if direction == "lt" else int(v > t)
        out[d] = {"present": present, "severity": present}
    out["white_balance"]["direction"] = m.get("direction", "neutral")
    return out


def mock_score(m: Dict[str, float]) -> float:
    """Fixed linear blend of physical quality proxies (the distilled score head)."""
    blur = min(1.0, m["laplacian_var"] / 300.0)
    exp = 1.0 - min(1.0, m["over_ratio"] * 3 + m["under_ratio"] * 3)
    glow = 1.0 - min(1.0, m["largest_blob_ratio"] * 4)
    wb = 1.0 - min(1.0, m["cast_mag"] / 25.0)
    return round(100 * (0.35 * blur + 0.3 * exp + 0.2 * glow + 0.15 * wb), 2)


def measure_defects_cached(paths, thresholds: Dict[str, Dict[str, Any]], defects,
                           cache_dir=None) -> Dict[str, Dict[str, Any]]:
    """{path: {defect: {...}}} using the threshold rules, cached to disk."""
    import json as _json
    from pathlib import Path as _Path

    cache: Dict[str, Any] = {}
    cache_file = None
    if cache_dir is not None:
        cache_file = _Path(cache_dir) / "measure_cache.json"
        if cache_file.exists():
            try:
                cache = _json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
    thr_sig = ";".join(f"{d}:{thresholds[d]['feature']}:{thresholds[d]['direction']}:{thresholds[d]['threshold']}"
                       for d in defects)
    out: Dict[str, Any] = {}
    dirty = False
    for p in paths:
        key = f"{thr_sig}|{p}"
        if key in cache:
            out[p] = cache[key]
            continue
        m = measure(p)["global"]
        out[p] = defects_from_thresholds(m, thresholds, defects)
        cache[key] = out[p]
        dirty = True
    if cache_file is not None and dirty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(_json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out
