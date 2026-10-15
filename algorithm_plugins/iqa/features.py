# -*- coding: utf-8 -*-
"""Degradation-aware IQA features (plugin-owned, dependency-light).

The frozen benchmark's score GT is a *severity-quantised* pseudo-MOS
(see ``iqa_agent/benchmark.py``): every defective image scores by its labelled
severity (light 66 / medium 46 / heavy 28) and good images score 88-95.9.
Predicting that target therefore reduces to estimating, from pixels alone:

* a **good vs defective** decision, and
* the **degradation severity** (a 3-level ordinal scale).

The generic 16-d ``measure_lib`` features undershoot because they measure
*content* as much as *degradation*.  This module adds the physical estimators
that isolate the degradation itself:

* multi-scale Laplacian decay + a Gaussian **blur-sigma** estimate fitted in the
  frequency domain (``log P(f) = c - a log f - b f^2``, ``sigma = sqrt(b/2pi^2)``);
* histogram **clipping / gain** statistics with per-channel breakdown;
* robust **white-balance** gains (grey-world and max-RGB illuminant);
* **noise sigma** from a high-pass residual (MAD/0.6745);
* **haze** via the dark channel and RMS contrast;
* **glare/flare** blob area and local-contrast halo statistics;
* edge-density at several gradient thresholds (blur/content separation).

Values are deterministic for a fixed decoder; the extractor caches to disk and
parallelises across the 610/410 frozen-split images.
"""

from __future__ import annotations

import numpy as np, cv2


# bump when the numeric feature set changes so on-disk caches are invalidated
FEATURE_VERSION = "v2-occ"


def _load(path, max_side=768):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img


def gauss_sigma_from_spectrum(gray):
    """Estimate Gaussian blur sigma (px) by fitting log power spectrum vs f^2."""
    g = gray.astype(np.float32) / 255.0
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * win))
    P = (np.abs(F) ** 2)
    h, w = P.shape
    cy, cx = h//2, w//2
    y, x = np.mgrid[0:h, 0:w]
    fy = (y - cy) / h; fx = (x - cx) / w
    r = np.sqrt(fy**2 + fx**2)
    # radial bins
    nb = 40
    rb = np.linspace(0, 0.5, nb+1)
    prof = np.zeros(nb); cnt = np.zeros(nb)
    idx = np.digitize(r.ravel(), rb) - 1
    pv = P.ravel()
    for b in range(nb):
        m = idx == b
        if m.sum() > 5:
            prof[b] = pv[m].mean(); cnt[b] = m.sum()
    good = cnt > 5
    f = 0.5*(rb[:-1]+rb[1:])[good]
    p = prof[good]
    if len(f) < 6:
        return 0.0, 0.0
    yv = np.log(p + 1e-12)
    # model: log P = c - a*log f - b*f^2 ; sigma_ms = sqrt(b/(2*pi^2))
    f2 = f**2; lf = np.log(f + 1e-6)
    A = np.stack([np.ones_like(f), lf, f2], 1)
    try:
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    except Exception:
        return 0.0, 0.0
    b = max(coef[2], 0.0)
    sigma = float(np.sqrt(b / (2*np.pi**2)))
    # high-freq energy fraction (blur kills it)
    hf = float(np.exp(yv[-1]) / (np.exp(yv).mean() + 1e-12))
    return float(sigma), hf


def lap_at_scale(gray, sigma):
    if sigma <= 0:
        g = gray
    else:
        k = int(2 * round(3 * sigma) + 1)
        g = cv2.GaussianBlur(gray, (k, k), sigma)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def noise_sigma(gray):
    """Robust high-frequency noise estimate via median abs of Laplacian."""
    r = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(np.median(np.abs(r)) / 0.6745 * np.sqrt(np.pi/2))


def dark_channel(img, ksize=15):
    mn = img.min(axis=2).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(mn, kernel)


def occlusion_features(gray):
    """Local/regional statistics that expose a pasted or covered region.

    The benchmark's ``occlusion`` images are the source scene with a region
    darkened / replaced, so the discriminative cues are *regional*: an abnormal
    dark-pixel mass concentrated in a compact connected component, corner- or
    tile-wise darkness imbalance, and sharp paste boundaries.  Global blur or
    exposure statistics are blind to these.
    """
    g = gray.astype(np.float32)
    H, W = g.shape
    tot = float(H * W)
    out = {}
    for t in (20, 35, 55, 80):
        out[f"dk_frac_{t}"] = float((g < t).mean())
    dk = (g < 55).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(dk, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        out["dk_cc_max"] = float(areas.max() / tot)
        out["dk_cc_top3"] = float(np.sort(areas)[-3:].sum() / tot)
        out["dk_cc_n"] = float((areas > 0.002 * tot).sum())
    else:
        out["dk_cc_max"] = out["dk_cc_top3"] = out["dk_cc_n"] = 0.0
    ch, cw = max(H // 4, 1), max(W // 4, 1)
    def m(y0, y1, x0, x1):
        return float(g[y0:y1, x0:x1].mean())
    out["dark_tl"] = m(0, ch, 0, cw)
    out["dark_tr"] = m(0, ch, W - cw, W)
    out["dark_bl"] = m(H - ch, H, 0, cw)
    out["dark_br"] = m(H - ch, H, W - cw, W)
    out["dark_ctr"] = m(H // 3, max(2 * H // 3, H // 3 + 1), W // 3, max(2 * W // 3, W // 3 + 1))
    out["dark_ring"] = float(np.mean([out["dark_tl"], out["dark_tr"], out["dark_bl"], out["dark_br"]]))
    out["dark_corner_ctr"] = out["dark_ring"] - out["dark_ctr"]
    tiles = [float(g[i * H // 4:(i + 1) * H // 4 or 1, j * W // 4:(j + 1) * W // 4 or 1].mean())
             for i in range(4) for j in range(4)]
    out["dark_tile_std"] = float(np.std(tiles))
    out["dark_tile_min"] = float(np.min(tiles))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    out["edge_strong"] = float((mag > 200).mean())
    return out


def jpeg_blockiness(gray):
    """Block-boundary energy ratio: re-encoded (degraded) images show 8px grid steps."""
    g = gray.astype(np.float32)
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 9:
        return 0.0
    on = d[:, 7::8].mean()
    off = float(d.mean()) + 1e-6
    return float(on / off)


def rich(path):
    img = _load(path)
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf = gray.astype(np.float32)
    out = {}
    # --- blur / sharpness profile ---
    ls = [lap_at_scale(gray, s) for s in (0.0, 1.0, 2.0, 4.0, 8.0)]
    out["lap_s0"], out["lap_s1"], out["lap_s2"], out["lap_s4"], out["lap_s8"] = ls
    base = ls[0] + 1e-6
    out["lap_ratio_1"] = ls[1]/base; out["lap_ratio_2"] = ls[2]/base
    out["lap_ratio_4"] = ls[3]/base; out["lap_ratio_8"] = ls[4]/base
    out["log_lap"] = float(np.log1p(ls[0]))
    sig, hf = gauss_sigma_from_spectrum(gray)
    out["blur_sigma"] = sig; out["hf_energy"] = hf
    # normalized sharpness (content-independent)
    out["lap_over_std"] = ls[0] / (float(gf.std()) ** 2 + 1e-6)
    out["lap_over_range"] = ls[0] / (float(gf.max()-gf.min())**2 + 1e-6)
    # --- exposure ---
    out.update({"mean": float(gf.mean()), "std": float(gf.std()),
                "med": float(np.median(gf)), "p01": float(np.percentile(gf,1)),
                "p05": float(np.percentile(gf,5)), "p95": float(np.percentile(gf,95)),
                "p99": float(np.percentile(gf,99)),
                "clip_hi": float((gf>=254).mean()), "clip_lo": float((gf<=1).mean()),
                "clip_hi250": float((gf>=250).mean()), "clip_lo5": float((gf<=5).mean())})
    # per-channel clipping (wb/over)
    for i, ch in enumerate("bgr"):
        a = img[:, :, i].astype(np.float32)
        out[f"clip_hi_{ch}"] = float((a>=254).mean())
        out[f"clip_lo_{ch}"] = float((a<=1).mean())
    # --- white balance ---
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    mean3 = (mb+mg+mr)/3 + 1e-6
    out["wbratio_r"] = mr/mean3 - 1; out["wbratio_g"] = mg/mean3 - 1; out["wbratio_b"] = mb/mean3 - 1
    out["wb_span"] = (max(mr,mg,mb)-min(mr,mg,mb))/mean3
    # max-RGB illuminant
    p95 = [float(np.percentile(img[:,:,i],95)) for i in range(3)]
    m95 = max(p95) + 1e-6
    out["maxrgb_r"] = p95[2]/m95; out["maxrgb_g"] = p95[1]/m95; out["maxrgb_b"] = p95[0]/m95
    out["maxrgb_dev"] = float(np.std(np.array(p95)/m95))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    out["lab_a"] = float(lab[:,:,1].mean()-128); out["lab_b"] = float(lab[:,:,2].mean()-128)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out["sat_mean"] = float(hsv[:,:,1].mean()); out["val_mean"] = float(hsv[:,:,2].mean())
    out["sat_clip"] = float((hsv[:,:,1]>=250).mean())
    # --- noise (measured on a blur-tolerant residual) ---
    out["noise_sigma"] = noise_sigma(gray)
    out["noise_over_std"] = out["noise_sigma"]/(float(gf.std())+1e-6)
    # local noise: std of (img - median) in 3x3
    med = cv2.medianBlur(img, 3).astype(np.float32)
    out["hf_detail"] = float(np.abs(img.astype(np.float32)-med).mean())
    # --- haze / contrast ---
    dc = dark_channel(img)
    out["dark_channel"] = float(dc.mean()); out["dark_channel_p99"] = float(np.percentile(dc,99))
    out["dc_min"] = float(dc.min())
    out["rms_contrast"] = float(gf.std())/(float(gf.mean())+1e-6)
    hist = cv2.calcHist([gray],[0],None,[64],[0,256]).ravel(); hist/= hist.sum()+1e-9
    out["entropy"] = float(-(hist*np.log(hist+1e-9)).sum())
    # --- glare / flare ---
    v = hsv[:,:,2]; s = hsv[:,:,1]
    bright = ((v>=245)&(s<=40)).astype(np.uint8)
    out["bright_ratio"] = float(bright.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    big = 0.0
    if n > 1:
        areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
        big = float(areas[0]/bright.size)
    out["blob_ratio"] = big
    # halo: local contrast reduction around bright blobs (flare haze)
    mb2 = cv2.blur(gf, (31,31))
    out["local_contrast"] = float(np.abs(gf-mb2).mean())
    out["local_contrast_norm"] = out["local_contrast"]/(float(gf.std())+1e-6)
    # --- center region ---
    h, w = gray.shape
    c = gray[int(.25*h):int(.75*h), int(.25*w):int(.75*w)]
    out["center_lap"] = float(cv2.Laplacian(c, cv2.CV_64F).var())
    out["center_std"] = float(c.astype(np.float32).std())
    out["center_mean"] = float(c.mean())
    out["center_bright"] = float((c>=250).mean())
    # --- edge density at multiple thresholds (blur/content separability) ---
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2+gy**2)
    for t in (20, 50, 100):
        out[f"edge_den_{t}"] = float((mag > t).mean())
    out["mag_mean"] = float(mag.mean())
    # --- occlusion / paste-region regional statistics ---
    out.update(occlusion_features(gray))
    out["jpeg_blockiness"] = jpeg_blockiness(gray)
    out["path"] = path
    return out


def feature_names():
    """Ordered numeric feature names (excludes the ``path`` key)."""
    _init_order()
    return [k for k in _FEATURE_ORDER]


def extract_many(paths, workers: int = 8, cache_path=None):
    """Feature matrix (n_paths, n_features) for *paths*, cached when requested."""
    import json as _json
    from pathlib import Path as _P
    if cache_path is not None and _P(cache_path).exists():
        try:
            blob = _json.loads(_P(cache_path).read_text(encoding="utf-8"))
            rows = blob.get("rows") if isinstance(blob, dict) else blob
            ver = blob.get("version") if isinstance(blob, dict) else FEATURE_VERSION
            if (ver == FEATURE_VERSION and isinstance(rows, dict)
                    and all(p in rows for p in paths)):
                return matrix_from_rows([rows[p] for p in paths]), list(feature_names())
        except Exception:
            pass
    if len(paths) > 32 and workers > 1:
        from multiprocessing import Pool
        with Pool(int(workers)) as pool:
            rows = pool.map(_safe_rich, list(paths), chunksize=4)
    else:
        rows = [_safe_rich(p) for p in paths]
    if cache_path is not None:
        try:
            by_path = {r["path"]: r for r in rows}
            _P(cache_path).write_text(
                _json.dumps({"version": FEATURE_VERSION, "rows": by_path}), encoding="utf-8")
        except Exception:
            pass
    return matrix_from_rows(rows), list(feature_names())


def _safe_rich(path):
    try:
        return rich(path)
    except Exception:
        return {"path": path, "error": "failed"}


def matrix_from_rows(rows):
    import numpy as _np
    names = list(feature_names())
    out = _np.zeros((len(rows), len(names)), dtype=_np.float32)
    for i, row in enumerate(rows):
        for j, k in enumerate(names):
            try:
                out[i, j] = float(row.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                out[i, j] = 0.0
    return out


# The numeric feature order, captured from a synthetic 32x32 probe so it stays
# in sync with ``rich`` without shipping a frozen list.
def _probe_names():
    import numpy as _np
    import tempfile as _tf
    import os as _os
    import cv2 as _cv2
    d = _tf.mkdtemp()
    p = _os.path.join(d, "probe.png")
    _cv2.imwrite(p, (_np.random.RandomState(0).rand(48, 48, 3) * 255).astype("uint8"))
    names = [k for k in rich(p) if k != "path" and not isinstance(rich(p)[k], str)]
    return names


_FEATURE_ORDER = []


def _init_order():
    global _FEATURE_ORDER
    if not _FEATURE_ORDER:
        _FEATURE_ORDER = _probe_names()


if __name__ == "__main__":
    import sys, json
    from multiprocessing import Pool
    paths = json.load(open(sys.argv[1]))
    with Pool(int(sys.argv[3])) as pool:
        res = pool.map(rich, paths, chunksize=4)
    json.dump(res, open(sys.argv[2], "w"))
    print("done", len(res))
