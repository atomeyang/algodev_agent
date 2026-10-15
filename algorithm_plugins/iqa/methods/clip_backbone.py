# -*- coding: utf-8 -*-
"""MobileCLIP2-S0 backbone helpers shared by the GPU IQA methods.

* ``reference_model_class`` loads ``iqa_adapter/model.py`` straight from the
  parent workspace (bypassing its ``__init__`` so no heavy data deps are pulled)
  and gives us the *exact* architecture the trained checkpoints expect.
* ``extract_features`` runs the frozen encoder once over a list of images and
  caches the 512-d embeddings, so training multi-task heads is seconds not
  minutes and re-runs of the loop are cheap.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODEL_PY = _REPO_ROOT.parent / "iqa_agent" / "model_src" / "iqa_adapter" / "model.py"


def reference_model_class():
    """Load ``MobileCLIP2IQA`` from the iqa_agent model_src by file path."""
    from .. import config
    candidates = [
        config.iqa_agent_dir() / "model_src" / "iqa_adapter" / "model.py",
        _MODEL_PY,
    ]
    for path in candidates:
        if path.exists():
            name = "iqa_ref_model"
            if name in sys.modules:
                return sys.modules[name].MobileCLIP2IQA
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module.MobileCLIP2IQA
    raise FileNotFoundError(f"iqa_adapter/model.py not found near {_REPO_ROOT.parent}")


def load_trained_adapter(ckpt_path: str | Path, pretrained_path: str | Path, device):
    """Build MobileCLIP2IQA and load a trained checkpoint (encoder + LoRA + head)."""
    import torch
    MobileCLIP2IQA = reference_model_class()
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    has_lora = any(("lora_a" in k or "lora_b" in k) for k in raw)
    model = MobileCLIP2IQA(pretrained=str(pretrained_path), image_size=256, use_lora=has_lora)
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    model.freeze_encoder()
    return model, {"has_lora": bool(has_lora), "n_missing": len(missing), "n_unexpected": len(unexpected)}


def build_encoder(pretrained_path: str | Path, image_size: int = 256):
    """Frozen MobileCLIP2-S0 encoder + its validation transform."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="MobileCLIP2-S0", pretrained=str(pretrained_path),
        force_image_size=image_size,
        image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0),
        image_interpolation="bicubic", image_resize_mode="shortest")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, preprocess


def _cache_key(paths: List[str], tag: str) -> str:
    h = hashlib.sha1()
    h.update(tag.encode())
    for p in paths:
        h.update(str(p).encode())
    return h.hexdigest()[:16]


def extract_features(encoder, preprocess, paths: List[str], device, *,
                     cache_dir: Optional[Path] = None, tag: str = "", batch_size: int = 32,
                     num_workers: int = 4) -> np.ndarray:
    """Frozen-encoder features for *paths*, cached under *cache_dir* when given."""
    import torch
    from PIL import Image

    feats: Optional[np.ndarray] = None
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"feat_{_cache_key(paths, tag)}.npz"
        if cache_file.exists():
            try:
                feats = np.load(cache_file)["feat"]
                if feats.shape[0] == len(paths):
                    return feats.astype(np.float32)
            except Exception:
                feats = None

    def _batch(imgs):
        tensors = [preprocess(im) for im in imgs]
        return torch.stack(tensors).to(device)

    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            imgs = []
            for p in chunk:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    im = Image.new("RGB", (256, 256), (0, 0, 0))
                imgs.append(im)
            x = _batch(imgs)
            f = encoder.encode_image(x)
            out.append(f.float().cpu().numpy())
    feats = np.concatenate(out, axis=0).astype(np.float32)
    if cache_file is not None:
        try:
            np.savez_compressed(cache_file, feat=feats)
        except Exception:
            pass
    return feats


__all__ = ["reference_model_class", "load_trained_adapter", "build_encoder", "extract_features"]
