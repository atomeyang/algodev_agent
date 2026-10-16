"""Independent synthetic train/save/reload probe for generated tabular methods."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import sys


def probe(contract_path: str) -> None:
    import numpy as np
    from algorithm_plugins.common.plugin import MethodContext, load_method
    from algorithm_plugins.demo_tabular.methods import build_dataset
    from ..paths import safe_join, sha256_file
    contract = json.loads(Path(contract_path).read_text())
    objective = json.loads(contract["objective"])
    entry = Path(contract["entry"]).resolve()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temp:
        root = Path(temp)
        train_data, test_data, trained, inferred = [root / name for name in ("train", "inference", "fit", "pred")]
        for path in (test_data, trained, inferred):
            path.mkdir(parents=True)
        build_dataset(train_data, n=80, seed=193)
        with np.load(train_data / "test.npz") as data:
            features = data["X"]
        np.savez(test_data / "test.npz", X=features)
        config = dict(objective["candidate"].get("params", {}))
        for key in ("epochs", "max_epochs", "mlp_max_iter"):
            config[key] = min(int(config.get(key, 2)), 2)
        ctx = MethodContext(root, train_data, trained, entry.parent, 193, "test")
        result = load_method(entry).run(ctx, config)
        if not isinstance(result, dict) or not result.get("weights_ref"):
            raise ValueError("method must return a saved weights_ref")
        weights = safe_join(trained, result["weights_ref"])
        digest = sha256_file(weights)
        prediction = np.load(trained / "pred_test.npy")
        ctx = MethodContext(root, test_data, inferred, entry.parent, 193, "test")
        load_method(entry).predict(ctx, config, weights)
        reloaded = np.load(inferred / "pred_test.npy")
        if reloaded.shape != (len(features),) or not np.isfinite(reloaded).all():
            raise ValueError("reload predictions have invalid shape or nonfinite values")
        if not np.allclose(prediction, reloaded, atol=1e-6, rtol=1e-5):
            raise ValueError("saved/reloaded predictions differ")
        if sha256_file(weights) != digest:
            raise ValueError("inference mutated trained weights")


if __name__ == "__main__":
    probe(sys.argv[1])
