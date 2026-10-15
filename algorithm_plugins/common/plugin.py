"""Business-plugin and method-adapter interfaces.

A *business plugin* validates requirements and data and declares the allowed
processing, metrics, failure slices and deployment measurement scope. A
*method adapter* exposes prepare/train/predict/export/profile/capabilities.
Both are plain Python objects so tasks can reference a single canonical
implementation by version instead of copying code (guide 4.1).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol


@dataclass
class MethodContext:
    """Filesystem context handed to a method implementation."""

    task_root: Path
    data_dir: Path
    out_dir: Path
    method_dir: Path
    seed: int = 0
    split: str = "val"
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_root": str(self.task_root),
            "data_dir": str(self.data_dir),
            "out_dir": str(self.out_dir),
            "method_dir": str(self.method_dir),
            "seed": self.seed,
            "split": self.split,
        }


@dataclass
class MethodSpec:
    name: str
    version: str
    family: str
    entry: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


class BusinessPlugin(Protocol):
    domain: str

    def validate_requirement(self, spec: Dict[str, Any]) -> List[str]: ...

    def validate_dataset(self, manifest: Dict[str, Any]) -> List[str]: ...

    def default_slices(self) -> List[str]: ...

    def deployment_scope(self) -> Dict[str, Any]: ...


class MethodAdapter(Protocol):
    def prepare(self, ctx: MethodContext, config: Dict[str, Any]) -> Dict[str, Any]: ...

    def train(self, ctx: MethodContext, config: Dict[str, Any]) -> Dict[str, Any]: ...

    def predict(self, ctx: MethodContext, config: Dict[str, Any]) -> Dict[str, Any]: ...

    def export(self, ctx: MethodContext, config: Dict[str, Any]) -> Dict[str, Any]: ...

    def profile(self, ctx: MethodContext, config: Dict[str, Any]) -> Dict[str, Any]: ...

    def capabilities(self) -> Dict[str, Any]: ...


def load_method(entry: str) -> Any:
    """Load a method entry module from an explicit file path.

    We never mutate ``sys.path``; the module is bound by absolute file path so
    behaviour is independent of the current working directory.
    """
    path = Path(entry)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"method entry not found: {path}")
    spec = importlib.util.spec_from_file_location(f"algo_method_{path.stem}_{abs(hash(str(path))) % 10**8}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load method entry {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_method_entries(model_pool: Path) -> List[MethodSpec]:
    specs: List[MethodSpec] = []
    if not model_pool.exists():
        return specs
    for d in sorted(model_pool.iterdir()):
        entry = d / "method.py"
        if d.is_dir() and entry.exists():
            specs.append(MethodSpec(name=d.name, version="v1", family=d.name, entry=str(entry)))
    return specs
