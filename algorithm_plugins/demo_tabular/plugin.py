"""Business plugin for the synthetic tabular demo domain."""

from __future__ import annotations

from typing import Any, Dict, List


class DemoPlugin:
    domain = "demo_tabular"

    def validate_requirement(self, spec: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        if not spec.get("primary_metric", {}).get("name"):
            problems.append("primary metric name is required")
        return problems

    def validate_dataset(self, manifest: Dict[str, Any]) -> List[str]:
        problems: List[str] = []
        if manifest.get("synthetic") and not manifest.get("seed", 0) >= 0:
            problems.append("synthetic dataset must declare a seed")
        return problems

    def default_slices(self) -> List[str]:
        return ["low_light", "high_light"]

    def deployment_scope(self) -> Dict[str, Any]:
        return {"device": "cpu", "runtime": "numpy/sklearn", "note": "offline demo only"}
