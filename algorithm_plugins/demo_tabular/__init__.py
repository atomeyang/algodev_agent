"""Deterministic synthetic tabular-regression demo plugin (synthetic_demo marker)."""

from .methods import METHODS, build_dataset, run, slice_masks  # noqa: F401

__all__ = ["METHODS", "build_dataset", "run", "slice_masks"]
