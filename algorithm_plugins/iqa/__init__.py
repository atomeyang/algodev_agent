# -*- coding: utf-8 -*-
"""IQA business plugin: light dual-output image-quality assessment.

Exposed so ``algorithm_plugins.registry`` can resolve the ``iqa`` domain and
other modules can ``from algorithm_plugins.iqa import config, metrics, data``.
"""

from . import config, data, metrics  # noqa: F401
from .plugin import PLUGIN, IqaPlugin  # noqa: F401

__all__ = ["PLUGIN", "IqaPlugin", "config", "data", "metrics"]
