# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'measure_threshold' (plugin measure_threshold.py)."""
from algorithm_plugins.iqa.methods import measure_threshold as _impl

METHOD = 'measure_threshold'


def run(ctx, config):
    return _impl.run(ctx, config)
