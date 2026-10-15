# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'multitask_adapter' (plugin multitask_adapter.py)."""
from algorithm_plugins.iqa.methods import multitask_adapter as _impl

METHOD = 'multitask_adapter'


def run(ctx, config):
    return _impl.run(ctx, config)
