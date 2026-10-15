# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'score_moe_meta' (plugin score_moe.py)."""
from algorithm_plugins.iqa.methods import score_moe as _impl

METHOD = 'score_moe_meta'


def run(ctx, config):
    return _impl.run(ctx, config)
