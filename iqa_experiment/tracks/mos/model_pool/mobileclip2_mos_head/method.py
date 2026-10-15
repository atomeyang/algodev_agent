# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'mobileclip2_mos_head' (plugin score_mos_transfer.py)."""
from algorithm_plugins.iqa.methods import score_mos_transfer as _impl

METHOD = 'mobileclip2_mos_head'


def run(ctx, config):
    return _impl.run(ctx, config)
