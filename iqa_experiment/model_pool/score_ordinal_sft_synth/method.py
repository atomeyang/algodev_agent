# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'score_ordinal_sft_synth' (plugin score_ordinal_sft.py)."""
from algorithm_plugins.iqa.methods import score_ordinal_sft as _impl

METHOD = 'score_ordinal_sft_synth'


def run(ctx, config):
    return _impl.run(ctx, config)
