# -*- coding: utf-8 -*-
"""Auto-materialised entry for IQA method 'mobileclip2_v7_score' (plugin mobileclip2_v7_score.py)."""
from algorithm_plugins.iqa.methods import mobileclip2_v7_score as _impl

METHOD = 'mobileclip2_v7_score'


def run(ctx, config):
    return _impl.run(ctx, config)

def predict(ctx, config, weights_path):
    return _impl.predict(ctx, config, weights_path)
