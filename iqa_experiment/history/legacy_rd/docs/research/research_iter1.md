# IQA research & design (iteration 1)

> OFFLINE MODE: no live literature search or cloud model was used. Candidates and thresholds come from the local IQA workspace + the requirement report; all results are labelled with their real provenance (frozen pseudo-MOS benchmark).

## Business objective
轻量双输出 IQA 模型研发：ROM < 60MB fp16，输出 0-100 质量分与 blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签；缺陷 Macro-F1 >= 0.85，PLCC/SRCC >= 0.90，分档一致率 >= 0.90，hardcase Macro-F1 >= 0.85，参考 iqa_agent 的冻结基准与 MobileCLIP2-S0 方案。

## Hard constraints
- ROM < 60 MB fp16 (deployment drops the text tower)
- dual output: 0-100 quality score + 5 defect multi-labels (blur/over_exp/under_exp/flare/white_balance)

## Acceptance floors
- defect Macro-F1 >= 0.85  (primary)
- PLCC >= 0.9, SRCC >= 0.9, hardcase Macro-F1 >= 0.85, grade agreement >= 0.9

## Frozen evaluation reality check
- eval positives: blur 150, over_exp 75, under_exp 75, flare 9, **white_balance 0**
- the all-class Macro-F1 is therefore capped at 0.8 even with perfect F1 on the other four classes; the report also tracks `defect_macro_f1_cov` over classes with support.

## Candidates
- **measure_threshold** (rule): Calibrated physical features (Laplacian/over-under ratio/blob/cast) with F1-maximising thresholds; zero-ROM distilled defect head. (expected gain: reproduces the dev-calibrated reference defect F1; cost: 65 s CPU)
- **mobileclip2_v7_score** (transfer): Trained MobileCLIP2-S0 adapter (KONIQ) for the score + measurement defect head. Reproduction baseline for the frozen protocol. (expected gain: real continuous score (PLCC), 22.9 MB fp16; cost: ~2 min GPU)
- **multitask_adapter** (finetune): Frozen MobileCLIP2-S0 + trainable score head + 5-way multi-label defect head trained on the dev split (GPU). (expected gain: targets the defect Macro-F1 gate directly; cost: ~3-5 min GPU)

## Weak classes flagged this iteration
- blur, over_exp, under_exp, flare