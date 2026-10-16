> RESEARCH MODE: research reasoning produced by the cloud model `deepseek-flash`; new implementations and data experiments are permitted; acceptance targets remain fixed.

## LLM research analysis

### Measured facts (this evidence packet)
- Dev release preview (seed 7, pseudo benchmark, subject-group split): train 325 rows / development_test 85; defect support train = blur 72, over_exp 36, under_exp 36, flare 7, white_balance 0; dev_test = blur 18, over_exp 9, under_exp 9, flare 3, white_balance 0; score range 28.0-95.9. final_labels_inspected = false.
- Frozen-protocol reproduction baseline `mobileclip2_v7_score` (KONIQ-trained MobileCLIP2-S0 adapter score head + zero-ROM calibrated measurement defect head) is documented as macro-F1 0.424, PLCC 0.476 on the frozen eval.
- `multitask_adapter.py` DEFAULTS use 60 epochs, above the task max_epochs=40; `config.py` sets MIN_CLASS_SUPPORT=5, floors 0.85/0.90, and resolves pretrained/adapter checkpoints locally (raises FileNotFoundError when absent).
- All eight supplied papers are abstract-only leads (evidence_level=abstract_only_not_reproduced), retrieved 2026-10-15 (publication dates 2018-2026); two were returned empty (recall_lessons: none; search_assets model: none).

### Interpretation (gate feasibility)
With white_balance support 0 in both splits, per-class F1 for that class is undefined (convention scores 0), so 5-class defect_macro_f1 is capped at 4/5 = 0.80 even with four perfect classes: the 0.85 defect gate is structurally unattainable without new labels, and flare (3 dev_test positives < MIN_CLASS_SUPPORT) is statistically unmeasurable. Only PLCC/SRCC/grade_agreement and supported-class defect F1 (blur/over_exp/under_exp) are measurable this run. The exposed historical test set means replication, not independent acceptance.

### Experiments selected (iteration 1)
C1 = reproduce the baseline protocol unchanged (reference point). C2 = new coded method `multitask_adapter_rank`: frozen MobileCLIP2-S0 encoder + shared trunk + score head + 5-logit defect head, trained with Huber + pairwise rank + soft-PLCC loss (leads: ev_65aa194cfa multi-task MOS with PLCC loss; ev_6fb3c850a1 learning-to-rank) and class-balanced focal BCE with prior logit adjustment, val-only isotonic calibration and val-only thresholds, fp16 export (~23.4 MB << 60 MB). Data pilots: 120 hflip-augmented train rows and a 4-sample weak-annotation probe for the missing white_balance class.

### Controls / ablations
A0 baseline; A1 C2 code without augmentation; A2 C2 code with hflip augment (isolates data); A3 score head only (rank/PLCC weights = 0, defect head disabled); A4 fixed 0.5 thresholds and identity calibration vs val-tuned. Same frozen encoder, seed 7 and subject-disjoint val for all; eval labels never touched; headline metrics only from the independent evaluator.

### Falsification and known limits
A val-PLCC gain < 0.02 over A1 falsifies the loss change; a flare/white_balance recall drop under hflip falsifies label-preserving augmentation for those classes; a macro-F1 5-class claim above 0.80 without new labels would indicate a protocol/metric error. Pseudo-MOS is not human MOS (noise level unknown), and ROM here is an fp16 parameter estimate, not a packaged/latency measurement.

### Unknowns
Grade-agreement and hardcase_macro_f1 definitions/slices are not supplied (hardcase unmeasurable as specified); whether grading uses the pseudo benchmark (325/85) or the KonIQ release (610 eval rows) is ambiguous; defect-label provenance is unstated; no registered trusted external source exists for `extend`.