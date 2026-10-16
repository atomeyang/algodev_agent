> RESEARCH MODE: research reasoning produced by the cloud model `deepseek-flash`; new implementations and data experiments are permitted; acceptance targets remain fixed.

## LLM research analysis

### What iteration 1 actually measured
Nothing. Two execution failures block the record. (1) `run_b6e0bd3aa9a9_v1_mobileclip2_v7_score` died with `FileNotFoundError` — the MobileCLIP2-S0 backbone and `iqa_adapter_v7` checkpoint are absent locally, and `search_assets` returned an empty asset list. (2) `multitask_adapter_rank` never produced `method.py` (ast-parse failed, run timed out). Fact: the deep-backbone route is blocked by missing local assets in this environment; per the failure-condition list from iteration 1 this is an infrastructure outage, not evidence that backbone features are unhelpful. Fix budget is exhausted (2/2), so iteration 2 must be a minimal, signature-agnostic diff.

### Frozen-split facts (from release v1 and plugin code)
- train = 445 rows = 325 + 120 hflip-augmented rows; development_test = 85 rows.
- `white_balance` has 0 positives in train and dev_test → 5-class defect macro-F1 ceiling is 4/5 = 0.80 < 0.85. `flare` has 3 dev_test positives < MIN_CLASS_SUPPORT=5 → unmeasurable. Annotation budget is spent (4/4), so this cannot be repaired this iteration.
- Score ground truth is a severity-quantised pseudo-MOS: good 88.0–95.9 from a filename hash, defective 66/46/28 by severity (`features.py`, `score_moe.py` docstrings).

### Interpretation (explicitly separate from measurement)
The binding constraint is the good-vs-defective gate. Plugin docs report a single global regressor topping out near PLCC 0.66, an oracle gate reaching ≈0.85, and a realistic gate at ROC-AUC ≈0.85. Those numbers are documentation claims, not measured runs in this task; still they imply PLCC 0.85 sits at the ceiling for this pseudo-MOS target, because the top level carries pixel-independent hash noise. Expect: PLCC/SRCC possibly reachable only if the gate is near-oracle; `defect_macro_f1` and `hardcase_macro_f1` gates structurally unreachable; `grade_agreement` is measurable and the most likely gate to pass; ROM is trivially satisfiable on this zero-ROM path.

### Iteration-2 decision
Pivot to the reference zero-ROM, CPU-only `score_moe` family (no external checkpoint). Information value: we have never measured this reference on our own frozen split, and the requested augmentation intervention is already materialised in release v1 but has never been trained on. Candidate c2 is a small monkeypatch shim that (a) changes only `DEFAULTS` keys verified at import time, (b) applies a dev-out-of-fold monotone calibration to the blended score, and (c) exposes a dose flag so augmentation can be ablated at 0/120/240 flipped rows if rows are tagged in the release.

### Controls and falsification
A0 = c1 baseline. A1 = c2 code, dose 0 (isolates code/calibration). A2 = dose 120, A3 = dose 240 (dose-response; only if tagging exists, else declared unmeasurable). A4 = identity calibration vs isotonic (isolates calibration). A5 = plugin-default defect thresholds vs dev-OOF tuned thresholds. Every variant shares the frozen eval split, seed 7, the evaluator's metric code and a ROM check.

### Honest limits
Not a new independent acceptance test — the historical benchmark is exposed. Pseudo-MOS is not human MOS. On-device latency and packaged ROM are unmeasured; ROM here is a parameter/table estimate only. Calibration+augmentation cannot lift a structurally unmeasurable class.