# IQA research & design (iteration 2)

> OFFLINE MODE: no live literature search or cloud model was used. Candidates and thresholds come from the local IQA workspace + the requirement report; all results are labelled with their real provenance (frozen pseudo-MOS benchmark).

## Business objective
轻量双输出 IQA 模型研发（ROM < 60MB fp16）：输出 0-100 质量分与 blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签；主指标 PLCC >= 0.85，SRCC >= 0.85，缺陷 Macro-F1 >= 0.85，分档一致率 >= 0.90，hardcase Macro-F1 >= 0.85；参考 iqa_agent 冻结基准与 MobileCLIP2-S0 方案，逐轮迭代直至达标。

## Hard constraints
- ROM < 60 MB fp16 (deployment drops the text tower)
- dual output: 0-100 quality score + 5 defect multi-labels (blur/over_exp/under_exp/flare/white_balance)

## Acceptance floors
- **PLCC >= 0.85  (primary gate this run)**, with defect Macro-F1 etc. as secondary gates
- secondary floors: SRCC >= 0.9, defect Macro-F1 >= 0.85, hardcase Macro-F1 >= 0.85, grade agreement >= 0.9

## Frozen evaluation reality check
- eval positives: blur 150, over_exp 75, under_exp 75, flare 9, **white_balance 0**
- the all-class Macro-F1 is therefore capped at 0.8 even with perfect F1 on the other four classes; the report also tracks `defect_macro_f1_cov` over classes with support.

## Candidates
- **score_moe_meta** (score-head+MoE+meta): Same three-view head with a finer blend grid and one more ensemble seed; escalates the MoE variant. (expected gain: small extra CV-selected gain; cost: ~4-6 min CPU)

## Weak classes flagged this iteration
- blur, under_exp, flare

## 1. 当前产出精度状态 (iteration 2)
- source: agent run
- best model: `m_score_moe`

| gate | current | floor | direction | verdict |
| --- | --- | --- | --- | --- |
| score PLCC (`plcc`) | 0.6525 | 0.90 | max | **fail** |
| defect Macro-F1 (`defect_macro_f1`) | 0.4240 | 0.85 | max | **fail** |
| defect Macro-F1 (covered classes only) (`defect_macro_f1_cov`) | 0.5300 | 0.85 | max | **fail** |
| score SRCC (`srcc`) | 0.6626 | 0.90 | max | **fail** |
| hardcase Macro-F1 (`hardcase_macro_f1`) | 0.5188 | 0.85 | max | **fail** |
| grade agreement (`grade_agreement`) | 0.6016 | 0.90 | max | **fail** |
| ROM (fp16) (`rom_mb`) | 22.9200 | 60.00 | min (budget) | **pass** |

- verdict: **6 gate(s) failing** -> plcc, defect_macro_f1, defect_macro_f1_cov, srcc, hardcase_macro_f1, grade_agreement
- frozen-eval positives: blur 150, over_exp 75, under_exp 75, flare 9, white_balance 0
- NOTE: white_balance has 0 positives on the frozen eval, so the all-class Macro-F1 is structurally capped at 0.80 even with perfect F1 on the other four classes; `defect_macro_f1_cov` is the view that can actually reach 0.85.
- NOTE: flare has only 9 eval positives -> wide confidence interval, no statistically reliable read on the flare head.
- NOTE: score ground truth is severity-quantised pseudo-MOS, not a real MOS panel: PLCC/SRCC read low and any PLCC discrepancy across runs must be re-checked before it is believed.
- NOTE: primary metric is PLCC; the score targets are a severity-quantised pseudo-MOS (good images get a filename-hash score in 88.0-95.9 that is not derivable from pixels), so a high PLCC requires near-perfect 4-level severity classification and has a structural ceiling below the defect-oriented framing suggests.

## 2. 模型 & 损失函数 & 训练策略优化计划
### opt-score-rank-loss - score PLCC / SRCC (model + loss)
- change: Train the score head with a pairwise ranking term on top of MSE (loss = MSE(score, mos) + lambda * margin-rank loss over (i, j) pairs with |mos_i - mos_j| > delta), because PLCC/SRCC are rank statistics while MSE only optimises calibration.
- parameters: `rank_loss`=pairwise Logistic/MarginRankingLoss; `lambda`=0.3 -> 1.0; `delta`=0.1 (ignore near-ties in the pseudo-MOS); `head`=score MLP 256->1
- why now: plcc/srcc are the gates with the largest gap.
- expected effect: rank correlation up without retraining the encoder; calibration reported separately
- risk: pairwise loss can collapse the score scale -> watch MAE/grade agreement
### opt-defect-asymmetric-loss - defect Macro-F1 (weak classes: blur, under_exp, flare) (loss)
- change: Replace the plain BCE with class-balanced asymmetric focal BCE: pos_weight_d = clip(N_neg / N_pos, 1.0, max_pos_weight) computed per class on the train split, focal gamma 1.5, and a per-class decision threshold tuned on the subject-disjoint val split by F1 maximisation.
- parameters: `pos_weight_scale`=1.5-2.5; `max_pos_weight`=40.0; `focal_gamma`=1.5; `threshold_tuning`=val-split F1 max, one threshold per class; `label_smoothing`=0.02
- why now: rare classes (flare / under_exp) sit at low recall.
- expected effect: rare-class recall up; guard precision with the per-class threshold
- risk: over-predicting rare classes inflates FP on the frequent classes
### opt-train-strategy - stability + generalisation on the 410-row dev split (training strategy)
- change: Longer, better-regularised head training: warmup 5% + cosine decay, EMA of head weights, early stop on val macro-F1 with patience 20, gradient accumulation to an effective batch >= 64, fp16 autocast, and encoder unfreezing scheduled so the last N transformer blocks train only after the head has stabilised.
- parameters: `epochs`=120-200 (budget max_epochs=200); `warmup`=5%; `cosine`=True; `ema_decay`=0.999; `early_stop_patience`=20; `effective_batch`=64; `amp`=fp16; `unfreeze_last_blocks`=2
- why now: the current 40-120 epoch heads are the only trainable part, so schedule quality is cheap to improve
- expected effect: lower variance across iterations; PLCC/F1 less seed-sensitive
- risk: longer training overfits a 410-row dev split -> keep the subject-disjoint val gate
### opt-head-ensemble-flare - flare (0.000 on the rule head, 0.706 on the multitask head) (model)
- change: Keep the trained 5-way defect head as the primary head but distil the calibrated rule/measure features into it (feature-level distillation on flare), so the single deployed head does not regress to the rule head's 0 FLARE F1.
- parameters: `method`=feature distillation; `rule_features`=blob/ratio/cast statistics from measure_lib; `head`=concat(adapter_feat, rule_feat) -> 5 logits
- why now: the rule head cannot separate flare at all while the adapter head can
- expected effect: flare F1 kept while ROM stays inside the budget
- risk: feature concatenation adds parameters -> re-check the 60 MB fp16 ROM gate
### opt-encoder-resolution - ROM vs accuracy head-room (model trade-off)
- change: Evaluate MobileCLIP2-S0 at 256 vs 336 input, and S0 vs the next-smaller/larger encoder variant, and pick the largest encoder that still fits <60 MB fp16 with the heads (11.46 M params at S0 -> ~22.9 MB fp16, so there is head-room).
- parameters: `encoders`=['MobileCLIP2-S0', 'MobileCLIP2-S2 (if it fits)']; `resolutions`=[256, 336]; `rom_gate_mb`=60.0
- why now: score accuracy is the binding gate now, and ROM has head-room
- expected effect: higher score correlation without breaking the ROM constraint
- risk: bigger encoder raises per-sample latency (not measured in this run)
### opt-eval-hygiene - trustworthy numbers (evaluation)
- change: Re-verify the frozen metric path against iqa_agent.tools.iqa_model.evaluate for the exact checkpoint/preprocessing pair before acting on any PLCC delta; report PLCC/SRCC with bootstrap CIs and fix the seed set (>= 3 seeds, report mean +- std).
- parameters: `seeds`=3; `bootstrap`=1000; `target_ci`=0.95
- why now: the run reported PLCC 0.4338 where the reference baseline reports 0.4757 for the same model/split/metric code; that gap is unresolved and must be explained first
- expected effect: no decisions taken on noise
- risk: more seeds cost GPU time

## 3. 数据标注调优计划 & 数据扩增计划
### 3.1 类别正样本覆盖 (frozen eval)
| defect | current positives | target | gap | measurable |
| --- | --- | --- | --- | --- |
| 模糊 `blur` | 150 | 150 | 0 | yes |
| 过曝 `over_exp` | 75 | 75 | 0 | yes |
| 欠曝 `under_exp` | 75 | 150 | 75 | yes |
| 炫光 `flare` | 9 | 30 | 21 | yes |
| 白平衡 `white_balance` | 0 | 30 | 30 | NO |

### 3.2 数据标注调优计划 (本地辅助标注模型 + 人工复核)
- goal: Turn the local VLM from an unvalidated helper into a review-assisted labeller that raises the positive count for the classes the frozen eval cannot measure (white_balance 0, flare 9), without ever treating a model output as ground truth.
- local model: `qwen3.8-27b` at `local_largemodel/qwen3.8-27b` (llama.cpp llama-server, OpenAI-compatible, endpoint `http://127.0.0.1:8084/v1`, modalities text, image)
- client: `algorithm_plugins/iqa/annotation.py` + `scripts/iqa_annotate.py`, prompt_version `iqa-prelabel-v1`
- invariant: review_required is always true - pre-labels never enter a dataset version unreviewed
- measured pilot agreement (real runs of `scripts/iqa_annotate.py`):
  - `iqa_annotations/prelabels_eval_iqa-prelabel-v1.agreement.json` on 20 paired images; severity_MAE=0.500; wb_direction_acc=n/a; blur P=0.250/R=0.200/F1=0.222; over_exp P=0.750/R=1.000/F1=0.857; under_exp P=1.000/R=0.500/F1=0.667; flare P=0.000/R=0.000/F1=0.000; white_balance P=0.000/R=0.000/F1=0.000
- reading: strong on over-exposure (F1 0.857), weak on blur (0.222) and it over-fires flare / white_balance, so v1 is a candidate generator for the missing positives and NOT a stand-alone labeller; every pre-label needs human review.
- tuning: prompt v2 for blur: ask the model for an edge-energy / high-frequency check and a centre-vs-corner crop comparison before deciding blur; v1 blur P/R was 0.25/0.20 on the 20-image pilot.
- tuning: add an explicit negative guard for flare and white_balance: require a described light source / cast direction; v1 flagged 4 flare FP and 6 white_balance FP on the same 20 images, so its positive rate is unusable without the guard.
- tuning: require the wb direction (cool/warm/green/magenta) and reject any white_balance=present without a direction; wb_direction_acc was not measurable in v1 (0 GT positives).
- tuning: run the pilot per class in batches of 20-30, keep the prompt_version in provenance, and only promote a prompt version when its agreement on a 30-image human-reviewed set beats the previous one.
- review protocol: rare_classes=flare, white_balance -> 100% human review of every pre-label; other_classes=>= 30% stratified human review (stratified by model confidence); accept_rule=a pre-label is accepted only after a human accept/edit; disagreement is logged and drives the next prompt version; audit=store model_id + prompt_version + reviewer + accept/edit/reject in the row provenance; kpi=human review time per accepted label, and the pre-label accept rate per class

### 3.3 数据扩增计划
- goal: Fill the measurable-support gaps for the classes the frozen eval cannot score, then re-freeze.
- rule: augment only the train split; the frozen eval is never augmented in the same protocol version
- rule: every augmented sample keeps provenance (source image id + transform + parameters)
- rule: bump dataset_version and re-run the reference baseline after any augmentation release
- **white_balance** (target 30): channel-gain (von Kries-style) transforms of clean images at measured colour temperatures, verified by estimating the cast direction back; real captures collected with grey-card reference for ground-truth direction -- cover cool/warm/green/magenta roughly balanced; a class with 0 positives today is the single biggest structural cap on the macro-F1 gate.
- **flare** (target 30): physically-plausible veiling-glare / ghost overlays whose PSF is estimated from the real flare positives; vary intensity in the same 0/1/2/3 severity scale as the taxonomy -- mix synthetic with real captures; keep a small human-reviewed synthetic holdout to catch an overlay artifact the metric would otherwise reward.
- **under_exp** (target 150): gamma/curve-based darkening with highlight preservation, severity-matched to the existing labels
- **blur** (target 150): motion-blur kernels (direction + length) and defocus disks, severity-matched
- **over_exp** (target 75): clipping curve / specular-region growth, severity-matched
- acceptance: after augmentation, re-run measure_threshold + the best adapter on the re-frozen eval
- acceptance: only promote the new dataset version if the reference baseline improves or holds on every class that was already measurable