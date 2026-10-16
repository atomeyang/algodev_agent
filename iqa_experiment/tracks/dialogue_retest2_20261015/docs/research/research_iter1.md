> RESEARCH MODE: research reasoning produced by the cloud model `deepseek-flash`; new implementations and data experiments are permitted; acceptance targets remain fixed.

## LLM research analysis

### 证据基线与已测事实
- 冻结基准预览（seed 7，development_only，非 frozen）：train 325，dev 85；缺陷支持数 train blur 72 / over_exp 36 / under_exp 36 / flare 7 / white_balance 0，dev blur 18 / over 9 / under 9 / flare 3 / white_balance 0。
- `metrics.py` 规定 GT 正例 < `MIN_CLASS_SUPPORT`(=5) 的类别 F1 按 0 计并按 5 类取宏平均 → **全类 defect_macro_f1 上界被结构性地限制在 4/5 = 0.80 < 0.85**；可诚实地报告的是 `defect_macro_f1_cov`（覆盖类）。white_balance 在 train 与 dev 均为 0 支持，本轮无法验证该类别。
- `score_moe.py` 文档记录了 dev 上的测量：单一全局回归器 PLCC ≈0.66；good/defective 门控的 oracle 版本（仅诊断）≈0.85，真实门控 ROC-AUC ≈0.85。且 good 图分的 GT 由文件名 hash 生成（与像素无关）。**含义（解释，非事实）**：PLCC/SRCC ≥0.85 即便在 oracle 门控下也仅勉强达到，属高风险不可达；本轮要重新实测而不是继承该结论。
- ROM 一致性：MobileCLIP2-S0 visual 11.41M 参数 → fp16 ≈22.8MB < 60MB，额外头可忽略；ROM 仅为参数估算，非打包实测。
- 配置不一致：`config.py` 内 PLCC/SRCC floor 为 0.90，task_spec 为 0.85。不得按更松或更严的自定义 floor 调参；以 evaluator 上报为准。

### 候选与选择理由
1) `mobileclip2_v7_score`（reuse，基线复现）：零风险可复现参照点，用于确认协议在本地复现，并给出后续 Δ 的分母。历史反馈指出无默认最新模型，故不继承任何历史最好值。
2) `multitask_adapter`（modify，真实编码+保存权重）：伪 MOS 是 4 级量化（good/light/medium/heavy），故把 score 头改为 4 级 ordinal 分类取期望 `E[score]=ΣP(k)·level_k`，与 L1 回归项联训，并加 pairwise ranking 损失（仅用同 split 内分数可分的图片对，避免用 hash 噪声层构造对）。缺陷头保留 5 独立 logit + 类别加权 BCE，阈值仅在 dev/val 上调（subject-group 不相交）。

### 数据干预（train 专用，不动 dev/final 真值）
- hflip 标签保持增强 240 张；flare 正例重采样加权；对 train 做 4 张小规模本地弱标注以检验 white_balance 是否有任何出现。
- 全部操作限定 `split=train`，禁止跨 subject group 复制，禁止把弱标注当真值或用于 final_test。

### 控制/消融与信息价值
基线复现、无增强、仅 L1（去 ordinal/ranking）、固定 0.5 阈值、仅 dev 调阈，五个控制可把「模型改动」「数据增强」「阈值策略」三种效应分离。信息价值：即使 PLCC 不可达，也只能通过该消融才知道是特征/损失还是标签结构（hash 噪声、0 支持类）造成的上限。

### 不可测与不可宣称
- defect_macro_f1（5 类）不可测 → 不可宣称达标；hardcase 亦受覆盖类限制。
- 历史测试集已暴露，本轮是流程复现，不是新的独立业务验收。
- 无 on-device 时延/ROM 打包测量。

# IQA research & design (iteration 1)

> OFFLINE MODE: no live literature search or cloud model was used. Candidates and thresholds come from the local IQA workspace + the requirement report; all results are labelled with their real provenance (frozen pseudo-MOS benchmark).

## Business objective
轻量双输出 IQA 自主研发验证。使用当前本地 IQA pseudo-MOS 基准，输出 0-100 质量分和 blur/over_exp/under_exp/flare/white_balance 五类缺陷。主指标 PLCC >= 0.85，SRCC >= 0.85，defect_macro_f1 >= 0.85，grade_agreement >= 0.90，hardcase_macro_f1 >= 0.85，ROM < 60 MB。自主研究模型结构、训练策略、loss 和数据方案，产生可执行的代码与数据实验，依据开发集结果再研究、改进并迭代，不等待审批。不能通过降低指标、替换测试集、删除困难样本或修改测试真值取得达标。原始样本及历史产物只读，训练集增强和少量本地弱标注可以实验。历史测试集已暴露，不能宣称本次复测是新的独立业务验收。即使某些门槛受标签覆盖限制，仍在预算内测量代码和数据干预对可测指标的影响，最终明确哪些达标、哪些未达标、哪些不可测。历史模型结果不代表理论上限。

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
- **mobileclip2_v7_score** (transfer): Trained MobileCLIP2-S0 adapter (KONIQ) for the score + measurement defect head. Reproduction baseline for the frozen protocol. (expected gain: real continuous score (PLCC), 22.9 MB fp16; cost: ~2 min GPU)
- **score_moe** (score-head+MoE): Three-view score head: global ET/RF regression, a 4-class posterior E[score], and a coarse-degradation-family MoE (gate + router + per-family experts), blended with weights chosen by subject-group CV on dev only. (expected gain: best honest PLCC frontier; auditable decomposition; cost: ~2-4 min CPU)

## Feedback driving this iteration
- fb_77f8e2461abc: best model: None (expected: 请研究并比较模型结构、训练方案、loss 和训练数据增强的干预。首轮请包含一个可复现基线与一个需要实际编码、保存权重的新方法，并实施一个小规模训练数据增强实验。后续用实测失败项驱动下一轮研究，有价值时检验排序损失或类别平衡策略，并保留消融；不降低验收标准。)

## 1. 当前产出精度状态 (iteration 1)
- source: frozen registry baseline (start point; no agent run yet)
- best model: `measure_threshold / mobileclip2_v7_score (reference)`

| gate | current | floor | direction | verdict |
| --- | --- | --- | --- | --- |
| score PLCC (`plcc`) | 0.4757 | 0.85 | max | **fail** |
| defect Macro-F1 (`defect_macro_f1`) | 0.4240 | 0.85 | max | **fail** |
| defect Macro-F1 (covered classes only) (`defect_macro_f1_cov`) | n/a | 0.85 | max | **not measured** |
| score SRCC (`srcc`) | 0.5085 | 0.90 | max | **fail** |
| hardcase Macro-F1 (`hardcase_macro_f1`) | 0.5188 | 0.85 | max | **fail** |
| grade agreement (`grade_agreement`) | n/a | 0.90 | max | **not measured** |
| ROM (fp16) (`rom_mb`) | n/a | 60.00 | min (budget) | **not measured** |

- verdict: **4 gate(s) failing** -> plcc, defect_macro_f1, srcc, hardcase_macro_f1
- frozen-eval positives: blur 150, over_exp 75, under_exp 75, flare 9, white_balance 0
- NOTE: white_balance has 0 positives on the frozen eval, so the all-class Macro-F1 is structurally capped at 0.80 even with perfect F1 on the other four classes; `defect_macro_f1_cov` is the view that can actually reach 0.85.
- NOTE: flare has only 9 eval positives -> wide confidence interval, no statistically reliable read on the flare head.
- NOTE: iteration 1 has no agent run yet, so the table shows the frozen registry baseline as the starting point.
- NOTE: score ground truth is severity-quantised pseudo-MOS, not a real MOS panel: PLCC/SRCC read low and any PLCC discrepancy across runs must be re-checked before it is believed.
- NOTE: primary metric is PLCC; the score targets are a severity-quantised pseudo-MOS (good images get a filename-hash score in 88.0-95.9 that is not derivable from pixels), so a high PLCC requires near-perfect 4-level severity classification and has a structural ceiling below the defect-oriented framing suggests.

## 2. 模型 & 损失函数 & 训练策略优化计划
### opt-score-rank-loss - score PLCC / SRCC (model + loss)
- change: Train the score head with a pairwise ranking term on top of MSE (loss = MSE(score, mos) + lambda * margin-rank loss over (i, j) pairs with |mos_i - mos_j| > delta), because PLCC/SRCC are rank statistics while MSE only optimises calibration.
- parameters: `rank_loss`=pairwise Logistic/MarginRankingLoss; `lambda`=0.3 -> 1.0; `delta`=0.1 (ignore near-ties in the pseudo-MOS); `head`=score MLP 256->1
- why now: plcc/srcc are the gates with the largest gap.
- expected effect: rank correlation up without retraining the encoder; calibration reported separately
- risk: pairwise loss can collapse the score scale -> watch MAE/grade agreement
### opt-defect-asymmetric-loss - defect Macro-F1 (weak classes: tbd) (loss)
- change: Replace the plain BCE with class-balanced asymmetric focal BCE: pos_weight_d = clip(N_neg / N_pos, 1.0, max_pos_weight) computed per class on the train split, focal gamma 1.5, and a per-class decision threshold tuned on the subject-disjoint val split by F1 maximisation.
- parameters: `pos_weight_scale`=1.5-2.5; `max_pos_weight`=40.0; `focal_gamma`=1.5; `threshold_tuning`=val-split F1 max, one threshold per class; `label_smoothing`=0.02
- why now: generalises the class-balance fix for any class the loop flags next.
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