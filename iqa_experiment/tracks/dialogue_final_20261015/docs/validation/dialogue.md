# IQA Live Agent Dialogue



Operator utterances are scripted. Agent responses are real.



## User (2026-10-15T08:07:04.037135+00:00)

轻量双输出 IQA 自主研发验证。使用当前本地 IQA pseudo-MOS 基准，输出 0-100 质量分和 blur/over_exp/under_exp/flare/white_balance 五类缺陷。主指标 PLCC >= 0.85，SRCC >= 0.85，defect_macro_f1 >= 0.85，grade_agreement >= 0.90，hardcase_macro_f1 >= 0.85，ROM < 60 MB。自主研究模型结构、训练策略、loss 和数据方案，产生可执行的代码与数据实验，依据开发集结果再研究、改进并迭代，不等待审批。不能通过降低指标、替换测试集、删除困难样本或修改测试真值取得达标。原始样本及历史产物只读，训练集增强和少量本地弱标注可以实验。历史测试集已暴露，不能宣称本次复测是新的独立业务验收。即使某些门槛受标签覆盖限制，仍在预算内测量代码和数据干预对可测指标的影响，最终明确哪些达标、哪些未达标、哪些不可测。历史模型结果不代表理论上限。



### Agent

````json
{
  "task_id": "iqa_dialogue_final_20261015",
  "task_slug": "iqa_dialogue_final_20261015",
  "accepted": true
}
````



## User (2026-10-15T08:07:04.209055+00:00)

请确认这个任务的完整验收条件。模型、训练策略、loss、数据各方面如何展开研究？当前尚未运行实验，请不要引用历史成绩冒充本轮结果。



### Agent

# 验收条件与下一步研究展开（基于当前 live context，无实验结果）

## 1. 冻结验收条件（`frozen_acceptance_targets`，6 项同时成立）
`plcc >= 0.85; srcc >= 0.85; defect_macro_f1 >= 0.85; grade_agreement >= 0.9; hardcase_macro_f1 >= 0.85; rom_mb < 60.0`

| 指标 | 方向 | 门槛 | strict | protocol_id | measured_on |
|---|---|---|---|---|---|
| plcc（primary） | max | 0.85 | 否 | 空 | null |
| srcc | max | 0.85 | 否 | 空 | null |
| defect_macro_f1 | max | 0.85 | 否 | 空 | null |
| grade_agreement | max | 0.9 | 否 | 空 | null |
| hardcase_macro_f1 | max | 0.85 | 否 | 空 | null |
| rom_mb | min | 60.0 | **是** | 空 | null |

业务侧附加判定规则：不得通过降低指标、替换测试集、删除困难样本或修改测试真值达标；历史测试集已暴露，本轮结果不能宣称是新的独立业务验收；原始样本与历史产物只读。

## 2. 硬约束（blocking）
- `rom_mb_fp16 = 60 MB`（blocking=true）
- `dual_output = score+defects`（blocking=true）

## 3. 预算（尚未消耗：`budget_used = {reserved:{}, used:{}}`）
研究调用 3 / 研究步 12 / 编码调用 8 / 修复尝试 2 / 实验 9 / epoch 40 / 墙钟 2700 s / 云 token 400k / 无提升 3 / 迭代 3 / 候选 2 / 数据操作 8 / 新增样本 240 / 标注样本 4 / 输出文件 2048 MB。

## 4. 当前状态（不要误读历史）
`phase=draft`、`iteration=0/3`、`active_plan=None`、`best_model_id=None`、`experiment_jobs=[]`。**本轮没有任何已测指标**，上表 6 项均为待测门槛，不是成绩。

## 5. 研究展开建议（受 2 候选 / 3 迭代 / 9 实验约束）

**模型（dual-output，候选 ≤2）**
- 共享轻量 backbone + 双头：回归头（0–100 分数）+ 多标签头（blur / over_exp / under_exp / flare / white_balance）。
- 参数量需先按 fp16 反推 ROM 上限（纯算术：60 MB ÷ 2 B ≈ 3.1e7 参数级），这是**推导上界，不是测量值**；两个候选建议在宽度/深度或初始化来源上做一组对照，而不是串行试很多结构（候选上限=2）。

**损失（对齐指标而非只对齐 MSE）**
- 分数：可微 PLCC/SRCC 近似（相关性损失或排序损失）与回归损失的组合，避免只优化 MSE 却达不到 PLCC/SRCC。
- 缺陷：五类多标签，注意类别不平衡（macro-F1 对稀有类敏感）。
- 分级：`grade_agreement` 是分档一致性，宜用 ordinal / 边界一致类损失，而非连续回归的副产品。
- 难例：`hardcase_macro_f1` 需难例加权或重采样；**只能加权/复制，不能删除困难样本**。

**训练策略**
- 墙钟 2700 s、epoch ≤40 是硬顶：先短程侦察（少量 epoch）确认损失与头是否学得动，再决定是否加长；早停参考 `max_no_improve=3`。
- 选型只看开发集，测试集不参与任何调参决策。

**数据**
- 允许：训练集增强、少量本地弱标注（≤4 标注样本 / ≤240 新增 / ≤8 次数据操作）。
- 禁止：改测试真值、删难例、换测试集。
- 必须在结论中标注：pseudo-MOS 不是独立采集的人类 MOS。

## 6. 当前无法确认 / 不可测的点（context 未给出，勿假设）
- **测量协议未定义**：全部 `protocol_id` 为空、`measured_on=null`，分档与困难例的具体判定口径未知；`grade_agreement`、`hardcase_macro_f1` 是否受标签覆盖限制、覆盖到什么程度，context 未说明。
- **ROM 计量口径未定**：`declared_unknowns` 明确「on-device latency 与 packaged ROM 未测，ROM 为 fp16 参数估计」，而 `rom_mb` 是 strict=blocking。**fp16 参数估计能否作为该项判定依据，需要你明确**；若要求实测打包 ROM，则本轮可能判定为不可测。
- **max_memory_mb=0** 的含义（未设限还是禁用）在 context 中未说明。
- **业务达标/未达标/不可测的最终清单**：无法预判，必须等本轮实验产出开发集指标后才能给出。

## 7. 控制
查看当前状态：`查看状态`；如需停止新增步骤：`暂停任务`（仅阻止新步骤，不保证中断/检查点已有的训练进程），`恢复任务` 恢复，`取消任务` 取消（取消后不可恢复，需另建任务）。我不会直接改状态文件。

需要我确认协议口径与 ROM 计量方式后，再进入计划编排吗？



## User (2026-10-15T08:07:21.341814+00:00)

如何暂停任务？



### Agent

# 暂停任务

直接发送指令：`暂停任务`

## 当前状态（本次暂停会发生什么）
- `paused: False`，`phase=draft`，`iteration=0`，`experiment_jobs=[]`，`active_plan=None`。
- 即当前没有任何正在运行的实验作业，暂停后只会**阻止后续新步骤**被启动。

## 注意（context 明示的边界）
- 暂停**不保证**中断或对正在运行的训练作业做检查点；它只阻止新步骤。
- 恢复用 `恢复任务`；取消用 `取消任务`（取消后不可恢复，需另建任务）；查看状态用 `查看状态`。
- 已停止/已接受/已失败的任务**不能恢复**；如需继续要另建任务。
- 不直接编辑状态文件。

## 解除暂停
发送 `恢复任务`（当前 `paused=False`，无需恢复；若你已暂停则用它回到可执行状态）。

预算消耗提示：本任务已用 `cloud_tokens = 3958`，其余预算项尚无消耗记录。



## User (2026-10-15T08:07:24.288996+00:00)

请研究并比较模型结构、训练方案、loss 和训练数据增强的干预。首轮请包含一个可复现基线与一个需要实际编码、保存权重的新方法，并实施一个小规模训练数据增强实验。后续用实测失败项驱动下一轮研究，有价值时检验排序损失或类别平衡策略，并保留消融；不降低验收标准。



### Agent

改进目标已记录，将在下一研究轮次评估；现有验收目标和数据授权保持冻结。



## User (2026-10-15T08:07:24.434440+00:00)

暂停任务



### Agent

````json
{
  "intent": {
    "action": "pause",
    "confidence": 1,
    "reason": "",
    "goal": "",
    "source": "rules"
  },
  "command_id": "cmd_0a444e0dc2d5",
  "state": {
    "schema_version": "1.0",
    "task_id": "iqa_dialogue_final_20261015",
    "phase": "paused",
    "iteration": 0,
    "state_version": 1,
    "active_plan_id": null,
    "pending_approval": null,
    "best_model_id": null,
    "stop_reason": null,
    "paused": true,
    "budget_used": {
      "reserved": {
        "cloud_tokens": 0.0
      },
      "used": {
        "cloud_tokens": 7052
      }
    },
    "updated_at": "2026-10-15T08:07:24.440242+00:00"
  }
}
````



## User (2026-10-15T08:07:24.546016+00:00)

恢复任务



### Agent

````json
{
  "intent": {
    "action": "resume",
    "confidence": 1,
    "reason": "",
    "goal": "",
    "source": "rules"
  },
  "command_id": "cmd_60ccf7c75657",
  "state": {
    "schema_version": "1.0",
    "task_id": "iqa_dialogue_final_20261015",
    "phase": "executing",
    "iteration": 0,
    "state_version": 2,
    "active_plan_id": null,
    "pending_approval": null,
    "best_model_id": null,
    "stop_reason": null,
    "paused": false,
    "budget_used": {
      "reserved": {
        "cloud_tokens": 0.0
      },
      "used": {
        "cloud_tokens": 7052
      }
    },
    "updated_at": "2026-10-15T08:07:24.554375+00:00"
  }
}
````



## User (2026-10-15T08:09:23.073181+00:00)

请解释当前计划的模型结构、训练策略、loss、数据操作、对照实验和失败条件。请区分已运行结果与预期收益。



### Agent

# 当前计划解读（plan `p_b6e0bd3aa9a9_001` v1）

## 0. 先分清「已运行」与「预期」

| 事实 | 状态 |
|---|---|
| `experiment_jobs: []` | **本轮没有任何作业在跑/跑完** |
| `best_model_id: None` | 无选中模型 |
| `iteration: 0 / 3`，`phase: executing` | 计划已激活但未产出结果 |
| 计划条目全部标注 `candidate_plan_not_execution` / `data_plan_not_execution` | 这些是**待执行的候选写法**，不是运行记录 |
| 预算已耗 `cloud_tokens 49242`、`research_calls 1` | 只发生了一次研究调用（文献检索），**不含任何训练实验** |
| 设计报告中的 macro-F1 0.424 / PLCC 0.476 | 报告文档中**记录的历史基线数值**，非本轮实测；`final_labels_inspected=false` |

下文一切收益均为**预期/假设**，未验证。

## 1. 模型结构（2 个候选，`max_candidates=2`）

**C1 `mobileclip2_v7_score` — action=reuse，is_baseline=true**
- 不改代码，按默认配置复现参考方法：MobileCLIP2-S0 视觉塔 + 既有 adapter 分数头（KONIQ 训练权重）+ 零 ROM 的「测量规则」缺陷头（阈值规则，不占参数）。
- 分数：`clip(adapter_head(features)*100, 0, 100)`；缺陷由 `measure_lib` 阈值给出。
- 关键依赖：pretrained backbone 与 `config.adapter_ckpt()` 必须能解析到本地文件，否则抛 `FileNotFoundError` —— 按计划规定这应**记为基础设施失败，而不是负面结果**。

**C2 `multitask_adapter_rank` — action=create，is_baseline=false（需实际编码并保存权重）**
- 冻结 MobileCLIP2-S0 编码器（11.41M 参数，≈22.8 MB fp16）→ 512-d 嵌入。
- 共享主干：`Linear(512→256) + LayerNorm + GELU + Dropout(0.15)`。
- 双头（满足 `dual_output` 硬约束）：score 头 `Linear(256→1)` 回归 `z=(score-50)/25`；defect 头 `Linear(256→5)` 输出 blur/over_exp/under_exp/flare/white_balance 逻辑值。
- 只训练主干+两头，编码器保持冻结（特征缓存，仅增强样本重新抽取）。
- ROM：**fp16 参数估计 ≈23.4 MB < 60 MB**（计划内算术估计，非打包/延迟实测——`declared_unknowns` 已声明）。

## 2. 训练策略（C2）
- AdamW，lr 2e-3，weight_decay 1e-2，cosine 调度，batch 64。
- `epochs ≤ 35`（覆盖文件默认 60，因任务 `max_epochs=40`），早停 patience 8。
- 验证：从确定性 dev 划分中再切 25% **subject-disjoint** val，seed 7；选择指标 `0.5*val_PLCC + 0.5*val_supported_macroF1`（仅统计 val 正例 ≥ `MIN_CLASS_SUPPORT=5` 的类）。
- 预测阶段：等张（isotonic）校准与每类阈值**只在 val 上拟合**，eval 标签不参与调参；导出 fp16 state_dict。
- 墙钟上限 2700 s、`max_no_improve=3` 是整体预算约束（计划未把它写进 C2 参数，属执行层约束）。

## 3. Loss（C2，组合式，对齐验收指标）
- `0.5 * Huber(delta=0.2)` on z —— 主回归项。
- `+ 0.3 * pairwise hinge rank loss`，仅在 in-batch `|Δscore| ≥ 5` 的样本对上，margin 0.1（对齐 SRCC）。
- `+ 0.2 * (1 - batch PLCC)`，可微相关项（对齐 PLCC）。
- `+ 1.0 * class-balanced focal BCE`，gamma 1.5，per-class `pos_weight` 截断到 [1,10]，先验 logit 调整 tau 0.5（对齐 macro-F1）。

## 4. 数据操作（2 项，均只作用于 train）
1. **augment**：120 条 hflip 增强训练行，label-preserving，seed 7。数量刻意低于 240 上限，为 iteration 2 的类别重采样留余量。假设：hflip 对总分及 blur/over_exp/under_exp/white_balance 在当前严重度下标签保持。
2. **annotate**：≤4 条本地弱标注（`labels_are_weak=true`，min_confidence 0.7），固定缺陷 JSON 口径，仅作训练辅助监督，绝不作为测试真值。目的是给 train 中支持为 0 的 white_balance / 稀少的 flare 补一点正例。计划自身已声明 4 条样本**无法**解决类别覆盖问题。
- 边界：`development_test`/`final_test` 不动；不删难例、不改真值、不换测试集。

## 5. 对照实验（A0–A4，同冻结编码器、seed 7、同一 subject-disjoint val）
| 编号 | 内容 | 隔离变量 |
|---|---|---|
| A0 | C1 基线原样复现 | 参考点 |
| A1 | C2 代码，无增强 | 方法/损失 |
| A2 | C2 + hflip 120 条 | 数据干预 |
| A3 | 仅 score 头（rank/PLCC 权重=0，缺陷头关闭） | 多任务与损失分量 |
| A4 | 固定阈值 0.5 + 恒等校准 vs val 调优 | 校准/阈值策略 |

## 6. 失败条件（计划显式声明）
- C1：backbone/adapter 权重缺失 → `FileNotFoundError`，报基础设施失败；**eval 切分歧义**——插件 `data.py` 记 KonIQ（610 eval 行），dev preview 只有 85 行，测得切分必须记录，否则数字不可比。
- C2：val PLCC 相对 A1 增益 < 0.02 → 损失改动不成立，退回 L1/BCE 并转向数据侧；rank loss（margin 0.1）若放大 pseudo-MOS 噪声 → val SRCC 相对 A1 下降且 val L1 下降。
- 设计报告补充的可证伪点：hflip 若使 flare/white_balance 召回下降 → 该增强对这两类不保标签；若在无新标签情况下 5 类 macro-F1 报出 > 0.80 → 说明协议/指标有错。

## 7. 已知门槛的结构性问题（影响「预期收益」的可信边界）
- 按设计报告：train 与 dev_test 的 **white_balance 支持均为 0**，该类 F1 无法定义（约定记 0）→ 5 类 `defect_macro_f1` 上限 4/5 = **0.80 < 0.85**，在不新增标签的前提下**结构性不可达**；flare（dev_test 3 例 < MIN_CLASS_SUPPORT）统计上不可测。
- 因此本轮可测的只有 PLCC / SRCC / grade_agreement 与支持类缺陷 F1（blur/over_exp/under_exp）。`grade_agreement`、`hardcase_macro_f1` 的判定口径未在 context 中给出，hardcase 按现状不可测。
- 历史测试集已暴露 → 即使达标也只能是**流程/复现验证**，不能宣称新的独立业务验收；pseudo-MOS 非独立采集的人类 MOS。

## 8. 要获得实测数字需要什么
当前没有命令能给出指标——`experiment_jobs` 为空。可执行 `查看状态` 查看是否已排入/启动作业；指标必须来自独立评测器的实验产物（`runs/<run_id>/preds.jsonl` 等）。我无法从现有 context 推断任何收益数值。



## User (2026-10-15T08:44:29.779007+00:00)

上一轮有哪些实测改进和失败？这些结果如何影响本轮研究、模型、loss、训练及数据计划？



### Agent

# 上一轮（iteration 1）实测结果

## 实测发生了什么

| 项目 | 记录 |
|---|---|
| 唯一作业 | `run_b6e0bd3aa9a9_v1_mobileclip2_v7_score` |
| 状态 | **failed** |
| 错误 | `FileNotFoundError('missing backbone (/mnt/data/.../mobileclip2_s0.pt) or adapter (plugin default (config.adapter_ckpt()))')` |
| 产出指标 | 无。`best_model_id=None`，PLCC / SRCC / defect_macro_f1 / grade_agreement / hardcase_macro_f1 / rom_mb 全部未测 |
| 预算消耗 | experiments 1/9、**fix_attempts 2/2（已用尽）**、coding_calls 3/8、research_calls 1/3、data_operations 2/8、added_samples 120/240、**annotation_samples 4/4（已用尽）**、cloud_tokens 58190/400000 |

**实测改进：0 项。** 本轮没有产生任何可比指标。
**实测失败：1 项，且属基础设施失败**（权重路径解析不到），按本任务规则**不能记为"方法无效"这类负面结果**。

需要注意：设计报告中的 `macro-F1 0.424 / PLCC 0.476` 是**文档记录的历史基线数值**（`final_labels_inspected=false`），不是本轮测量值。

数据侧存在不一致：预算计数器显示 120 条增强样本与 4 条弱标注已被计费，但 `experiment_jobs` 中只有那一个失败作业，且两条数据计划仍标 `data_plan_not_execution`。**这两项干预是否真正落地为产物，context 无法确认**（见"未知"）。

## 这些结果如何影响本轮（iteration 2）

当前 `iteration=2`、`phase=executing`，但 `active_plan` 仍是 `p_b6e0bd3aa9a9_001` v1，即展示的是首轮计划，未反映第二轮调整。

**1) 模型 / 依赖 —— 最优先，且是硬阻塞**
- C2 `multitask_adapter_rank` 复用的是**同一个冻结 MobileCLIP2-S0 编码器**。若 backbone 文件确实缺失，C2 会以同样方式失败，损失全部编码与训练预算。
- 因此本轮第一步不是训练，而是**先确认 backbone 与 adapter checkpoint 能否解析**。
- **fix_attempts 已 2/2 用尽**，不能再走修复循环；只能动用剩余 `coding_calls`（5 次）做代码级回退，或更换编码器/方案。若确认权重不可得 → 记为基础设施失败，并说明该路线在本环境不可执行。

**2) eval 切分歧义仍未解决**
- 插件 `data.py` 记 KonIQ（610 eval 行），dev preview 只有 85 行。任何测出的数字必须记录用了哪个切分，否则不可比。上一轮没有产生可记录的分辨证据。

**3) loss —— 不做无依据的增减**
- 一个数字都没测到，C2 的两条证伪条件（val PLCC 增益 <0.02；rank loss 推低 val SRCC）**本轮仍完全未被检验**。
- 结论：保留 A1（无增强）/ A3（仅 score 头）消融原样，**不要



## User (2026-10-15T08:48:51.631509+00:00)

请解释当前计划的模型结构、训练策略、loss、数据操作、对照实验和失败条件。请区分已运行结果与预期收益。



### Agent

# 计划解读：`p_b6e0bd3aa9a9_002` (v2)

## 0. 先分清「已运行」与「计划/预期」

| 项目 | 事实 |
|---|---|
| `active_plan` | `p_b6e0bd3aa9a9_002` (v2)，methods = `['score_moe', 'score_moe_calibrated_aug']` |
| 已运行作业 | 仅 1 个：`run_b6e0bd3aa9a9_v1_mobileclip2_v7_score` → **failed**（`FileNotFoundError`：缺 backbone `mobileclip2_s0.pt` / adapter） |
| 实测指标 | **无**。PLCC / SRCC / defect_macro_f1 / grade_agreement / hardcase_macro_f1 / rom_mb 全部未测；`best_model_id=None` |
| v2 的两个候选 | 全部标注 `candidate_plan_not_execution`，**尚未执行** |
| `data_plan_not_execution` | hflip 增强仍是**待执行计划** |
| 预算已耗 | experiments 1/9、**fix_attempts 2/2（用尽）**、coding_calls 3/8、research_calls 2/3、data_operations 2/8、added_samples 120/240、**annotation_samples 4/4（用尽）**、cloud_tokens 169651/400000、iteration 2/3 |

注意：v1 的 `mobileclip2_v7_score`（失败的那个）与 `multitask_adapter_rank` 已被 v2 取代；v1 的 `multitask_adapter_rank` 代码从未产出 `method.py`。

---

## 1. 模型结构

**C1 `score_moe`（action=reuse，is_baseline=true，零 ROM、纯 CPU）**
- 不改代码，直接跑 `algorithm_plugins/iqa/methods/score_moe.py`。
- 三个互补视图，**只在 dev 上拟合**：
  1. 全局回归器 ExtraTrees + RF；
  2. 4 类分类器，按 `E[score] = Σ_k P(level=k)·level_mean` 解码为分数；
  3. 显式门控 `P(defective|x)` × 退化族路由 × 每严重度专家。
- 输入：82 维退化特征 + 16 维物理测量 + 可选文件大小元数据；由 subject-group CV 在 dev 上选择混合权重。
- 缺陷输出：来自**零 ROM 的测量规则**（`measure_lib` 阈值），不占参数。

**C2 `score_moe_calibrated_aug`（action=modify，is_baseline=false）**
- 新文件 `model_pool/score_moe_calibrated_dose/method.py`，**导入并复用冻结的插件模块**，重导出 `run(*a, **k)/predict(*a, **k)`。
- 导入时只对 `score_moe.DEFAULTS` 中**已存在的键**按白名单改值，缺失键丢弃并记录，绝不臆造。
- 包装 `predict` 的分数输出：单调校准映射（`calibration.json`，由 dev out-of-fold 预测拟合的 isotonic 断点；退化方案 = 仅用训练折拟合的 4 级单调映射），并 clip 到 [0,100]；**5 个缺陷标志原样透传**。
- ROM 口径：该路径零 ROM（无神经网络权重）；导出物是 `preds.jsonl + calibration.json`，**没有 fp16 state_dict** —— 计划已显式声明这是导出偏离。

## 2. 训练策略
- **C1**：无梯度训练。确定性 seed 7、4 折 CV、3 个集成种子、`min_family_rows=15`、每个估计器 ≤40 迭代；**eval 标签永不读取**。
- **C2**：CPU-only、seed 7、dev 上 subject-group CV、`cv_folds=5`、`ensemble_seeds=5`、`blend_step=0.05`、`min_family_rows=12`、`use_metadata=true`、每个估计器 ≤40 迭代；eval 标签不参与调参。选型指标 `0.5*dev_oof_PLCC + 0.5*dev_oof_supported_macroF1`。
- 全局约束：`max_wall_clock_s=2700`、`epochs≤40`、`max_no_improve=3`。

## 3. Loss
- **两个候选都没有新损失函数。** C1 不是梯度训练，靠 dev-only CV 选混合权重（`blend_step`）；C2 明确保留基座 sklearn 集成（ExtraTrees/RF 回归、4 类分类、门控/路由/严重度专家），以便把任何变化归因于**超参、校准、数据剂量**，而不是新目标函数。
- （v1 曾计划 Huber + pairwise hinge rank + 可微 PLCC + focal BCE 的组合损失，但对应候选未产出可运行代码；v2 已放弃该方向。）

## 4. 数据操作（`data_plan_not_execution`，仅作用于 train）
- 类型：**augment**；`hflip`，label-preserving，seed 11。
- 剂量对照 `[0, 120, 240]`，取与 iteration 1（seed 7 那批）**不重叠**的子集；使用 240 上限中剩余的 120。
- 约束：`development_test` / `final_test` 不动；不删难例、不改真值、不换测试集。
- **前提条件**：需要 release 给增强行打标签，剂量消融才可执行；否则该项应声明为**不可测**。
- 注：`budget_used.added_samples = 120/240`，但增强计划仍标 `data_plan_not_execution` —— 产物是否真正落地，context 无法确认。

## 5. 对照实验
| 编号 | 内容 | 隔离变量 |
|---|---|---|
| A0 | C1 基线原样复现 | 参考点 |
| A1 | C2 代码，剂量 0 | 代码 + 校准 |
| A2 | 剂量 120 | 数据剂量 |
| A3 | 剂量 240 | 数据剂量（仅当打标存在，否则不可测） |
| A4 | 恒等校准 vs isotonic | 校准 |
| A5 | 插件默认缺陷阈值 vs dev-OOF 调优阈值 | 阈值策略 |

全部共享同一冻结 eval 切分、seed 7、评测器指标代码与



## User (2026-10-15T08:51:51.876358+00:00)

本轮最终是否满足全部 IQA 验收条件？请逐项给出本轮实测值、门槛、差距与证据，说明模型、训练、loss、数据方案分别有哪些实际执行，以及尚未验证的环节。



### Agent

# 本轮验收结论：**0/6 项达标可判定 —— 不是"未达标"，而是"未测"**

任务已 `phase=stopped`（`stop_reason: task wall-clock budget exhausted`），`best_model_id=None`，全部 `frozen_acceptance_targets` 的 `measured_on=null`。本轮没有任何可比的本轮实测指标，因此**不能声称满足全部验收条件，也不能声称未达标**——正确表述是"未可判定"。

## 1. 逐项对照

| 指标 | 门槛 | 本轮实测值 | 差距 | 证据 |
|---|---|---|---|---|
| plcc（primary） | ≥0.85(max) | **未测** | 未知 | 唯一作业 `run_..._v1_mobileclip2_v7_score` = failed |
| srcc | ≥0.85 | **未测** | 未知 | 同上，无 preds 产出 |
| defect_macro_f1 | ≥0.85 | **未测** | 未知 | 另受标签覆盖限制（见下） |
| grade_agreement | ≥0.90 | **未测** | 未知 | — |
| hardcase_macro_f1 | ≥0.85 | **未测** | 未知 | 设计报告记为不可测（见下） |
| rom_mb（strict, blocking） | <60 | **未实测** | 未知 | 仅有 fp16 参数估计，`declared_unknowns` 明确未测打包 ROM |

**唯一一条实测记录**（`experiment_jobs`，共 1 条）：
`status=failed`，`FileNotFoundError('missing backbone (/mnt/data/.../mobileclip2_s0.pt) or adapter (plugin default (config.adapter_ckpt()))')`。按本轮计划自身的失败条件，这属**基础设施失败，不得记为"方法无效"**。

**设计报告给出的限制性事实**（属文档级结论，非本轮实测指标，需在结论中标注来源）：
- `white_balance` 在 train 与 development_test 均为 0 正例 → 5 类 defect macro-F1 上限 4/5 = 0.80，低于 0.85；
- `flare` 仅 3 个 dev_test 正例 < `MIN_CLASS_SUPPORT=5` → 归为不可测；
- 标注预算已耗尽（4/4），本轮无法修补覆盖。

## 2. 模型 / 训练 / loss / 数据 实际执行了什么

| 方面 | 实际执行 | 状态 |
|---|---|---|
| **模型** | C1 `mobileclip2_v7_score`（reuse 基线）启动了但失败；C2 `multitask_adapter_rank` 未产出 `method.py` | **无有效模型**；`best_model_id=None` |
| **模型（v2）** | `score_moe`、`score_moe_calibrated_aug` | 仅 **`candidate_plan_not_execution`**，无运行记录 |
| **训练** | 无任何训练完成的作业（`experiments 1/9`，且该 1 次为失败） | **未执行** |
| **loss** | 设计报告提出的 Huber+z 归一化 + pairwise rank + 可微 PLCC + 类平衡 focal BCE | 仅计划，**其两条证伪条件从未被检验** |
| **数据** | 计数显示 `added_samples 240/240`、`annotation_samples 4/4`、`data_operations 3/8` 已消耗；但两条 data plan 仍标 `data_plan_not_execution`，且 `experiment_jobs` 只有 1 个失败作业 | **干预是否真正落盘为产物，context 无法确认** |

预算侧另录 `coding_calls 3/8`、`fix_attempts 2/2`（已用尽）、`research_calls 2/3`、`cloud_tokens 183019/400000`、`experiments 1/9`。

## 3. 尚未验证的环节

1. 全部 6 项验收指标本身，以及 A0–A5 对照（含 v2 的 A0–A5 设计）——**一个都未运行**。
2. **权重可得性**：backbone `mobileclip2_s0.pt` 与 adapter checkpoint 是否可解析，是本轮阻塞根因，`search_assets` 返回空资产列表；context 未提供用于探测该路径的命令。
3. **eval 切分口径**：插件 `data.py` 记 KonIQ 610 eval 行，dev preview 只有 85 行；测出的数字若未记录切分则不可比。本轮无分辨证据。
4. **ROM 计量口径**：`rom_mb` 是 strict + blocking，但本轮只有 fp16 参数估计；on-device 延迟与打包 ROM 均未测。
5. **数据干预落地性**：240 增强样本 / 4 弱标注是否真的生成，需核对产物。
6. 设计报告中 `macro-F1 0.424 / PLCC 0.476` 是**文档记录的历史基线数值**，不是本轮实测成绩。

## 4. 业务结论清单（本轮可给出的部分）

- **达标**：无（无一项被测量，不能声明达标）。
- **未达标**：不可判定（无测量值支撑）；但 `defect_macro_f1`、`hardcase_macro_f1` 在现有标签覆盖下有**结构性不可达/不可测**的文档级证据。
- **不可测**：`hardcase_macro_f1`（flare 支持不足）、`rom_mb` 的实测口径、切分协议未定下的所有指标。
- 本轮结果**不构成新的独立业务验收**（历史测试集已暴露；pseudo-MOS 非独立采集的人类 MOS）。

## 5. 控制

`查看状态` 可复核当前状态。注意：任务已 `stopped`，**不能恢复**，也**不能在本任务内再取任何指标**；如需新试次须另建任务。`暂停任务` 仅阻止新步骤，且此处已无运行中作业可中断。我不会直接编辑状态文件。
