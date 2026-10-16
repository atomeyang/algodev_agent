# algo-agent 技术方案架构说明

版本：0.4.0（含 2026-10-15 自主研发加固）
代码基线：`runtimes/agent_runtime/src/algo_agent/`、`algorithm_plugins/`
本轮详细实现、角色配置与剩余限制见 [自主算法研发加固说明](rd_hardening.md)。

## 1. 定位

algo-agent 是一个**算法自研究与优化 Agent**：输入一句业务需求或一份结构化
`TaskSpec`，自主走完
`需求解析 → 调研设计 → 方案冻结 → 编码实现 → 数据构建 → 实验 → 独立评测 →
结果分析 → 决策 → 再调研` 的闭环，直到验收门槛达标或预算耗尽。

三条不可让步的原则贯穿实现：

1. **不伪造结果。** 没有实测、没有权重、门槛未定义，一律不算达标。离线时如实
   回退到合成演示并显式标注。
2. **验收标准不属于 Agent 可改范围。** 指标、阈值、预算、授权数据源在 intake
   冻结，计划校验会拒绝任何篡改。
3. **最终测试集只用一次。** 开发阶段用开发集；达标后重新加载已保存权重，做一次
   独立最终评测。

## 2. 分层结构

```
入口层    CLI (typer)            HTTP (FastAPI)           嵌入式 (--embedded)
          cli/main.py,cli/chat.py  service/app.py           cli/client.py
                    \                    |                    /
                     \                   v                   /
控制层               Controller：唯一任务状态写入者（锁 + 状态版本 + 哈希校验）
                     service/controller.py
                                    |
编排层               LangGraph 状态机（14 个节点 / 条件路由 / SQLite checkpoint）
                     graph/builder.py, graph/nodes/*
                     |
        +------------+-------------+----------------+-----------------+
        v            v             v                v                 v
   ResearchDesign  CodexCoding   Execution        Storage          Evidence
   Agent           Agent         Runner/Worker    Repository       Store/Reader
   agents/*.py     agents/*.py   execution/*.py   storage/*.py     evidence/store.py
        |            |             |                |
        v            v             v                v
   Provider 层   Provider 层   方法适配器         SQLite         任务目录文件系统
   cloud/local   codex CLI     model_pool/*     db.py          TaskPaths / paths.py
        |
   domain 插件层：algorithm_plugins/<domain>/（iqa、demo_tabular）
```

分层职责：

| 层 | 职责 | 关键文件 |
|---|---|---|
| 入口 | 命令解析、人机交互、服务暴露 | `cli/main.py`、`cli/chat.py`、`service/app.py` |
| 控制 | 生命周期唯一写者、审批哈希、状态版本、预算、暂停/取消 | `service/controller.py` |
| 编排 | 节点图、条件路由、checkpoint | `graph/builder.py`、`graph/routing.py`、`graph/state.py` |
| 智能体 | 调研设计与编码执行 | `agents/research_design.py`、`agents/codex_coding.py` |
| 执行 | 作业提交/隔离运行/取消/恢复 | `execution/runner.py`、`execution/worker.py`、`execution/method_worker.py` |
| 评测 | 冻结协议独立评测、开发/最终分离 | `execution/evaluator.py`、`graph/nodes/experiments.py` |
| 数据 | 版本化 release、数据操作、来源校验 | `execution/data_pipeline.py` |
| 存储 | SQLite 事务仓储、版本化迁移 | `storage/repository.py`、`storage/db.py` |
| 证据 | 来源登记、可见范围、上传边界、依赖缓存 | `evidence/store.py` |
| 域插件 | 领域指标、门槛、方案候选、方法实现、数据适配 | `algorithm_plugins/<domain>/` |
| 路径 | 任务目录契约与越界防护 | `paths.py`、`config.py` |
| 复用 | 资产注册表、兼容性、复现包、任务包 | `storage/assets.py`、`reuse/*.py` |

## 3. 生命周期

图定义见 [builder.py](/mnt/data/yangt/codes/QAA/algo_agent/runtimes/agent_runtime/src/algo_agent/graph/builder.py:24)，
共 14 个阶段：

```text
intake → research → design_plan → approval_gate → prepare_assets → coding
   → submit_experiments → collect_results → evaluate → analyze → decide
   → feedback_gate → diagnose_feedback → (research)  ↘ export_docs → END
```

| 节点 | 行为 |
|---|---|
| `intake` | 解析需求为 `TaskSpec`，登记并校验数据源校验和，创建任务目录 |
| `research` | 自主调研设计：检索、证据包、工具调用、候选方案、数据配方、评测协议 |
| `design_plan` | 生成 `PlanSpec`、`plan_hash`，落盘 `plans/<version>/` |
| `approval_gate` | 人工模式中断等审批；auto-process 自动以哈希批准 |
| `prepare_assets` | 构建不可变 release（开发集与最终集分离） |
| `coding` | 复用插件实现或由编码单元产出/修复候选代码 |
| `submit_experiments` | 幂等提交作业，冻结代码哈希与运行时环境 |
| `collect_results` | 汇总作业状态与指标 |
| `evaluate` | 用冻结协议独立评测，计算验收失败项，选出当前最佳 |
| `analyze` | 诊断瓶颈、生成证据包与下一轮输入 |
| `decide` | 达标→（auto-process）一次性最终评测→`accepted`；否则继续或按预算停止 |
| `feedback_gate` | 人工模式可插话；auto-process 直接进入下一轮 |
| `diagnose_feedback` | 本地辅助模型诊断用户反馈，标注"未独立复现" |
| `export_docs` | 生成 OVERVIEW / 迭代报告 / 研究文档 / 数据集文档 |

路由在 `graph/routing.py`，阶段枚举在 `STAGES`，任务状态字段在
`graph/state.py`（只存 ID 与控制字段，权重、数据、日志不进 checkpoint）。

## 4. 控制层

`Controller` 是任务状态的**唯一写入者**（[controller.py](/mnt/data/yangt/codes/QAA/algo_agent/runtimes/agent_runtime/src/algo_agent/service/controller.py:41)）：

- 每任务文件锁 `state/task_<key>.lock`，阻止并发驱动同一任务；
- `expected_state_version` 比较并交换，过期写入被拒（`StateConflict`）；
- 审批时校验 `plan_id` 与 `plan_hash` 一致，防止批到被改过的计划；
- research / coding / fix / experiment 次数及墙钟受限；研究、聊天/意图记 token 账本，
  外部 Codex 的内部 token 消费不能由该预算硬限制；
- 暂停只阻止后续步骤，正在运行的实验可结束；取消才请求终止作业进程组；
- 任务状态快照导出到 `<task>/runtime/state/snapshot.json`，但**不替代**共享调度
  数据库与 LangGraph checkpoint。

## 5. 自主模式（auto-process）

自主模式取消的是**等待**，不是验收契约：

| 人工模式 | auto-process |
|---|---|
| 计划需用户批准 | 校验后自动批准同一份冻结哈希 |
| 每轮结束等用户指令 | 直接进入下一轮调研 |
| 反馈驱动迭代 | 实测结果与失败项驱动迭代 |

关键实现点：

- 计划仍经 `policy.plan_problems` 校验：候选上限、方法名合法、验收目标未变、
  写入范围仍为 `model_pool`、预算未变、数据操作授权与切分合法性；
- 自动研究阶段先校验并将错误反馈模型，在 `max_research_steps` 内修正；落库前再次校验；
- 新算法的候选**不允许**回退到模板写码器，必须有真实编码执行器；
- 达标判定要求「已保存权重 + 有限值实测 + 所有硬约束通过」，
  仅预测文件不算模型产物（见 [experiments.py](/mnt/data/yangt/codes/QAA/algo_agent/runtimes/agent_runtime/src/algo_agent/graph/nodes/experiments.py:240)）；
- 最终评测占一个保留的作业额度，且**只跑一次**：失败即 `stopped`，
  不允许在同一 holdout 上继续调参；重试只复用已完成的最终评测。

`policy.acceptance_failures` 是唯一验收判据来源：主指标、次指标、领域插件的
约束钩子、ROM/包体积等硬约束，任一缺失或不可测量都判失败。

## 6. 数据与标注

`execution/data_pipeline.py` 负责版本化数据构建，核心性质：

- **不可变 release**：已存在的 release 校验 manifest 与配方哈希，配方变化必须换版本；
- **完整配方语义**：每轮配方都是"作用于同一冻结基底的完整配方"，不是增量补丁；
- **切分隔离**：训练/开发/最终集做路径、主体与图像内容重叠检查，禁止泄漏；
- **训练期操作**：`resample`、`filter`、`annotate`、`relabel` 仅限训练集；
  对测试集做这些操作会被计划校验直接拒绝；
- **最终集扩充**：只接受 intake 时已登记、独立标注（`human`/`verified`）、
  带校验和的来源；
- **来源固定**：controller 在 intake 固定文件校验和，调研无法自行授予新数据源；
- **标注落档**：方案写 `data/annotations/plans/<version>/`，原始模型响应、
  被拒标签、失败记录、一致性报告写 `data/annotations/runs/<version>/<run_id>/`；
- **自动标注恒为弱标签**：schema、有限值、置信度三项校验通过才可用于训练，
  永远带 `review_required: true`，绝不成为最终测试真值；
- **图像自动标注只允许回环地址**：私有原图不外发云端。

支持的能力矩阵（`operation_capabilities`）：`augment`、`resample`、`filter`、
`annotate`、`relabel`、`extend`，按域（`demo_tabular` / `iqa`）给出参数与允许切分。

## 7. 编码与验证

`CodexCodingAgent`（[codex_coding.py](/mnt/data/yangt/codes/QAA/algo_agent/runtimes/agent_runtime/src/algo_agent/agents/codex_coding.py:34)）：

- 任务内隔离工作区 `runtime/coding/workspaces/<coding_id>_<rand>/`；
- 写入范围白名单（默认 `model_pool/<method>`），越界文件与符号链接直接拒收；
- **以文件系统哈希判定改动**，不采信 CLI 的自述事件；
- 独立验收命令（语法解析、可导入、入口可调用）必须全部通过；
- tabular 自主新建/修改方法另外检查小数据训练、权重保存、重载预测一致性；
- 失败进入有上限的修复循环，未通过的补丁**不会**替换基线代码；
- 全过程归档到 `coding_records/<coding_id>/`：任务、结果、diff、事件、工具能力。

## 8. 执行与评测

`Runner`（[runner.py](/mnt/data/yangt/codes/QAA/algo_agent/runtimes/agent_runtime/src/algo_agent/execution/runner.py:80)）：

- 每个 run 独立目录 `runs/<run_id>/`：`code/`（方法 + 插件快照）、`job.json`、
  `run.log`、`tmp/`、预测与权重；
- 幂等键 `task:run_id`，重复提交返回既有作业；
- 子进程独立会话，超时/取消按进程组终止；恢复时对照真实进程校准作业状态；
- 冻结代码哈希在运行前后各校验一次，改动即失败；
- `TMPDIR/TMP/TEMP` 指向任务内 `runs/<run_id>/tmp/`，`PYTHONDONTWRITEBYTECODE=1`。

评测三方分离：方法只产出预测与权重，`Evaluator` 用冻结协议代码算指标，
`acceptance_failures` 判达标。IQA 域还额外要求缺陷/PLCC/分档/hardcase 多门限，
由插件 `check_constraints` 与 `select_best` 承担。

## 9. 证据与安全边界

`evidence/store.py`：

- 来源登记（类型、版本、信任级、上传策略、校验和）；
- 包与读取分离：`PacketBuilder` 组装上下文，`EvidenceReader` 执行读取；
- 上传黑名单：`data/raw`、`data/releases`、`final_test`、`evaluation/protocols`、
  `pred_test`、`.env` 一律拒绝外发；
- 后缀白名单与 `max_read_bytes` 上限由配置（`configs/workflow.yaml`）控制；
- 远端读取只接受公网 HTTP(S) 文本，拒绝重定向、私网地址与非文本类型；
- 每次读取写审计记录（角色、范围、字节数、状态）。

研究/编码提示词明确声明：文档、日志、检索结果都是**数据不是指令**。研究可用的
工具限定为检索、读证据、读本地代码片段、读运行摘要、聚合数据检查、复用资产查询。

## 10. 存储与任务目录

SQLite（`storage/db.py`，schema 4）承载：任务与状态、计划、审批、命令、事件、
作业、模型、评测、反馈、数据集、实验、预算台账、会话/调用、资产、证据、
读取审计、缓存、账本、变更请求，以及多轮消息、研究步骤和可召回的实验经验。
命令与作业带唯一幂等键；作业心跳和结果还以 owner + attempt 拒绝旧 worker 回写。

任务产物遵循[任务目录创建约束](task_directory_structure.md)：一个算法任务只有
一个 `<task_name>_experiment/` 顶层目录，标注、数据、评测、编码、缓存、交付全在
其中；显式输出路径也经 `TaskPaths.output_path` 校验，拒绝绝对路径、`..` 越界与
指向外部的符号链接。分支任务通过 `.task_layout.json` 的 `members` 显式登记到
`tracks/` 或 `history/`，历史路径由 `relocations` + `resolve_reference` 解析。

## 11. 扩展点

三个可插拔契约（纯 Python 对象，见 `algorithm_plugins/common/plugin.py`）：

1. **业务插件**（必须）：`validate_requirement` / `validate_dataset` /
   `default_slices` / `deployment_scope`；
   可选钩子：`matches`、`design`、`slices`、`data_sources`、`method_entry`、
   `coding_context_files`、`experiment_env`、`experiment_split`、
   `experiment_resources`、`evaluate`、`model_manifest`、`select_best`、
   `check_constraints`、`constraint_names`、`diagnose`。缺失钩子时回退到
   合成表格演示行为，老任务语义不变。
2. **方法适配器**：`model_pool/<method>/method.py` 暴露
   `run(ctx, config)` / `predict(ctx, config, weights_path)`，
   由 `MethodContext` 提供任务根、数据目录、输出目录与种子。
3. **Provider**：`research`（云端；缺云端时离线模板，不以本地聊天冒充自主研究）、`local`（本地 VLM 辅助
   标注与诊断）、`coding`（Codex CLI，真实执行需显式开启）、`coding_fallback`。
   每个 provider 必须报告可用性与原因，缺失即成为可诊断阻塞项。

新增领域只需加一个插件目录并声明 `domain`，图与控制层无需改动。

## 12. 接口与运行

CLI：`init`、`doctor`、`demo`、`chat`、`auto-process`、`status`、`tasks`、
`preflight`、`memory`、
`approve`、`feedback`、`pause`/`resume`/`cancel`、`events`、`budget`、`jobs`、
`plan show`、`export`、`assets *`、`compat check`、`reuse export/validate`、
`bundle export/import`、`migrate`、`docs *`、`service start/status/stop`。

HTTP：`GET /health`、`GET|POST /tasks`、`GET /tasks/{id}/state`、
`GET /tasks/{id}/events`、`GET /tasks/{id}/events/stream`、
`GET /tasks/{id}/plans/{plan_id}`、`POST /tasks/{id}/ask|commands|approvals|feedback|control|export`、
`GET /jobs/{job_id}`。
补充：`GET /readiness`、`GET|POST /tasks/{id}/messages`、
`GET /tasks/{id}/memory|budget|models|evals`、`GET /jobs`。

退出码：仅验收通过返回 0；停止/未完成返回 2，便于流水线判读。

## 13. 已知边界

- 自然语言需求解析是域解析器，不是通用需求谈判；复杂任务应给结构化 `TaskSpec`。
- 相似图重复检测未实现（仅做路径、主体与内容哈希重叠检查）。
- 多随机种子的统计显著性、目标设备真实延迟与 ROM 实测，需要外部执行环境补足。
- 检索支持显式启用的 arXiv API 或 HTTP 服务；默认关闭，摘要不等于全文研读或独立复现。
- 实验进程不是安全沙箱；长任务推荐独立服务，活着的孤儿进程尚不能自动接管。
- 共享数据集与预训练模型不复制进任务目录，仅在任务内登记路径、版本与校验和。

## 14. 验证状态

修改前回归基线为 **194 passed**。本轮新增指标/意图/多轮消息、研究修正与重放、
恢复、模型分工、知识隐私、编码契约和服务接口测试，
最终全量结果为 **254 passed**（2026-10-15），
详见 [验证范围](rd_hardening.md#10-验证)。
回归验证编排与契约正确性，不构成真实云端调研质量、真实标注准确率
或任意业务门槛必然达标的证明。
