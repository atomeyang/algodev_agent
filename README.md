# algo-agent — 算法自研究与优化 Agent

一个对话式、可审计的算法自研究 / 优化 Agent：把一句业务需求，走完
**需求澄清 → 研究设计 → 方案评审 → 编码 → 实验 → 评估 → 分析 → 决策 → 沉淀复用**
的完整闭环。生命周期由 LangGraph 驱动，控制层负责审批哈希、状态版本、预算与暂停/取消语义，
全部状态落 SQLite；离线时如实回退到合成演示，**绝不伪造结果**。

指导说明：`docs/codex_development_guide_v1.2.docx`、`docs/agent_architecture_design_v0.4.docx`。

## 目录结构

```
configs/                      # workflow.yaml：预算、审批、上传策略、provider 默认值
local_largemodel/             # 本地辅助大模型（Qwen3-27B 等）：weights/serving/configs/adapters
  _template/                  # 复制成 local_largemodel/<model_id>/ 使用
runtimes/                     # 平台运行时代码与共享调度状态，不存任务实验产物
  agent_runtime/              # Agent 本体源码：src/algo_agent/（pip 包 algo-agent）
  state/                      # sqlite 状态库、checkpoint、备份
algorithm_plugins/            # 可复用算法组件（common、demo_tabular）
shared_assets/                # 已注册复用资产（recipes/datasets/models/evaluation/knowledge）
tests/                        # pytest 测试套件
docs/                         # 交付文档（agent_status.docx、version_notes_v*.docx）
<task_name>_experiment/        # 一个算法任务唯一的顶层目录，包含全部任务产物
  .task_layout.json            # 任务归属、内部 track 和历史路径映射
  DIRECTORY_RULES.md           # 自动创建的任务目录约束
  task.yaml                   # 需求与执行策略
  plans/                      # 方案迭代、筛选、审批记录
  data/annotations/{plans,runs,reviews}/ # 标注方案、原始响应、中间结果和复核
  data/{raw,recipes,splits,releases}/    # 数据来源、加工配方和冻结版本
  evaluation/{protocols,scripts,intermediate,results}/
  model_pool/                 # 当前任务的算法实现
  runs/                       # 实验代码快照、训练日志、预测和中间产物
  runtime/{coding,tmp,cache,state}/     # 编码工作区、临时文件和状态快照
  coding_records/             # 编码修改与验证记录
  evidence/                   # 研究证据与缓存
  docs/                       # 任务研究报告与迭代分析
  tracks/                     # 同一任务的不同研究/评测分支
  history/                    # 历史实验及迁移清单所指向的原始证据
  deliverables/               # 交付物
```

> 创建约束：同一算法任务不得在平台根旁建 `*_annotations`、`*_rd` 或其他分散产物目录。
> 标注、数据扩增、评测的所有迭代和中间产物，以及编码临时工作区，都必须通过 `TaskPaths` 写入任务目录。
> 规则、迁移和平台共享资源边界见 [任务目录结构约束](docs/task_directory_structure.md)；
> 此约束更新早期设计文档中把任务编码工作区放在平台 `runtimes/` 的约定。
> `local_largemodel/` 是指导说明中的规范目录名，`local_auxmodels/` 作为兼容别名仍可被解析。
> `init` 会在平台根下创建 `local_largemodel/` 与 `runtimes/` 骨架，`doctor` 里对应检查项为 `local_models`。

## 环境与安装

需要 Python ≥ 3.11。推荐用两个 conda 环境（开发 / 运行隔离）：

```bash
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

# 开发环境（含 pytest）
conda create -n algo_agent_dev python=3.11 -y
conda run -n algo_agent_dev python -m pip install $MIRROR -e ".[dev]"

# 运行环境（仅运行依赖）
conda create -n algo_agent_runtime python=3.11 -y
conda run -n algo_agent_runtime python -m pip install $MIRROR .
```

> 国内网络建议走镜像源（上面的清华源）；默认 PyPI 在本机实测会长时间卡住。

不想装包时，也可以直接源码运行（系统 Python 具备依赖即可）：

```bash
PYTHONPATH=runtimes/agent_runtime/src:. python3 -m algo_agent.cli.main --help
```

## 快速开始

```bash
algo-agent --root ./agentic_algorithm init        # 建目录、默认配置、数据库、交付文档
algo-agent --root ./agentic_algorithm doctor      # 环境自检：provider / 目录 / 本地模型
algo-agent --root ./agentic_algorithm demo --offline   # 离线跑通一整轮闭环
algo-agent --root ./agentic_algorithm chat --task demo_task
```

`init` 幂等：已存在的文件不会被覆盖，并会生成 `docs/agent_status.docx` 与
`docs/version_notes_v1.0.docx`（可用 `docs status` / `docs version-notes` 重新生成）。

### Auto-Process 无人审批模式

```bash
# 新任务：自动研究、批准计划、改码、数据实验、训练、测试、分析、迭代
ALGO_OFFLINE=1 algo-agent --embedded auto-process "CPU rmse <= 0.5" --task auto_demo --demo

# 真实任务建议明确指定指标、预算与已授权数据源，避免自然语言解析歧义
algo-agent auto-process --spec configs/auto_process.example.json

# 恢复已经采用 auto-process 的任务，不重建已完成的实验
algo-agent auto-process --task <auto_task_id>
```

该模式不等待计划审批，也不在轮次之间等待用户回复。可以自主创建/修改算法，
研究并执行训练数据增强、重采样、过滤、标注规则实验、重标注和测试集扩充。
验收门槛、资源预算和数据源授权保持固定；资料中的文字不是新的用户指令。
无有效评测、非有限指标、缺少模型权重或任一验收条件不满足，都不能报达标。

迭代只查看开发测试集；最终测试在开发指标达标后，**加载选定权重推理一次，不重新训练**。
失败会如实停止，不继续利用最终测试结果调参。缺失验收标准或业务适配器也会明确报告，
不会通过自动降低门槛来“达标”。历史手动任务保留原行为，不能冒充新模式的独立验收。

运行条件、数据操作 schema、标注可信度、版本记录和已知边界见
[Auto-Process 使用说明](docs/auto_process.md)。新流程已使用临时数据和模拟研究/编码提供方测试；
这不是对真实业务复杂算法必然达标的保证，也不是一次新的云端研究或 GPU 达标报告。

### 对话式 REPL

```bash
algo-agent chat                 # 新建会话
algo-agent chat --task <task_id>  # 附着到已有任务
```

REPL 命令：`/help` `/new <需求>` `/use <task>` `/tasks` `/status` `/plan` `/approve`
`/ask <问题>`（调用实时对话模型：优先云端 DeepSeek-V4-Flash，否则本地 Qwen，均无则退回状态拼装）
`/improve <目标>`（对话式驱动研究/编码子 Agent 继续改进）`/report [--full]`
`/feedback` `/pause` `/resume` `/cancel` `/budget` `/events` `/export` `/attach` `/quit`。

### 独立命令行

```bash
algo-agent status --task <id>          # 阶段、当前方案、最优模型
algo-agent tasks                       # 任务列表
algo-agent plan show --task <id>       # 方案详情 + plan_hash
algo-agent approve --task <id> --plan <plan_id> --hash <plan_hash>
algo-agent feedback --task <id> --model <model_id> --scenario "低光照" --actual "RMSE 0.8"
algo-agent pause|resume|cancel --task <id>
algo-agent events --task <id>          # 事件流（审计）
algo-agent budget --task <id>          # 预算/用量/账本
algo-agent jobs --task <id>            # 作业状态
algo-agent export --task <id>          # 重新生成该任务的 DOCX 记录
```

## 复用与交付（M4/M5）

```bash
# 资产注册表：不可变、幂等、发布需真实验证记录
algo-agent assets list [--type recipe] [--status reproduced]
algo-agent assets show <asset_id> [--version v1]
algo-agent assets register --file manifest.json [--body path ...]   # 同内容重复注册幂等
algo-agent assets publish <asset_id> --version v1 --validation validation.json
algo-agent assets deprecate <asset_id> --version v1 --reason "superseded"

# 兼容性：显式字段全部匹配才 compatible，任何不可核对项一律 unknown（不会静默放行）
algo-agent compat check --asset <id> --version v1 \
  --inputs rgb_image --outputs quality_score --device mobile \
  --license synthetic --distribution mobile_photos

# 最小复现包：冻结依赖闭包/数据引用/种子/入口/期望结果与容差
# DIR 为任务相对路径，且必须在 deliverables/ 内；默认 deliverables/reproduction/
algo-agent reuse export --task <id> [--out DIR]
algo-agent reuse validate --task <id> [--package DIR] [--actual-metrics metrics.json]

# 任务包：哈希校验的可移植快照（原始数据/权重为外部哈希引用，凭证过滤）
# 默认 deliverables/bundles/<bundle_id>/；DIR 同样必须在任务 deliverables/ 内
algo-agent bundle export --task <id> [--out DIR]
algo-agent bundle import <bundle_dir> [--as <new_task_id>] [--dry-run]

# 状态迁移：带备份与回滚，跨 graph 版本拒绝恢复
algo-agent migrate [--no-backup]

# 交付文档
algo-agent docs status                 # 重新生成 docs/agent_status.docx
algo-agent docs version-notes --version v1.0
```

## IQA 业务插件（端到端实跑）

`algorithm_plugins/iqa/` 是第一个真实业务插件（轻量双输出 IQA：0–100 质量分 +
blur/over_exp/under_exp/flare/white_balance 五类缺陷多标签，ROM < 60 MB fp16）。
它直接复用 `../iqa_agent` 的真实资产：预训练 MobileCLIP2-S0 编码器与冻结指标代码，指标实现逐位对齐。

插件支持**两条互不混淆的评测轨道**：

- **pseudo（默认）**：内部伪标注冻结集（610 张，severity 量化 pseudo-MOS），用于回归与缺陷侧 / ROM 复现，
  实测 PLCC 上限约 0.73（伪标注构造导致结构性不可达，证据见 `iqa_experiment/docs/iqa_run_report.md` §0b）。
- **real-MOS**：`IQA_BENCHMARK=koniq` 时切到 **KonIQ-10k 真实人类 MOS**，
  官方 7058 / 1000 / 2015 train / val / test 划分，评测在冻结官方 test 上完成。

```bash
# 轨道一：内部 pseudo 基准（默认）
ALGO_STATE_DIR=/root/algo_agent_state_iqa \
  python3 scripts/run_iqa_loop.py --root . --task-id iqa --max-iterations 3

# 轨道二：真实 MOS 达标跑（PLCC ≥ 0.85，KonIQ-10k 官方测试集）
env IQA_BENCHMARK=koniq IQA_FEATURE_NPZ=/root/iqa_assets/koniq_mobileclip2s0_256.npz \
  ALGO_STATE_DIR=/root/algo_agent_state_iqa_mos PYTHONPATH=runtimes/agent_runtime/src:. \
  python3 scripts/run_iqa_loop.py --root . --task-id iqa_mos --parent-task iqa --track mos --max-iterations 2
# IQA 仅有一个顶层任务目录：iqa_experiment/；MOS 分支在其中的 tracks/mos/
```

**real-MOS 达标结果**：冻结 MobileCLIP2-S0 512-d L2 特征 + 闭式岭回归头
（方法 `mobileclip2_mos_head`；同类还有 `mobileclip2_mos_mlp` 作为迭代 2 阶梯），
在 KonIQ-10k 官方 2015 张测试集上测得 **PLCC 0.8886 / SRCC 0.8712**，
ROM 22.92 MB fp16，Evaluator 判定 `pass=true`、`failures=[]`，Agent 第 1 轮即接受
（`iqa_experiment/tracks/mos/`，历史记录 `phase=accepted`）。

研究设计报告（`<task_dir>/docs/research/research_iter*.md`）每轮都会给出任务方需要的三件事：
**①当前产出精度状态**（逐项对照验收门槛的实测表）、**②模型 & 损失函数 & 训练策略优化计划**、
**③数据标注调优计划 & 数据扩增计划**（含本地辅助标注模型 `local_largemodel/qwen3.8-27b`
与人工复核闭环）。结构化副本同时写入 `research_iter*.json` 的 `extras` 字段。

> IQA 目录归属：主任务在 `iqa_experiment/`，MOS 分支在 `tracks/mos/`，
> 早期实跑在 `history/legacy_rd/`，标注试点在 `data/annotations/runs/legacy/pilot/`。
> 保留 `iqa_mos`、`iqa_rd` 的历史身份，不混合其验收结果；路径登记阻止再次创建同名顶层目录。

实跑证据、真实指标、达标差距与对 Agent 的改进见
[`IQA 实跑报告`](iqa_experiment/docs/iqa_run_report.md)。要点：pseudo 轨道 3 轮 5 个模型全部在 610 张冻结集上独立评测，
ROM 达标、其余门槛未达成并定位到根因；real-MOS 轨道 `mobileclip2_mos_head` 达成 PLCC 0.8886 ≥ 0.85。

## 离线模式与诚实回退

- `ALGO_OFFLINE=1`（或既未配置云端密钥、也未配置本地端点）时，研究设计走离线 provider，
  实验结果标注为 `synthetic_demo`，**不会**声称做过云端推理、GPU 训练或业务验证。
- 只配置了本地端点（`ALGO_LOCAL_BASE_URL`）也算在线：研究设计仍优先云端 DeepSeek，
  但没有云端密钥时退回插件级确定性设计；对话回复改用本地模型；`ALGO_OFFLINE=1` 仍可强制全离线。
- **研究设计模型**默认 `deepseek-flash`：手动模式保持插件候选选择行为；
  `auto-process` 可通过工具补充证据，提出新的算法实现、代码修改与可执行数据操作。
  自动模式下云端失败会明确报告，不把离线模板说成成功的自主研究。
- 任何无法核对的信息（硬件档位、业务分布、许可、未提供模型版本的反馈）都记为
  `unknown` / `missing_evidence`，并要求真实复跑后才能升级。
- 资产只有在提供真实 `validation` 记录时才允许发布，且一次只升一级。

## 接入真实能力

复制 `.env.example` 为 `.env` 并填写：

| 能力 | 变量 | 说明 |
| --- | --- | --- |
| 云端研究/设计 + 对话 | `ALGO_CLOUD_BASE_URL` `ALGO_CLOUD_API_KEY` `ALGO_CLOUD_MODEL` | OpenAI 兼容端点；默认 `deepseek-flash`（DeepSeek-V4-Flash） |
| 本地辅助模型 | `ALGO_LOCAL_BASE_URL` `ALGO_LOCAL_MODEL` | 提取/标注/诊断 + 无云端密钥时的对话回复，见 `local_largemodel/README.md` |
| 编码工具 | `ALGO_CODEX_BIN` `ALGO_CODEX_ALLOW_REAL` | 置 `ALGO_CODEX_ALLOW_REAL=1` 才允许真实 Codex 执行 |
| 控制服务 | `ALGO_SERVICE_HOST` `ALGO_SERVICE_PORT` | 默认 `127.0.0.1:8765` |

配置后运行 `algo-agent doctor` 确认模式由 OFFLINE 变为 LIVE。

### 控制服务与恢复

```bash
algo-agent service start --foreground        # 前台运行 HTTP 控制服务
algo-agent --service-url http://127.0.0.1:8765 --root <root> status --task <id>
```

控制服务与 CLI 共用同一份业务逻辑与 SQLite 状态。进程重启后用同一个 `--root` 重新
构造 Controller，即可从 checkpoint 看到历史任务与图状态（见 `tests/test_recovery.py`）。

**暂停 / 取消语义**：`pause` 只阻止新步骤启动，不杀正在运行的作业；`resume` 继续；
`cancel` 才会取消未完成的作业，已保存的资产**永不删除**。

## 共享 / FUSE 文件系统的兼容处理

若平台根目录位于 `ossfs2` 等 FUSE 挂载（例如本项目所在的 `/mnt/data`），
**SQLite 无法在该目录上创建数据库**（`sqlite3.connect` 报 `disk I/O error`），
**python-docx 也无法直接就地写 zip**（保存时报 `Invalid argument`）。平台已内置处理：

- **状态目录自动回退**：`Settings.state_dir` 会探测 `runtimes/state` 能否承载
  SQLite；不能时自动改用 `~/.algo_agent/state/<root 哈希>`。`algo-agent ... init`
  打印的 `database:` 行即真实生效路径。
- **显式覆盖**：`ALGO_STATE_DIR=/abs/path` 直接指定状态目录；
  `ALGO_STATE_FALLBACK_DIR=/abs/base` 只改回退根（目录名仍为根哈希）。
- **DOCX 安全写盘**：所有 `docs_export` 输出先序列化到内存，再一次性顺序写入并
  `fsync`，因此 `docs/*.docx` 在 FUSE 上仍可被 python-docx / Word 正常打开。

> 影响：状态库不在仓库内，`git`/`rsync` 备份平台根时需单独备份 `state_dir`；
> 换个 `--root` 会得到不同的哈希目录。跨机器迁移请用 `bundle export`。
> 若 `docs/*.docx` 出现 0 字节或损坏文件，说明它们是在修复前生成的——去掉 `init` 的
> 保留逻辑重新生成即可：`algo-agent docs status` / `docs version-notes`。

## 测试

```bash
PYTHONPATH=runtimes/agent_runtime/src:. python3 -m pytest tests -q
# 或使用开发环境
conda run --no-capture-output -n algo_agent_dev python -m pytest tests -q -p no:cacheprovider
```

共 **142** 项测试（含 `tests/test_iqa_plugin.py`、`tests/test_iqa_plans.py`、
`tests/test_iqa_annotation.py`、`tests/test_iqa_score_head.py`、
`tests/test_iqa_mos_benchmark.py`、`tests/test_chat_grounding.py`、
`tests/test_providers_wiring.py`、`tests/test_chat_driving.py` 等 IQA 路由/指标/导出/约束/三份计划/本地标注/打分头/真实 MOS/模型接线与对话驱动用例），覆盖：需求解析与路径安全、资产不可变/注册幂等（含 `--body` 校验和）、
发布阶梯与依赖闭包、审批哈希与状态版本、命令幂等、数据发布冻结校验、反馈版本归因、
预算与停机规则、恢复、证据读取范围、兼容性判定与缓存失效、复用/复现/任务包/迁移。

## FAQ

- **`local_largemodel/` 里有可直接用的标注模型吗？** 有：`local_largemodel/qwen3.8-27b/`
  （llama.cpp 服务、OpenAI 兼容端点 `127.0.0.1:8084/v1`、多模态、Apache-2.0）。
  用 `scripts/iqa_annotate.py` 产出候选预标注，再经人工复核后入库；
  详见 `local_largemodel/README.md`。
- **`*_runtime` 旧目录怎么办？** 首次运行会迁移到 `runtimes/`，无需手工搬动。
- **能直接上生产数据吗？** 先在 `.env` 配置云端/本地 provider 并跑 `doctor`；离线结果
  只用于流程演示，不可作为业务结论。
- **反馈一定要给 model 版本吗？** 是。未给版本时不会默认取最新模型，只返回候选并要求确认。
- **数据库为什么不在仓库里？** 见上文「共享 / FUSE 文件系统的兼容处理」：FUSE 挂载
  不能承载 SQLite，会自动回退到 `~/.algo_agent/state/<hash>`，可用 `ALGO_STATE_DIR` 固定。
