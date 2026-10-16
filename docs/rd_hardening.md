# 自主算法研发加固说明

更新：2026-10-15。本文说明本轮代码落地后的行为。
[能力审查](current_capability_review.md)保留为修改前基线，不应再把其中所有缺口当作现状。

## 1. 能力定位

系统现在具备更完整的有界研发闭环：

```text
对话 / 结构化需求
  -> 意图分类与控制指令校验
  -> 冻结指标、预算、授权数据源
  -> 检索证据 + 历史实验 + 知识检索
  -> 研究模型提出算法、数据操作和实验计划
  -> 计划校验，失败时在研究预算内修正
  -> auto-process 自动批准计划哈希
  -> 版本化数据构建 + 编码与独立契约检查
  -> 训练、权重保存、开发评测
  -> 失败分析 + 知识记录 -> 下一轮研究
  -> 开发达标后一次最终评测 -> 接受或停止
```

这是“可以自主尝试解决复杂任务”的工程能力，不是“任意复杂算法必然达标”的承诺。
自动模式不等用户审批，但也不能自行降低门槛、授予新数据权限或解除预算。
预算耗尽、持续无改进、依赖不可用和最终测试失败均会明确停止。
本轮没有启动真实模型生成、业务训练或改写历史 IQA 实验。

## 2. 对话、意图与模型分工

| 职责 | 组件 | 模型选择 | 本轮变化 |
| --- | --- | --- | --- |
| 对话 | `Controller.answer` | `ALGO_CHAT_MODEL`，否则云端默认；无云端时可用本地模型 | 按任务及 conversation_id 保存多轮消息，回答有界的实时上下文 |
| 意图 | `service/intents.py` + 可选模型提议 | `ALGO_INTENT_MODEL`，否则聊天 provider | 结构化 action/confidence；确定性控制授权，低置信度澄清 |
| 调研、方案、下一轮改进 | `ResearchDesignAgent` | `ALGO_RESEARCH_MODEL`，否则 `ALGO_CLOUD_MODEL` | 证据工具、计划修正、步骤缓存、历史负结果召回 |
| 结果辅助诊断 | `QwenLocalProvider.diagnose` | `ALGO_LOCAL_MODEL` | 输入包含失败项、指标、切片和任务契约；解释标为未验证 |
| 编码和修复 | `CodexCodingAgent` -> `CodexCliExecutor` | `ALGO_CODING_MODEL`；缺省读取 CLI 顶层 model 配置快照 | 显式传入模型，记录请求/返回身份及可获得的 usage |
| 图像预标注 | IQA annotation 模块 | 本地 VLM 的独立端点配置 | 保留 rubric 实验、弱标签、复核状态和来源记录 |
| 训练和推理 | `Runner` / `Worker` / 方法插件 | 候选算法模型，不是对话 LLM | 租约续期、过期恢复、旧 worker 结果拒收、资源上限 |
| 指标和验收 | `Evaluator` + policy | 不使用 LLM 判分 | 主次指标、严格不等式、有限值和独立最终评测 |
| 编排 | Controller + LangGraph | 确定性程序 | 服务启动时恢复任务，保留暂停、取消、预算和 checkpoint |

模型别名是配置，不等于厂商版本或能力认证。2026-10-15 本机只读预检：
chat / intent / research 为 `deepseek-flash`，local 为 `qwen3.8-27b`，
coding 配置快照为 `gpt-6-astra`；检索未配置，真实编码关闭，`live_verified=false`。
本轮没有更换模型或修改 `.env` / CLI 凭证。

### 意图识别的边界

- “暂停任务”“恢复任务”“取消任务”按明确控制意图执行。
- “如何暂停任务？”“不要停止训练”只作为问题，不执行控制。
- “继续优化”记为改进反馈；不修改原任务验收标准。
- 模型输出不能指定另一个 task_id、批准计划或授权含糊的破坏性操作。
- 不识别的消息走查询或澄清，优先避免误操作；不是覆盖任意语言表述的通用规划模型。
- 已结束的任务不会被“继续优化”偷偷重新打开；继续研发需新任务与独立验收安排。
- 复杂需求仍建议结构化 `TaskSpec`。本轮没有实现完整的多轮需求协商、字段补全与规格修订界面。

## 3. 需求和计划

通用指标解析现在绑定指标附近的比较符、数值和单位。例如：
`CPU latency <= 20 ms, accuracy >= 0.95` 不再把 20 当准确率门槛；
`PLCC/SRCC >= .9` 可拆成两个目标；百分比、秒/毫秒、GB/MB/KB 能归一化。
显式主指标优先，否则预测质量指标优先于延迟和包体积。
严格 `<` / `>` 不接受恰好等于阈值的结果。
矛盾方向、不同的重复门槛、越界值、缺少次指标阈值会阻止 auto-process 启动。
IQA 保留领域解析器，其显式指标也经过通用解析校正。

研究得到方案后先用与最终计划相同的 `build_plan_spec` / `plan_problems` 校验。
错误及有界草案反馈到研究模型，剩余步数内允许修复；没有额外无限重试。
修复消耗模型调用的 token 预算和 `max_research_steps`，不额外消耗一个研究轮次。
真正落库、冻结哈希、自动批准前还会再次校验。

## 4. 调研检索

检索后端默认关闭，已有 `ALGO_SEARCH_URL` 的接入方式兼容保留。
可显式选择：

```yaml
# configs/workflow.yaml
search:
  backend: arxiv
  sort_by: relevance  # 或 submittedDate / lastUpdatedDate
```

也可用 `ALGO_SEARCH_BACKEND=arxiv` 或 `http`；HTTP 后端读取
`ALGO_SEARCH_URL`，返回格式为 `{"results":[{"url":"...","title":"...","summary":"..."}]}`。
优先级：非空/显式 backend 环境配置、已有 HTTP URL、workflow 默认。
如需禁止已配置的 HTTP 检索，用 `ALGO_SEARCH_BACKEND=disabled`。

arXiv 使用官方 Atom API 和结构化 XML 解析；限制响应体、结果数量、超时，
同一平台多个进程间至少间隔 3 秒。结果保留标题、摘要、作者、发表/更新日期、
源地址及证据 ID。网络或解析失败明确记录“未取得证据”，不制造文献。
目前不自动抓取 PDF 全文，也不把摘要当已复现结论。
`submittedDate` 只影响排序，不代表已经覆盖所有前沿工作。

`TaskSpec.research_queries` 用于提供不含隐私的技术检索词，第一个用于初始检索，
其余供研究模型进一步调用工具。缺省只发送 domain 关键词，不发送完整业务描述。
显式启用公开检索代表允许发送技术查询；模型仍可能生成敏感关键词，
对保密任务应使用内部 HTTP 检索服务或关闭公开检索。

检索和工具观察按任务、轮次和输入键持久化，供中断重放；不是跨任务永久文献缓存。
新一轮可重新查询。`require_live_research=true` 时，启动前要求云端研究、检索、
编码开关及 CLI 均已配置；实际初始检索没有结果也会停止，不用离线模板冒充实时研究。
该开关是严格研发预检，尽管名字是 research，也会要求编码能力。

## 5. 数据、编码与评测

数据本身是可研究的实验变量，不仅是固定输入：
训练增强、重采样、过滤、自动标注、重标注以及符合授权的测试扩充都可由研究提出、
执行、评测，然后根据结果改进。详细 schema 见 [Auto-Process](auto_process.md)。

保护仍然生效：

- 自动标签只可作为弱训练标签，不升级为人类确认的最终真值。
- 数据源在任务创建时授权和固定哈希；模型不能自行读取任意新目录。
- 不允许对测试标签自动重标、为提分删样本或调整验收门槛。
- 新算法必须由真实编码执行器实现，不能用模板替代成功的代码开发。
- tabular 自动新建/修改方法增加独立的“小数据训练、保存权重、重载预测一致性”检查。
- 编码失败有有限修复；运行时失败进入下一轮研究，不当作有效指标。
- 只在相同开发评测指纹下比较成绩；新测试覆盖面不是免费的指标提升。
- 最终评测只加载选定权重，不重新训练；失败后不再针对该最终集继续优化。

## 6. 长时间运行与恢复

推荐独立服务承载任务，REPL 只负责交互：

```bash
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent service start
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent \
  --service-url http://127.0.0.1:8765 chat --auto-process
```

`/mode auto` 仅影响之后新建的任务。新任务和普通改进消息启动后台执行，
不让 REPL 随训练一直等待。`/ask` 仍需等待模型回答；
人工模式的 `/approve` 和直接 feedback 接口仍可能同步等待，未改为全面流式 UI。
没有后台服务时 embedded chat 的线程属于 CLI 进程，退出终端不保证任务继续；
不要将它作为长期无人值守部署方式。

恢复增强：

1. 服务 lifespan 启动时恢复非终态、未暂停的自主任务，不等首次 HTTP 请求。
2. 已领取但还没有 PID 的过期租约会重新排队。
3. 作业租约定期续期；结果和心跳必须匹配 owner + attempt，旧 worker 不能覆盖新结果。
4. Linux boot ID + 进程 start ticks 防止 PID 复用被误认为旧任务还活着。
5. 活进程不会仅因一次过期心跳就重复开跑；取消请求优先于成功回写。
6. 固定初始研究帧、完成的模型响应及工具观察可重用，减少 checkpoint 重放的重复调用。

仍不能称为完全无人值守的分布式调度系统：
活着的孤儿训练进程不能自动重新接管；崩溃在外部调用与持久化之间仍有不确定窗口；
预算预留可能保守滞留。最终评测的不可确定崩溃仍保守停止。
建议使用进程监护、持久化本地 SQLite、备份和告警，不声称 exactly-once 外部执行。
墙钟预算从任务创建时间计算，暂停时间也计入；超时任务不能靠恢复绕过预算。

## 7. 上下文和知识沉淀

控制数据库升至 schema 4，以加列/建表方式兼容旧数据库，未重置任务：

| 存储 | 内容 | 提示词中的使用 |
| --- | --- | --- |
| `messages` | task_id / conversation_id / role / content / metadata | 最近最多 16 条，单条与总体长度受限 |
| `research_steps` | 输入指纹、初始帧、检索、模型响应、工具观察、完成结果 | 同任务同轮次重放，不重复计已完成模型步骤 |
| `lessons` | 假设、控制、参数、seed、权重哈希、指标、失败项、数据操作、证据 | 同领域词项相关性召回，旧的相关负结果可优先于最新结果 |
| 图 checkpoint | 生命周期节点与控制状态 | 恢复流程，不承担全部知识记忆 |
| 任务文件 | 研究文档、计划、日志、数据版本、编码归档 | 用 ID / 路径关联，不把原始数据整体塞进提示词 |

每轮分析生成：

```text
<task>_experiment/docs/knowledge/
  README.md
  lesson_<hash>.json
  lesson_<hash>.md
```

Markdown 包括假设/对照、配置与种子、实际结果、失败、未验证解释、
评测指纹、来源引用及运行日志链接。JSON 可供其它工具消费。
`algo-agent memory --task <id>` / `/memory` / HTTP memory 接口查看召回结果；
`algo-agent export --task <id>` 同时导出知识索引和既有 DOCX。
文档导出已修复 task_id 与 task_slug 不同的情况，指标列按实际 TaskSpec 展示。

默认知识仅对本任务可检索。`memory.share_lessons: true` 允许之后记录的经验供
同领域任务召回；共享内容可能含需求摘要、代码配置和失败信息，开启前应确认保密范围。
该配置是发布策略，关闭不撤回已经标记 shared 的历史记录；没有新增逐条撤回 UI。
本地 Markdown 不会自动上传外部平台。文档跨任务共享时应连同授权的引用和复现材料传递。
复用资产会给出兼容性判定，读取正文要求 `upload_policy=open` 且哈希正确。
只检查复现包或用户提交的指标，不再称为已经独立执行复现。

这里使用有界摘要和词项检索，不是无限上下文，也不是语义向量库或已验证因果知识库。
共享经验只提供下一实验的依据，不自动成为另一任务的验收证据。

## 8. 运行入口和可观测性

```bash
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent preflight --strict
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent auto-process --spec configs/auto_process.example.json
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent memory --task auto_regression
```

第二条命令的相对 spec 路径按当前工作目录解析；示例本身是合成回归，不是业务验收配置。
`preflight` 不发起模型生成或验证认证；仅配置齐全时 `configuration_ready=true`。
真实研究/编码/标注质量仍需独立 smoke test 和业务实验确认。
配置角色变量见 `.env.example`，API key 不应写进任务文档。

本机默认 PATH 尚未安装 `algo-agent` 命令。已验证可在平台目录直接用源码入口：

```bash
cd /mnt/data/yangt/codes/QAA/algo_agent
PYTHONPATH=runtimes/agent_runtime/src:. /usr/local/bin/python3 -m algo_agent.cli.main preflight --strict
```

将最后的 `preflight --strict` 替换为 `service start`、`chat --auto-process`
或 `memory --task <id>` 即为对应入口。不要把预检缺失项视作已完成真实认证。

新增/补齐服务接口：
`GET /readiness`、`GET|POST /tasks/{id}/messages`、
`GET /tasks/{id}/memory|budget|models|evals`、`GET /jobs?task_id=...`。
服务模式的 CLI 不再以任务列表替代作业列表，也能看到真实模型和评测记录。
显式指定 service URL 时不会静默退回 embedded；自动发现只连接同一平台 root。
`/health` 是轻量进程健康接口，不再触发慢模型探测。

本轮没有新增 Web UI、团队权限系统、移动端或流式聊天。
竞品比较与可借鉴的研究/执行/交互模式仍见[修改前审查](current_capability_review.md)；
本轮优先实现可验证的流程可靠性和 CLI 可观测性，不以界面相似代替研发能力。

## 9. 资源与安全边界

`max_memory_mb` 对实验 worker 设置 Linux `RLIMIT_AS`；
`max_output_file_mb` 设置 `RLIMIT_FSIZE`。默认 0，不修改现有实验的资源行为。
前者是每个进程的虚拟地址空间限制，后者是单文件大小限制，
均不是 GPU 显存配额、整个进程树的内存限制或总磁盘配额。
实验环境会过滤常见 API_KEY/TOKEN/SECRET/PASSWORD 等凭证变量。

**这些改动不构成敌对代码安全沙箱。** 生成代码仍使用宿主用户权限；
强隔离、网络限制、文件系统权限、GPU 配额需在外部容器/执行环境落实。
服务默认回环监听，没有新增公网认证授权，不能直接裸露到公网。

`max_cloud_tokens` 覆盖研究及聊天/意图辅助调用的账本和保守缺省计费；
外部 Codex 自身 token 仅在事件提供时记录，不能靠该字段硬限制其消费。
Codex 仍受编码次数、修复次数及墙钟约束。未提供 usage 时不会记成零消耗。

## 10. 验证

2026-10-15 最终全量结果：**254 passed in 114.65s**，相对基线新增 60 项。
运行时 Python 语法、JSON/YAML 示例配置以及 CLI 帮助入口也已检查通过。

原有 194 项回归为修改前基线；新增
`tests/test_rd_hardening.py` 覆盖指标绑定、控制安全、多轮会话、模型分工、
租约/PID恢复、旧 worker 拒收、文献解析、计划修正与重放、知识隐私、
导出指标、编码契约及服务接口。全量命令：

```bash
PYTHONDONTWRITEBYTECODE=1 ALGO_CODEX_ALLOW_REAL=0 \
PYTHONPATH=runtimes/agent_runtime/src:. \
python3 -m pytest tests -o addopts='' -q -p no:cacheprovider
```

测试使用临时数据库、临时数据、真实本地子进程及模拟研究/编码服务。
它们不能证明真实云端研究质量、前沿方案覆盖率、标注准确率、复杂任务达标率、
多随机种子统计显著性或目标设备性能。
