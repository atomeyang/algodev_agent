# 算法任务目录创建约束

## 唯一归属

每个算法任务只能有一个顶层目录：`<规范化任务名>_experiment/`。
同一任务从需求分析、调研、计划筛选、编码、训练、数据扩增、标注、
测试、结果分析到交付的全部生成文件及中间目录，均归属于此目录。
不同模型、数据版本、标注方案或评测轨道不构成另建顶层任务目录的理由。

平台 `--root` 始终指平台根；传入任务目录或其子目录时，解析回平台根，
不得在任务内再初始化平台骨架。任务 slug 不带 `_experiment` 后缀。

## 标准结构

| 任务内路径 | 用途 |
|---|---|
| `task.yaml`、`plans/<version>/` | 需求、策略、计划候选、筛选、审批记录 |
| `docs/research/`、`evidence/` | 调研报告、来源证据、检索中间结果 |
| `data/raw/`、`data/recipes/`、`data/splits/` | 来源登记、扩增配方、划分 |
| `data/releases/<version>/` | 不可变训练/开发数据版本和校验清单 |
| `data/annotations/plans/<version>/<attempt>.json` | 标注 rubric、提示词版本、模型与操作方案 |
| `data/annotations/runs/<version>/<attempt>/` | 原始模型响应、被拒标签、失败记录、一致性报告 |
| `data/annotations/reviews/` | 复核结果、标签调整依据 |
| `evaluation/protocols/<version>/` | 各版评测协议、冻结标签 |
| `evaluation/scripts/`、`evaluation/intermediate/` | 评测实现与辅助中间产物 |
| `evaluation/results/<eval_id>/` | 每次指标、切片、资源测量和验收结果 |
| `model_pool/`、`runs/<run_id>/`、`model_versions/` | 实现、代码快照、训练日志、预测、权重与版本 |
| `runtime/coding/workspaces/`、`coding_records/` | 隔离编码工作区及修改验证记录 |
| `runtime/tmp/`、`runtime/cache/`、`runtime/state/` | 临时文件、缓存、任务状态导出 |
| `docs/iterations/`、`feedback/`、`deliverables/` | 分析、反馈、交付 |
| `tracks/<track>/`、`history/<name>/` | 内部分支、历史记录，内部可使用同样的资产结构 |

已有任务内的缓存或历史文件不因采用这份规范而强制改写其内容。
标注每次运行使用独立 ID，禁止默认覆盖上一次结果。失败响应也需留存。
目录收拢不改变训练/开发/最终测试隔离，不允许将自动弱标签当成最终测试真值。

## 实现约束

- 所有平台管理的任务输出由 `TaskPaths` 定位；显式 `--out` 也必须通过归属检查。
- `safe_join` 和 `output_path` 拒绝 `..` 越界、外部绝对路径及指向外部的符号链接。
- `ensure_task_dirs` 创建完整目录和 `DIRECTORY_RULES.md`；目录规则同时传入研究和编码请求。
- 编码工作区以及实验/编码子进程的临时目录留在任务内，不写平台级实验输出目录。
- 内部分支通过所属任务 `.task_layout.json` 的 `members` 显式登记。
  根据名字相似度自动合并会混淆任务身份，因此禁止隐式猜测。
- 一个成员不能归属于多个父任务；若成员路径和旧顶层目录同时存在，拒绝执行。
- 调度数据库仍是平台共享基础设施；任务创建和运行边界会导出
  `runtime/state/snapshot.json`，但该快照不冒充完整 LangGraph 恢复数据库。
- 任务包默认导出到 `deliverables/bundles/<bundle_id>/`，复现包到 `deliverables/reproduction/`。
  显式输出路径也不能离开本任务的 `deliverables/`，导出不得递归打包旧任务包或运行临时目录。

新建独立任务：`algo-agent auto-process --task example --spec requirement.json`。
在已登记父任务内部新建分支：

```bash
algo-agent auto-process --task example_variant --parent-task example --track variant --spec requirement.json
```

独立标注：

```bash
python3 scripts/iqa_annotate.py --root . --task iqa --version v2 --split dev --limit 20
```

## 共享资源边界

Agent 源码、可复用插件、平台配置、共享调度数据库、通用辅助大模型及公共原始数据集，
可以保留在共享位置；它们不是某次任务生成的产物。
任务应在自己的来源清单记录实际使用的路径、版本和校验值。
不得把任务的新标注、扩增图片、模型训练输出或评测缓存写回共享来源。
原始数据和预训练模型不会仅因目录整理而复制一份或删除外部原件。

SQLite 在不支持文件锁的挂载盘上仍可使用已有本地回退目录；
这是数据库的存储限制，不允许据此把实验、标注和编码文件散落到平台外。

## 已有 IQA 迁移

`scripts/consolidate_iqa.py` 默认为预览，`--apply` 才执行：

| 旧位置（平台相对路径） | 新位置（`iqa_experiment/` 内） |
|---|---|
| `iqa_annotations/` | `data/annotations/runs/legacy/pilot/` |
| `iqa_mos_experiment/` | `tracks/mos/` |
| `iqa_rd/iqa_rd_experiment/` | `history/legacy_rd/` |
| `iqa_rd/iqa_run_summary.json` | `history/legacy_rd/iqa_run_summary.json` |
| `iqa_rd/` 的剩余旧平台骨架 | `history/legacy_rd/legacy_platform/` |
| `docs/iqa_run_report.md` | `docs/iqa_run_report.md` |

迁移不合并同名文件、不删除实验、不重算历史指标、不修改原始 JSON/DOCX 内容；
逐文件哈希校验清单保存在 `runtime/migrations/consolidation.json`。
原记录中的路径属于历史证据，使用 `TaskPaths.resolve_reference` 解析搬迁后的实际位置。
对外部程序直接使用的旧绝对路径，需要按映射更新调用方；不保留平台根软链接别名。
历史任务仍需原有调度数据库才能原样续跑，文件迁移不会自动合并不同状态库。
