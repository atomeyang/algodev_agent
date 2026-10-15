# IQA 任务文件索引

IQA 任务在平台根下只有本目录 `iqa_experiment/`。

| 目录 | 内容 |
|---|---|
| `plans/`、`data/`、`runs/`、`evaluation/`、`docs/` | 主任务的计划、数据、实验与分析 |
| `data/annotations/plans/` | 标注方案版本 |
| `data/annotations/runs/` | 标注原始响应、中间产物、一致性评测 |
| `data/annotations/runs/legacy/pilot/` | 原 `iqa_annotations/` 的真实试点产物 |
| `tracks/mos/` | 原 `iqa_mos_experiment/`，独立的真实 MOS 评分分支 |
| `history/legacy_rd/` | 原 `iqa_rd/iqa_rd_experiment/`，早期实跑 |
| `history/legacy_rd/legacy_platform/` | 原 `iqa_rd/` 的剩余旧骨架与说明 |
| `runtime/coding/workspaces/` | 后续编码临时工作区 |
| `runtime/migrations/consolidation.json` | 本次搬迁的逐文件哈希校验清单 |

主任务、MOS 分支和早期任务保留各自的任务 ID，不混合评测协议或达标结论。
每个分支自己的标注和评测产物存放在自己的 `data/annotations/` 与 `evaluation/` 中。

## 约束与迁移

- [自动创建的目录规则](DIRECTORY_RULES.md)
- [平台目录创建约束](../docs/task_directory_structure.md)
- [历史实跑报告](docs/iqa_run_report.md)
- [主任务历史摘要](iqa_run_summary.json)
- [MOS 历史摘要](tracks/mos/iqa_run_summary.json)
- [早期任务历史摘要](history/legacy_rd/iqa_run_summary.json)

2026-10-15 的迁移保留了原始文件内容，校验了 169 个搬迁文件。
旧报告、日志、JSON 中的旧路径仍是原始证据；新位置通过 `.task_layout.json`
的 `relocations` 以及 `TaskPaths.resolve_reference()` 查找。
目录归属通过 `members` 登记：`iqa_mos -> tracks/mos`，
`iqa_rd -> history/legacy_rd`。不要重建顶层旧目录或软链接别名。

这里保存的是历史运行结果，目录迁移本身不代表重新训练、重新验收或任务仍在运行。
