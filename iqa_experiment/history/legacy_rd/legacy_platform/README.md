# 历史记录：布局修复前的第一次 IQA 实跑（legacy）

这个目录是 **第一次** IQA 端到端实跑留下的目录，产生于 `run_iqa_loop.py`
布局修复之前：当时脚本把 `--root` 当成了「任务目录」传入
（`--root ./iqa_rd --task-id iqa_rd`），于是 `TaskPaths` 又在其下拼了一层
`iqa_rd_experiment/`，同时 `init_platform` 还在 `iqa_rd/` 下建了一套空的
平台骨架（`algorithm_plugins/`、`runtimes/`、`shared_assets/` …）。

- 真正的任务资产在 `iqa_rd/iqa_rd_experiment/`（3 轮迭代、5 个模型、冻结评测结果）。
- 该次实跑的结论见 [`../docs/iqa_run_report.md`](../docs/iqa_run_report.md)。

## 正确的约定（指导说明 §4）

`--root` 是**平台根**，任务资产落在 `<平台根>/<task_name>_experiment/`：

```bash
python3 scripts/run_iqa_loop.py --root . --task-id iqa --max-iterations 3
# -> ./iqa_experiment/
```

修复后的实跑记录见 `../iqa_experiment/`。为保留真实证据，本目录（含那一套空骨架）
不做删除；新任务请不要再产生这种嵌套布局（`run_iqa_loop.py` 现在会在 `--root`
以 `_experiment` 结尾时自动上溯到平台根，并打印提示）。
