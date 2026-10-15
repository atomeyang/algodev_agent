# Auto-Process

This mode removes approval waits, not the acceptance contract. It runs:

1. Validate the task's requirements and fixed thresholds.
2. Research with bounded search, evidence reading, source inspection, run summaries,
   aggregate data inspection and reusable-asset lookup.
3. Select candidates and data hypotheses; validate and automatically approve the plan hash.
4. Build a versioned development dataset and implement/revise candidate code.
5. Independently verify code, execute bounded jobs and evaluate stored predictions.
6. Diagnose failures and research again until development gates pass or a limit stops the task.
7. Reload the selected weights for a one-shot independent final evaluation.

The final state is `accepted` only with measured, finite passing results and a
saved model artifact. Failed runs, missing data, missing weights, undefined targets,
exhausted budgets and unavailable required providers are never treated as success.

## Entry Points

```bash
ALGO_OFFLINE=1 algo-agent --embedded auto-process "CPU rmse <= 0.5" --task auto_demo --demo
algo-agent auto-process --spec configs/auto_process.example.json
algo-agent auto-process --task auto_demo
```

The example spec is synthetic regression, not an IQA production configuration.
For IQA use `domain: "iqa"`, real PLCC/SRCC/defect targets and an appropriate benchmark
configuration. `IQA_BENCHMARK=koniq` selects the real-MOS track. Explicit structured
metrics are recommended for complex tasks; natural-language parsing is still a
domain parser, not a general requirements negotiation system.

HTTP: `POST /tasks` accepts `auto_process`, `task_spec`, `domain`, `methods_hint`
and `run`. The service and embedded CLI use the same controller.
The CLI returns exit code 0 only for acceptance; stopped/incomplete tasks return 2.

Mode, budget and method hints are task-scoped and persisted. Resuming a checkpoint
does not repeat completed jobs. Per-task process locks prevent competing controllers
from driving the same task concurrently. An existing manual task with plans must
use a new task ID: its previous test feedback cannot become unseen final evidence.

## Real Providers

- Research: configure `ALGO_CLOUD_API_KEY`, `ALGO_CLOUD_BASE_URL` and optionally
  `ALGO_CLOUD_MODEL`. Configure `ALGO_SEARCH_URL` for live source discovery.
- Coding: enable `ALGO_CODEX_ALLOW_REAL=1` and authenticate the installed CLI.
  New/modified algorithms cannot fall back to a template writer.
- IQA annotation: configure the existing local VLM endpoint. Automatic image
  annotation accepts only loopback endpoints. Private raw images are not sent
  to cloud research.
- Without these services, the deterministic offline demo can verify orchestration,
  but it is not autonomous literature research or evidence of algorithm discovery.

The coding prompt contains the task contract, candidate objective, prior measured
failures and plugin context. Actual filesystem hashes, not CLI event claims,
determine changes. Syntax/import checks run independently; failed verification
feeds a bounded repair attempt. Failed or out-of-scope patches do not replace the
task's baseline code. Runtime failures are subsequently fed into research.

## Data Experiments

Research emits `data_operations`, stored in `PlanSpec.data_recipe.operations`.
Every operation has `operation_id`, `kind`, `split`, `hypothesis`, optional
`source_id`, and concrete `params`. Recipes are **complete recipes applied to the
same frozen base**, not incremental patches to the previous round. Carry desired
earlier operations forward explicitly when making a new plan.

Supported operations:

| Kind | Domain / Parameters | Allowed Splits |
| --- | --- | --- |
| `augment` | Tabular: `count`, `noise_std`, `seed` | `train` |
| `augment` | IQA: `transform: "hflip"`, `label_preserving: true`, `count`, `seed` | `train`, `development_test` |
| `resample` | `count`, `seed`; tabular `slice: "low_light"`, `weight`; IQA `defect`, `weight` | `train` |
| `filter` | Tabular `feature`, `minimum`, `maximum`; IQA `score_min`, `score_max` | `train` |
| `annotate` | IQA local VLM: `limit`, `min_confidence`, optional `rubric` | `train` |
| `relabel` | IQA source JSONL labels keyed by image path | `train` |
| `extend` | Registered JSONL image records or NPZ `{X,y}` | all, with trust checks |

Example research output:

```json
{
  "data_operations": [
    {
      "operation_id": "balance_blur",
      "kind": "resample",
      "split": "train",
      "hypothesis": "More blur positives reduce the observed recall gap.",
      "params": {"defect": "blur", "weight": 2.0, "count": 600, "seed": 7}
    },
    {
      "operation_id": "severity_rubric",
      "kind": "annotate",
      "split": "train",
      "hypothesis": "A more explicit rubric reduces label inconsistency.",
      "params": {"limit": 30, "min_confidence": 0.9, "rubric": "Distinguish fine texture from blur."}
    }
  ]
}
```

The taxonomy/output schema remains fixed. Rubric changes get a prompt hash.
Automatic annotation must pass schema/finite-value/confidence checks and is always
recorded as **weak**, `review_required: true`. It can be used for training without
waiting for review, but it never becomes verified test ground truth. The agent
must measure whether the label intervention actually helped on development data.

External extension/relabel sources must be registered in the initial TaskSpec:

```json
{
  "data_sources": [
    {
      "source_id": "extra_human_labels",
      "path": "/absolute/path/to/labels.jsonl",
      "label_kind": "human",
      "independent": true
    }
  ]
}
```

The controller pins the source file checksum at intake. Research cannot grant
itself new filesystem sources, human-label status or independence. Final test
extension is allowed only with an initially authorised, independently labelled
`human`/`verified` source. Filtering, resampling and automatic relabelling of test
sets are rejected. Source image paths must exist locally.

Raw data is never modified. Releases record recipes, before/after counts, source
hashes, label provenance and annotation model/prompt information. IQA checks
path, subject and exact image-content overlap between splits; near-duplicate
detection beyond those checks is not implemented. Source image hashes are checked
again before/after jobs. Invalid data experiments feed failure information into
the next research round.

## Development Versus Final

For tabular data, original validation becomes the development test; original test
is held out. IQA uses the existing development train/validation split: only the
outer training rows are available to methods, with an inner split for tuning.
Outer validation labels and final labels are kept outside the method's release.

Changing development-test coverage changes its fingerprint. Scores from different
test fingerprints are not ranked as if they were the same benchmark. For the same
fingerprint, a weaker new candidate cannot replace a better prior result.

Final inference gets no training release. Method contract:

```python
def run(ctx, config):
    # Save a trained artifact inside ctx.out_dir.
    return {"weights_ref": "model.joblib"}

def predict(ctx, config, weights_path):
    # Reload exact weights, write predictions, never fit here.
    return {"inference_only": True}
```

Jobs snapshot method/plugin source, config and seed. Models have task/iteration
IDs and real weight hashes. Built-in tabular methods and the IQA MOS ridge/MLP
head support reload inference. Other IQA methods must implement artifact export
and `predict` before they can pass automatic final acceptance. The MOS head
records its frozen backbone/feature-cache dependency; that dependency must remain
available and unchanged.

Final test failure ends this task. Continuing R&D requires a fresh independent
holdout, not another round against exposed final-test results. A crash after
reserving final evaluation but before saving its result stops conservatively;
the system does not silently repeat that one-shot evaluation.

## Limits and Evidence

Limits include research rounds/model-tool steps, coding calls/repair attempts,
experiment count, configured epochs, wall time, no-improvement rounds, data
operations, added samples and annotation samples. Attempts are charged before
execution, including failures. One experiment slot is reserved for final inference.
`max_cloud_tokens` currently accounts for the research provider's calls, including
conservative charges when usage is unavailable. The external coding CLI's internal
token usage is not a hard-bounded part of that token ledger; its call count and
wall time are bounded. Model code must respect the supplied epoch config.

`pause` prevents further stages; running experiment processes may finish.
`cancel` terminates running process groups. Explicit pause/cancel always takes
precedence over auto-process.

This is local process isolation, **not an adversarial-code sandbox**. A generated
method still runs under the host account; use an externally restricted container
for untrusted code, secrets, network denial and hard GPU/RAM limits. Directory
separation and checksums detect accidental leakage/modification, but do not prove
that malicious code cannot read other host paths. Arbitrary domains need their
own data/evaluation/export adapters; built-in automatic data adapters are currently
tabular demo and IQA.

The regression suite uses temporary datasets, real subprocesses and test-double
research/coding/annotation services. It tests autonomous iteration, code generation
and repair plumbing, data operations, split isolation, independent evaluation,
budgets and cancellation. It does not establish real cloud research quality,
real annotation accuracy, multi-seed statistical significance, target-device
latency/ROM, or guaranteed achievement of an arbitrary business threshold.

## 任务产物归属

auto-process 的所有任务产物遵循 [任务目录结构约束](task_directory_structure.md)：
一个算法任务只有一个 `<task_name>_experiment/` 顶层目录。标注方案与原始响应、
数据扩增、每轮评测协议及中间结果、编码工作区和临时文件均放在任务内部。
相关分支使用 `--parent-task` 与 `--track` 显式登记，不另建同任务的顶层输出目录。
