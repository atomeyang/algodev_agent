# IQA Dialogue Validation

Task: `iqa_dialogue_final_20261015`; phase: `stopped`; stop reason: task wall-clock budget exhausted

This is a live-provider workflow/replication run on an already exposed benchmark, not fresh business acceptance.

## Observed Workflow Checks

- question_did_not_pause_or_start: `True`

- explicit_pause: `True`

- explicit_resume: `True`

- live_cloud_design_completed: `True`

- live_search_returned_refs: `True`

- multiple_research_iterations: `True`

- live_code_applied: `False`

- data_operations_executed: `True`

- experiments_succeeded: `False`

- independent_development_metrics: `False`

- saved_weights: `False`

- final_evaluation_executed: `False`

## Measured Development Gates

````json

[
  {
    "name": "plcc",
    "threshold": 0.85,
    "direction": "max",
    "strict": false,
    "development_value": null,
    "development_meets": null
  },
  {
    "name": "srcc",
    "threshold": 0.85,
    "direction": "max",
    "strict": false,
    "development_value": null,
    "development_meets": null
  },
  {
    "name": "defect_macro_f1",
    "threshold": 0.85,
    "direction": "max",
    "strict": false,
    "development_value": null,
    "development_meets": null
  },
  {
    "name": "grade_agreement",
    "threshold": 0.9,
    "direction": "max",
    "strict": false,
    "development_value": null,
    "development_meets": null
  },
  {
    "name": "hardcase_macro_f1",
    "threshold": 0.85,
    "direction": "max",
    "strict": false,
    "development_value": null,
    "development_meets": null
  },
  {
    "name": "rom_mb",
    "threshold": 60.0,
    "direction": "min",
    "strict": true,
    "development_value": null,
    "development_meets": null
  }
]

````

## Evidence

- [Actual conversation](dialogue.md)

- [Machine-readable results and plans](summary.json)

- [Initial frozen contract and readiness](setup.json)

- [Research knowledge](../knowledge/README.md)

No flag above proves research quality, causal attribution or production readiness. Inspect the real research, code diff, training logs and evaluation records.
