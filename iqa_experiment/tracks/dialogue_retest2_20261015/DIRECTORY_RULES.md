# Task Directory Contract

- One algorithm task owns exactly one top-level <task_slug>_experiment directory.
- All task-generated files, including temporary files, belong inside that directory.
- Keep requirements/plans in plans/, research in docs/research/ and evidence/.
- Keep annotation schemes in data/annotations/plans/<version>/; keep raw responses,
  rejected labels, reviews and agreement reports in data/annotations/runs/<version>/<run_id>/.
- Keep data recipes, splits and immutable releases in data/; never overwrite a release.
- Keep evaluation protocols, scripts, intermediate outputs and results in evaluation/.
  Preserve development/final-test isolation and never tune on final-test feedback.
- Keep experiment code, logs and predictions in runs/<run_id>/; methods in model_pool/.
- Keep coding workspaces and temporary/cache files in runtime/, records in coding_records/.
- Related tracks live in tracks/<track>/ and old runs in history/, not sibling task roots.
- Use TaskPaths for output paths. Do not create per-task output under the platform root,
  platform runtimes/, plugin directories, shared_assets/ or local_largemodel/.
- Platform code/services and genuinely shared read-only datasets/models may live outside
  the task; record their location/version/checksum in data/raw or runtime dependencies.
- The shared scheduler database/checkpointer is platform infrastructure, not an output
  directory. Task-local state exports do not replace its live transactions.
