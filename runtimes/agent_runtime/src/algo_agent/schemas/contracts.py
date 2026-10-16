"""Canonical, schema_versioned data contracts (guide section 5).

All structures carry ``schema_version``; IDs are separate from file names and
timestamps are timezone-aware ISO strings stored in UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    SCHEMA_VERSION,
    ApprovalMode,
    AssetStatus,
    AssetType,
    Compatibility,
    ImportStatus,
    Intent,
    JobStatus,
    ResultStatus,
    TaskPhase,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Budget / metrics / constraints
# --------------------------------------------------------------------------- #
class Budget(Contract):
    max_research_calls: int = 3
    max_coding_calls: int = 4
    max_fix_attempts: int = 3
    max_experiments: int = 24
    max_epochs: int = 200
    max_wall_clock_s: float = 3600.0
    max_cloud_tokens: int = 400_000
    max_no_improve: int = 4
    max_iterations: int = 6
    max_research_steps: int = Field(default=8, ge=1, le=50)
    max_candidates: int = Field(default=3, ge=1, le=10)
    max_data_operations: int = Field(default=24, ge=0)
    max_added_samples: int = Field(default=10000, ge=0)
    max_annotation_samples: int = Field(default=1000, ge=0)
    max_memory_mb: int = Field(default=0, ge=0)
    max_output_file_mb: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def nonnegative_limits(self):
        import math
        for name in type(self).model_fields:
            if name.startswith("max_"):
                value = getattr(self, name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
        return self


class DataSource(Contract):
    source_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,79}$")
    path: str
    checksum: str = ""
    label_kind: Literal["human", "verified", "weak", "unlabelled"] = "unlabelled"
    independent: bool = False


class DataOperation(Contract):
    """An executable hypothesis, not an instruction to mutate raw data."""
    operation_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,79}$")
    kind: Literal["augment", "resample", "filter", "annotate", "relabel", "extend"]
    split: Literal["train", "development_test", "final_test"] = "train"
    hypothesis: str = Field(min_length=1)
    source_id: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class MetricTarget(Contract):
    name: str
    direction: Literal["min", "max"] = "min"
    threshold: Optional[float] = None
    unit: str = ""
    protocol_id: str = ""
    measured_on: Optional[str] = None
    unknown: bool = False
    strict: bool = False

    @model_validator(mode="after")
    def finite_threshold(self):
        import math
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("metric threshold must be finite")
        return self

    def better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "min" else a > b

    def meets(self, value: float) -> Optional[bool]:
        if self.threshold is None:
            return None
        if self.strict:
            return value < self.threshold if self.direction == "min" else value > self.threshold
        return value <= self.threshold if self.direction == "min" else value >= self.threshold


class HardConstraint(Contract):
    name: str
    value: Any = None
    unit: str = ""
    measurement_scope: str = ""
    blocking: bool = True


class TaskSpec(Contract):
    task_id: str
    task_slug: str
    # Business domain -> algorithm_plugins/<domain>/plugin.py (guide 4.1).
    domain: str = "demo_tabular"
    business: str = ""
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    target_hardware: Optional[str] = None
    precision: Optional[str] = None
    hard_constraints: List[HardConstraint] = Field(default_factory=list)
    primary_metric: MetricTarget
    secondary_metrics: List[MetricTarget] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    unknowns: List[str] = Field(default_factory=list)
    demo: bool = False
    synthetic: bool = False
    source: str = "natural_language"
    auto_process: bool = False
    require_live_research: bool = False
    research_queries: List[str] = Field(default_factory=list, max_length=4)
    methods_hint: List[str] = Field(default_factory=list)
    data_sources: List[DataSource] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Plan / approval
# --------------------------------------------------------------------------- #
class Candidate(Contract):
    candidate_id: str
    method: str
    family: str = ""
    description: str = ""
    expected_gain: str = ""
    cost_estimate: str = ""
    failure_conditions: List[str] = Field(default_factory=list)
    ablations: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    is_baseline: bool = False
    source: str = ""  # open_source | reference | local
    action: Literal["reuse", "create", "modify"] = "reuse"
    implementation_goal: str = ""
    base_method: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class DataRecipe(Contract):
    recipe_id: str
    description: str = ""
    source: str = ""
    transforms: List[str] = Field(default_factory=list)
    splits: Dict[str, float] = Field(default_factory=dict)
    seed: int = 0
    augmentation: List[str] = Field(default_factory=list)
    isolation: str = ""
    operations: List[DataOperation] = Field(default_factory=list)


class EvalProtocol(Contract):
    protocol_id: str
    primary_metric: str
    direction: str = "min"
    threshold: Optional[float] = None
    slices: List[str] = Field(default_factory=list)
    frozen: bool = True
    dataset_version: str = ""
    code_hash: str = ""
    notes: str = ""
    stage: Literal["development", "final", "legacy"] = "legacy"


class DagStep(Contract):
    step_id: str
    kind: str  # coding | experiment | evaluation | analysis
    depends_on: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


class ApprovalScope(Contract):
    allowed_methods: List[str] = Field(default_factory=list)
    allowed_data_sources: List[str] = Field(default_factory=list)
    allowed_recipe_changes: List[str] = Field(default_factory=list)
    parameter_space: Dict[str, Any] = Field(default_factory=dict)
    write_paths: List[str] = Field(default_factory=list)
    evaluation_protocol_id: str = ""
    upload_policy: str = "restricted"
    resource_budget: Budget = Field(default_factory=Budget)
    max_iterations: int = 6
    approval_mode: str = ApprovalMode.SCOPED_AUTO.value


class StopRules(Contract):
    require_all_hard_constraints: bool = True
    min_metric_delta: float = 0.0
    max_no_improve: int = 4


class PlanSpec(Contract):
    plan_id: str
    task_id: str
    version: str = "v1"
    parent_plan_id: Optional[str] = None
    summary: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    candidates: List[Candidate] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    controls: List[str] = Field(default_factory=list)
    data_recipe: DataRecipe
    eval_protocol: EvalProtocol
    execution_dag: List[DagStep] = Field(default_factory=list)
    change_scope: ApprovalScope = Field(default_factory=ApprovalScope)
    cost_estimate: str = ""
    stop_rules: StopRules = Field(default_factory=StopRules)
    created_at: str = Field(default_factory=utcnow)


class Approval(Contract):
    approval_id: str
    task_id: str
    plan_id: str
    plan_hash: str
    scope: ApprovalScope
    mode: str = ApprovalMode.SCOPED_AUTO.value
    approved_at: str = Field(default_factory=utcnow)
    state_version: int = 0


class Command(Contract):
    command_id: str
    task_id: Optional[str] = None
    conversation_id: str = ""
    intent: str = Intent.UNKNOWN.value
    text: str = ""
    expected_state_version: Optional[int] = None
    idempotency_key: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Data / coding / experiment / model / eval
# --------------------------------------------------------------------------- #
class DatasetManifest(Contract):
    dataset_id: str
    version: str
    source: str = ""
    raw_refs: List[str] = Field(default_factory=list)
    annotations: str = ""
    splits: Dict[str, int] = Field(default_factory=dict)
    recipe_id: str = ""
    code_ref: str = ""
    seed: int = 0
    checksums: Dict[str, str] = Field(default_factory=dict)
    external_checksums: Dict[str, str] = Field(default_factory=dict)
    license: str = ""
    upload_policy: str = "restricted"
    synthetic: bool = False
    task_id: str = ""
    provenance: List[Dict[str, Any]] = Field(default_factory=list)
    stage: str = "legacy"
    created_at: str = Field(default_factory=utcnow)


class CodingTaskSpec(Contract):
    coding_id: str
    task_id: str
    plan_id: str
    plan_hash: str = ""
    objective: str
    baseline_commit: str = ""
    workspace: str = ""
    write_paths: List[str] = Field(default_factory=list)
    acceptance_commands: List[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    context_files: List[str] = Field(default_factory=list)
    entry: str = ""
    require_real: bool = False


class Verification(Contract):
    command: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""


class CodingResult(Contract):
    coding_id: str
    status: str = ResultStatus.COMPLETED.value
    diff: str = ""
    changed_files: List[str] = Field(default_factory=list)
    tool: str = ""
    model: str = ""
    base_version: str = ""
    dependency_lock: str = ""
    verifications: List[Verification] = Field(default_factory=list)
    cost: Dict[str, Any] = Field(default_factory=dict)
    open_issues: List[str] = Field(default_factory=list)
    change_requests: List["ChangeRequest"] = Field(default_factory=list)
    archive_ref: str = ""
    archived_at: str = Field(default_factory=utcnow)


class ExperimentSpec(Contract):
    run_id: str
    task_id: str
    plan_id: str
    candidate_id: str
    method: str
    code_ref: str = ""
    data_version: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    env: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    hypothesis: str = ""
    parent_run_id: Optional[str] = None
    inputs_frozen: bool = True
    synthetic: bool = False


class ModelManifest(Contract):
    model_id: str
    task_id: str
    run_id: str
    code_ref: str = ""
    data_version: str = ""
    config_ref: str = ""
    env: str = ""
    seed: int = 0
    parent_model: Optional[str] = None
    weights_ref: str = ""
    weights_hash: str = ""
    export: Dict[str, Any] = Field(default_factory=dict)
    status: str = "candidate"  # candidate | recommended
    recommended: bool = False
    created_at: str = Field(default_factory=utcnow)


class EvalResult(Contract):
    eval_id: str
    task_id: str
    model_id: str
    protocol_id: str
    dataset_version: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    slices: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    resources: Dict[str, Any] = Field(default_factory=dict)
    failures: List[str] = Field(default_factory=list)
    passed: Optional[bool] = None
    synthetic: bool = False
    notes: str = ""
    stage: Literal["development", "final", "legacy"] = "legacy"
    created_at: str = Field(default_factory=utcnow)


class Feedback(Contract):
    feedback_id: str
    task_id: str
    model_id: Optional[str] = None
    source: str = "user"
    scenario: str = ""
    expected: str = ""
    actual: str = ""
    severity: str = "medium"
    repro: str = ""
    conditions: Dict[str, Any] = Field(default_factory=dict)
    upload_scope: str = "restricted"
    reproduced: Optional[bool] = None
    missing_evidence: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Evidence / agents
# --------------------------------------------------------------------------- #
class EvidenceRef(Contract):
    evidence_id: str
    uri: str
    locator: str = ""
    version: str = ""
    checksum: str = ""
    scope: str = "task"
    upload_policy: str = "restricted"
    source_type: str = "web"  # web | repo | paper | local | run
    trust: str = "medium"
    accessed_at: str = Field(default_factory=utcnow)


class EvidencePacket(Contract):
    packet_id: str
    task_id: str
    plan_id: str = ""
    baseline_id: str = ""
    requirement: str = ""
    metric_delta: Dict[str, float] = Field(default_factory=dict)
    slices: Dict[str, Any] = Field(default_factory=dict)
    failures: List[str] = Field(default_factory=list)
    counterexamples: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    refs: List[EvidenceRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)


class ReadAudit(Contract):
    audit_id: str
    role: str
    invocation_id: str = ""
    refs: List[str] = Field(default_factory=list)
    requested_range: str = ""
    actual_range: str = ""
    upload_bytes: int = 0
    status: str = "ok"
    usage: Dict[str, Any] = Field(default_factory=dict)


class Handoff(Contract):
    handoff_id: str
    from_role: str
    to_role: str
    objective: str
    input_asset_ids: List[str] = Field(default_factory=list)
    acceptance: List[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    parent_invocation: str = ""


class AgentInvocation(Contract):
    invocation_id: str
    task_id: str
    role: str
    plan_id: str = ""
    session_id: str = ""
    budget: Budget = Field(default_factory=Budget)
    archive_ref: str = ""
    created_at: str = Field(default_factory=utcnow)


class ChangeRequest(Contract):
    reason: str
    affected_constraints: List[str] = Field(default_factory=list)
    old_fields: Dict[str, Any] = Field(default_factory=dict)
    new_fields: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    estimated_cost: str = ""
    suggested_next_action: str = ""


class AgentResult(Contract):
    invocation_id: str
    status: str = ResultStatus.COMPLETED.value
    summary: str = ""
    assets: Dict[str, Any] = Field(default_factory=dict)
    verification: List[str] = Field(default_factory=list)
    cost: Dict[str, Any] = Field(default_factory=dict)
    change_requests: List[ChangeRequest] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Assets / reproduction / migration (v1.2)
# --------------------------------------------------------------------------- #
class AssetManifest(Contract):
    asset_id: str
    asset_version: str
    asset_type: str = AssetType.RECIPE.value
    schema_version: str = SCHEMA_VERSION
    source_task: str = ""
    artifact_refs: List[str] = Field(default_factory=list)
    checksums: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    license: str = ""
    upload_policy: str = "restricted"
    validation_status: str = AssetStatus.RESEARCHED.value
    validated_at: Optional[str] = None
    created_at: str = Field(default_factory=utcnow)


class RecipeSpec(Contract):
    recipe_id: str
    version: str = "v1"
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    applicable_distribution: str = ""
    preconditions: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    data_recipe: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    environment: str = ""
    weights_ref: str = ""
    evaluation_protocol: str = ""
    device: str = ""
    result_range: str = ""
    failure_boundary: str = ""
    evidence_refs: List[str] = Field(default_factory=list)


class CompatibilityResult(Contract):
    status: str = Compatibility.UNKNOWN.value
    reasons: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    adaptation: List[str] = Field(default_factory=list)
    verification_actions: List[str] = Field(default_factory=list)


class ReproductionManifest(Contract):
    manifest_id: str
    task_id: str
    asset_dependencies: List[str] = Field(default_factory=list)
    lock_file: str = ""
    data_refs: List[str] = Field(default_factory=list)
    weights_refs: List[str] = Field(default_factory=list)
    seed: int = 0
    exec_entries: List[str] = Field(default_factory=list)
    eval_entries: List[str] = Field(default_factory=list)
    expected_results: Dict[str, float] = Field(default_factory=dict)
    tolerance: float = 1e-6
    created_at: str = Field(default_factory=utcnow)


class ImportResult(Contract):
    status: str = ImportStatus.FAILED.value
    bundle_id: str = ""
    task_id: str = ""
    source_task: str = ""
    mapping: Dict[str, str] = Field(default_factory=dict)
    missing: List[str] = Field(default_factory=list)
    events: List[str] = Field(default_factory=list)
    new_run_ids: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runtime state
# --------------------------------------------------------------------------- #
class TaskStateSnapshot(Contract):
    task_id: str
    phase: str = TaskPhase.DRAFT.value
    iteration: int = 0
    state_version: int = 0
    active_plan_id: Optional[str] = None
    pending_approval: Optional[Dict[str, Any]] = None
    best_model_id: Optional[str] = None
    stop_reason: Optional[str] = None
    paused: bool = False
    budget_used: Dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utcnow)


class JobRecord(Contract):
    job_id: str
    task_id: str
    kind: str = "experiment"
    status: str = JobStatus.QUEUED.value
    idempotency_key: str = ""
    run_id: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    lease_owner: Optional[str] = None
    attempts: int = 0
    pid: Optional[int] = None
    pid_start: Optional[str] = None
    heartbeat_at: Optional[str] = None
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)


class Event(Contract):
    event_seq: int = 0
    task_id: str = ""
    kind: str = ""
    level: str = "info"
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)


CodingResult.model_rebuild()
