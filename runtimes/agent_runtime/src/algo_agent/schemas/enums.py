"""Enumerations shared across the agent platform.

Task phase and job status are deliberately separate (guide 6): a single
status field must never conflate lifecycle phase with execution state.
"""

from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = "1.0"
# LangGraph topology version. State must not be blindly restored across a
# different graph version (guide 12.3).
GRAPH_VERSION = "1.0"


class TaskPhase(str, Enum):
    DRAFT = "draft"
    RESEARCHING = "researching"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    PAUSED = "paused"
    ACCEPTED = "accepted"
    STOPPED = "stopped"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class AgentRole(str, Enum):
    RESEARCH_DESIGN = "research_design"
    CODEX_CODING = "codex_coding"
    LOCAL_DIAGNOSIS = "local_diagnosis"


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_PLAN_CHANGE = "needs_plan_change"
    BLOCKED = "blocked"
    FAILED = "failed"


class Intent(str, Enum):
    NEW_REQUIREMENT = "new_requirement"
    REQUIREMENT_REVISION = "requirement_revision"
    FEEDBACK = "feedback"
    QUERY = "query"
    APPROVAL = "approval"
    CONTROL = "control"
    UNKNOWN = "unknown"


class ApprovalMode(str, Enum):
    SCOPED_AUTO = "scoped_auto"
    PER_ITERATION = "per_iteration"


class Compatibility(str, Enum):
    COMPATIBLE = "compatible"
    ADAPTATION_REQUIRED = "adaptation_required"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class AssetStatus(str, Enum):
    RESEARCHED = "researched"
    REPRODUCED = "reproduced"
    BUSINESS_VALIDATED = "business_validated"
    REUSABLE = "reusable"
    DEPRECATED = "deprecated"


class AssetType(str, Enum):
    COMPONENT = "component"
    RECIPE = "recipe"
    DATASET = "dataset"
    MODEL = "model"
    EVALUATION = "evaluation"
    KNOWLEDGE = "knowledge"


class ImportStatus(str, Enum):
    READY = "ready"
    MISSING_DEPENDENCIES = "missing_dependencies"
    UNSUPPORTED_STATE = "unsupported_state"
    FAILED = "failed"
