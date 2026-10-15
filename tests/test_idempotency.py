"""Idempotency keys: a replayed command must not be executed twice (guide 11)."""

from __future__ import annotations

import uuid

from algo_agent.schemas.enums import Intent


def _task(controller) -> str:
    task_id = f"idem_{uuid.uuid4().hex[:8]}"
    controller.create_task("idempotency probe task", task_id=task_id, synthetic=True)
    return task_id


def test_replayed_command_returns_the_original_result(controller):
    task_id = _task(controller)
    key = f"key-{uuid.uuid4().hex}"
    first = controller.submit_command(task_id, Intent.QUERY.value, idempotency_key=key)
    second = controller.submit_command(task_id, Intent.QUERY.value, idempotency_key=key)
    assert second.get("idempotent_replay") is True
    assert second["command_id"] == first["command_id"]
    assert len([c for c in [controller.repo.get_command_by_key(key)] if c]) == 1


def test_distinct_keys_create_distinct_commands(controller):
    task_id = _task(controller)
    a = controller.submit_command(task_id, Intent.QUERY.value, idempotency_key=uuid.uuid4().hex)
    b = controller.submit_command(task_id, Intent.QUERY.value, idempotency_key=uuid.uuid4().hex)
    assert a["command_id"] != b["command_id"]


def test_repository_add_command_deduplicates_by_key(controller):
    from algo_agent.schemas import contracts as C
    from algo_agent.storage.repository import new_id

    key = f"repo-{uuid.uuid4().hex}"
    task_id = _task(controller)
    cmd = C.Command(command_id=new_id("cmd"), task_id=task_id, intent=Intent.QUERY.value,
                    idempotency_key=key)
    stored = controller.repo.add_command(cmd)
    again = controller.repo.add_command(C.Command(command_id=new_id("cmd"), task_id=task_id,
                                                  intent=Intent.QUERY.value, idempotency_key=key))
    assert stored.command_id == again.command_id == cmd.command_id
