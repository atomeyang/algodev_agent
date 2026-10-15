"""Persistent SQLite checkpointer adapter (guide 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings


def make_checkpointer(settings: Settings) -> Any:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(settings.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


__all__ = ["make_checkpointer"]
