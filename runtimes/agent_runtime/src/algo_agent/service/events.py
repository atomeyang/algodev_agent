"""Persistent event bus with polling and an SSE-ready stream (guide 3).

Event sequences are stored in SQLite so a client that disconnects can re-read
everything after a known ``event_seq``. The bus adds an in-process notification
hook so a local UI can react without busy-polling; it never prints raw training
log lines into the dialogue stream.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..storage.repository import Repository


class EventBus:
    def __init__(self, repo: Repository):
        self.repo = repo
        self._lock = threading.Lock()
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self._cv = threading.Condition(self._lock)
        self._last_seq = 0

    # ---- write path ---------------------------------------------------- #
    def publish(self, task_id: str, kind: str, message: str = "", level: str = "info",
                data: Optional[Dict[str, Any]] = None) -> int:
        seq = self.repo.append_event(task_id, kind, message, level=level, data=data)
        with self._cv:
            self._last_seq = max(self._last_seq, seq)
            self._cv.notify_all()
        event = {"event_seq": seq, "task_id": task_id, "kind": kind, "level": level,
                 "message": message, "data": data or {}}
        for cb in list(self._subs.get(task_id, [])):
            try:
                cb(event)
            except Exception:
                pass
        return seq

    def subscribe(self, task_id: str, cb: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs[task_id].append(cb)

    # ---- read path ----------------------------------------------------- #
    def poll(self, task_id: str, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        return self.repo.read_events(task_id, after=after, limit=limit)

    def wait_for(self, task_id: str, after: int, timeout: float = 1.0) -> List[Dict[str, Any]]:
        deadline = time.time() + timeout
        with self._cv:
            while self._last_seq <= after and time.time() < deadline:
                self._cv.wait(timeout=max(0.0, deadline - time.time()))
        return self.poll(task_id, after=after)

    def sse(self, task_id: str, after: int = 0, poll_s: float = 1.0,
            stop_when: Optional[Callable[[List[Dict[str, Any]]], bool]] = None) -> Iterator[str]:
        """Yield Server-Sent Events; safe to swap in for the polling client."""
        cursor = after
        while True:
            events = self.poll(task_id, after=cursor)
            for ev in events:
                cursor = ev["event_seq"]
                yield f"id: {cursor}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if stop_when is not None and stop_when(events):
                break
            if not events:
                time.sleep(poll_s)


__all__ = ["EventBus"]
