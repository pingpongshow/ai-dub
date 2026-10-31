"""Job queue and records (SQLite in appdata)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

QUEUED, RUNNING, WAITING, DONE, SKIPPED, FAILED, CANCELLED = (
    "queued", "running", "waiting", "done", "skipped", "failed", "cancelled")
REVIEW = "review"   # paused for the review screen; the queue moves on without it
ACTIVE = {QUEUED, RUNNING, WAITING, REVIEW}

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  path      TEXT NOT NULL,
  profile   TEXT NOT NULL,
  options   TEXT NOT NULL DEFAULT '{}',
  status    TEXT NOT NULL,
  position  REAL NOT NULL,
  step      TEXT NOT NULL DEFAULT '',
  progress  REAL NOT NULL DEFAULT 0,
  message   TEXT NOT NULL DEFAULT '',
  created   REAL NOT NULL,
  started   REAL,
  finished  REAL,
  record    TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS originals (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id    INTEGER,
  path      TEXT NOT NULL UNIQUE,   -- the set-aside file
  live_path TEXT NOT NULL,          -- the name it was set aside from
  kind      TEXT NOT NULL,          -- original | dubbed
  size      INTEGER NOT NULL,
  created   REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in ("options", "record"):
            if k in d:
                d[k] = json.loads(d[k] or "{}")
        return d

    # Jobs -----------------------------------------------------------------
    def add(self, path: str, profile: str, options: dict | None = None) -> int:
        with self._tx() as c:
            pos = c.execute("SELECT COALESCE(MAX(position), 0) + 1 FROM jobs").fetchone()[0]
            cur = c.execute(
                "INSERT INTO jobs (path, profile, options, status, position, created) VALUES (?,?,?,?,?,?)",
                (path, profile, json.dumps(options or {}), QUEUED, pos, time.time()))
            return int(cur.lastrowid)

    def get(self, job_id: int) -> dict | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, statuses: set[str] | None = None, limit: int = 500) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM jobs ORDER BY position, id LIMIT ?", (limit,)).fetchall()
        out = [self._row(r) for r in rows]
        return [j for j in out if not statuses or j["status"] in statuses]

    def active_for(self, path: str) -> dict | None:
        for j in self.list(ACTIVE):
            if j["path"] == path:
                return j
        return None

    def next_queued(self) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?, ?) ORDER BY position, id LIMIT 1",
                (QUEUED, WAITING)).fetchone()
        return self._row(r)

    def claim(self, job_id: int) -> bool:
        """Mark a queued job running; False if it was cancelled or removed meanwhile."""
        with self._tx() as c:
            cur = c.execute("UPDATE jobs SET status=? WHERE id=? AND status IN (?, ?)",
                            (RUNNING, job_id, QUEUED, WAITING))
            return cur.rowcount == 1

    def update(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        for k in ("options", "record"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._tx() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def move(self, job_id: int, direction: int) -> None:
        queued = [j for j in self.list({QUEUED, WAITING})]
        ids = [j["id"] for j in queued]
        if job_id not in ids:
            return
        i = ids.index(job_id)
        j = max(0, min(len(ids) - 1, i + direction))
        ids.insert(j, ids.pop(i))
        with self._tx() as c:
            for n, jid in enumerate(ids):
                c.execute("UPDATE jobs SET position=? WHERE id=?", (n + 1, jid))

    def requeue_front(self, job_id: int) -> None:
        """Back in the queue, ahead of everything still waiting (a reviewed job continues)."""
        with self._tx() as c:
            first = c.execute("SELECT COALESCE(MIN(position), 1) FROM jobs").fetchone()[0]
            c.execute("UPDATE jobs SET status=?, position=?, message=? WHERE id=?",
                      (QUEUED, first - 1, "continuing after review", job_id))

    def remove(self, job_id: int) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def recover(self) -> None:
        """After a restart, a job that was running resumes from its last finished step."""
        with self._tx() as c:
            c.execute("UPDATE jobs SET status=?, message='resuming after restart' WHERE status IN (?, ?)",
                      (QUEUED, RUNNING, WAITING))

    # Set-aside files ------------------------------------------------------
    def add_original(self, job_id: int | None, path: str, live_path: str, kind: str, size: int) -> None:
        with self._tx() as c:
            c.execute("INSERT OR REPLACE INTO originals (job_id, path, live_path, kind, size, created) "
                      "VALUES (?,?,?,?,?,?)", (job_id, path, live_path, kind, size, time.time()))

    def originals(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM originals ORDER BY created").fetchall()]

    def original(self, oid: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM originals WHERE id=?", (oid,)).fetchone()
        return dict(r) if r else None

    def drop_original(self, oid: int) -> None:
        with self._tx() as c:
            c.execute("DELETE FROM originals WHERE id=?", (oid,))
