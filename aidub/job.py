"""The running job: working folder, checkpoints, logging, workers, cancel."""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import gpu
from .config import Config
from .workers import Worker


class Cancelled(Exception):
    pass


class Skipped(Exception):
    pass


class Paused(Exception):
    """The job waits for the user (the review screen); its work is kept and it resumes
    from the step that paused."""


class Job:
    def __init__(self, cfg: Config, job_id: int | str, src: Path, profile, options: dict | None = None,
                 *, on_status: Callable[[dict], None] | None = None,
                 cancel: threading.Event | None = None):
        self.cfg = cfg
        self.id = job_id
        self.src = Path(src)
        self.profile = profile
        self.options = options or {}
        self.work = cfg.work / f"job-{job_id}"
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_file = self.work / "state.json"
        self.state: dict = json.loads(self.state_file.read_text()) if self.state_file.exists() else {
            "done": [], "data": {}, "timings": {}}
        self.on_status = on_status or (lambda d: None)
        self.cancel_event = cancel or threading.Event()
        self.halt_event = threading.Event()  # container shutdown: stop, keep the work, resume later
        self.log_file = self.work / "job.log"
        self._worker: Worker | None = None
        self.step = ""
        self.models: dict[str, str] = self.state["data"].get("models", {})
        self.qc: dict = self.state["data"].get("qc", {})

    # Files ------------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dir(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.work.rglob("*") if f.is_file()) if self.work.exists() else 0

    # Checkpoints --------------------------------------------------------------
    def is_done(self, step: str) -> bool:
        return step in self.state["done"]

    def mark_done(self, step: str, seconds: float) -> None:
        if step not in self.state["done"]:
            self.state["done"].append(step)
        self.state["timings"][step] = round(seconds, 1)
        self.save()

    def save(self) -> None:
        self.state["data"]["models"] = self.models
        self.state["data"]["qc"] = self.qc
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        tmp.replace(self.state_file)

    def __getitem__(self, key: str) -> Any:
        return self.state["data"][key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.state["data"].get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state["data"][key] = value
        self.save()

    # Status -------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def status(self, message: str = "", progress: float | None = None, **extra: Any) -> None:
        d = {"step": self.step, "message": message, **extra}
        if progress is not None:
            d["progress"] = max(0.0, min(1.0, progress))
        if message:
            self.log(f"[{self.step}] {message}")
        self.on_status(d)

    def check(self) -> None:
        if self.cancel_event.is_set() or self.halt_event.is_set():
            raise Cancelled()

    # Workers and the GPU --------------------------------------------------------
    def wait_gpu(self, stage: str) -> None:
        need = int((self.cfg.get("gpu.min_free_mb") or {}).get(stage, 4000))
        waited = False

        def on_wait(msg: str) -> None:
            nonlocal waited
            waited = True
            self.status(msg, waiting=True)

        gpu.wait_for(need, poll_s=int(self.cfg.get("gpu.poll_seconds", 30)), on_wait=on_wait,
                     cancelled=self.cancel_event.is_set, enabled=bool(self.cfg.get("gpu.wait", True)))
        self.check()
        if waited:
            self.status("GPU free, continuing", waiting=False)

    def worker(self, stage: str, *, gpu_stage: str | None = None) -> Worker:
        """Start the worker configured for `stage`. Caller uses it as a context manager."""
        name = self.cfg.get(f"stages.{stage}.worker")
        if not name:
            raise RuntimeError(f"no worker configured for stage '{stage}'")
        self.wait_gpu(gpu_stage or stage)
        w = Worker(self.cfg, name, log=self.log)
        self._worker = w
        return w

    def kill_worker(self) -> None:
        if self._worker:
            self._worker.kill()

    def remove_work(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)
