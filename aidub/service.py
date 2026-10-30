"""The queue runner: one job at a time on the GPU, in queue order."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import jobs as J
from . import media, originals, pipeline
from .config import Config
from .job import Cancelled, Job, Skipped
from .profiles import Profiles, guess_name

log = logging.getLogger("aidub")


class Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.appdata.mkdir(parents=True, exist_ok=True)
        cfg.work.mkdir(parents=True, exist_ok=True)
        self.store = J.Store(cfg.appdata / "aidub.db")
        self.profiles = Profiles(cfg.appdata)
        self.live: dict[int, dict] = {}            # job id -> latest status from the running job
        self._cancel: dict[int, threading.Event] = {}
        self._current: Job | None = None
        self._remove_after: set[int] = set()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # Paths ----------------------------------------------------------------------------------
    def media_roots(self) -> list[Path]:
        return [Path(m["path"]) for m in self.cfg.media_roots]

    def check_path(self, path: str) -> Path:
        """Only files inside the configured media folders are accepted."""
        p = Path(path).resolve()
        for root in self.media_roots():
            try:
                p.relative_to(root.resolve())
                return p
            except ValueError:
                continue
        raise PermissionError("outside the media folders")

    # Queue ----------------------------------------------------------------------------------
    def add(self, path: str, profile: str | None = None, options: dict | None = None) -> int:
        p = self.check_path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        if p.suffix.lower() not in media.VIDEO_EXT:
            raise ValueError("not a video file")
        if self.store.active_for(str(p)):
            raise ValueError("already in the queue")
        name = profile or guess_name(p, self.media_roots())
        prof = self.profiles.get(name) or self.profiles.get_or_create(
            name, {"honorifics": self.cfg.get("translate.honorifics", "keep"),
                   "style": self.cfg.get("translate.style", "natural")})
        jid = self.store.add(str(p), prof.slug, options or {})
        self._wake.set()
        return jid

    def inspect(self, path: str) -> dict:
        """What the queue form needs: audio tracks, skip-rule verdict, guessed profile."""
        from .ingest import classify
        p = self.check_path(path)
        pr = media.probe(p)
        info = classify(pr)
        info["profile_guess"] = guess_name(p, self.media_roots())
        info["duration"] = pr.duration
        info["container"] = pr.container
        return info

    def cancel(self, jid: int) -> None:
        j = self.store.get(jid)
        if not j:
            return
        if j["status"] in (J.QUEUED, J.WAITING) and jid not in self._cancel:
            self._finish_cancel(jid)
        ev = self._cancel.get(jid)
        if ev:
            ev.set()
            if self._current and self._current.id == jid:
                self._current.kill_worker()

    def _finish_cancel(self, jid: int) -> None:
        shutil.rmtree(self.cfg.work / f"job-{jid}", ignore_errors=True)
        self.store.update(jid, status=J.CANCELLED, finished=time.time(), message="cancelled")

    def retry(self, jid: int) -> None:
        j = self.store.get(jid)
        if j and j["status"] in (J.FAILED, J.CANCELLED) and not self.store.active_for(j["path"]):
            self.store.update(jid, status=J.QUEUED, message="retrying", progress=0)
            self._wake.set()

    def remove(self, jid: int) -> None:
        j = self.store.get(jid)
        if not j:
            return
        if j["status"] in J.ACTIVE:
            self.cancel(jid)
            if jid in self._cancel:
                self._remove_after.add(jid)  # the runner removes it when it stops
                return
        shutil.rmtree(self.cfg.work / f"job-{jid}", ignore_errors=True)
        self.store.remove(jid)

    def work_size(self, jid: int) -> int:
        d = self.cfg.work / f"job-{jid}"
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0

    # Runner ---------------------------------------------------------------------------------
    def start(self) -> None:
        self.store.recover()
        self._thread = threading.Thread(target=self._loop, name="aidub-runner", daemon=True)
        self._thread.start()
        threading.Thread(target=self._scan_originals, daemon=True).start()

    def stop(self) -> None:
        """Container shutdown: the running job stays 'running' and resumes after restart."""
        self._stop.set()
        self._wake.set()
        if self._current:
            self._current.halt_event.set()
            self._current.kill_worker()

    def _scan_originals(self) -> None:
        try:
            originals.scan(self.store, self.media_roots())
        except Exception:
            log.exception("scanning for set-aside files failed")

    def _loop(self) -> None:
        while not self._stop.is_set():
            j = self.store.next_queued()
            if not j:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            self.run_one(j)

    def run_one(self, j: dict) -> None:
        jid = j["id"]
        cancel = threading.Event()
        self._cancel[jid] = cancel
        if not self.store.claim(jid):  # cancelled or removed after it was picked
            self._cancel.pop(jid, None)
            return
        prof = self.profiles.get(j["profile"]) or self.profiles.get_or_create(j["profile"])

        def on_status(d: dict) -> None:
            self.live[jid] = {**self.live.get(jid, {}), **d, "t": time.time()}
            fields = {"step": d.get("step", ""), "message": d.get("message", "")}
            if d.get("waiting") is True:
                fields["status"] = J.WAITING
            elif d.get("waiting") is False:
                fields["status"] = J.RUNNING
            if "step_index" in d:
                fields["progress"] = d["step_index"] / d.get("steps", 1)
            elif "progress" in d and "step" in d:
                idx = pipeline.STEP_NAMES.index(d["step"]) if d["step"] in pipeline.STEP_NAMES else 0
                fields["progress"] = (idx + d["progress"]) / len(pipeline.STEPS)
            self.store.update(jid, **{k: v for k, v in fields.items() if v != "" or k == "message"})

        self.store.update(jid, status=J.RUNNING, started=j.get("started") or time.time(), message="starting")
        job = Job(self.cfg, jid, Path(j["path"]), prof, j["options"], on_status=on_status, cancel=cancel)
        self._current = job
        try:
            rec = pipeline.run(job)
            delivered = rec.get("delivered") or {}
            if delivered.get("set_aside"):
                aside = Path(delivered["set_aside"])
                self.store.add_original(jid, str(aside), str(originals.live_name(aside)), delivered["set_aside_kind"],
                                        aside.stat().st_size if aside.exists() else 0)
            job.remove_work()
            self.store.update(jid, status=J.DONE, finished=time.time(), progress=1.0,
                              message="done", record=rec)
        except Skipped as exc:
            job.remove_work()
            self.store.update(jid, status=J.SKIPPED, finished=time.time(), message=f"skipped: {exc}",
                              record=pipeline.record(job))
        except Cancelled:
            if self._stop.is_set():
                job.log("stopped by shutdown; will resume")
                return
            job.remove_work()
            self.store.update(jid, status=J.CANCELLED, finished=time.time(), message="cancelled")
        except Exception as exc:
            if self._stop.is_set():
                job.log("stopped by shutdown; will resume")
                return
            if cancel.is_set():  # a killed worker surfaces as an error; it was a cancel
                job.remove_work()
                self.store.update(jid, status=J.CANCELLED, finished=time.time(), message="cancelled")
                return
            job.log(traceback.format_exc())
            msg = f"failed at '{pipeline.TITLES.get(job.step, job.step)}': {exc}"
            rec = pipeline.record(job)
            rec["error"] = str(exc)
            rec["work_bytes"] = job.size()
            self.store.update(jid, status=J.FAILED, finished=time.time(), message=msg[:800], record=rec)
            if not self.cfg.get("cleanup.keep_failed_work", True):
                job.remove_work()
        finally:
            self._current = None
            self._cancel.pop(jid, None)
            self.live.pop(jid, None)
            if jid in self._remove_after:
                self._remove_after.discard(jid)
                job.remove_work()
                self.store.remove(jid)
