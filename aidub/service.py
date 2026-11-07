"""The queue runner: one job at a time on the GPU, in queue order."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import jobs as J
from . import media, originals, pipeline
from .config import Config
from .job import Cancelled, Job, NeedsSubtitles, Paused, Skipped
from .profiles import Profiles, guess_name

log = logging.getLogger("aidub")


LINE_VOICE_MIN_S = 1.5   # shorter than this, a clip gives the voice model too little to go by


def _voice_kind(prof, ln) -> str:
    if not ln.voice:
        return "own"
    if "/refs/" in ln.voice and ln.voice.split("/refs/")[-1].split("/")[0] == prof.voice_source(ln.character):
        return "usual"
    return "other"


def _speakers(prof, lines) -> list[dict]:
    from collections import Counter
    count = Counter(ln.character for ln in lines if ln.character and ln.en)
    out = []
    for cid, n in count.most_common():
        ch = prof.characters.get(cid)
        if not ch:
            continue
        refs = prof.refs(cid)
        out.append({"id": cid, "name": ch.get("name", cid), "lines": n, "gender": ch.get("gender", ""),
                    "sample": refs[0]["file"] if refs else ""})
    return out


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
        self._to_check: queue.Queue = queue.Queue()
        self._check_lock = threading.Lock()

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
        self._to_check.put(jid)
        self._wake.set()
        return jid

    def _subtitle_problem(self, j: dict) -> str:
        """Why this file must wait for the user before any work starts on it; empty if it needn't."""
        from .ingest import classify
        from .subtitles.preflight import missing_english
        opts = j.get("options") or {}
        if (not self.cfg.get("subtitles.require", True) or opts.get("speech_ok") or opts.get("subtitle_pick")
                or opts.get("subs_checked") or opts.get("preview_s")):
            return ""
        state = self.cfg.work / f"job-{j['id']}" / "state.json"
        if state.exists() and "subtitles" in json.loads(state.read_text()).get("done", []):
            return ""  # already past this
        path = Path(j["path"])
        pr = media.probe(path)
        info = classify(pr)
        # The skip rules end this job at once: nothing to ask about.
        if not opts.get("redo") and (info["aidub_present"] or (info["english_audio"] and self.cfg.get("ingest.skip_english_audio", True))):
            return ""
        return missing_english(self.cfg, path, pr)

    def _hold_if_no_subtitles(self, jid: int) -> bool:
        """True if the job waits for the user. Checked once per job."""
        with self._check_lock:
            j = self.store.get(jid)
            if not j or j["status"] not in (J.QUEUED, J.WAITING):
                return False
            try:
                why = self._subtitle_problem(j)
            except Exception as exc:  # an unreadable file fails with its own message when it runs
                log.warning("subtitle check of %s: %s", j["path"], exc)
                return False
            if why:
                self.store.update(jid, status=J.REVIEW, message=why, record={
                    "needs_subtitles": why, "subtitle_choices": self._subtitle_choices(j["path"]), "paused_at": time.time()})
                return True
            self.store.update(jid, options={**(j.get("options") or {}), "subs_checked": True})
            return False

    def _subtitle_choices(self, path: str) -> list[dict]:
        from .subtitles.preflight import choices
        try:
            return choices(Path(path))
        except Exception as exc:
            log.warning("subtitles of %s: %s", path, exc)
            return []

    def _check_added(self) -> None:
        """Files just added are checked straight away, so one that needs an answer says so
        while it is still far down the queue."""
        while True:
            self._hold_if_no_subtitles(self._to_check.get())

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
        if j["status"] in (J.QUEUED, J.WAITING, J.REVIEW) and jid not in self._cancel:
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

    # Review -------------------------------------------------------------------------------
    def review_view(self, jid: int) -> dict:
        from . import review
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        v = review.view(self.cfg, j, self.profiles.get(j["profile"]))
        if j["status"] == J.REVIEW and v.get("ready"):
            n = len(v.get("flags", []))
            msg = f"waiting for review: {n} to check" if n else "waiting for review"
            if msg != j.get("message"):
                self.store.update(jid, message=msg)  # the queue shows what the screen shows
        return v

    def review_save(self, jid: int, decisions: dict, start: bool = False) -> None:
        from . import review
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        work = self.cfg.work / f"job-{jid}"
        if j["status"] == J.DONE and (j.get("record") or {}).get("verify") == "pending" and review.can_redub(work):
            if start:
                raise ValueError("this dub is finished: redub it with the changes instead")
            review.save_decisions(work, {**decisions, "started": True})
            return
        if j["status"] != J.REVIEW:
            raise ValueError("this job isn't waiting for review")
        review.save_decisions(work, {**decisions, "started": start})
        if start:
            self.store.requeue_front(jid)
            self._wake.set()

    def _needing_subtitles(self, jid: int) -> dict:
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        if j["status"] != J.REVIEW or not (j.get("record") or {}).get("needs_subtitles"):
            raise ValueError("this job isn't waiting for subtitles")
        return j

    def allow_speech(self, jid: int) -> None:
        """The user's override for this one file: dub it from the speech."""
        j = self._needing_subtitles(jid)
        self.store.update(jid, options={**(j.get("options") or {}), "speech_ok": True}, message="dubbing from the speech, as you chose")
        self.store.requeue_front(jid)
        self._wake.set()

    def use_subtitle(self, jid: int, subtitle: str) -> None:
        """The user's choice for this one file: dub from this subtitle, translated if it isn't English.
        From the subtitle step on: it has to be read and matched to the speech."""
        from .pipeline import STEP_NAMES
        j = self._needing_subtitles(jid)
        if subtitle not in {c["id"] for c in self._subtitle_choices(j["path"])}:
            raise ValueError("that subtitle isn't there any more")
        f = self.cfg.work / f"job-{jid}" / "state.json"
        if f.exists():
            st = json.loads(f.read_text())
            st["done"] = [x for x in st.get("done", []) if x in STEP_NAMES[:STEP_NAMES.index("subtitles")]]
            f.write_text(json.dumps(st, ensure_ascii=False))
        self.store.update(jid, options={**(j.get("options") or {}), "subtitle_pick": subtitle}, message="dubbing from the subtitle you chose")
        self.store.requeue_front(jid)
        self._wake.set()

    def search_subtitles(self, jid: int, language: str) -> None:
        """Ask Bazarr for a subtitle in this language; it can take minutes, so in the background.
        The card lists whatever arrives; an English one means the file can simply go on."""
        j = self._needing_subtitles(jid)
        if not 2 <= len(language) <= 3 or not language.isalpha():
            raise ValueError("choose a language")
        rec = j.get("record") or {}
        self.store.update(jid, record={**rec, "subtitle_search": language}, message=f"asking Bazarr for {language} subtitles")

        def work():
            from .subtitles.preflight import _english_next_to, search
            why = search(self.cfg, Path(j["path"]), language)
            now = self.store.get(jid)
            if not now or now["status"] != J.REVIEW:
                return
            if not why and language == "en" and _english_next_to(Path(j["path"])):
                self.recheck_subtitles(jid)
                return
            r = {k: v for k, v in (now.get("record") or {}).items() if k != "subtitle_search"}
            r["subtitle_choices"] = self._subtitle_choices(j["path"])
            r["subtitle_search_result"] = why or f"Bazarr searched for {language}: anything it found is in the list"
            self.store.update(jid, record=r, message=now.get("record", {}).get("needs_subtitles", ""))
        threading.Thread(target=work, name="aidub-subtitle-search", daemon=True).start()

    def recheck_subtitles(self, jid: int) -> None:
        """Look again (a subtitle added next to the file, or Bazarr has one now): back to the
        subtitle step, keeping the analysis before it."""
        from .pipeline import STEP_NAMES
        j = self._needing_subtitles(jid)
        f = self.cfg.work / f"job-{jid}" / "state.json"
        if f.exists():
            st = json.loads(f.read_text())
            st["done"] = [x for x in st.get("done", []) if x in STEP_NAMES[:STEP_NAMES.index("subtitles")]]
            f.write_text(json.dumps(st, ensure_ascii=False))
        opts = {k: v for k, v in (j.get("options") or {}).items() if k != "subs_checked"}
        self.store.update(jid, options=opts, message="looking for subtitles again")
        self.store.requeue_front(jid)
        self._to_check.put(jid)
        self._wake.set()

    def review_redub(self, jid: int, decisions: dict) -> str:
        """Dub a finished episode again with the review's changes: the original goes back,
        the job returns to its review step (the analysis is kept) and continues from there."""
        from . import review, verify
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        work = self.cfg.work / f"job-{jid}"
        rec = j.get("record") or {}
        if j["status"] != J.DONE or rec.get("verify") != "pending" or not review.can_redub(work):
            raise ValueError("this dub can't be redone from its review (its working files are gone)")
        if self.store.active_for(j["path"]):
            raise ValueError("this file is already in the queue")
        review.save_decisions(work, {**decisions, "started": True})
        verify.put_back(self.store, jid)
        review.reset_for_redub(work, Path(j["path"]))
        rec["verify"] = ""
        rec["redubbed"] = rec.get("redubbed", 0) + 1
        self.store.update(jid, record=rec, finished=None, progress=0)
        self.store.requeue_front(jid)
        self.store.update(jid, message="dubbing again with your review changes")
        self._wake.set()
        return "the original is back; dubbing again with your changes"

    def redo_lines(self, jid: int, changes: dict[int, dict]) -> str:
        """Voice a few lines of a finished dub again (with new English or speaker), keeping
        every other take: the original goes back, the job returns to just before voicing
        and only these lines are voiced; the check, mix and delivery run as before."""
        from . import review, verify
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        work = self.cfg.work / f"job-{jid}"
        rec = j.get("record") or {}
        if j["status"] != J.DONE or rec.get("verify") != "pending" or not review.can_redo_lines(work):
            raise ValueError("this dub's takes aren't kept any more: redub it from its review instead")
        if not changes:
            raise ValueError("choose the lines to redo")
        if self.store.active_for(j["path"]):
            raise ValueError("this file is already in the queue")
        # Remember the English and speakers for any later redub of the whole episode.
        dec = review.load_decisions(work)
        for lid, ch in changes.items():
            mine = dec.setdefault("lines", {}).setdefault(str(lid), {})
            mine.update({k: v for k, v in ch.items() if k in ("en", "character") and v})
        review.save_decisions(work, {**dec, "started": True})
        verify.put_back(self.store, jid)
        done = review.reset_for_lines(work, Path(j["path"]), changes, self.profiles.get(j["profile"]))
        rec["verify"] = ""
        self.store.update(jid, record=rec, finished=None, progress=0)
        self.store.requeue_front(jid)
        msg = f"voicing {len(done)} line{'s' if len(done) != 1 else ''} again"
        self.store.update(jid, message=msg)
        self._wake.set()
        return f"the original is back; {msg}, then the dub is put together again"

    def review_hold(self, jid: int) -> str:
        """Stop a queued or running job at its review step, even with nothing flagged."""
        from . import review
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        if j["status"] not in (J.QUEUED, J.RUNNING, J.WAITING) or (j.get("step") or "") in review._after_review():
            raise ValueError("this job is already past its review step; once it's done, review it and redub")
        opts = {**(j.get("options") or {}), "review": "always"}
        self.store.update(jid, options=opts)
        cur = self._current
        if cur is not None and str(cur.id) == str(jid):
            cur.options["review"] = "always"
        return "it will stop for review before anything is translated"

    def review_media(self, jid: int) -> dict:
        from . import review
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        work = self.cfg.work / f"job-{jid}"
        state = work / "state.json"
        if not (work / "lines.json").exists() or not state.exists():
            raise ValueError("Ai Dub hasn't worked out the lines of this episode yet")
        import json
        return review.start_media(work, json.loads(state.read_text()).get("data", {}))

    def verify_list(self) -> list[dict]:
        from . import verify
        return verify.pending(self.store, self.cfg.work)

    def verify_view(self, jid: int) -> dict:
        from . import lines as L
        from . import verify
        out = verify.view(self.store, self.cfg.work, jid)
        j = self.store.get(jid)
        prof = self.profiles.get(j["profile"]) if j else None
        work = self.cfg.work / f"job-{jid}"
        if out.get("can_redo_lines") and prof and (work / "lines.json").exists():
            # Who speaks each line and from which voice, and every speaker with a clip to hear,
            # so a voice can be changed from the check page itself.
            lines = {ln.id: ln for ln in L.load(work / "lines.json")}
            out["lines"] = [row + [lines[row[0]].character, _voice_kind(prof, lines[row[0]])]
                            for row in out["lines"] if row[0] in lines]
            out["speakers"] = _speakers(prof, lines.values())
            out["profile"] = prof.slug
        return out

    def line_as_voice(self, jid: int, lid: int, cid: str = "") -> dict:
        """Make a line's original audio the clip a speaker is voiced from: for a voice that
        came out wrong when you can point to a line where it's right. Kept, so it is used
        first, and in this show's later episodes too."""
        from . import lines as L
        from . import review
        j = self.store.get(jid)
        if not j:
            raise KeyError(jid)
        work = self.cfg.work / f"job-{jid}"
        prof = self.profiles.get(j["profile"])
        lines = {ln.id: ln for ln in L.load(work / "lines.json")} if (work / "lines.json").exists() else {}
        ln = lines.get(int(lid))
        if not prof or not ln:
            raise ValueError("this dub's lines aren't kept any more")
        cid = cid or ln.character
        if cid not in prof.characters:
            raise ValueError("choose a speaker for this line first")
        if ln.duration < LINE_VOICE_MIN_S:
            raise ValueError(f"this line is too short to clone a voice from ({ln.duration:.1f} s); choose one of {LINE_VOICE_MIN_S:.1f} s or more")
        rel = review.line_as_ref(work, prof, ln, cid)
        prof.save()
        src = prof.voice_source(cid)
        usual = sorted(x.id for x in lines.values() if x.kind == L.SPEECH and x.voice and prof.voice_source(x.character) == src
                       and f"/refs/{src}/" in x.voice)
        return {"ok": True, "ref": rel, "character": cid, "name": prof.characters[cid].get("name", cid), "usual": usual,
                "message": f"this line is now {prof.characters[cid].get('name', cid)}'s voice"}

    def verify_verdict(self, jid: int, choice: str) -> dict:
        from . import verify
        out = verify.verdict(self, jid, choice)
        self._wake.set()
        return out

    def work_size(self, jid: int) -> int:
        d = self.cfg.work / f"job-{jid}"
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0

    # Runner ---------------------------------------------------------------------------------
    def start(self) -> None:
        self.store.recover()
        for j in self.store.list({J.QUEUED, J.WAITING}):
            self._to_check.put(j["id"])
        threading.Thread(target=self._check_added, name="aidub-subtitle-check", daemon=True).start()
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

    # Pausing the queue: the running job finishes, nothing new starts until resumed.
    def _pause_file(self) -> Path:
        return self.cfg.appdata / "queue.json"

    @property
    def paused(self) -> bool:
        f = self._pause_file()
        try:
            return bool(json.loads(f.read_text()).get("paused")) if f.exists() else False
        except Exception:
            return False

    def set_paused(self, on: bool) -> None:
        self._pause_file().write_text(json.dumps({"paused": bool(on), "at": time.time()}))
        self._wake.set()

    def start_now(self, jid: int, on: bool = True) -> None:
        """Run this title next, even while the queue is paused. It stays marked if it comes back to
        the queue after its review, so that it can finish."""
        j = self.store.get(jid)
        if not j or j["status"] not in (J.QUEUED, J.WAITING):
            raise ValueError("only a title waiting in the queue can be started")
        opts = {k: v for k, v in (j.get("options") or {}).items() if k != "start_now"}
        self.store.update(jid, options={**opts, **({"start_now": True} if on else {})})
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            j = self.store.next_up(self.paused)
            if not j:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            self.run_one(j)

    def run_one(self, j: dict) -> None:
        jid = j["id"]
        if self._hold_if_no_subtitles(jid):
            return
        j = self.store.get(jid) or j
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
            from . import review, verify
            dec = review.load_decisions(job.work)
            if dec.get("started"):
                rec["review_decisions"] = {k: dec.get(k) for k in ("regions", "cues", "lines")}
            if self.cfg.get("verify.enabled", True) and delivered.get("set_aside"):
                try:
                    job.status("preparing the check")
                    verify.prepare(job, rec)
                    rec["verify"] = "pending"
                except Exception as exc:
                    job.log(f"check files not kept: {exc}")
                    rec["check_error"] = str(exc)[:300]  # the log goes with the working folder
                    job.remove_work()
            else:
                job.remove_work()
            self.store.update(jid, status=J.DONE, finished=time.time(), progress=1.0,
                              message="done", record=rec)
        except Paused as exc:
            job.log(f"paused: {exc}")
            self.store.update(jid, status=J.REVIEW, message=str(exc), record={
                **pipeline.record(job), "paused_at": time.time(),
                **({"needs_subtitles": exc.why, "subtitle_choices": self._subtitle_choices(j["path"])}
                   if isinstance(exc, NeedsSubtitles) else {})})
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
