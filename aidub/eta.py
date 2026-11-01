"""How long a queued or running job still has to go, from the step times of earlier
jobs: each step's seconds of work per second of video, for the same show when it has
history, else across every show."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import media
from .pipeline import STEP_NAMES

WAITING = {"review"}       # a pause for the user, not work
_durations: dict[str, float] = {}
_done: dict[str, tuple[float, list[str]]] = {}


def duration_of(path: str) -> float:
    if path not in _durations:
        try:
            _durations[path] = float(media.probe(path).duration or 0)
        except Exception:
            _durations[path] = 0.0
    return _durations[path]


def rates(history: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Median seconds per second of video, per step: by show, and over all shows."""
    per: dict[str, dict[str, list[float]]] = {}
    every: dict[str, list[float]] = {}
    for j in history:
        r = j.get("record") or {}
        t = r.get("timings") or {}
        if j.get("status") != "done" or not t:
            continue
        dur = float(r.get("duration") or 0) or duration_of(j["path"])
        if dur <= 60:
            continue
        for step, sec in t.items():
            per.setdefault(j["profile"], {}).setdefault(step, []).append(float(sec) / dur)
            every.setdefault(step, []).append(float(sec) / dur)

    def med(d):
        return {k: statistics.median(v) for k, v in d.items()}
    return {p: med(d) for p, d in per.items()}, med(every)


def _done_steps(work: Path) -> list[str]:
    """Steps a resumed job (a redo from its review or of a few lines) has already done."""
    f = work / "state.json"
    try:
        m = f.stat().st_mtime
    except OSError:
        return []
    key = str(f)
    if key not in _done or _done[key][0] != m:
        try:
            _done[key] = (m, list(json.loads(f.read_text()).get("done", [])))
        except Exception:
            _done[key] = (m, [])
    return _done[key][1]


def remaining(j: dict, live: dict, work: Path, per: dict, every: dict) -> float | None:
    """Seconds of work left for this job, or None with no history to go by."""
    r = per.get(j["profile"]) or every
    if not r:
        return None
    dur = float(live.get("duration") or 0) or duration_of(j["path"])
    if dur <= 0:
        return None
    done = set(_done_steps(work))
    step = j.get("step") or ""
    running = j.get("status") in ("running", "waiting")
    left = 0.0
    for s in STEP_NAMES:
        if s in done or s in WAITING:
            continue
        cost = r.get(s, every.get(s, 0.0)) * dur
        if running and s == step:
            frac = live.get("progress") if live.get("step") == step else None
            cost *= 1.0 - min(max(float(frac), 0.0), 1.0) if isinstance(frac, (int, float)) else 0.5
        left += cost
    return round(left)
