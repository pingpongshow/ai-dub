"""Shot changes from the video and pauses in the dialogue: cuts and scene windows."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from . import media
from .job import Job

_PTS = re.compile(r"pts_time:([0-9.]+)")


def shot_cuts(src: Path, threshold: float, duration: float | None = None) -> list[float]:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "0", "-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:v:0", "-an", "-sn",
            "-vf", f"scale=192:-2,select='gt(scene,{threshold})',metadata=print:file=-",
            "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return sorted({round(float(m.group(1)), 3) for m in _PTS.finditer(proc.stdout)})


def speech_pauses(wav: Path, min_pause: float, hop: float = 0.05) -> list[tuple[float, float]]:
    """Silent stretches in the dialogue stem (energy below an adaptive floor)."""
    data, sr = media.read_wav(wav, mono=True)
    x = data[:, 0]
    n = int(sr * hop)
    if len(x) < n:
        return []
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    floor = max(np.percentile(db, 20) + 6, np.percentile(db, 95) - 35)
    quiet = db < floor
    out, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= min_pause:
                out.append((start * hop, i * hop))
            start = None
    if start is not None and (len(quiet) - start) * hop >= min_pause:
        out.append((start * hop, len(quiet) * hop))
    return out


def windows(duration: float, cuts: list[float], pauses: list[tuple[float, float]],
            max_len: float, min_len: float = 20.0) -> list[list[float]]:
    """Split the timeline into windows no longer than `max_len`, preferring
    boundaries where a pause and a shot change meet, then pauses, then cuts."""
    candidates: list[tuple[float, int]] = [(t, 2) for t in cuts]
    for a, b in pauses:
        in_pause = [c for c in cuts if a - 0.5 <= c <= b + 0.5]
        candidates.append((in_pause[0], 0) if in_pause else ((a + b) / 2, 1))
    out, start = [], 0.0
    while duration - start > max_len:
        # (never leaving a last window too short to diarize)
        lo, hi = start + min_len, min(start + max_len, duration - min_len)
        inside = [(rank, -t, t) for t, rank in candidates if lo <= t <= hi]
        cut = min(inside)[2] if inside else hi
        out.append([round(start, 3), round(cut, 3)])
        start = cut
    out.append([round(start, 3), round(duration, 3)])
    return out


def scene_of(t: float, cuts: list[float]) -> int:
    return int(np.searchsorted(np.asarray(cuts), t, side="right"))


def run(job: Job) -> None:
    ing = job["ingest"]
    cuts: list[float] = []
    if ing["has_video"]:
        job.status("finding shot changes")
        cuts = shot_cuts(Path(ing["source"]), float(job.cfg.get("ingest.scene_threshold", 0.35)),
                         ing["duration"] if ing.get("preview_s") else None)
    pauses = speech_pauses(job.path("audio", "dialogue.wav"), float(job.cfg.get("ingest.scene_min_pause_s", 1.5)))
    win = windows(ing["duration"], cuts, pauses, float(job.cfg.get("stages.diarize.window_max_s", 120)))
    job.set("scenes", {"cuts": cuts, "pauses": [list(p) for p in pauses], "windows": win})
    job.status(f"{len(cuts)} shot changes, {len(win)} windows")
