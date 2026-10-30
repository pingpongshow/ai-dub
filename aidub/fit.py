"""Timing-fit rules: target windows, speed factors, stretch limits, trimming.

Speed factor semantics follow the voice model: the factor scales the spoken
length, so factor = window / measured; below 1.0 speaks faster.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lines as L
from . import media


@dataclass
class FitRules:
    borrow: float = 0.25
    short_borrow: float = 0.5      # very short lines may borrow more following silence
    short_line: float = 1.0        # ... when the original line is shorter than this
    factor_min: float = 0.85
    factor_max: float = 1.15
    stretch_max: float = 1.12      # at most this much faster by time-stretch
    stretch_min: float = 0.92      # never slower than this by time-stretch
    tolerance: float = 0.08
    short_factor_min: float = 0.75  # one- or two-word lines may be spoken this much faster
    short_stretch_max: float = 1.25

    @classmethod
    def from_config(cls, cfg) -> "FitRules":
        g = lambda k, d: float(cfg.get(f"fit.{k}", d))
        return cls(borrow=g("borrow_ms", 250) / 1000, short_borrow=g("short_line_borrow_ms", 500) / 1000,
                   short_line=g("short_line_s", 1.0), factor_min=g("factor_min", 0.85),
                   factor_max=g("factor_max", 1.15), stretch_max=g("stretch_max", 1.12),
                   stretch_min=g("stretch_min", 0.92), tolerance=g("tolerance_ms", 80) / 1000,
                   short_factor_min=g("short_factor_min", 0.75), short_stretch_max=g("short_stretch_max", 1.25))

    def for_short_text(self) -> "FitRules":
        """Looser limits for one- or two-word lines, where a faster delivery isn't noticeable."""
        from dataclasses import replace
        return replace(self, factor_min=min(self.factor_min, self.short_factor_min),
                       stretch_max=max(self.stretch_max, self.short_stretch_max))


def free_timing(cfg) -> bool:
    """fit.timing: start: each English line starts with the original line and runs its
    natural length (lip movement ignored); nothing is shortened, sped up or cut to fit."""
    return str(cfg.get("fit.timing", "window")) == "start"


def windows(lines: list[L.Line], cuts: list[float], rules: FitRules, duration: float) -> dict[int, float]:
    """Seconds available to each dubbed line: its own span plus up to `borrow` of
    following silence (`short_borrow` for very short lines), never past the same
    character's next line or a shot change."""
    cuts_arr = np.asarray(sorted(cuts)) if cuts else np.asarray([])
    out = {}
    by_time = sorted(lines, key=lambda l: l.start)
    for i, ln in enumerate(by_time):
        limit = ln.end + (rules.short_borrow if ln.duration < rules.short_line else rules.borrow)
        for nxt in by_time[i + 1:]:
            if nxt.start >= limit:
                break
            if nxt.character == ln.character or nxt.kind != L.SPEECH or not nxt.overlap:
                limit = min(limit, max(ln.end, nxt.start))
        if cuts_arr.size:
            k = np.searchsorted(cuts_arr, ln.end - 0.05)
            if k < cuts_arr.size and cuts_arr[k] < limit:
                limit = max(ln.end, float(cuts_arr[k]))
        out[ln.id] = round(min(limit, duration) - ln.start, 3)
    return out


def speech_bounds(data: np.ndarray, sr: int, start_db: float = 18.0, end_db: float = 15.0,
                  decay_db: float = 25.0, max_decay: float = 0.15) -> tuple[int, int]:
    """Where the speech is, judged against the take's own speech level (the 90th
    percentile of 20 ms frame loudness): starts where a frame comes within
    `start_db` of it, ends at the last frame within `end_db`, plus at most
    `max_decay` seconds of natural decay above `decay_db` below it. Voice models
    pad takes with low-level noise and long quiet tails; this cuts both."""
    x = data if data.ndim == 1 else data.mean(axis=1)
    hop = max(1, int(sr * 0.02))
    if x.size < hop:
        return 0, int(x.size)
    frames = x[: x.size // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    ref = float(np.percentile(db, 90))
    on = np.nonzero(db > ref - start_db)[0]
    core = np.nonzero(db > ref - end_db)[0]
    if on.size == 0 or core.size == 0:
        return 0, 0
    first, last = int(on[0]), int(core[-1])
    k = last + 1
    while k < len(db) and (k - last) * 0.02 <= max_decay and db[k] > ref - decay_db:
        k += 1
    return first * hop, min(int(x.size), k * hop)


def trim(path: Path, out: Path | None = None, pad: float = 0.02) -> float:
    """Trim leading/trailing silence in place (or to `out`); return the new duration."""
    data, sr = media.read_wav(path)
    a, b = speech_bounds(data, sr)
    if b <= a:
        return 0.0
    p = int(pad * sr)
    data = data[max(0, a - p): min(len(data), b + p)].copy()
    f = min(int(0.015 * sr), len(data) // 4)
    if f > 0:
        ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None] if data.ndim == 2 else np.linspace(0, 1, f, dtype=np.float32)
        data[:f] *= ramp
        data[-f:] *= ramp[::-1]
    media.write_wav(out or path, data, sr)
    return len(data) / sr


def needs_fit(measured: float, window: float, original: float, rules: FitRules) -> str:
    """'over', 'under' or '' (fits)."""
    if measured > window + rules.tolerance:
        return "over"
    if measured < 0.8 * original - rules.tolerance and original > 1.0:
        return "under"
    return ""


def factor_for(measured: float, target: float, rules: FitRules) -> float | None:
    """Speed factor that turns `measured` into `target`, or None outside the natural range."""
    if measured <= 0:
        return None
    f = target / measured
    if f < rules.factor_min or f > rules.factor_max:
        return None
    return round(f, 4)


def stretch_ratio(measured: float, window: float, rules: FitRules) -> float | None:
    """Tempo ratio (>1 faster) to fit by time-stretch, or None if beyond limits."""
    if measured <= window:
        return 1.0
    r = measured / window
    return round(r, 4) if r <= rules.stretch_max else None


def time_stretch(src: Path, dst: Path, tempo: float) -> Path:
    """Change tempo with pitch preserved: rubberband if present, else ffmpeg atempo."""
    if abs(tempo - 1.0) < 1e-3:
        shutil.copyfile(src, dst)
        return dst
    if shutil.which("rubberband"):
        subprocess.run(["rubberband", "-q", "-3", "--tempo", f"{tempo:.5f}", str(src), str(dst)],
                       check=True, capture_output=True)
    else:
        media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-af", f"atempo={tempo:.5f}",
                   "-c:a", "pcm_f32le", str(dst)])
    return dst
