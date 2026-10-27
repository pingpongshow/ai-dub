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
    factor_min: float = 0.85
    factor_max: float = 1.15
    stretch_max: float = 1.12      # at most this much faster by time-stretch
    stretch_min: float = 0.92      # never slower than this by time-stretch
    tolerance: float = 0.08

    @classmethod
    def from_config(cls, cfg) -> "FitRules":
        g = lambda k, d: float(cfg.get(f"fit.{k}", d))
        return cls(borrow=g("borrow_ms", 250) / 1000, factor_min=g("factor_min", 0.85),
                   factor_max=g("factor_max", 1.15), stretch_max=g("stretch_max", 1.12),
                   stretch_min=g("stretch_min", 0.92), tolerance=g("tolerance_ms", 80) / 1000)


def windows(lines: list[L.Line], cuts: list[float], rules: FitRules, duration: float) -> dict[int, float]:
    """Seconds available to each dubbed line: its own span plus up to `borrow` of
    following silence, never past the same character's next line or a shot change."""
    cuts_arr = np.asarray(sorted(cuts)) if cuts else np.asarray([])
    out = {}
    by_time = sorted(lines, key=lambda l: l.start)
    for i, ln in enumerate(by_time):
        limit = ln.end + rules.borrow
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


def speech_bounds(data: np.ndarray, sr: int, floor_db: float = -42.0) -> tuple[int, int]:
    x = np.abs(data if data.ndim == 1 else data.mean(axis=1))
    if x.size == 0:
        return 0, 0
    peak = float(x.max()) or 1.0
    hop = max(1, int(sr * 0.01))
    frames = x[: x.size // hop * hop].reshape(-1, hop).max(axis=1) if x.size >= hop else x
    thr = peak * 10 ** (floor_db / 20)
    idx = np.nonzero(frames > thr)[0]
    if idx.size == 0:
        return 0, 0
    return int(idx[0] * hop), int(min(x.size, (idx[-1] + 1) * hop))


def trim(path: Path, out: Path | None = None, pad: float = 0.03) -> float:
    """Trim leading/trailing silence in place (or to `out`); return the new duration."""
    data, sr = media.read_wav(path)
    a, b = speech_bounds(data, sr)
    if b <= a:
        return 0.0
    p = int(pad * sr)
    data = data[max(0, a - p): min(len(data), b + p)]
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
