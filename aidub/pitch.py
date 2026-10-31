"""The dialogue's pitch over time (FCPE, from the events step) and what it tells.

Singing holds notes: the pitch, smoothed past vibrato, stays put for a fifth of a
second and more, again and again, while speech glides between syllables. Over
the ±10 s around a line, the share of voiced time spent on held notes told the
sung lines of two Midnight Diner episodes from the spoken ones (all 67 sung
lines at 0.4 or more, against 3 % of spoken lines, most of them over a song).
How high a line sits (its median, in semitones) tells a man's short reply from
a woman's when the diarizer put both in one speaker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SMOOTH = 15        # frames (10 ms) averaged: flattens vibrato, keeps note changes
SLOPE_SPAN = 5     # frames either side the pitch slope is measured over
STEADY = 8.0       # semitones a second: slower than this is a held note ...
HOLD = 20          # ... once it has held this many frames (0.2 s)
MIN_VOICED_S = 0.4


class Pitch:
    def __init__(self, f0: np.ndarray, hop: float = 0.01):
        self.hop = hop
        self.st = np.full(f0.shape, np.nan, dtype=np.float32)
        v = f0 > 0
        self.st[v] = 12 * np.log2(f0[v] / 55.0)
        voiced = ~np.isnan(self.st)
        held = held_frames(self.st)
        self._voiced = np.concatenate([[0], np.cumsum(voiced)])
        self._held = np.concatenate([[0], np.cumsum(held)])

    @classmethod
    def load(cls, job) -> "Pitch | None":
        info = job.get("pitch") or {}
        p = Path(info.get("path", ""))
        if not info or not p.exists():
            return None
        return cls(np.load(p).astype(np.float32), float(info.get("hop", 0.01)))

    def _span(self, a: float, b: float) -> tuple[int, int]:
        n = self.st.size
        return min(n, max(0, int(a / self.hop))), min(n, max(0, int(b / self.hop)))

    def voiced_s(self, a: float, b: float) -> float:
        i, j = self._span(a, b)
        return float(self._voiced[j] - self._voiced[i]) * self.hop

    def held_share(self, a: float, b: float, reach: float = 0.0) -> float | None:
        """Share of the voiced time between a - reach and b + reach spent on held notes
        (None with too little voice to judge)."""
        i, j = self._span(a - reach, b + reach)
        voiced = self._voiced[j] - self._voiced[i]
        if voiced * self.hop < MIN_VOICED_S:
            return None
        return float(self._held[j] - self._held[i]) / float(voiced)

    def median(self, a: float, b: float) -> float | None:
        i, j = self._span(a, b)
        seg = self.st[i:j]
        seg = seg[~np.isnan(seg)]
        return float(np.median(seg)) if seg.size >= 8 else None


def held_frames(st: np.ndarray) -> np.ndarray:
    """Frames on a held note: inside a voiced run, the smoothed pitch moving slower than
    STEADY semitones a second for at least HOLD frames."""
    voiced = ~np.isnan(st)
    mask = np.zeros(st.size, bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        run = st[s:e]
        if run.size < max(HOLD, SMOOTH + 2 * SLOPE_SPAN + 1):
            continue
        smooth = np.convolve(run, np.ones(SMOOTH) / SMOOTH, mode="valid")
        slope = np.abs(smooth[2 * SLOPE_SPAN:] - smooth[:-2 * SLOPE_SPAN]) / (2 * SLOPE_SPAN * 0.01)
        steady = slope < STEADY
        off = s + (SMOOTH - 1) // 2 + SLOPE_SPAN
        se = np.flatnonzero(np.diff(np.concatenate([[0], steady.astype(np.int8), [0]])))
        for x, y in zip(se[::2], se[1::2]):
            if y - x >= HOLD - SMOOTH:
                mask[off + x: off + y] = True
    return mask
