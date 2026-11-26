"""Where the original voice really is around each line, from the dialogue's pitch and level.

A line's times come from the words recognition and the aligner found, and the aligner is
sometimes wrong: 何するんだい ("What are you doing?") in Woman in the Dunes got 0.3 s at
3857.24, while the actor says it from 3855.9 to 3857.0. The mix muted only that 0.3 s, so the
Japanese played, and the English came in 1.3 s late. And a line's first or last sound (a
breath of a word, a trailing "ne") is often left out of its words.

retime: a line whose times are far too short for what it says (under half the time it
needs) takes them from the voice heard around it, and drops its word times (fit.timing: sync
then follows the voice as one phrase, not the misplaced words).
extents: how far each dubbed line's voice runs on past its times, joined to it, so the
original is muted under all of it.
"""

from __future__ import annotations

import re
from bisect import bisect_right

import numpy as np

from . import lines as L
from . import media
from .job import Job
from .pitch import Pitch

BRIDGE_S = 0.15        # an unvoiced consonant or a breath inside speech, at most
BELOW_USUAL_DB = 22.0  # voice: at least this loud, against the film's usual line peak
REACH_S = 1.0          # a line's voice runs on at most this far past its times ...
SEARCH_S = 1.5         # ... and a line timed far too short looks this far back for its voice
TOO_SHORT = 0.5        # share of the time a line needs, under which its times are wrong
GAP_S = 0.03           # kept clear of the lines either side

_KANA = re.compile(r"[぀-ヿ가-힯]")
_HAN = re.compile(r"[一-鿿]")
_VOWELS = re.compile(r"[aeiouyàâäéèêëîïôöùûüáíóú]+", re.I)


def needed_s(text: str) -> float:
    """About how long a line takes to say: Japanese at ~8 morae a second (a kanji ~2 morae),
    other languages at ~6 syllables a second."""
    kana, han = len(_KANA.findall(text)), len(_HAN.findall(text))
    if kana or han:
        return (kana + 2 * han) / 8.0
    return len(_VOWELS.findall(text)) / 6.0


class Voice:
    """Voiced frames of the dialogue (10 ms): a pitch and a speaking level."""

    def __init__(self, voiced: np.ndarray, hop: float = 0.01):
        self.hop = hop
        self.on = voiced.astype(bool)

    @classmethod
    def load(cls, job: Job, lines: list[L.Line]) -> "Voice | None":
        p = Pitch.load(job)
        wav = job.path("audio", "dialogue16k.wav")
        if p is None or not wav.exists():
            return None
        x, sr = media.read_wav(wav, mono=True)
        return cls.from_signal(x[:, 0], sr, ~np.isnan(p.st), p.hop, lines)

    @classmethod
    def from_signal(cls, x: np.ndarray, sr: int, pitched: np.ndarray, hop: float,
                    lines: list[L.Line]) -> "Voice":
        step = int(round(sr * hop))
        n = min(len(pitched), len(x) // step)
        frames = x[: n * step].astype(np.float64).reshape(n, step)
        db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
        peaks = [db[int(l.start / hop): max(int(l.start / hop) + 1, int(l.end / hop))].max()
                 for l in lines if l.kind == L.SPEECH and int(l.start / hop) < n]
        usual = float(np.median(peaks)) if peaks else float(np.percentile(db, 90))
        return cls(pitched[:n] & (db > usual - BELOW_USUAL_DB), hop)

    def _i(self, t: float) -> int:
        return min(len(self.on), max(0, int(round(t / self.hop))))

    def runs(self, a: float, b: float) -> list[tuple[float, float]]:
        """Voiced stretches in [a, b], pauses under BRIDGE_S joined."""
        i, j = self._i(a), self._i(b)
        idx = np.flatnonzero(self.on[i:j]) + i
        if not idx.size:
            return []
        out, s, last = [], idx[0], idx[0]
        bridge = int(BRIDGE_S / self.hop)
        for k in idx[1:]:
            if k - last > bridge:
                out.append((s * self.hop, (last + 1) * self.hop))
                s = k
            last = k
        out.append((s * self.hop, (last + 1) * self.hop))
        return out

    def joined(self, a: float, b: float, floor: float, ceil: float) -> tuple[float, float]:
        """[a, b] widened over voice that runs straight on from it (pauses under BRIDGE_S),
        at most REACH_S each way and never past floor or ceil."""
        lo, hi = a, b
        for s, e in reversed(self.runs(max(floor, a - REACH_S), a + BRIDGE_S)):
            if e >= lo - BRIDGE_S:
                lo = min(lo, s)
        for s, e in self.runs(b - BRIDGE_S, min(ceil, b + REACH_S)):
            if s <= hi + BRIDGE_S:
                hi = max(hi, e)
        return max(floor, lo), min(ceil, hi)


def _neighbours(lines: list[L.Line]):
    """floor(ln): the end of the line before it; ceil(ln): the start of the line after it."""
    by_start = sorted(lines, key=lambda l: l.start)
    starts = [l.start for l in by_start]
    ends = sorted(l.end for l in lines)

    def floor(ln: L.Line) -> float:
        k = bisect_right(ends, ln.start - 1e-3) - 1
        return ends[k] + GAP_S if k >= 0 else 0.0

    def ceil(ln: L.Line) -> float:
        k = bisect_right(starts, ln.end - 1e-3)
        return starts[k] - GAP_S if k < len(starts) else float("inf")

    return floor, ceil


def retime(job: Job, lines: list[L.Line], voice: "Voice | None" = None) -> int:
    """Lines timed far too short for what they say take their times from the voice around them."""
    todo = [l for l in lines if l.dubbed and l.source_text and needed_s(l.source_text) >= 0.5
            and l.duration < TOO_SHORT * needed_s(l.source_text)]
    if not todo:
        return 0
    voice = voice or Voice.load(job, lines)
    if voice is None:
        return 0
    floor, ceil = _neighbours(lines)
    moved = 0
    for ln in todo:
        lo, hi = max(floor(ln), ln.start - SEARCH_S), min(ceil(ln), ln.end + REACH_S)
        runs = voice.runs(lo, hi)
        if not runs:
            continue
        a, b = min(ln.start, runs[0][0]), max(ln.end, runs[-1][1])
        if b - a <= ln.duration + 0.15:
            continue
        job.log(f"line {ln.id} ({ln.source_text[:20]}): its voice is at {a:.2f}-{b:.2f}, "
                f"not {ln.start:.2f}-{ln.end:.2f}")
        ln.start, ln.end = round(a, 3), round(b, 3)
        ln.words = []  # they were where the aligner put them, not where they're said
        moved += 1
    if moved:
        job.qc["lines_retimed_by_voice"] = moved
    return moved


def extents(job: Job, lines: list[L.Line], dubbed: list[L.Line],
            voice: "Voice | None" = None) -> dict[int, tuple[float, float]]:
    """Each dubbed line's span widened over the original voice joined to it."""
    voice = voice or Voice.load(job, lines)
    if voice is None:
        return {l.id: (l.start, l.end) for l in dubbed}
    floor, ceil = _neighbours(lines)
    return {l.id: voice.joined(l.start, l.end, floor(l), ceil(l)) for l in dubbed}
