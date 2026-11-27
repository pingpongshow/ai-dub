"""Where the original voice really is around each line, from the dialogue's pitch and level.

A line's times come from the words recognition and the aligner found, and the aligner is
sometimes wrong: 何するんだい ("What are you doing?") in Woman in the Dunes got 0.3 s at
3857.24, while the actor says it from 3855.9 to 3857.0. The mix muted only that 0.3 s, so the
Japanese played, and the English came in 1.3 s late. And a line's first or last sound (a
breath of a word, a trailing "ne") is often left out of its words.

retime: a line whose times are far too short for what it says (under half the time it
needs) takes them from the voice heard around it, and drops its word times (fit.timing: sync
then follows the voice as one phrase, not the misplaced words). So does a line far shorter
than the voice under its own subtitle: recognition sometimes hears only a scrap of what is
said (そうそう, 0.3 s, under "That reminds me:" with 1.8 s of voice), so the time its text
needs says nothing; 14 lines of Woman in the Dunes left most of their Japanese playing so.
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
LOUD_DB = 8.0          # ... and this near it, voice with or without a pitch: a shout or a harsh
                       # word has none to the pitch tracker (バカバカしい, Woman in the Dunes 53:54,
                       # at -21 dB began 0.25 s before its line and played under the dub)
REACH_S = 1.0          # a line's voice runs on at most this far past its times ...
SEARCH_S = 1.5         # ... and a line timed far too short looks this far back for its voice
TOO_SHORT = 0.5        # share of the time a line needs, under which its times are wrong
GAP_S = 0.03           # kept clear of the lines either side
CUE_REACH_S = 0.3      # a subtitle's voice: inside its times, this far either side
CUE_SHARE = 0.5        # a line covering under this share of the voice under its own subtitle ...
CUE_MORE_S = 0.4       # ... and at least this much less of it, is timed from that voice

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
    """Voiced frames of the dialogue (10 ms): a pitch and a speaking level, or as loud as
    the film's lines usually are at their loudest (the dialogue stem: little else is that loud)."""

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
        return cls((pitched[:n] & (db > usual - BELOW_USUAL_DB)) | (db > usual - LOUD_DB), hop)

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


def _own_cues(lines: list[L.Line], cues: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """{line id: cue} for subtitles over exactly one speech line (theirs alone, so the voice
    under them is that line's)."""
    speech = [l for l in lines if l.kind == L.SPEECH]
    out = {}
    for a, b in cues:
        on = [l for l in speech if min(b, l.end) - max(a, l.start) > 0]
        if len(on) == 1:
            out[on[0].id] = (a, b)
    return out


def _cues(job: Job) -> list[tuple[float, float]]:
    try:
        from .subtitles.step import load_chosen
        from .subtitles.model import DIALOGUE
        return [(q.start, q.end) for q in load_chosen(job, "english") if q.kind == DIALOGUE and q.text.strip()]
    except Exception:  # no subtitles chosen (or none at all)
        return []


def retime(job: Job, lines: list[L.Line], voice: "Voice | None" = None,
           cues: list[tuple[float, float]] | None = None) -> int:
    """Lines timed far too short for what they say, or for the voice under their own subtitle,
    take their times from the voice around them."""
    dubbed = [l for l in lines if l.dubbed and l.source_text]
    short = {l.id for l in dubbed if needed_s(l.source_text) >= 0.5 and l.duration < TOO_SHORT * needed_s(l.source_text)}
    own = _own_cues(lines, _cues(job) if cues is None else cues)
    todo = [l for l in dubbed if l.id in short or l.id in own]
    if not todo:
        return 0
    voice = voice or Voice.load(job, lines)
    if voice is None:
        return 0
    floor, ceil = _neighbours(lines)
    moved = 0
    for ln in todo:
        runs: list[tuple[float, float]] = []
        if ln.id in own:
            a, b = own[ln.id]
            runs = voice.runs(max(floor(ln), a - CUE_REACH_S), min(ceil(ln), b + CUE_REACH_S))
            heard = sum(e - s for s, e in runs)
            if not (ln.duration < CUE_SHARE * heard and heard - ln.duration >= CUE_MORE_S):
                runs = []
        if not runs and ln.id in short:
            runs = voice.runs(max(floor(ln), ln.start - SEARCH_S), min(ceil(ln), ln.end + REACH_S))
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
