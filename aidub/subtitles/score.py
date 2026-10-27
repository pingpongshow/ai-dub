"""Syncing a subtitle to the speech and scoring how well it matches.

Scores are numbers in 0..1:
  timing    - after the best offset (and frame-rate scale), how much subtitle
              time lands on speech; penalised for drift or no clear best offset
  coverage  - how much detected speech the subtitle covers (catches signs-only
              and forced tracks)
  content   - agreement with what was actually said: character error rate for
              a source-language subtitle, cross-lingual similarity for English
"""

from __future__ import annotations

import difflib

import numpy as np

from .clean import norm_chars
from .model import DIALOGUE, Cue

RES = 0.05  # seconds per mask step
SCALES = [1.0, 25 / 23.976, 23.976 / 25, 24 / 23.976, 23.976 / 24, 25 / 24, 24 / 25]
MAX_OFFSET = 30.0


def mask(spans: list[tuple[float, float]], duration: float) -> np.ndarray:
    m = np.zeros(int(duration / RES) + 2, dtype=np.float32)
    for a, b in spans:
        i, j = max(0, int(a / RES)), min(len(m), int(b / RES))
        if j > i:
            m[i:j] = 1.0
    return m


def best_offset(cues: list[Cue], speech: np.ndarray, duration: float) -> tuple[float, float, float, float]:
    """(offset, scale, on_speech_fraction, sharpness) maximising overlap with speech."""
    from scipy.signal import fftconvolve
    dial = [q for q in cues if q.kind == DIALOGUE and q.text]
    if not dial:
        return 0.0, 1.0, 0.0, 0.0
    best = (0.0, 1.0, -1.0, 0.0)
    lag_max = int(MAX_OFFSET / RES)
    for scale in SCALES:
        m = mask([(q.start * scale, q.end * scale) for q in dial], duration + MAX_OFFSET)
        total = m.sum()
        if total <= 0:
            continue
        sp = np.zeros_like(m)
        sp[: min(len(sp), len(speech))] = speech[: len(sp)]
        corr = fftconvolve(sp, m[::-1], mode="full")
        zero = len(m) - 1
        lo, hi = max(0, zero - lag_max), min(len(corr), zero + lag_max + 1)
        window = corr[lo:hi]
        k = int(np.argmax(window))
        frac = float(window[k] / total)
        off = (lo + k - zero) * RES
        # sharpness: best peak vs best value at least 2 s away
        far = np.concatenate([window[: max(0, k - int(2 / RES))], window[k + int(2 / RES):]])
        sharp = 1.0 - (float(far.max()) / float(window[k]) if far.size and window[k] > 0 else 0.0)
        if frac > best[2] + 0.01 or (abs(frac - best[2]) <= 0.01 and scale == 1.0):
            best = (off, scale, frac, sharp)
    return best[0], best[1], max(0.0, best[2]), best[3]


def apply_sync(cues: list[Cue], offset: float, scale: float) -> list[Cue]:
    return [Cue(start=q.start * scale + offset, end=q.end * scale + offset, text=q.text, raw=q.raw,
                name=q.name, style=q.style, kind=q.kind) for q in cues]


def drift(cues: list[Cue], speech: np.ndarray, duration: float) -> float:
    """Difference between the best offsets of the first and second half (seconds)."""
    dial = [q for q in cues if q.kind == DIALOGUE and q.text]
    if len(dial) < 20:
        return 0.0
    half = len(dial) // 2
    a = best_offset(dial[:half], speech, duration)[0]
    b = best_offset(dial[half:], speech, duration)[0]
    return abs(a - b)


def coverage(cues: list[Cue], speech: np.ndarray, duration: float) -> float:
    m = mask([(q.start, q.end) for q in cues if q.kind == DIALOGUE and q.text], duration)
    n = min(len(m), len(speech))
    total = speech[:n].sum()
    return float((m[:n] * speech[:n]).sum() / total) if total > 0 else 0.0


def cer(ref: str, hyp: str) -> float:
    a, b = norm_chars(ref), norm_chars(hyp)
    if not a:
        return 1.0 if b else 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    return 1.0 - matched / max(len(a), len(b))


def asr_text_between(asr_segments: list[dict], a: float, b: float) -> str:
    words = [w["text"] for s in asr_segments for w in s.get("words", [])
             if w["end"] > a and w["start"] < b and (min(w["end"], b) - max(w["start"], a)) >= 0.5 * (w["end"] - w["start"])]
    return "".join(words) if words else ""


def content_source(cues: list[Cue], asr_segments: list[dict]) -> float:
    vals = []
    for q in cues:
        if q.kind != DIALOGUE or not q.text:
            continue
        hyp = asr_text_between(asr_segments, q.start - 0.3, q.end + 0.3)
        vals.append(1.0 - min(1.0, cer(q.text, hyp)))
    return float(np.median(vals)) if vals else 0.0


def discrimination(true_sims: list[float], shifted_sims: list[float]) -> float:
    """How reliably each subtitle line matches the speech at its own time better
    than speech from elsewhere (a self-calibrating content score, 0..1).

    A matching subtitle wins nearly every comparison; a wrong episode wins about
    half by chance, which maps to 0."""
    if not true_sims or len(true_sims) != len(shifted_sims):
        return 0.0
    wins = sum(1.0 if t > f else 0.5 if t == f else 0.0 for t, f in zip(true_sims, shifted_sims))
    rate = wins / len(true_sims)
    return float(np.clip((rate - 0.5) / 0.4, 0.0, 1.0))


def combine(timing: float, cov: float, content: float, drift_s: float, sharp: float) -> float:
    t = timing * (1.0 if drift_s < 0.5 else max(0.3, 1.0 - drift_s / 4))
    t *= 0.6 + 0.4 * min(1.0, sharp * 2)
    total = 0.35 * t + 0.25 * cov + 0.40 * content
    if cov < 0.25:  # signs-only or forced track
        total *= 0.5
    return round(float(total), 4)
