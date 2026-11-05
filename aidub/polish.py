"""Smooths a voiced line for the mix: the voice model's output can be harsh, rough or uneven, and
clean in a way the scene isn't. Harsh and hissy moments are turned down only while they last, the
rough top end is smoothed, the level is evened, and a little room is added. Nothing here changes
which words are said, and the finished track is still listened back to afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve, istft, stft

WINDOW_S = 0.02


@dataclass(frozen=True)
class Polish:
    harsh_db: float      # most taken off 2.5-5 kHz when it stands out from the line's usual
    hiss_db: float       # the same for 5-10 kHz: "s" and hiss
    smooth: float        # how much of the sound above 4 kHz is averaged over time, 0-1
    ratio: float         # evening the level: 1 is none
    room_db: float       # how loud the room is against the voice


LEVELS = {"gentle": Polish(harsh_db=2.5, hiss_db=4.0, smooth=0.35, ratio=1.5, room_db=-24.0),
          "strong": Polish(harsh_db=5.0, hiss_db=7.0, smooth=0.6, ratio=2.2, room_db=-18.0)}


def level_for(name) -> Polish | None:
    """The setting's filter; None for off (YAML reads a bare off as false)."""
    return LEVELS.get(str(name).lower())


def _turn_down(power: np.ndarray, band: np.ndarray, most_db: float) -> np.ndarray:
    """dB per frame for one band: nothing at the line's usual level, up to `most_db` off at 6 dB over it."""
    level = 10 * np.log10(power[band].sum(axis=0) + 1e-14)
    voiced = level > level.max() - 40
    over = level - (float(np.median(level[voiced])) if voiced.any() else 0.0)
    down = -np.clip(over / 6.0, 0.0, 1.0) * most_db
    # Falls at once, recovers over about 50 ms: a gain that follows every frame would flutter.
    for k in range(1, len(down)):
        down[k] = min(down[k], down[k - 1] * 0.7)
    return down


def _tone(x: np.ndarray, sr: int, p: Polish) -> np.ndarray:
    size = int(sr * WINDOW_S)
    if len(x) < size * 2:
        return x
    _, _, spec = stft(x, sr, nperseg=size, noverlap=size * 3 // 4, boundary="even", padded=True)
    freqs = np.fft.rfftfreq(size, 1 / sr)
    power = np.abs(spec) ** 2
    gains = np.ones(spec.shape[1:])[None, :] * np.ones((len(freqs), 1))
    for lo, hi, most in ((2500, 5000, p.harsh_db), (5000, 10000, p.hiss_db)):
        band = (freqs >= lo) & (freqs < hi)
        if band.any():
            gains[band] = 10 ** (_turn_down(power, band, most) / 20)
    high = freqs >= 4000
    if p.smooth and high.any() and spec.shape[1] >= 3:
        mag = np.abs(spec[high])
        padded = np.pad(mag, ((0, 0), (1, 1)), mode="edge")
        average = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3
        gains[high] *= ((1 - p.smooth) * mag + p.smooth * average) / (mag + 1e-9)
    _, y = istft(spec * gains, sr, nperseg=size, noverlap=size * 3 // 4, boundary=True)
    return np.resize(y, len(x)).astype(np.float32)


def _even(x: np.ndarray, sr: int, ratio: float, most_db: float = 6.0) -> np.ndarray:
    """Brings the loudest moments towards the line's usual level, by at most `most_db`."""
    if ratio <= 1 or len(x) < sr * 0.05:
        return x
    hop = int(sr * 0.01)
    n = len(x) // hop
    level = 10 * np.log10((x[: n * hop] ** 2).reshape(n, hop).mean(axis=1) + 1e-12)
    voiced = level > level.max() - 30
    usual = float(np.median(level[voiced]))
    down = -np.clip((level - usual) * (1 - 1 / ratio), 0.0, most_db)
    for k in range(1, n):  # comes down in about 20 ms, recovers over about 300 ms
        keep = 0.97 if down[k] > down[k - 1] else 0.5
        down[k] = down[k - 1] * keep + down[k] * (1 - keep)
    at = np.arange(len(x)) / hop
    return (x * 10 ** (np.interp(at, np.arange(n) + 0.5, down) / 20)).astype(np.float32)


@lru_cache(maxsize=4)
def _decay(sr: int, length_s: float = 0.25, low: float = 300.0, high: float = 5000.0) -> np.ndarray:
    """A short decay of noise from `low` to `high` Hz, at unit level within that band. No bass: it
    would add up with the voice's own low notes and thin or muddy them. The same for every line:
    varying noise would change the sound from line to line."""
    n = int(sr * length_s)
    noise = np.fft.rfft(np.random.default_rng(1).normal(0, 1, n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    noise[(freqs < low) | (freqs > high)] = 0.0
    reply = np.fft.irfft(noise, n=n) * np.exp(-6.9 * np.arange(n) / n)
    reply[: int(sr * 0.008)] = 0.0
    return reply / np.sqrt((reply ** 2).sum()) * np.sqrt((high - low) / (sr / 2))


def _room(x: np.ndarray, sr: int, room_db: float) -> np.ndarray:
    """A little room after the voice, longer than `x` by its tail."""
    out = fftconvolve(x, _decay(sr) * 10 ** (room_db / 20)).astype(np.float32)
    out[: len(x)] += x
    return out


def even_within(x: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Levels a line within itself, for mix.even_within."""
    return _even(x.astype(np.float32), sr, ratio, most_db=12.0)


def polish(x: np.ndarray, sr: int, p: Polish) -> np.ndarray:
    return _room(_even(_tone(x.astype(np.float32), sr, p), sr, p.ratio), sr, p.room_db)
