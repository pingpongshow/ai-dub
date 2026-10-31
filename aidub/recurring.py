"""Openings and endings found by matching an episode's audio against its siblings.

A show's opening and ending theme are the same recording in every episode, so a
stretch of this episode's sound that also plays in the episode next to it is a
titles sequence. Audio fingerprints in the Philips (Haitsma-Kalker) style, 32
bits a frame from the energy differences of 33 bands, are compared over the
first and last minutes. Everything runs locally; the siblings' fingerprints are
cached in the show's profile folder.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np

from . import media

SR = 5512
FRAME = 2048          # ~0.37 s window
HOP = 0.05            # seconds between fingerprint frames
BANDS = np.geomspace(300, 2000, 34)
START_S = 600.0       # an opening lies in the first 10 minutes ...
END_S = 480.0         # ... an ending in the last 8
MATCH_BER = 0.35      # bit errors per frame below this: the same sound
MIN_RUN_S = 20.0      # a recurring stretch shorter than this isn't a titles sequence
SMOOTH_S = 2.0        # bit errors averaged over this long when following a match
EPISODE = re.compile(r"[Ss](\d+)[ ._-]*[Ee](\d+)|(?<![a-z\d])(\d{1,3})(?=\D*$)")


def fingerprint(x: np.ndarray) -> np.ndarray:
    """32-bit sub-fingerprints (uint32) of mono audio at SR, one every HOP seconds."""
    hop = int(HOP * SR)
    if len(x) < FRAME + hop:
        return np.zeros(0, np.uint32)
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float32), FRAME)[::hop] * np.hanning(FRAME)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    idx = np.searchsorted(freqs, BANDS)
    energy = np.stack([spec[:, a:max(a + 1, b)].sum(axis=1) for a, b in zip(idx[:-1], idx[1:])], axis=1)
    d = energy[:, :-1] - energy[:, 1:]                     # 32 band differences
    bits = (d[1:] - d[:-1]) > 0                            # ... changing over time
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32).ravel()


def _ber(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a, b)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 4), axis=1).sum(axis=1) / 32.0


def common(a: np.ndarray, b: np.ndarray) -> list[tuple[float, float]]:
    """Stretches of `a` (start, end in seconds from a's start) that also play somewhere in `b`."""
    if not a.size or not b.size:
        return []
    where: dict[int, list[int]] = {}
    for j, v in enumerate(b.tolist()):
        where.setdefault(v, []).append(j)
    votes: dict[int, int] = {}
    for i, v in enumerate(a.tolist()):
        for j in where.get(v, ())[:8]:
            votes[j - i] = votes.get(j - i, 0) + 1
    out = []
    k = max(1, int(SMOOTH_S / HOP))
    for off, n in sorted(votes.items(), key=lambda kv: -kv[1])[:5]:
        if n < 20:
            break
        lo, hi = max(0, -off), min(a.size, b.size - off)
        if hi - lo < k:
            continue
        ber = _ber(a[lo:hi], b[lo + off:hi + off])
        same = np.convolve(ber, np.ones(k) / k, mode="same") < MATCH_BER
        edges = np.flatnonzero(np.diff(np.concatenate([[0], same.astype(np.int8), [0]])))
        for s, e in zip(edges[::2], edges[1::2]):
            if (e - s) * HOP >= MIN_RUN_S:
                out.append(((lo + s) * HOP, (lo + e) * HOP))
    return merge(out)


def merge(spans: list[tuple[float, float]], gap: float = 1.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _part(wav: Path, start: float, seconds: float) -> np.ndarray:
    """A stretch of a WAV file, mono at SR, without reading the rest."""
    import soundfile as sf
    with sf.SoundFile(str(wav)) as f:
        f.seek(min(f.frames, int(start * f.samplerate)))
        x = f.read(int(seconds * f.samplerate), dtype="float32", always_2d=True).mean(axis=1, keepdims=True)
        sr = f.samplerate
    return media.resample(x, sr, SR)[:, 0]


def _audio(path: Path, start: float, seconds: float, stream: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        wav = media.extract_audio(path, stream, Path(d) / "fp.wav", rate=SR, channels=1, start=start, duration=seconds)
        x, _ = media.read_wav(wav, mono=True)
    return x[:, 0]


def _sibling_prints(path: Path, cache: Path) -> dict | None:
    """Start and end fingerprints of another episode (its original-language track), cached."""
    st = path.stat()
    key = hashlib.sha1(f"{path.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]
    f = cache / f"{key}.npz"
    if f.exists():
        d = np.load(f)
        return {"start": d["start"], "end": d["end"], "end_at": float(d["end_at"])}
    try:
        pr = media.probe(path)
    except Exception:
        return None
    tracks = [s for s in pr.streams if s.type == "audio" and not s.is_aidub]
    if not tracks or not pr.duration:
        return None
    stream = tracks[0].index
    end_at = max(0.0, pr.duration - END_S)
    out = {"start": fingerprint(_audio(path, 0.0, START_S, stream)),
           "end": fingerprint(_audio(path, end_at, END_S, stream)), "end_at": end_at}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


def _number(name: str) -> tuple[int, int] | None:
    m = EPISODE.search(Path(name).stem)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2))) if m.group(1) else (0, int(m.group(3)))


def siblings(src: Path, n: int = 2) -> list[Path]:
    """Up to n other episodes in the same folder, nearest episode numbers first."""
    me = _number(src.name)
    files = [p for p in src.parent.iterdir()
             if p != src and p.is_file() and p.suffix.lower() in media.VIDEO_EXT and not p.name.startswith(".")]
    if me:
        files = [p for p in files if _number(p.name)]
        files.sort(key=lambda p: (_number(p.name)[0] != me[0], abs(_number(p.name)[1] - me[1]), p.name))
    else:
        files.sort(key=lambda p: p.name)
    return files[:n]


def find(job) -> list[list[float]]:
    """Recurring stretches (opening and ending themes) of this episode, in seconds."""
    duration = float(job["ingest"]["duration"])
    source = job.path("audio", "source.wav")
    end_at = max(0.0, duration - END_S)
    mine_start = fingerprint(_part(source, 0.0, START_S))
    mine_end = fingerprint(_part(source, end_at, END_S))
    cache = job.profile.root / "fingerprints" if job.profile else job.work / "fingerprints"
    found: list[tuple[float, float]] = []
    used = []
    for sib in siblings(Path(job.src)):
        other = _sibling_prints(sib, cache)
        if other is None:
            continue
        used.append(sib.name)
        found += common(mine_start, np.concatenate([other["start"], other["end"]]))
        found += [(a + end_at, b + end_at) for a, b in common(mine_end, np.concatenate([other["end"], other["start"]]))]
    spans = [[round(a, 2), round(min(b, duration), 2)] for a, b in merge(found)]
    job.set("recurring", {"spans": spans, "compared_with": used})
    return spans
