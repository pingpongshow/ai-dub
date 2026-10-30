"""Step 9: mix the dubbed lines over the music & effects.

The original dialogue stem stays everywhere except under dubbed lines, so
wordless sounds, English lines, songs and lines that fell back are carried
straight into the mix at their original times. Each dubbed line is matched to
the original line's loudness, rough EQ and stereo position. The result is
normalised to the original track's integrated loudness (EBU R128) and, for
5.1 and up, folded down to stereo with standard coefficients.
"""

from __future__ import annotations

import json

import numpy as np
import pyloudnorm as pyln

from . import lines as L
from . import media
from .fit import free_timing
from .job import Job

STEP = "mix"
SR = 48000
BANDS = [(0, 300), (300, 3000), (3000, 24000)]


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad = np.zeros((n - len(x),) + x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, pad])


def band_gains_db(ref: np.ndarray, sig: np.ndarray, sr: int, max_db: float) -> list[float]:
    def energy(x):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return [float(spec[(f >= a) & (f < b)].sum()) + 1e-12 for a, b in BANDS]
    if len(ref) < 512 or len(sig) < 512:
        return [0.0] * len(BANDS)
    er, es = energy(ref), energy(sig)
    tr, ts = sum(er), sum(es)
    out = []
    for a, b in zip(er, es):
        g = 10 * np.log10((a / tr) / (b / ts))
        out.append(float(np.clip(g, -max_db, max_db)))
    return out


def apply_band_gains(x: np.ndarray, sr: int, gains_db: list[float]) -> np.ndarray:
    if not any(abs(g) > 0.1 for g in gains_db):
        return x
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    centres = [150.0, 1000.0, 8000.0]
    curve = np.interp(np.log10(np.maximum(f, 20)), np.log10(centres), gains_db)
    return np.fft.irfft(spec * 10 ** (curve / 20), n=n).astype(np.float32)


def active_rms(x: np.ndarray, sr: int) -> float:
    hop = int(sr * 0.02)
    if len(x) < hop:
        return float(np.sqrt(np.mean(x ** 2) + 1e-12))
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    e = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    loud = e[e > e.max() * 0.1]
    return float(np.sqrt(np.mean(loud ** 2))) if loud.size else float(e.mean())


def fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp if x.ndim == 1 else ramp[:, None]
        x[-n:] *= ramp[::-1] if x.ndim == 1 else ramp[::-1, None]
    return x


def outside(span: tuple[float, float], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of `span` not covered by any of `holes`."""
    parts = [span]
    for h0, h1 in holes:
        parts = [q for a, b in parts for q in ((a, min(b, h0)), (max(a, h1), b)) if q[1] - q[0] > 1e-3]
    return parts


def outside_all(spans: list[tuple[float, float]], holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [q for sp in spans for q in outside(sp, holes)]


def keep_gain(n: int, spans: list[tuple[float, float]], ramp_s: float) -> np.ndarray:
    """1 where the original dialogue is kept, 0 under dubbed lines, with short ramps."""
    g = np.ones(n, dtype=np.float32)
    r = int(ramp_s * SR)
    for a, b in spans:
        i, j = max(0, int(a * SR)), min(n, int(b * SR))
        if j <= i:
            continue
        g[i:j] = 0.0
        if r:
            lo = max(0, i - r)
            g[lo:i] = np.minimum(g[lo:i], np.linspace(1, 0, i - lo, dtype=np.float32))
            hi = min(n, j + r)
            g[j:hi] = np.minimum(g[j:hi], np.linspace(0, 1, hi - j, dtype=np.float32))
    return g


def soft_limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    c = 10 ** (ceiling_db / 20)
    over = np.abs(x) > c * 0.9
    if over.any():
        knee = c * 0.9
        y = x.copy()
        s = np.sign(x[over])
        y[over] = s * (knee + (c - knee) * np.tanh((np.abs(x[over]) - knee) / (c - knee)))
        return y
    return x


def run(job: Job) -> None:
    lines = L.load(job.path("lines.json"))
    stems = json.loads(job.path("audio", "stems.json").read_text())
    src, src_sr = media.read_wav(job.path("audio", "source.wav"))
    n_ch = src.shape[1]
    total = int(job["ingest"]["duration"] * SR) + SR
    cfg = job.cfg
    fade_n = int(float(cfg.get("mix.fade_ms", 12)) / 1000 * SR)

    groups = {}
    for name, st in stems.items():
        d, dsr = media.read_wav(st["dialogue"])
        m, msr = media.read_wav(st["mne"])
        d, m = _fit_len(media.resample(d, dsr, SR), total), _fit_len(media.resample(m, msr, SR), total)
        if len(st["channels"]) == 1:
            d, m = d.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)
        groups[name] = {"dlg": d, "mne": m, "dub": np.zeros_like(d)}

    dubbed = sorted((ln for ln in lines if ln.take and ln.kind == L.SPEECH and not ln.fallback), key=lambda l: l.start)
    free = free_timing(job)
    gap = float(cfg.get("fit.free_gap_ms", 120)) / 1000
    spans, waited, busy_until = [], [], 0.0
    # Lines left in the original (short reactions, songs, lines with no usable take) stay
    # audible even where a dub runs on over them; only the dubbed lines' own speech is muted.
    dub_ids = {l.id for l in dubbed}
    kept = [(l.start, l.end) for l in lines if l.id not in dub_ids]
    own = [(l.start, l.end) for l in dubbed]
    over_kept = placed = 0
    for ln in dubbed:
        take, tsr = media.read_wav(ln.take, mono=True)
        x = media.resample(take, tsr, SR)[:, 0].copy()
        start = ln.start
        if free:
            # Natural length: the line starts with the original, unless the English line
            # before it is still speaking; then it follows that one (never two at once).
            start = max(ln.start, busy_until + gap)
            if start > ln.start + 1e-3:
                waited.append(start - ln.start)
            busy_until = start + len(x) / SR
            ln.fit["placed"] = round(start, 3)
        a = min(int(start * SR), total)  # free timing can push a line past the end
        b = min(total, a + len(x))
        x = x[: b - a]
        if len(x) == 0:
            continue
        # Share of this line in each channel group, from the original dialogue.
        energies = {g: float(np.mean(v["dlg"][int(ln.start * SR): int(ln.end * SR)] ** 2) + 1e-12)
                    for g, v in groups.items()}
        etot = sum(energies.values())
        ref = sum(v["dlg"][int(ln.start * SR): int(ln.end * SR)].mean(axis=1) for v in groups.values())
        if cfg.get("mix.eq_match", True):
            x = apply_band_gains(x, SR, band_gains_db(ref, x, SR, float(cfg.get("mix.eq_max_db", 6.0))))
        target = active_rms(ref, SR) if len(ref) else 10 ** (ln.loudness_db / 20)
        x *= target / max(active_rms(x, SR), 1e-6)
        x = fade(x.astype(np.float32), fade_n)
        for g, v in groups.items():
            share = np.sqrt(energies[g] / etot)
            if share < 0.05:
                continue
            seg = v["dlg"][int(ln.start * SR): int(ln.end * SR)]
            if v["dub"].shape[1] == 2:
                l, r = float(np.sqrt(np.mean(seg[:, 0] ** 2) + 1e-12)), float(np.sqrt(np.mean(seg[:, 1] ** 2) + 1e-12))
                pan = (r - l) / (r + l)
                theta = (pan + 1) * np.pi / 4
                v["dub"][a:b, 0] += x * share * np.cos(theta) * np.sqrt(2)
                v["dub"][a:b, 1] += x * share * np.sin(theta) * np.sqrt(2)
            else:
                v["dub"][a:b, 0] += x * share
        placed += 1
        spans.append((min(ln.start, start) - 0.04, ln.end + 0.04))
        run_on = (ln.end + 0.04, start + len(x) / SR + 0.04)
        if run_on[1] > run_on[0]:
            holes = [(k0, k1) for k0, k1 in kept if k1 > run_on[0] and k0 < run_on[1]]
            over_kept += bool(holes)
            spans.extend(outside(run_on, outside_all(holes, own)))

    g = keep_gain(total, spans, ramp_s=0.03)[:, None]
    duck = 10 ** (-abs(float(cfg.get("mix.extra_duck_db", 0.0))) / 20)
    duck_env = np.where(g < 1.0, duck, 1.0).astype(np.float32) if duck < 1.0 else 1.0
    for v in groups.values():
        v["out"] = v["mne"] * duck_env + v["dlg"] * g + v["dub"]

    if n_ch >= 6 and "front" in groups and "centre" in groups:
        rest = _fit_len(media.resample(src[:, 3:], src_sr, SR), total)  # LFE, back, side
        fl, fr = groups["front"]["out"][:, 0], groups["front"]["out"][:, 1]
        fc = groups["centre"]["out"][:, 0]
        surr = rest[:, 1:]  # skip LFE (index 3 in the source)
        sl = surr[:, 0::2].sum(axis=1) if surr.shape[1] else 0
        sr_ = surr[:, 1::2].sum(axis=1) if surr.shape[1] else 0
        k = 1 / np.sqrt(2)
        out = np.stack([fl + k * fc + k * sl, fr + k * fc + k * sr_], axis=1)
        orig_st = np.stack([src[:, 0] + k * src[:, 2] + k * src[:, 4::2].sum(axis=1),
                            src[:, 1] + k * src[:, 2] + k * src[:, 5::2].sum(axis=1)], axis=1)
    else:
        out = groups["main"]["out"]
        if out.shape[1] == 1:
            out = np.repeat(out, 2, axis=1)
        orig_st = src if src.shape[1] >= 2 else np.repeat(src, 2, axis=1)
        orig_st = orig_st[:, :2]

    out = out[: int(job["ingest"]["duration"] * SR)]
    meter = pyln.Meter(SR)
    orig48 = media.resample(orig_st.astype(np.float32), src_sr, SR)
    try:
        target = meter.integrated_loudness(orig48)
        now = meter.integrated_loudness(out)
        if np.isfinite(target) and np.isfinite(now):
            out = out * 10 ** ((target - now) / 20)
        job.qc["loudness_lufs"] = round(float(target), 1)
    except ValueError:
        pass
    out = soft_limit(out.astype(np.float32))
    media.write_wav(job.path("mix", "dub.wav"), out, SR)
    job.qc["dubbed_lines"] = placed
    if free:
        job.qc["free_timing"] = {"lines_waited": len(waited), "max_wait_s": round(max(waited, default=0.0), 2),
                                 "ran_over_kept_lines": over_kept,
                                 "mean_wait_s": round(sum(waited) / len(waited), 2) if waited else 0.0}
        L.save(lines, job.path("lines.json"))
    job.status(f"mixed {placed} dubbed lines")
