"""Step 9: mix the dubbed lines over the music & effects.

The original dialogue stem stays everywhere except under dubbed lines, so
wordless sounds, English lines, songs and lines that fell back are carried
straight into the mix at their original times. Each dubbed line is matched to
the original line's loudness, rough EQ and stereo position. The result is
normalised to the original track's integrated loudness (EBU R128). A 5.1 original
keeps its layout; other layouts over stereo are folded down with standard coefficients.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from . import fit as F
from . import lines as L
from . import media
from . import polish as P
from .fit import free_timing, pace
from .job import Job

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


def even_levels(starts: list[float], levels: list[float], spread: float, around_s: float = 30.0) -> list[float]:
    """Brings lines' levels closer together: each keeps `spread` of its difference in dB from the
    scene's usual level (1: as the film has them, 0: all alike). The film's own gaps sound natural
    with its actors, but in one dub voice they stand out, and each voice seems spliced in. Shouts
    and whispers stay louder and quieter than the rest."""
    if spread >= 1 or not levels:
        return list(levels)
    db = 20 * np.log10(np.maximum(levels, 1e-6))
    at = np.array(starts)
    out = []
    for t, d in zip(at, db):
        usual = float(np.median(db[np.abs(at - t) <= around_s]))
        out.append(float(10 ** ((usual + (d - usual) * spread) / 20)))
    return out


def between_lines(n: int, lines: list[L.Line], before: float = 0.15, after: float = 0.35) -> np.ndarray:
    """True where no line is speaking or dying away."""
    free = np.ones(n, dtype=bool)
    for ln in lines:
        free[max(0, int((ln.start - before) * SR)): max(0, int((ln.end + after) * SR))] = False
    return free


def _per_block(x: np.ndarray, step: int, stat) -> np.ndarray:
    """`stat` of each `step` samples, a slice at a time."""
    n = len(x) // step
    parts = []
    for k in range(0, n, 65536):
        part = x[k * step: min(k + 65536, n) * step]
        parts.append(stat(part.reshape(-1, step, *x.shape[1:])))
    return np.concatenate(parts) if parts else np.zeros(0)


def room_tone(dlg: np.ndarray, g: np.ndarray, free: np.ndarray, grain_s: float = 0.2, search_s: float = 10.0) -> np.ndarray:
    """The scene's own background for where the dialogue stem is muted. In a noisy scene separation
    leaves much of the noise in that stem, as loud as the rest of the background, so muting it under
    each line makes the noise dip as the voice starts: the voice sounds spliced in. Short random grains
    from the quiet moments nearby, overlapped so the level holds; nothing where there are none."""
    out = np.zeros_like(dlg)
    muted = g < 1.0
    edges = np.flatnonzero(np.diff(muted.astype(np.int8)))
    starts, ends = list(edges[~muted[edges]] + 1), list(edges[muted[edges]] + 1)
    if muted[0]:
        starts.insert(0, 0)
    if muted[-1]:
        ends.append(len(muted))
    size = int(grain_s * SR)
    hop, step = size // 2, max(1, size // 4)
    window = np.sin(np.pi * (np.arange(size) + 0.5) / size)[:, None].astype(np.float32)  # the level holds at half overlap
    # Places a whole grain fits between lines, and how loud each is (a film is far too long for per-sample sums).
    blocks = size // step
    clear = _per_block(free, step, lambda b: b.all(axis=1))
    power = _per_block(dlg, step, lambda b: (b ** 2).mean(axis=(1, 2)))
    if len(clear) < blocks:
        return out
    fits = np.convolve(clear.astype(np.int32), np.ones(blocks, dtype=np.int32), mode="valid") == blocks
    places = np.flatnonzero(fits)
    if not len(places):
        return out
    level = np.convolve(power, np.ones(blocks) / blocks, mode="valid")[places]
    places = places * step
    rnd = np.random.default_rng(0)
    for a, b in zip(starts, ends):
        lo, hi = np.searchsorted(places, [a - int(search_s * SR), b + int(search_s * SR)])
        if hi - lo < 4:
            continue
        near, loud = places[lo:hi], level[lo:hi]
        # Only the quietest moments: in a busy scene most "gaps" hold voices no line covers, and
        # repeated under a dubbed line their fragments sound like a ghostly echo of speech.
        near = near[loud <= float(np.percentile(loud, 15))]
        if len(near) < 3:
            continue
        for at in range(max(0, a - hop), min(b + hop, len(out) - size), hop):
            k = int(rnd.choice(near))
            out[at:at + size] += dlg[k:k + size] * window
    return out


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


def start_floors(job: Job, lines: list[L.Line], dubbed: list[L.Line]) -> list[float]:
    """How early each dubbed line may begin: not back across a shot change (the voice would
    start in the shot before), nor over a line left in the original."""
    cuts = sorted((job.get("scenes") or {}).get("cuts", []))
    dub_ids = {l.id for l in dubbed}
    kept = sorted(l.end for l in lines if l.id not in dub_ids and l.kind in (L.SPEECH, L.ENGLISH, L.WORDLESS, L.SONG))
    out = []
    for ln in dubbed:
        k = bisect.bisect_right(cuts, ln.start + 0.05) - 1
        e = bisect.bisect_right(kept, ln.start) - 1
        out.append(max(min(ln.start, cuts[k]) if k >= 0 else float("-inf"), kept[e] if e >= 0 else float("-inf")))
    return out


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
    # (the dub never goes in the surrounds, and they don't count towards a line's level or position)
    around = None
    if job.path("audio", "surround.json").exists():
        st = json.loads(job.path("audio", "surround.json").read_text())
        d, dsr = media.read_wav(st["dialogue"])
        m, msr = media.read_wav(st["mne"])
        around = {"dlg": _fit_len(media.resample(d, dsr, SR), total), "mne": _fit_len(media.resample(m, msr, SR), total)}

    dubbed = sorted((ln for ln in lines if ln.take and ln.kind == L.SPEECH and not ln.fallback), key=lambda l: l.start)
    free = free_timing(job)
    gap = float(cfg.get("fit.free_gap_ms", 120)) / 1000
    takes = []
    for ln in dubbed:
        take, tsr = media.read_wav(ln.take, mono=True)
        takes.append(media.resample(take, tsr, SR)[:, 0].copy())
    plan = pace([(ln.start, len(x) / SR) for ln, x in zip(dubbed, takes)], gap,
                float(cfg.get("fit.free_max_late_ms", 350)) / 1000, float(cfg.get("fit.force_stretch_max", 1.3)),
                early=float(cfg.get("fit.free_early_ms", 400)) / 1000, floors=start_floors(job, lines, dubbed),
                gentle_tempo=float(cfg.get("fit.free_gentle_tempo", 1.12))) if free else []
    spans, waited, sped = [], [], 0
    # A line whose original voice is nearly gone (left in the music by separation, or
    # whispered under it) would be dubbed as quietly: no quieter than this under the film's usual level.
    voice_refs = [sum(v["dlg"][int(ln.start * SR): int(ln.end * SR)].mean(axis=1) for v in groups.values()) for ln in dubbed]
    usual = float(np.median([active_rms(r, SR) for r in voice_refs if len(r)] or [0.0]))
    wanted = even_levels([ln.start for ln in dubbed],
                         [active_rms(r, SR) if len(r) else 10 ** (ln.loudness_db / 20) for r, ln in zip(voice_refs, dubbed)],
                         float(cfg.get("mix.level_spread", 0.0)))
    least = usual * 10 ** (float(cfg.get("mix.min_level_db", -12.0)) / 20)
    raised = 0
    # Lines left in the original (short reactions, songs, lines with no usable take) stay
    # audible even where a dub runs on over them; only the dubbed lines' own speech is muted.
    dub_ids = {l.id for l in dubbed}
    kept = [(l.start, l.end) for l in lines if l.id not in dub_ids]
    own = [(l.start, l.end) for l in dubbed]
    over_kept = placed = 0
    smoothing = P.level_for(cfg.get("mix.polish", "off"))
    even_within = float(cfg.get("mix.even_within", 4.0))
    for i, ln in enumerate(dubbed):
        x = takes[i]
        start = ln.start
        if free:
            # Natural length: the line starts with the original, unless the English line
            # before it is still speaking; then it follows that one (never two at once),
            # and that one was sped up if it would have made this one late.
            start, tempo = plan[i]
            if tempo > 1.001:
                fast = F.time_stretch(Path(ln.take), job.path("mix", "paced", f"{ln.id}.wav"), tempo)
                y, ysr = media.read_wav(fast, mono=True)
                x = media.resample(y, ysr, SR)[:, 0].copy()
                ln.fit["paced_tempo"] = tempo
                sped += 1
            if start > ln.start + 1e-3:
                waited.append(start - ln.start)
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
        if smoothing:
            x = P.polish(x, SR, smoothing)[: total - a]  # (now longer by the room's tail)
            b = a + len(x)
        if even_within > 1:
            x = P.even_within(x, SR, even_within)
        target = wanted[i]
        raised += target < least
        x *= min(max(target, least) / max(active_rms(x, SR), 1e-6), 30.0)
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
    quiet = between_lines(total, lines) if cfg.get("mix.room_tone", True) else None
    # A long 5.1 film runs to tens of GB here: each stem goes as soon as it's in its channel's mix.
    for v in [*groups.values(), *([around] if around else [])]:
        mixed = v.pop("mne")
        if duck < 1.0:
            mixed *= duck_env
        if quiet is not None:
            room = room_tone(v["dlg"], g[:, 0], quiet)
            room *= 1 - g
            mixed += room
            del room
        kept_part = v.pop("dlg")
        kept_part *= g
        mixed += kept_part
        del kept_part
        if "dub" in v:
            mixed += v.pop("dub")
        v["out"] = mixed
    del quiet

    surround = n_ch == 6 and bool(cfg.get("deliver.surround", True))
    full = None
    if n_ch >= 6 and "front" in groups and "centre" in groups:
        rest = _fit_len(media.resample(src[:, 3:], src_sr, SR), total)  # LFE, back, side
        fl, fr = groups["front"]["out"][:, 0], groups["front"]["out"][:, 1]
        fc = groups["centre"]["out"][:, 0]
        surr = rest[:, 1:]  # skip LFE (index 3 in the source)
        if around:
            surr[:, :2] = around.pop("out")
        sl = surr[:, 0::2].sum(axis=1) if surr.shape[1] else 0
        sr_ = surr[:, 1::2].sum(axis=1) if surr.shape[1] else 0
        k = 1 / np.sqrt(2)
        if surround:  # the original's own layout, with the dub in its front and centre channels
            full = np.stack([fl, fr, fc, rest[:, 0], surr[:, 0], surr[:, 1]], axis=1)
        out = np.stack([fl + k * fc + k * sl, fr + k * fc + k * sr_], axis=1)
        orig_st = np.stack([src[:, 0] + k * src[:, 2] + k * src[:, 4::2].sum(axis=1),
                            src[:, 1] + k * src[:, 2] + k * src[:, 5::2].sum(axis=1)], axis=1)
        del rest, surr, sl, sr_, fl, fr, fc
        groups.clear()
    else:
        out = groups["main"]["out"]
        if out.shape[1] == 1:
            out = np.repeat(out, 2, axis=1)
        orig_st = src if src.shape[1] >= 2 else np.repeat(src, 2, axis=1)
        orig_st = orig_st[:, :2]

    del src
    out = out[: int(job["ingest"]["duration"] * SR)]
    if full is not None:
        full = full[: len(out)]
    meter = pyln.Meter(SR)
    orig48 = media.resample(orig_st.astype(np.float32, copy=False), src_sr, SR)
    del orig_st
    try:
        target = meter.integrated_loudness(orig48)
        now = meter.integrated_loudness(out)
        if np.isfinite(target) and np.isfinite(now):
            out *= 10 ** ((target - now) / 20)
            if full is not None:
                full *= 10 ** ((target - now) / 20)
        job.qc["loudness_lufs"] = round(float(target), 1)
    except ValueError:
        pass
    del orig48
    out = soft_limit((full if full is not None else out).astype(np.float32, copy=False))
    media.write_wav(job.path("mix", "dub.wav"), out, SR)
    job.qc["dub_channels"] = out.shape[1]
    job.qc["dubbed_lines"] = placed
    job.qc["quiet_lines_raised"] = raised
    if free:
        job.qc["free_timing"] = {"lines_waited": len(waited), "max_wait_s": round(max(waited, default=0.0), 2),
                                 "lines_early": sum(s < ln.start - 0.02 for (s, _), ln in zip(plan, dubbed)),
                                 "sped_up": sped, "ran_over_kept_lines": over_kept,
                                 "mean_wait_s": round(sum(waited) / len(waited), 2) if waited else 0.0}
        L.save(lines, job.path("lines.json"))
    job.status(f"mixed {placed} dubbed lines")
