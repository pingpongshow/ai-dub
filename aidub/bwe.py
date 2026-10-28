"""Step 7b: match the original's sound quality.

Voice models mostly output 22-24 kHz audio, which can sound dull next to full-band
dialogue. A bandwidth-extension model restores the top end (48 kHz out), but
only when the original dialogue actually has more top end than the voice:
streaming sources are often band-limited already, and extending the dub there
makes it brighter than the scene around it.

Modes (`stages.bwe.mode`): auto (default; compare and decide), always, never.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import lines as L
from . import media
from .job import Job

STEP = "bwe"


def high_band_share(x: np.ndarray, sr: int, cutoff: float = 8000.0) -> float:
    """Share of spectral energy above `cutoff` (mono signal)."""
    if len(x) < 1024 or sr <= 2 * cutoff:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spec[f >= cutoff].sum() / max(spec.sum(), 1e-12))


def original_is_brighter(job: Job, lines: list[L.Line], margin: float = 1.25) -> tuple[bool, float, float]:
    """Compare the top end of the original dialogue with the voice takes."""
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    speech = [ln for ln in lines if ln.kind == L.SPEECH][:60]
    orig = [high_band_share(dlg[int(ln.start * sr): int(ln.end * sr), 0], sr) for ln in speech]
    takes = []
    for ln in [l for l in lines if l.take][:60]:
        x, tsr = media.read_wav(ln.take, mono=True)
        takes.append(high_band_share(media.resample(x, tsr, 48000)[:, 0], 48000))
    o = float(np.median(orig)) if orig else 0.0
    t = float(np.median(takes)) if takes else 0.0
    return o > t * margin, o, t


def run(job: Job) -> None:
    caps = job.get("tts_caps", {})
    mode = str(job.cfg.get("stages.bwe.mode", "auto")).lower()
    if not job.cfg.get("stages.bwe.enabled", True) or mode == "never":
        job.status("bandwidth extension is off")
        return
    if int(caps.get("sample_rate", 24000)) >= 44100:
        job.status("voice model is already full-band")
        return
    lines = L.load(job.path("lines.json"))
    if mode == "auto":
        brighter, o, t = original_is_brighter(job, lines)
        job.qc["bandwidth"] = {"original_above_8k": round(o, 4), "voice_above_8k": round(t, 4), "extended": brighter}
        if not brighter:
            job.status("original dialogue isn't brighter than the voice; no extension needed")
            return
    items = []
    for ln in lines:
        if ln.take and ln.take.endswith("final.wav"):
            out = Path(ln.take).with_name("final48.wav")
            items.append((ln, {"input": ln.take, "output": str(out)}))
    if not items:
        return
    with job.worker("bwe") as w:
        info = w.call("load", {})
        job.models["bandwidth_extension"] = info.get("model", "")
        for i in range(0, len(items), 16):
            job.check()
            w.call("upsample", {"items": [x for _, x in items[i:i + 16]]})
            job.status(f"{min(i + 16, len(items))}/{len(items)} lines", progress=min(1.0, (i + 16) / len(items)))
    for ln, x in items:
        if Path(x["output"]).exists():
            ln.take = x["output"]
    L.save(lines, job.path("lines.json"))
