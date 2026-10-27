"""Step 7b: match the original's sound quality.

Voice models mostly output 22-24 kHz audio, dull next to 48 kHz music and
effects. A bandwidth-extension model restores the top end (48 kHz out).
Skipped when the voice model already outputs 44.1/48 kHz or the stage is off.
"""

from __future__ import annotations

from pathlib import Path

from . import lines as L
from .job import Job

STEP = "bwe"


def run(job: Job) -> None:
    caps = job.get("tts_caps", {})
    if not job.cfg.get("stages.bwe.enabled", True):
        job.status("bandwidth extension is off")
        return
    if int(caps.get("sample_rate", 24000)) >= 44100:
        job.status("voice model is already full-band")
        return
    lines = L.load(job.path("lines.json"))
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
