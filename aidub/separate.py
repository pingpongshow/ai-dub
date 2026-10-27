"""Step 1: split the audio into a dialogue stem and a music & effects stem.

Stereo/mono: the whole track is separated. 5.1 and up: the centre channel is
separated on its own and the front left/right pair together (to catch panned
dialogue); the other channels are kept as they are.

Outputs in audio/:
  dialogue.wav  - analysis copy: all dialogue folded to stereo (48 kHz)
  stems.json    - per channel group: {"channels": [...], "dialogue": path, "mne": path}
"""

from __future__ import annotations

import json

import numpy as np

from . import media
from .job import Job

STEP = "separate"


def channel_groups(n: int) -> list[tuple[str, list[int]]]:
    if n >= 6:
        return [("front", [0, 1]), ("centre", [2])]
    return [("main", list(range(n)))]


def run(job: Job) -> None:
    src = job.path("audio", "source.wav")
    data, sr = media.read_wav(src)
    n = data.shape[1]
    groups = channel_groups(n)
    stems = {}
    cfg = job.cfg.get("stages.separate")
    with job.worker("separate") as w:
        w.call("load", {"model": cfg.get("model")})
        job.models["separation"] = cfg.get("model", "")
        for gi, (name, chans) in enumerate(groups):
            job.check()
            part = data[:, chans]
            if part.shape[1] == 1:
                part = np.repeat(part, 2, axis=1)  # separators expect stereo
            inp = media.write_wav(job.path("audio", f"in_{name}.wav"), part, sr)
            job.status(f"separating {name} channels", progress=gi / len(groups))
            res = w.call("separate", {"input": str(inp), "out_dir": str(job.dir("audio", f"sep_{name}"))},
                         on_progress=lambda p, m: job.status(m, progress=(gi + p) / len(groups)))
            stems[name] = {"channels": chans, "dialogue": res["vocals"], "mne": res["instrumental"]}
            inp.unlink(missing_ok=True)

    # Analysis copy of the dialogue, folded to stereo at the source rate.
    fold = None
    for name, st in stems.items():
        d, dsr = media.read_wav(st["dialogue"])
        d = media.resample(d, dsr, sr)
        if len(st["channels"]) == 1:
            d = d.mean(axis=1, keepdims=True).repeat(2, axis=1) * 0.707
        fold = d if fold is None else _add(fold, d)
    media.write_wav(job.path("audio", "dialogue.wav"), fold, sr)
    job.path("audio", "stems.json").write_text(json.dumps(stems, indent=1))
    job.set("separate", {"channels": n, "groups": [g for g, _ in groups]})


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    out = np.zeros((n, a.shape[1]), dtype=np.float32)
    out[: len(a)] += a
    out[: len(b)] += b
    return out
