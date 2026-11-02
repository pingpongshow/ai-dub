"""Listening to the finished English track before it is delivered.

The takes are checked before they are mixed, so nothing else hears what actually
reaches the viewer: a track gone silent (a film's last hour once did), a line sped up
in the mix, or a line lost under music. This step measures the track against the
original second by second and recognises every dubbed line where the mix placed it.
A track that is broken stops the job, with the original untouched; a line that only
got hard to hear goes on the check screen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import lines as L
from . import media
from .job import Job

SILENT_DB = -70.0     # as good as nothing
SOUND_DB = -50.0      # the original is playing something
SILENT_RUN_S = 20.0   # a pause in a mix can't last this long while the original plays on
EDGE_S = 0.15


class ReadbackError(RuntimeError):
    pass


def clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def levels(path: Path, hop: float = 1.0) -> np.ndarray:
    """Level in dB of each `hop` seconds of a long audio file, without loading it all."""
    out = []
    with sf.SoundFile(str(path)) as f:
        for block in f.blocks(blocksize=int(f.samplerate * hop), dtype="float32", always_2d=True):
            out.append(10 * np.log10(float(np.mean(block ** 2)) + 1e-12))
    return np.asarray(out)


def silent_runs(dub: np.ndarray, original: np.ndarray, min_run: float = SILENT_RUN_S) -> list[tuple[int, int]]:
    """(from, to) seconds where the dub is silent while the original plays. A dub shorter
    than the original counts as silent for the rest."""
    dub = np.concatenate([dub, np.full(max(0, len(original) - len(dub)), -120.0)])
    gone = (dub[: len(original)] < SILENT_DB) & (original > SOUND_DB)
    # Quiet seconds in the original don't end a run: only the dub playing does.
    playing = dub[: len(original)] >= SILENT_DB
    runs, start, last = [], None, None
    for i in range(len(original)):
        if playing[i]:
            if start is not None and last - start + 1 >= min_run:
                runs.append((start, last + 1))
            start, last = None, None
        elif gone[i]:
            start = i if start is None else start
            last = i
    if start is not None and last - start + 1 >= min_run:
        runs.append((start, last + 1))
    return runs


def placed(ln: L.Line) -> tuple[float, float]:
    """Where a dubbed line plays in the mix."""
    fit = ln.fit or {}
    start = fit.get("placed") if fit.get("placed") is not None else ln.start
    length = float(fit.get("final") or ln.duration) / float(fit.get("paced_tempo") or 1.0)
    return float(start), float(start) + length


def voice_channels(data: np.ndarray) -> np.ndarray:
    """In a 5.1 track the voices are in front: the surrounds and LFE would only hide them."""
    return data[:, :3].mean(axis=1, keepdims=True) if data.shape[1] >= 6 else data


def hard_to_hear(text: str, heard: str, missed: list[str]) -> bool:
    """Key words lost in the mix: any in a short line, two or more in a long one. (A line
    the recogniser hears with nothing of it at all counts whatever its length.)"""
    from .speak import _words
    words = _words(text)
    if not words:
        return False
    if not _words(heard):
        return True
    return len(missed) >= (1 if len(words) <= 6 else 2)


def run(job: Job) -> None:
    if not job.cfg.get("readback.enabled", True):
        return
    from . import speak
    from .pronounce import load as load_lexicon
    dub_path = job.path("mix", "dub.wav")
    job.status("comparing the track with the original")
    dub, original = levels(dub_path), levels(job.path("audio", "source.wav"))
    runs = silent_runs(dub, original)
    job.qc["readback"] = {"silent": [[a, b] for a, b in runs]}
    if runs and not job.options.get("preview_s"):
        where = ", ".join(f"{clock(a)} to {clock(b)}" for a, b in runs[:3])
        raise ReadbackError(f"the English track is silent from {where} while the original plays; nothing was delivered")

    lines = L.load(job.path("lines.json"))
    todo = [ln for ln in lines if ln.kind == L.SPEECH and ln.take and not ln.fallback and ln.en]
    if not todo or not job.cfg.get("stages.recognize.worker"):
        return
    clips = job.dir("mix", "readback")
    items = []
    with sf.SoundFile(str(dub_path)) as f:
        sr = f.samplerate
        for ln in todo:
            a, b = placed(ln)
            f.seek(min(f.frames, max(0, int((a - EDGE_S) * sr))))
            data = voice_channels(f.read(int((b - a + 2 * EDGE_S) * sr), dtype="float32", always_2d=True))
            p = clips / f"{ln.id:05d}.wav"
            media.write_wav(p, data if len(data) else np.zeros((sr // 10, data.shape[1]), np.float32), sr)
            items.append(str(p))
    load_lexicon(job.cfg.models / "lexicon" / "cmudict.dict")
    names = speak.show_names(job.profile)
    heard: list[dict] = []
    cfg = job.cfg.get("stages.recognize")
    with job.worker("recognize") as w:
        w.call("load", {"model": cfg.get("model")})
        for i in range(0, len(items), 16):
            job.check()
            heard += w.call("transcribe", {"items": [{"path": p} for p in items[i:i + 16]], "language": "en",
                                           "timestamps": False})["results"]
            job.status(f"listening back {min(i + 16, len(items))}/{len(items)}", progress=min(1.0, (i + 16) / len(items)))
    lost = 0
    for ln, r in zip(todo, heard):
        text = r.get("text", "")
        missed = [] if speak.pronounce.only_sounds(speak._words(ln.en)) else [
            w for w in speak.missed_words(ln.en, text, names) if w != "not" or speak._negations(speak._words(ln.en))]
        ln.fit.update({"mix_heard": text, "mix_wer": round(speak.wer(ln.en, text, names), 3)})
        # (A neighbouring line at the clip's edge can't make a line count as lost: only its own words count.)
        if hard_to_hear(ln.en, text, missed) and not speak.pronounce.only_sounds(speak._words(ln.en)):
            ln.fit["mix_missed"] = missed or ["all of it"]
            lost += 1
        else:
            ln.fit.pop("mix_missed", None)
    L.save(lines, job.path("lines.json"))
    for p in items:
        Path(p).unlink(missing_ok=True)
    clear = 1 - lost / len(todo)
    job.qc["readback"].update({"lines": len(todo), "hard_to_hear": lost, "clear_share": round(clear, 4)})
    least = float(job.cfg.get("readback.min_clear", 0.5))
    if clear < least and len(todo) >= 20 and not job.options.get("preview_s"):
        raise ReadbackError(f"only {clear:.0%} of the English lines can be heard in the finished track; nothing was delivered")
    job.status(f"{len(todo) - lost} of {len(todo)} lines heard clearly in the finished track")
