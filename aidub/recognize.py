"""Step 3a: speech recognition with word timings on the dialogue stem.

Runs on speech regions from diarization, grouped into chunks of up to ~25 s,
so nothing is cut mid-word. Also detects the language per chunk.

Saves `asr`: {"language": "ja", "segments": [{start, end, text, language, words}]}
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import media
from .job import Job

STEP = "recognize"
CHUNK_MAX_S = 25.0
PAD_S = 0.15


def speech_chunks(segments: list[dict], duration: float, max_len: float = CHUNK_MAX_S,
                  gap: float = 0.6) -> list[tuple[float, float]]:
    """Merge diarized turns into chunks, splitting only in gaps between turns."""
    spans = sorted((max(0.0, s["start"] - PAD_S), min(duration, s["end"] + PAD_S)) for s in segments)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a - out[-1][1] <= gap and b - out[-1][0] <= max_len:
            out[-1][1] = max(out[-1][1], b)
        elif out and a < out[-1][1]:
            out[-1][1] = max(out[-1][1], b)  # overlapping turns stay together
        else:
            out.append([a, b])
    final = []
    for a, b in out:
        while b - a > max_len * 1.6:  # one very long turn: split evenly
            final.append((a, a + max_len))
            a += max_len
        final.append((a, b))
    return final


def run(job: Job) -> None:
    ing = job["ingest"]
    segs = job["diarization"]["segments"]
    chunks = speech_chunks(segs, ing["duration"])
    data, sr = media.read_wav(job.path("audio", "dialogue16k.wav"), mono=True)
    cfg = job.cfg.get("stages.recognize")
    lang_hint = ing.get("language") or None
    out = []
    with job.worker("recognize") as w:
        info = w.call("load", {"model": cfg.get("model"), "aligner": cfg.get("aligner")})
        job.models["recognition"] = info.get("model", cfg.get("model", ""))
        batch = []
        for i, (a, b) in enumerate(chunks):
            p = media.write_wav(job.path("asr", f"c{i:05d}.wav"), data[int(a * sr): int(b * sr)], sr)
            batch.append({"path": str(p), "offset": a, "end": b})
        for i in range(0, len(batch), 16):
            job.check()
            part = batch[i:i + 16]
            res = w.call("transcribe", {"items": [{"path": x["path"]} for x in part],
                                        "language": lang_hint, "timestamps": True})
            for x, r in zip(part, res["results"]):
                off = x["offset"]
                words = [{"text": wd["text"], "start": round(off + wd["start"], 3),
                          "end": round(off + wd["end"], 3)} for wd in r.get("words", [])]
                out.append({"start": words[0]["start"] if words else round(off, 3),
                            "end": words[-1]["end"] if words else round(x["end"], 3),
                            "text": r.get("text", "").strip(), "language": r.get("language", ""),
                            "words": words})
            job.status(f"{min(i + 16, len(batch))}/{len(batch)} chunks", progress=min(1.0, (i + 16) / len(batch)))
        for x in batch:
            Path(x["path"]).unlink(missing_ok=True)

    langs = Counter(s["language"] for s in out if s["language"] and s["text"])
    main = lang_hint or (langs.most_common(1)[0][0] if langs else "")
    job.set("asr", {"language": media.lang_code(main), "segments": out})
    job.status(f"{sum(len(s['words']) for s in out)} words, language {main or 'unknown'}")
