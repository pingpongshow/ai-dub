"""Step 3b: sound events, emotion and language per speaker turn (SenseVoice).

Finds laughs, cries, coughs, breaths and music (wordless sounds and songs to
keep), the emotion used when a line's own audio can't be the emotion
reference, and the language of each turn (lines already in English).

Saves `events`: list parallel to diarization segments:
  {"language": "ja", "emotion": "HAPPY", "events": ["Laughter"], "text": "..."}
"""

from __future__ import annotations

from .job import Job

STEP = "events"


def run(job: Job) -> None:
    segs = job["diarization"]["segments"]
    cfg = job.cfg.get("stages.events")
    out = []
    with job.worker("events") as w:
        info = w.call("load", {"model": cfg.get("model")})
        job.models["events"] = info.get("model", cfg.get("model", ""))
        spans = [[s["start"], s["end"]] for s in segs]
        for i in range(0, len(spans), 64):
            job.check()
            res = w.call("tag", {"input": str(job.path("audio", "dialogue16k.wav")), "segments": spans[i:i + 64]})
            out.extend(res["results"])
            job.status(f"{min(i + 64, len(spans))}/{len(spans)} turns", progress=min(1.0, (i + 64) / max(1, len(spans))))
    job.set("events", out)
    langs = {}
    for r in out:
        langs[r.get("language", "")] = langs.get(r.get("language", ""), 0) + 1
    job.status("languages per turn: " + ", ".join(f"{k or '?'} {v}" for k, v in sorted(langs.items(), key=lambda kv: -kv[1])))
