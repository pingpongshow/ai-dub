"""Step 2: who speaks when.

Diarization runs per scene window (the diarizer handles a limited number of
speakers per pass); speakers are then linked across windows by embedding.
Linking to the show's characters happens in the voice-bank step.

Saves `diarization`: {"segments": [{start, end, speaker, overlap, emb}],
                      "speakers": {S1: {embedding, seconds, gender, age}}}
"""

from __future__ import annotations

from collections import defaultdict

from . import media, speakers
from .job import Job

MIN_EMBED_S = 0.8


def _overlaps(segs: list[dict]) -> None:
    segs.sort(key=lambda s: s["start"])
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if b["speaker"] != a["speaker"]:
                a["overlap"] = b["overlap"] = True


def run(job: Job) -> None:
    win = job["scenes"]["windows"]
    dlg = job.path("audio", "dialogue.wav")
    data, sr = media.read_wav(dlg, mono=True)
    mono16 = media.resample(data, sr, 16000)
    full16 = media.write_wav(job.path("audio", "dialogue16k.wav"), mono16, 16000)

    # 1. Diarize each window.
    raw: list[dict] = []
    cfg = job.cfg.get("stages.diarize")
    with job.worker("diarize") as w:
        w.call("load", {"model": cfg.get("model")})
        job.models["diarization"] = cfg.get("model", "")
        for wi, (a, b) in enumerate(win):
            job.check()
            clip = mono16[int(a * 16000): int(b * 16000)]
            if len(clip) < 16000:
                continue
            p = media.write_wav(job.path("diar", f"w{wi:04d}.wav"), clip, 16000)
            res = w.call("diarize", {"input": str(p)})
            for s in res["segments"]:
                if s["end"] - s["start"] < 0.2:
                    continue
                raw.append({"start": round(a + s["start"], 3), "end": round(a + s["end"], 3),
                            "local": f"{wi}:{s['speaker']}", "window": wi, "overlap": False})
            p.unlink(missing_ok=True)
            job.status(f"window {wi + 1}/{len(win)}", progress=(wi + 1) / len(win))

    # 2. Embed segments; mark overlaps first so embeddings avoid them.
    for s in raw:
        s["speaker"] = s["local"]
    _overlaps(raw)
    to_embed = [s for s in raw if s["end"] - s["start"] >= MIN_EMBED_S and not s["overlap"]]
    with job.worker("speaker") as w:
        w.call("load", {})
        job.models["speaker_embedding"] = w.call("info", {}).get("embedding_model", "")
        res = w.call("embed", {"input": str(full16), "segments": [[s["start"], s["end"]] for s in to_embed]})
        for s, e in zip(to_embed, res["embeddings"]):
            s["emb"] = e

        # 3. One embedding per window-local speaker, then link across windows.
        by_local: dict[str, list[dict]] = defaultdict(list)
        for s in raw:
            by_local[s["local"]].append(s)
        locals_, vecs = [], []
        for loc, segs in by_local.items():
            good = [s for s in segs if s.get("emb")]
            if not good:
                continue
            locals_.append(loc)
            vecs.append(speakers.mean_embedding([s["emb"] for s in good], [s["end"] - s["start"] for s in good]))
        cannot = {(i, j) for i in range(len(locals_)) for j in range(len(locals_))
                  if i != j and locals_[i].split(":")[0] == locals_[j].split(":")[0]}
        labels = speakers.link(vecs, float(job.cfg.get("speakers.link_threshold", 0.62)), cannot)
        global_of = {loc: f"S{lab + 1}" for loc, lab in zip(locals_, labels)}

        # Window-local speakers too short to embed join the nearest-in-time window speaker.
        for s in raw:
            s["speaker"] = global_of.get(s["local"], "")
        for s in raw:
            if not s["speaker"]:
                near = min((t for t in raw if t["speaker"]), default=None,
                           key=lambda t: abs(t["start"] - s["start"]))
                s["speaker"] = near["speaker"] if near else "S0"

        # 4. Per global speaker: embedding, clean seconds, gender and age range.
        spk: dict[str, dict] = {}
        for name in sorted({s["speaker"] for s in raw}, key=lambda x: int(x[1:])):
            segs = [s for s in raw if s["speaker"] == name]
            good = [s for s in segs if s.get("emb")]
            spk[name] = {
                "embedding": speakers.mean_embedding([s["emb"] for s in good],
                                                     [s["end"] - s["start"] for s in good]) if good else [],
                "seconds": round(sum(s["end"] - s["start"] for s in segs if not s["overlap"]), 1),
            }
        ga_req = []
        for name in spk:
            segs = sorted((s for s in raw if s["speaker"] == name and not s["overlap"]),
                          key=lambda s: -(s["end"] - s["start"]))[:4]
            ga_req.append({"speaker": name, "segments": [[s["start"], s["end"]] for s in segs]})
        if ga_req:
            ga = w.call("gender_age", {"input": str(full16), "speakers": ga_req})
            for name, v in ga["speakers"].items():
                spk[name].update({"gender": v.get("gender", ""), "age": v.get("age", "")})
            job.models["gender_age"] = ga.get("model", "")

    for s in raw:
        s.pop("local", None)
    job.set("diarization", {"segments": sorted(raw, key=lambda s: s["start"]), "speakers": spk})
    job.status(f"{len(spk)} speakers in {len(raw)} turns")
