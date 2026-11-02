"""Who is talking on screen, for the lines where the sound alone can't say.

Diarization and voice matching only hear. Where a face is clearly the one talking,
the same face's other lines say which character it is: a line the voice check doubted
stays in that character's voice if its face agrees, instead of being voiced from its
own short clip, and a line given to the wrong speaker goes to the right one. Optional:
without the Models page's "Who is talking on screen" installed, this step is skipped.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from . import lines as L
from . import workers
from .job import Job

TALKING = 0.0         # a face's mean talking score over this is talking ...
OTHERS_BELOW = 0.0    # ... when no other face in the line is
SAME_FACE = 0.40      # face prints this alike are one person
MIN_VOTES = 4         # a face names a character from at least this many of its lines ...
MIN_SHARE = 0.6       # ... this many of them agreeing
BATCH = 40


def speaking_face(row: dict) -> list[float] | None:
    """The print of the one face clearly talking in a line; None if none or several are."""
    faces = sorted(row.get("faces", []), key=lambda f: -f["score"])
    if not faces or faces[0]["score"] <= TALKING or (len(faces) > 1 and faces[1]["score"] > OTHERS_BELOW):
        return None
    return faces[0].get("print") or None


def cluster(prints: list[list[float]], same: float = SAME_FACE) -> list[int]:
    """One id per face print: each joins the closest person so far, or starts a new one."""
    means: list[np.ndarray] = []
    counts: list[int] = []
    out = []
    for p in prints:
        v = np.asarray(p, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = [float(m @ v) / (np.linalg.norm(m) + 1e-9) for m in means]
        k = int(np.argmax(sims)) if sims else -1
        if k < 0 or sims[k] < same:
            means.append(v.copy())
            counts.append(1)
            out.append(len(means) - 1)
        else:
            means[k] = means[k] * counts[k] + v
            means[k] /= np.linalg.norm(means[k]) + 1e-9
            counts[k] += 1
            out.append(k)
    return out


def characters_by_face(lines: list[L.Line], rows: dict[int, dict]) -> dict[int, str]:
    """Line id -> the character its speaking face belongs to, where that face's lines agree."""
    ids, prints = [], []
    for ln in lines:
        p = speaking_face(rows.get(ln.id, {}))
        if p and ln.kind == L.SPEECH:
            ids.append(ln.id)
            prints.append(p)
    people = cluster(prints)
    by_id = {ln.id: ln for ln in lines}
    votes: dict[int, Counter] = defaultdict(Counter)
    for lid, person in zip(ids, people):
        if by_id[lid].character:
            votes[person][by_id[lid].character] += 1
    named = {}
    for person, v in votes.items():
        top, n = v.most_common(1)[0]
        if sum(v.values()) >= MIN_VOTES and n / sum(v.values()) >= MIN_SHARE:
            named[person] = top
    return {lid: named[p] for lid, p in zip(ids, people) if p in named}


def load(job: Job) -> dict[int, dict]:
    p = job.path("faces.json")
    return {r["id"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def run(job: Job) -> None:
    name = job.cfg.get("stages.faces.worker")
    video = (job.get("ingest") or {}).get("has_video", True)
    if not name or not job.cfg.get("stages.faces.enabled", True) or not video or not workers.env_ready(job.cfg, name):
        job.status("skipped (Who is talking on screen isn't installed)" if name else "skipped")
        return
    lines = L.load(job.path("lines.json"))
    todo = [{"id": ln.id, "start": ln.start, "end": ln.end} for ln in lines if ln.kind == L.SPEECH and ln.duration >= 0.6]
    rows: list[dict] = []
    with job.worker("faces", gpu_stage="faces") as w:
        w.call("load", {})
        for i in range(0, len(todo), BATCH):
            job.check()
            rows += w.call("analyse", {"video": str(job["ingest"]["source"]), "audio": str(job.path("audio", "dialogue16k.wav")),
                                       "lines": todo[i:i + BATCH]})["lines"]
            job.status(f"watching {min(i + BATCH, len(todo))}/{len(todo)} lines", progress=min(1.0, (i + BATCH) / len(todo)))
    job.path("faces.json").write_text(json.dumps(rows))
    clear = sum(speaking_face(r) is not None for r in rows)
    job.qc["faces"] = {"lines": len(rows), "one_face_talking": clear}
    job.status(f"one face clearly talking in {clear} of {len(rows)} lines")
