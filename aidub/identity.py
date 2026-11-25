"""One actor, one speaker: the diarizer's speakers linked again across the film, once
the faces are known.

Diarization runs per scene window and links each window's speakers by voice alone.
Old soundtracks with the music taken out give weak voiceprints (two clips of one
speaker came out 0.3-0.4 alike, two people in one scene 0.2), so one actor ended up
as many speakers: 106 in Tampopo, 67 in Mothra vs. Godzilla, each actor's lines
spread over two to six of them. Here, before the speakers are matched to characters:

- window speakers whose lines show the same face clearly talking are one person
  (when their voices don't plainly differ and nothing says they are two);
- the rest are linked by voice as before;
- what is left with only a few seconds joins the main speaker it sounds most like,
  never one of the other gender or one talking in the same window.

On those two films (judged by on-screen faces): lines of one face kept as one speaker
went from 56 to 84 % and from 42 to 63 %; the speakers found from 106 to 71 and from
67 to 39; lines of clearly different faces put under one speaker stayed at 2-4 %.

Two speakers of one window are still never joined: the diarizer told them apart
there, and it hears two people talking to each other far better than any voiceprint.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import faces, speakers
from . import lines as L
from .job import Job

FACE_SAME = 0.55       # two clearly talking faces this alike are one person ...
FACE_HITS = 2          # ... seen in at least this many pairs of their lines ...
FACE_SHARE = 0.3       # ... and in this share of the pairs
FACE_VOICE_MIN = 0.10  # unless their voices are this unlike


def local_of(seg: dict) -> str:
    """The window speaker a diarized turn came from (kept from the diarize step;
    older jobs: its window and speaker, which named one window speaker)."""
    return seg.get("local") or f"{seg.get('window', 0)}:{seg['speaker']}"


def _line_locals(lines: list[L.Line], segs: list[dict]) -> dict[int, str]:
    """Each speech line's window speaker: the turn of its speaker it overlaps most."""
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        by_spk[s["speaker"]].append(s)
    out = {}
    for ln in lines:
        if ln.kind != L.SPEECH or not ln.speaker:
            continue
        best, most = None, 0.0
        for s in by_spk.get(ln.speaker, ()):
            o = min(s["end"], ln.end) - max(s["start"], ln.start)
            if o > most:
                best, most = s, o
        if best is not None:
            out[ln.id] = local_of(best)
    return out


def face_links(keys: list[str], windows: list, genders: list[str], sim: np.ndarray,
               face_prints: dict[int, list]) -> list[tuple[int, int]]:
    """Pairs of window speakers (indexes) in different windows whose lines show one face talking."""
    prints = {i: speakers.normalize(v) for i, v in face_prints.items() if v}
    ids = sorted(prints)
    out = []
    for x, a in enumerate(ids):
        for b in ids[x + 1:]:
            if windows[a] == windows[b] or sim[a, b] < FACE_VOICE_MIN:
                continue
            if genders[a] and genders[b] and genders[a] != genders[b]:
                continue
            m = prints[a] @ prints[b].T >= FACE_SAME
            if int(m.sum()) >= FACE_HITS and float(m.mean()) >= FACE_SHARE:
                out.append((a, b))
    return out


def relink(job: Job, lines: list[L.Line]) -> dict[str, str]:
    """Link the window speakers again with the faces; rename the diarized speakers (S1 the
    one heard most) and the lines' speakers to match. Returns old speaker -> new speaker for
    the lines (an old speaker split by the faces goes to where most of it went)."""
    dia = job.get("diarization") or {}
    segs, old_spk = dia.get("segments") or [], dia.get("speakers") or {}
    if not segs:
        return {}
    cfg = job.cfg
    for s in segs:
        s["local"] = local_of(s)
    by_local: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        by_local[s["local"]].append(s)
    keys, vecs, secs, genders, windows = [], [], [], [], []
    for loc, ss in by_local.items():
        good = [s for s in ss if s.get("emb")]
        if not good:
            continue
        keys.append(loc)
        vecs.append(speakers.mean_embedding([s["emb"] for s in good], [s["end"] - s["start"] for s in good]))
        secs.append(sum(s["end"] - s["start"] for s in ss if not s.get("overlap")))
        genders.append((old_spk.get(ss[0]["speaker"]) or {}).get("gender", ""))
        windows.append(ss[0].get("window", loc.split(":")[0]))
    if not keys:
        return {}
    index = {k: i for i, k in enumerate(keys)}
    x = speakers.normalize(vecs)
    sim = x @ x.T
    cannot = {(i, j) for i in range(len(keys)) for j in range(len(keys)) if i != j and windows[i] == windows[j]}

    must: list[tuple[int, int]] = []
    if cfg.get("speakers.face_links", True):
        rows = faces.load(job)
        line_local = _line_locals(lines, segs)
        face_prints: dict[int, list] = defaultdict(list)
        for ln in lines:
            p = faces.speaking_face(rows.get(ln.id, {})) if rows else None
            if p and ln.id in line_local and line_local[ln.id] in index:
                face_prints[index[line_local[ln.id]]].append(p)
        must = face_links(keys, windows, genders, sim, face_prints)

    labels = speakers.link(vecs, float(cfg.get("speakers.link_threshold", 0.62)), cannot, must)
    absorb_s = float(cfg.get("speakers.absorb_s", 20.0))
    if absorb_s > 0:
        labels = speakers.absorb_small(vecs, labels, secs, genders, cannot, absorb_s,
                                       float(cfg.get("speakers.absorb_threshold", 0.45)))

    # S1 is the speaker heard most, as from the diarize step.
    total: Counter = Counter()
    for lab, s in zip(labels, secs):
        total[lab] += s
    name = {lab: f"S{n + 1}" for n, (lab, _) in enumerate(total.most_common())}
    new_of_local = {k: name[lab] for k, lab in zip(keys, labels)}
    # A window speaker with no turn long enough to embed goes where most of its old speaker went.
    went: dict[str, Counter] = defaultdict(Counter)
    for s in segs:
        if s["local"] in new_of_local:
            went[s["speaker"]][new_of_local[s["local"]]] += s["end"] - s["start"]
    old_to_new = {o: c.most_common(1)[0][0] for o, c in went.items()}
    line_local = _line_locals(lines, segs)

    for s in segs:
        s["was"] = s["speaker"]
        s["speaker"] = new_of_local.get(s["local"]) or old_to_new.get(s["speaker"], s["speaker"])
    for ln in lines:
        if ln.speaker:
            ln.speaker = new_of_local.get(line_local.get(ln.id, ""), "") or old_to_new.get(ln.speaker, ln.speaker)

    spk: dict[str, dict] = {}
    for new in sorted({s["speaker"] for s in segs}, key=lambda n: int(n[1:]) if n[1:].isdigit() else 0):
        mine = [s for s in segs if s["speaker"] == new]
        good = [s for s in mine if s.get("emb")]
        # Gender and age range: the most heard among the speakers it was made from.
        votes: dict[str, Counter] = {"gender": Counter(), "age": Counter()}
        for s in mine:
            o = old_spk.get(s.get("was", "")) or {}
            for k in votes:
                if o.get(k):
                    votes[k][o[k]] += s["end"] - s["start"]
        spk[new] = {
            "embedding": speakers.mean_embedding([s["emb"] for s in good], [s["end"] - s["start"] for s in good]) if good else [],
            "seconds": round(sum(s["end"] - s["start"] for s in mine if not s.get("overlap")), 1),
            **{k: (v.most_common(1)[0][0] if v else "") for k, v in votes.items()},
        }
    for s in segs:
        s.pop("was", None)
    dia["segments"], dia["speakers"] = segs, spk
    job.set("diarization", dia)
    job.qc["speakers_relinked"] = {"before": len(old_spk), "after": len(spk), "face_links": len(must)}
    job.log(f"one actor, one speaker: {len(old_spk)} speakers became {len(spk)} ({len(must)} linked by face)")
    return old_to_new
