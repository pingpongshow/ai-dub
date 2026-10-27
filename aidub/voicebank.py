"""Step 6: link this episode's speakers to the show's characters and update
the voice bank (reference clips), then decide minor-speaker voices.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

from . import lines as L
from . import media, speakers
from .job import Job
from .profiles import Profile

STEP = "voicebank"


def _bleed_db(dlg: np.ndarray, mne: np.ndarray) -> float:
    d = float(np.sqrt(np.mean(dlg ** 2) + 1e-12))
    m = float(np.sqrt(np.mean(mne ** 2) + 1e-12))
    return 20 * np.log10(m / d)


def link_characters(job: Job, prof: Profile, lines: list[L.Line]) -> dict[str, str]:
    """Map each diarized speaker (S1..) to a profile character id."""
    spk = job["diarization"]["speakers"]
    thr = float(job.cfg.get("speakers.profile_threshold", 0.66))
    names: dict[str, Counter] = defaultdict(Counter)
    for ln in lines:
        if ln.name_hint and ln.speaker:
            names[ln.speaker][ln.name_hint] += 1
    mapping: dict[str, str] = {}
    for s, info in sorted(spk.items(), key=lambda kv: -kv[1].get("seconds", 0)):
        hint = names[s].most_common(1)[0][0] if names[s] else ""
        cid = prof.by_name(hint) if hint else None
        if cid is None:
            cid, sim = prof.match(info.get("embedding", []), thr)
            if cid and hint and prof.characters[cid]["name"].startswith("Character "):
                prof.rename(cid, hint)
        if cid is None:
            cid = prof.new_character(hint)
        ch = prof.characters[cid]
        prof.update_embedding(cid, info.get("embedding", []), float(info.get("seconds", 0)))
        for k in ("gender", "age"):
            if info.get(k) and not ch.get(k):
                ch[k] = info[k]
        mapping[s] = cid
        # One voice, several named characters: named lines get their own character sharing the voice.
        for other, n in names[s].items():
            if other == hint or n < 2:
                continue
            oid = prof.by_name(other) or prof.new_character(other, voice_of=cid, gender=ch.get("gender", ""),
                                                            age=ch.get("age", ""))
            for ln in lines:
                if ln.speaker == s and ln.name_hint == other:
                    ln.character = oid
    for ln in lines:
        if not ln.character:
            ln.character = mapping.get(ln.speaker, "")
    return mapping


def update_refs(job: Job, prof: Profile, mapping: dict[str, str]) -> None:
    """Add this episode's cleanest clips to each character's references."""
    cfg = job.cfg
    lo, hi = float(cfg.get("speakers.ref_min_s", 3.0)), float(cfg.get("speakers.ref_max_s", 15.0))
    cap = int(cfg.get("speakers.max_refs_per_character", 8))
    segs = job["diarization"]["segments"]
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    stems = json.loads(job.path("audio", "stems.json").read_text())
    mne = None
    for st in stems.values():
        m, msr = media.read_wav(st["mne"], mono=True)
        m = media.resample(m, msr, sr)
        mne = m if mne is None else mne[: len(m)] + m[: len(mne)]
    by_char: dict[str, list[dict]] = defaultdict(list)
    short: dict[str, list[dict]] = defaultdict(list)
    for s in segs:
        d = s["end"] - s["start"]
        if s.get("overlap") or not s.get("emb") or d < 1.0 or d > hi:
            continue
        cid = mapping.get(s["speaker"])
        if not cid:
            continue
        cid = prof.voice_source(cid)
        centroid = prof.characters[cid].get("embedding") or []
        sim = speakers.cosine(s["emb"], centroid) if centroid else 0.5
        a, b = int(s["start"] * sr), int(s["end"] * sr)
        bleed = _bleed_db(dlg[a:b, 0], mne[a:b, 0]) if mne is not None else -20.0
        # Prefer consistent voice, little music under it, 6-10 s long.
        length_pref = 1.0 - min(1.0, abs(d - 8.0) / 8.0)
        score = 0.6 * sim + 0.3 * float(np.clip((-bleed - 3) / 25, 0, 1)) + 0.1 * length_pref
        cand = {"clips": [[s["start"], s["end"]]], "duration": d, "score": round(score, 4), "sim": sim,
                "bleed": round(bleed, 1), "emb": s["emb"]}
        (by_char if d >= lo else short)[cid].append(cand)

    # Characters who mostly speak in short lines: join several clean short lines
    # into one reference of 6-12 s.
    for cid, cands in short.items():
        if len(by_char[cid]) >= 2:
            continue
        pool = [c for c in sorted(cands, key=lambda c: -c["score"]) if c["sim"] >= 0.5]
        for _ in range(2):
            parts, total = [], 0.0
            while pool and total < 6.0:
                c = pool.pop(0)
                if total + c["duration"] > 12.0:
                    continue
                parts.append(c)
                total += c["duration"] + 0.15
            if total < lo:
                break
            by_char[cid].append({
                "clips": sorted(cl for c in parts for cl in c["clips"]), "duration": round(total, 2),
                "score": round(float(np.mean([c["score"] for c in parts])) - 0.05, 4),
                "sim": float(np.mean([c["sim"] for c in parts])),
                "bleed": round(float(np.mean([c["bleed"] for c in parts])), 1),
                "emb": speakers.mean_embedding([c["emb"] for c in parts])})

    gap = np.zeros(int(0.15 * sr), dtype=np.float32)
    for cid, cands in by_char.items():
        ch = prof.characters[cid]
        existing = ch.setdefault("refs", [])
        cands = [c for c in sorted(cands, key=lambda c: -c["score"]) if c["sim"] >= 0.5][:cap]
        for c in cands:
            key = f"{job.src.name}:{c['clips'][0][0]}:{len(c['clips'])}"
            rel = f"refs/{cid}/{hashlib.sha1(key.encode()).hexdigest()[:10]}.flac"
            if any(r["file"] == rel for r in existing):
                continue
            existing.append({"file": rel, "duration": round(c["duration"], 2), "score": c["score"],
                             "bleed_db": c["bleed"], "embedding": c["emb"], "episode": job.src.name,
                             "joined": len(c["clips"]) > 1, "locked": False, "added": time.time(),
                             "_clips": c["clips"]})
        keep = sorted(existing, key=lambda r: (not r.get("locked"), -r.get("score", 0)))[:max(cap, sum(
            1 for r in existing if r.get("locked")))]
        dropped = [r for r in existing if r not in keep]
        for r in dropped:
            (prof.root / r["file"]).unlink(missing_ok=True)
        for r in existing:
            clips = r.pop("_clips", None)
            if clips and r in keep:
                pieces = []
                for a, b in clips:
                    pieces += [dlg[int(a * sr): int(b * sr), 0], gap]
                seg = np.concatenate(pieces[:-1])
                path = prof.root / r["file"]
                path.parent.mkdir(parents=True, exist_ok=True)
                peak = float(np.max(np.abs(seg))) or 1.0
                sf.write(str(path), (seg * min(1.0, 0.9 / peak)).astype(np.float32), sr, format="FLAC",
                         subtype="PCM_16")
        ch["refs"] = keep


def generic_voices(job: Job) -> list[dict]:
    root = job.cfg.appdata / "voices" / "generic"
    out = []
    if root.exists():
        idx = root / "voices.json"
        if idx.exists():
            for v in json.loads(idx.read_text()):
                if (root / v["file"]).exists():
                    out.append({**v, "path": str(root / v["file"])})
    return out


def assign_voices(job: Job, prof: Profile, lines: list[L.Line]) -> None:
    """Pick the reference clip for every dubbed line, substituting for minor speakers."""
    min_s = float(job.cfg.get("speakers.min_clean_seconds", 4.0))
    scenes_of: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        scenes_of[prof.voice_source(ln.character)].add(ln.scene)
    generic = generic_voices(job)
    subs: dict[str, tuple[str, str]] = {}
    for cid in {ln.character for ln in lines if ln.kind == L.SPEECH and ln.character}:
        src = prof.voice_source(cid)
        if prof.clean_seconds(src) >= min_s:
            continue
        ch = prof.characters[cid]
        # 1. Another character from the show: same gender and age range, never in the same scene.
        best, best_sim = None, -2.0
        for oid, other in prof.characters.items():
            o_src = prof.voice_source(oid)
            if o_src == src or prof.clean_seconds(o_src) < min_s:
                continue
            if ch.get("gender") and other.get("gender") and ch["gender"] != other["gender"]:
                continue
            if ch.get("age") and other.get("age") and ch["age"] != other["age"]:
                continue
            if scenes_of[src] & scenes_of.get(o_src, set()):
                continue
            sim = speakers.cosine(ch["embedding"], other["embedding"]) if ch.get("embedding") and other.get("embedding") else 0
            if sim > best_sim:
                best, best_sim = oid, sim
        if best:
            ref = prof.refs(best)[0]
            subs[cid] = (str(prof.root / ref["file"]), f"voice of {prof.characters[best]['name']}")
            continue
        # 2. A designed generic voice of matching gender and age range.
        fits = [v for v in generic if (not ch.get("gender") or v.get("gender") == ch["gender"])
                and (not ch.get("age") or v.get("age") == ch["age"])] or generic
        if fits:
            v = fits[int(hashlib.sha1(f"{prof.slug}:{cid}".encode()).hexdigest(), 16) % len(fits)]
            subs[cid] = (v["path"], f"generic voice {v.get('name', '')}".strip())
    used = Counter()
    for ln in lines:
        if ln.kind != L.SPEECH:
            continue
        if ln.character in subs:
            ln.voice, ln.voice_note = subs[ln.character]
            used[ln.voice_note] += 1
            continue
        refs = prof.refs(ln.character)
        if refs:
            ln.voice = str(prof.root / refs[0]["file"])
        else:
            # Nothing usable at all: this episode's own audio of the line (last resort).
            ln.voice = ""
            ln.voice_note = "own line as reference"
    job.qc["minor_speaker_substitutions"] = dict(used)


def run(job: Job) -> None:
    prof: Profile = job.profile
    lines = L.load(job.path("lines.json"))
    mapping = link_characters(job, prof, lines)
    update_refs(job, prof, mapping)
    assign_voices(job, prof, lines)
    if job.src.name not in prof.data.setdefault("episodes", []):
        prof.data["episodes"].append(job.src.name)
    prof.save()
    L.save(lines, job.path("lines.json"))
    job.set("characters", {s: c for s, c in mapping.items()})
    names = sorted({prof.characters[c]["name"] for c in mapping.values() if c in prof.characters})
    job.status(f"{len(names)} characters: " + ", ".join(names[:12]))
