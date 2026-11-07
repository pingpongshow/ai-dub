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

from . import faces
from . import lines as L
from . import media, speakers
from .job import Job
from .pitch import Pitch
from .profiles import Profile

PITCH_SHORT_S = 0.8     # shorter lines are too short for the voice and gender checks ...
PITCH_OFF_ST = 9.0      # ... so one this far from its character's usual pitch ...
PITCH_FIT_ST = 3.0      # ... goes to a character of the other gender this close to it ...
PITCH_NEAR_S = 180.0    # ... heard within this long of the line


def _bleed_db(dlg: np.ndarray, mne: np.ndarray) -> float:
    d = float(np.sqrt(np.mean(dlg ** 2) + 1e-12))
    m = float(np.sqrt(np.mean(mne ** 2) + 1e-12))
    return 20 * np.log10(m / d)


REF_QUIET_DB = 12.0  # a reference clip's quiet parts this far under its speech, or it ranks last


def quiet_db(x: np.ndarray, sr: int) -> float:
    """How loud a clip's quiet parts are against its speech, in dB (below 0; nearer 0 = echo or noise)."""
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n < 10:
        return -99.0
    e = 10 * np.log10((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    return float(np.percentile(e, 20) - np.percentile(e, 90))


def character_sounds(lines: list[L.Line], voices: dict[int, dict]) -> dict[str, list]:
    """Each character's voiceprint from their lines long enough to judge, and failing those from
    the other lines of the diarized speakers they came from."""
    by_char: dict[str, list] = defaultdict(list)
    by_spk: dict[str, list] = defaultdict(list)
    spk_of: dict[str, set] = defaultdict(set)
    for ln in lines:
        if ln.kind != L.SPEECH or not ln.character:
            continue
        if ln.speaker:
            spk_of[ln.character].add(ln.speaker)
        if ln.id in voices:
            by_char[ln.character].append(voices[ln.id]["emb"])
            if ln.speaker:
                by_spk[ln.speaker].append(voices[ln.id]["emb"])
    out = {}
    for cid in spk_of.keys() | by_char.keys():
        prints = by_char.get(cid) or [e for s in spk_of.get(cid, ()) for e in by_spk.get(s, [])]
        if prints:
            out[cid] = speakers.mean_embedding(prints)
    return out


def _sounds_like_one(prof: Profile, sound: list, cid: str) -> float:
    refs = [r["embedding"] for r in prof.refs(cid) if r.get("embedding")]
    if refs:
        return float(np.mean([speakers.cosine(sound, r) for r in refs]))
    emb = prof.characters.get(cid, {}).get("embedding")
    return speakers.cosine(sound, emb) if emb else -1.0


def sounds_like(prof: Profile, sound: list, exclude: set[str] = frozenset()) -> list[tuple[float, str]]:
    """The show's characters with a voice of their own, most alike to this voiceprint first."""
    out = []
    for cid in prof.characters:
        src = prof.voice_source(cid)
        if src != cid or cid in exclude or not prof.refs(cid):
            continue
        out.append((_sounds_like_one(prof, sound, cid), cid))
    return sorted(out, reverse=True)


def clear_ref(prof: Profile, cid: str) -> bool:
    """Is the character's top clip free of echo and noise? A borrowed voice is lent to someone
    else's lines: one 8 dB clip, lent, made two lines sound ghostly however often they were redone."""
    refs = prof.refs(cid)
    if not refs:
        return False
    try:
        data, sr = sf.read(str(prof.root / refs[0]["file"]), dtype="float32", always_2d=True)
    except Exception:
        return True  # (can't tell: as before)
    return quiet_db(data.mean(axis=1), sr) <= -REF_QUIET_DB


def _at_least(a: float, b: float, seconds: float) -> list[float]:
    grow = max(0.0, seconds - (b - a)) / 2
    return [max(0.0, a - grow), b + grow]


def line_voices(job: Job, lines: list[L.Line]) -> dict[int, dict]:
    """A voice embedding and gender/age estimate for each speech line long enough to judge."""
    todo = [ln for ln in lines if ln.kind == L.SPEECH and ln.duration >= 0.6]
    if not todo or not job.cfg.get("stages.speaker.worker"):
        return {}
    audio = str(job.path("audio", "dialogue16k.wav"))
    with job.worker("speaker", gpu_stage="speaker") as w:
        w.call("load", {})
        embs = w.call("embed", {"input": audio, "segments": [[ln.start, ln.end] for ln in todo]})["embeddings"]
        # Gender and age need a second of audio: lines from 0.8 s are widened evenly to 1 s.
        ga = w.call("gender_age", {"input": audio, "speakers": [
            {"speaker": str(ln.id), "segments": [_at_least(ln.start, ln.end, 1.02)]} for ln in todo
            if ln.duration >= 0.8]})["speakers"]
    return {ln.id: {"emb": e, **(ga.get(str(ln.id)) or {})} for ln, e in zip(todo, embs) if e}


def same_person(a: list, b: list, ratio: float, least: int = 4) -> bool | None:
    """Do two sets of line voiceprints sound like one person? Their likeness to each other,
    against how alike each set is within itself: one speaker's lines split in two come out at
    0.89 to 1.04 of it, two different men at most 0.82 (on one film of 12 speakers). A speaker's
    average voiceprint can't tell: there two different men came out 0.76 alike and were
    merged into one character. None with too few lines to judge."""
    if len(a) < least or len(b) < least:
        return None
    A = np.asarray([speakers.normalize(x) for x in a], dtype=np.float32)
    B = np.asarray([speakers.normalize(x) for x in b], dtype=np.float32)

    def within(m: np.ndarray) -> float:
        n = len(m)
        return float(((m @ m.T).sum() - n) / (n * (n - 1)))
    wa, wb = within(A), within(B)
    if wa <= 0 or wb <= 0:
        return None
    return float((A @ B.T).mean()) / float(np.sqrt(wa * wb)) >= ratio


def closest_ref(prof: Profile, cid: str, emb: list | None) -> dict | None:
    """The character's reference clip that sounds most like this voiceprint, or their best
    clip without one. A character whose clips came from more than one person (a merge that
    should not have happened) then still speaks in the voice of the person on screen."""
    refs = prof.refs(cid)
    if not refs:
        return None
    kept = [r for r in refs if r.get("locked")]
    if kept:  # the clips you chose (Keep, or a line you said is their voice) are theirs
        refs = kept
    if not emb:
        return refs[0]
    scored = [(speakers.cosine(emb, r["embedding"]), r) for r in refs if r.get("embedding")]
    return max(scored, key=lambda x: x[0])[1] if scored else refs[0]


def link_characters(job: Job, prof: Profile, lines: list[L.Line], voices: dict[int, dict] | None = None) -> dict[str, str]:
    """Map each diarized speaker (S1..) to a profile character id.

    When the subtitles name several characters within one diarized speaker, they
    share a voice only if their lines really sound alike (one actor, several
    roles); otherwise diarization merged different people and each named
    character gets a voice of its own."""
    voices = voices or {}
    link_thr = float(job.cfg.get("speakers.link_threshold", 0.62))
    spk = job["diarization"]["speakers"]
    thr = float(job.cfg.get("speakers.profile_threshold", 0.66))
    names: dict[str, Counter] = defaultdict(Counter)
    for ln in lines:
        if ln.name_hint and ln.speaker:
            names[ln.speaker][ln.name_hint] += 1
    mapping: dict[str, str] = {}
    ratio = float(job.cfg.get("speakers.same_person_ratio", 0.86))
    prints: dict[str, list] = defaultdict(list)  # each diarized speaker's line voiceprints
    for ln in lines:
        if ln.speaker and ln.id in voices and ln.kind == L.SPEECH:
            prints[ln.speaker].append(voices[ln.id]["emb"])
    for s, info in sorted(spk.items(), key=lambda kv: -kv[1].get("seconds", 0)):
        hint = names[s].most_common(1)[0][0] if names[s] else ""
        cid = prof.by_name(hint) if hint else None
        if cid is None:
            # Matched by voice: the character must also sound like this speaker line by line, against
            # the speakers already given it in this episode. (Not against clips from other episodes:
            # recordings differ, and how much that lowers the likeness hasn't been measured.)
            ruled_out: set[str] = set()
            while True:
                cid, sim = prof.match(info.get("embedding", []), thr, exclude=ruled_out)
                if cid is None:
                    break
                theirs = [e for o, c in mapping.items() if c == cid for e in prints[o]]
                if same_person(prints[s], theirs, ratio) is not False:
                    break
                job.log(f"{s} sounds like {prof.characters[cid].get('name', cid)} overall but not line by line: a character of its own")
                ruled_out.add(cid)
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
        main_embs = [voices[ln.id]["emb"] for ln in lines
                     if ln.speaker == s and ln.name_hint in ("", hint) and ln.id in voices]
        main_emb = speakers.mean_embedding(main_embs) if main_embs else info.get("embedding", [])
        for other, n in names[s].items():
            if other == hint or n < 2:
                continue
            mine = [ln for ln in lines if ln.speaker == s and ln.name_hint == other]
            embs = [voices[ln.id]["emb"] for ln in mine if ln.id in voices]
            emb = speakers.mean_embedding(embs) if embs else []
            same_voice = bool(emb and main_emb) and speakers.cosine(emb, main_emb) >= link_thr
            genders = Counter(voices[ln.id].get("gender") for ln in mine if voices.get(ln.id, {}).get("gender"))
            gender = genders.most_common(1)[0][0] if genders else ""
            oid = prof.by_name(other)
            if oid is None:
                oid = prof.new_character(other, voice_of=cid if same_voice else None,
                                         gender=ch.get("gender", "") if same_voice else gender,
                                         age=ch.get("age", "") if same_voice else "")
            if not same_voice and emb:
                prof.update_embedding(oid, emb, float(sum(l.duration for l in mine)))
            for ln in mine:
                ln.character = oid
    for ln in lines:
        if not ln.character:
            ln.character = mapping.get(ln.speaker, "")
        if not ln.character and not ln.speaker and ln.name_hint:
            # A voice diarization didn't mark (a child's "Daddy!" under music): the
            # character the subtitle names, if already known, rather than the line's
            # own split second of audio. The voice check in assign_voices still applies.
            ln.character = prof.by_name(ln.name_hint) or ""
    return mapping


def update_refs(job: Job, prof: Profile, mapping: dict[str, str], lines: list[L.Line] | None = None,
                voices: dict[int, dict] | None = None) -> None:
    """Add this episode's cleanest clips to each character's references."""
    cfg = job.cfg
    lo, hi = float(cfg.get("speakers.ref_min_s", 3.0)), float(cfg.get("speakers.ref_max_s", 15.0))
    cap = int(cfg.get("speakers.max_refs_per_character", 8))
    segs = job["diarization"]["segments"]
    lines = lines or []
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    stems = json.loads(job.path("audio", "stems.json").read_text())
    mne = None
    for st in stems.values():
        m, msr = media.read_wav(st["mne"], mono=True)
        m = media.resample(m, msr, sr)
        mne = m if mne is None else mne[: len(m)] + m[: len(mne)]
    by_char: dict[str, list[dict]] = defaultdict(list)
    short: dict[str, list[dict]] = defaultdict(list)
    voices = voices or {}
    # Candidates are each character's own lines (so a merged diarization speaker
    # can't lend one character's clips to another); diarized turns are the
    # fallback when line voiceprints aren't available.
    spans = [(ln.start, ln.end, ln.character, voices[ln.id]["emb"]) for ln in lines
             if ln.kind == L.SPEECH and not ln.overlap and ln.id in voices and ln.character]
    if not spans:
        spans = [(t["start"], t["end"], mapping.get(t["speaker"]), t.get("emb")) for t in segs if not t.get("overlap")]
    for start, end, cid, emb in spans:
        d = end - start
        if not cid or not emb or d < 1.0 or d > hi:
            continue
        cid = prof.voice_source(cid)
        centroid = prof.characters[cid].get("embedding") or []
        sim = speakers.cosine(emb, centroid) if centroid else 0.5
        a, b = int(start * sr), int(end * sr)
        # Echoey or noisy between words: the voice model copies that into every take (one such
        # clip, 8 dB, made a borrowed voice sound ghostly). Ranked last, not refused: continuous
        # speech can read the same way, and a character with nothing else still needs a clip.
        echoey = quiet_db(dlg[a:b, 0], sr) > -REF_QUIET_DB
        bleed = _bleed_db(dlg[a:b, 0], mne[a:b, 0]) if mne is not None else -20.0
        # Prefer consistent voice, little music under it, 6-10 s long.
        length_pref = 1.0 - min(1.0, abs(d - 8.0) / 8.0)
        score = 0.6 * sim + 0.3 * float(np.clip((-bleed - 3) / 25, 0, 1)) + 0.1 * length_pref - (0.5 if echoey else 0.0)
        cand = {"clips": [[start, end]], "duration": d, "score": round(score, 4), "sim": sim,
                "bleed": round(bleed, 1), "emb": emb}
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


def own_voice_clips(job: Job, prof: Profile, lines: list[L.Line], min_s: float) -> dict[str, str]:
    """For each character, their longest clean line in this episode as a voice clip,
    for their lines too short to clone from when the voice bank has nothing yet."""
    best: dict[str, L.Line] = {}
    for ln in lines:
        if ln.kind != L.SPEECH or ln.overlap or not ln.character or ln.duration < min_s:
            continue
        cid = prof.voice_source(ln.character)
        if cid not in best or ln.duration > best[cid].duration:
            best[cid] = ln
    if not best:
        return {}
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    out = {}
    for cid, ln in best.items():
        a, b = int(max(0.0, ln.start - 0.05) * sr), int((ln.end + 0.05) * sr)
        out[cid] = str(media.write_wav(job.path("voices", f"{cid}.wav"), dlg[a:b], sr))
    return out


def face_fits(prof: Profile, cid: str, v: dict) -> bool:
    """The character a line's face says it is, if the line doesn't sound like the wrong gender
    or age for them."""
    ch = prof.characters.get(prof.voice_source(cid), {})
    return (not (v.get("gender") and ch.get("gender") and v["gender"] != ch["gender"])
            and not (v.get("age") and ch.get("age") and (v["age"] == "child") != (ch["age"] == "child")))


def assign_voices(job: Job, prof: Profile, lines: list[L.Line], voices: dict[int, dict] | None = None) -> None:
    """Pick the reference clip for every dubbed line.

    A character with enough clean speech uses their voice-bank clips. One without
    (a minor speaker, or a main character early in a show) speaks from the line's
    own audio, which is always the right person; a line too short for that uses
    the character's voice from their longest line in the episode. Another
    character's voice or a designed generic voice is the last resort, for a
    character with no line long enough to clone from at all. A line whose own audio clearly doesn't
    match the chosen voice (voice, gender, or child vs adult) is voiced from its
    own audio instead, so nobody speaks with someone else's voice."""
    min_s = float(job.cfg.get("speakers.min_clean_seconds", 4.0))
    own_min = float(job.cfg.get("speakers.own_line_min_s", 0.5))
    scenes_of: dict[str, set[int]] = defaultdict(set)
    for ln in lines:
        scenes_of[prof.voice_source(ln.character)].add(ln.scene)
    generic = generic_voices(job)
    subs: dict[str, tuple[str, str]] = {}
    sounds = character_sounds(lines, voices or {})
    suggestions: dict[str, list] = {}
    for cid in {ln.character for ln in lines if ln.kind == L.SPEECH and ln.character}:
        src = prof.voice_source(cid)
        if prof.clean_seconds(src) >= min_s:
            continue
        ch = prof.characters[cid]
        sound = sounds.get(cid)
        if sound:
            suggestions[cid] = sounds_like(prof, sound, exclude={src})[:3]
        # 1. Another character from the show, never one in the same scene: the one this speaker
        # sounds most like, from any of their lines long enough to judge. Only with nothing to go by,
        # the same gender and age range: a gender guessed from a few seconds lent a woman's voice to
        # a man's lines. (Sound can't name the person reliably from so little either; the review
        # page shows the closest voices for you to choose from.)
        best, best_sim = None, -2.0
        for oid, other in prof.characters.items():
            o_src = prof.voice_source(oid)
            if o_src == src or prof.clean_seconds(o_src) < min_s or not clear_ref(prof, o_src):
                continue
            if not sound and ch.get("gender") and other.get("gender") and ch["gender"] != other["gender"]:
                continue
            if ch.get("age") and other.get("age") and (
                    (ch["age"] == "child") != (other["age"] == "child") if sound else ch["age"] != other["age"]):
                continue
            if scenes_of[src] & scenes_of.get(o_src, set()):
                continue
            if sound:
                sim = _sounds_like_one(prof, sound, o_src)
            else:
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
    job.set("voice_suggestions", {c: [[o, prof.characters[o].get("name", o), round(s, 2)] for s, o in v]
                                  for c, v in suggestions.items() if v})
    used = Counter()
    voices = voices or {}
    min_match = float(job.cfg.get("speakers.line_voice_min", 0.35))
    own = 0
    by_face = faces.characters_by_face(lines, faces.load(job))
    moved_by_face = 0
    prefer_own = bool(job.cfg.get("speakers.prefer_own_clip", True))
    by_speaker: dict[str, list] = defaultdict(list)
    for ln in lines:
        if ln.speaker and ln.id in voices:
            by_speaker[ln.speaker].append(voices[ln.id]["emb"])
    speaker_print = {s: speakers.mean_embedding(v) for s, v in by_speaker.items()}
    elsewhere = own_voice_clips(job, prof, lines, own_min)
    for ln in lines:
        if ln.kind != L.SPEECH:
            continue
        refs = prof.refs(ln.character)
        thin = ln.character in subs or prof.clean_seconds(prof.voice_source(ln.character)) < min_s
        if thin and ln.duration >= own_min and not ln.overlap:
            # Too little of this character in the bank: the line's own audio is the truest voice.
            ln.voice, ln.voice_note = "", "own line as reference"
        elif refs:
            # The clip closest to this line's voice, or to its speaker's when it is too short to judge.
            near = closest_ref(prof, ln.character, (voices.get(ln.id) or {}).get("emb") or speaker_print.get(ln.speaker))
            ln.voice = str(prof.root / near["file"])
        elif prof.voice_source(ln.character) in elsewhere:
            # A line too short to clone from: the same character's voice from another of their lines.
            ln.voice, ln.voice_note = elsewhere[prof.voice_source(ln.character)], "own voice from another line"
        elif ln.character in subs:
            ln.voice, ln.voice_note = subs[ln.character]
            used[ln.voice_note] += 1
        else:
            ln.voice, ln.voice_note = "", "own line as reference"
        # The chosen voice must sound like the person speaking this line.
        v = voices.get(ln.id)
        ch = prof.characters.get(prof.voice_source(ln.character), {}) if ln.character else {}
        if v and ln.voice:
            ref_emb = ch.get("embedding") if ln.character not in subs else None
            if ln.character in subs:
                note = ln.voice_note
                ref_emb = next((c.get("embedding") for c in prof.characters.values()
                                if note.endswith(c.get("name", "\0"))), None)
            wrong_voice = bool(ref_emb) and speakers.cosine(v["emb"], ref_emb) < min_match
            wrong_gender = bool(v.get("gender") and ch.get("gender") and v["gender"] != ch["gender"])
            wrong_age = bool(v.get("age") and ch.get("age") and (v["age"] == "child") != (ch["age"] == "child"))
            if wrong_voice or wrong_gender or wrong_age:
                ln.voice = ""  # speak with the line's own audio as the voice reference
                # (By ear, blind, on 24 such lines: this was right on 21, the character's usual voice
                # on 6. So the picture may say whose line it is, but the voice stays the line's own.)
                face = by_face.get(ln.id)
                if face and face != ln.character and face_fits(prof, face, v):
                    ln.character = face
                    moved_by_face += 1
                what = "gender" if wrong_gender else "age" if wrong_age else "voice"
                ln.voice_note = f"own line as reference ({what} didn't match {ch.get('name') or 'the character'})"
                own += 1
        # By ear, blind, over 48 lines: voiced from their own clip 36 acceptable, in the character's
        # usual voice 16. A line too short to clone from, or with a second voice in it, stays as above.
        if prefer_own and ln.voice and ln.duration >= own_min and not ln.overlap:
            ln.voice, ln.voice_note = "", "own line as reference"
    if by_face:
        job.qc["faces_moved_speaker"] = moved_by_face
        job.qc["faces_one_person"] = [[prof.characters.get(c, {}).get("name", c) for c in pair]
                                      for pair in faces.split_people(lines, faces.load(job))]
    job.qc["minor_speaker_substitutions"] = dict(used)
    job.qc["lines_voiced_from_own_audio"] = own + sum(1 for ln in lines if ln.voice_note == "own line as reference")


def mixed_characters(prof: Profile, lines: list[L.Line], voices: dict[int, dict], ratio: float) -> list[dict]:
    """Characters whose lines come from diarized speakers who don't sound like one person
    (a name in the subtitles can put two people under one character), for the review."""
    out = []
    by_char: dict[str, dict[str, list[L.Line]]] = defaultdict(lambda: defaultdict(list))
    for ln in lines:
        if ln.kind == L.SPEECH and ln.character and ln.speaker:
            by_char[ln.character][ln.speaker].append(ln)
    for cid, groups in by_char.items():
        main = max(groups, key=lambda s: len(groups[s]))
        mains = [voices[ln.id]["emb"] for ln in groups[main] if ln.id in voices]
        for s, ls in groups.items():
            if s != main and same_person([voices[ln.id]["emb"] for ln in ls if ln.id in voices], mains, ratio) is False:
                out.append({"character": cid, "name": prof.characters.get(cid, {}).get("name", cid),
                            "lines": [ln.id for ln in ls], "main_lines": len(groups[main])})
    return out


def voice_for(prof: Profile, ln: L.Line) -> None:
    """The voice for a line the review screen gave to another speaker: their voice-bank
    clip, or the line's own audio when they have none yet."""
    refs = prof.refs(ln.character)
    if refs:
        ln.voice, ln.voice_note = str(prof.root / refs[0]["file"]), "your choice"
    else:
        ln.voice, ln.voice_note = "", "own line as reference (your choice)"


def move_by_pitch(job: Job, prof: Profile, lines: list[L.Line]) -> int:
    """Short replies the diarizer gave to the other person in a two-hander ("Uh? For
    what?" in a man's voice, filed under the woman he talks to). Too short for the
    voice and gender checks, they are told by pitch: a line an octave or so from its
    character's usual pitch moves to the character of the other gender heard nearby
    whose usual pitch it matches (the one heard most around it, if several do). Only a
    line far below its character's pitch moves."""
    p = Pitch.load(job)
    if p is None:
        return 0
    speech = [ln for ln in lines if ln.kind == L.SPEECH and ln.character]
    med = {ln.id: p.median(ln.start, ln.end) for ln in speech}
    per: dict[str, list[float]] = defaultdict(list)
    for ln in speech:
        if med[ln.id] is not None:
            per[ln.character].append(med[ln.id])
    usual = {c: float(np.median(v)) for c, v in per.items() if len(v) >= 3}
    gender = {c: prof.characters.get(c, {}).get("gender", "") for c in usual}
    everywhere = Counter(ln.character for ln in speech)
    moved = 0
    for ln in speech:
        m, own = med[ln.id], usual.get(ln.character)
        # Only far lower: a raised voice or a second voice over it lifts a line's pitch,
        # but nothing but a man speaking puts a woman's line an octave down.
        if m is None or own is None or ln.duration >= PITCH_SHORT_S or own - m < PITCH_OFF_ST:
            continue
        if not gender.get(ln.character):
            continue
        near = Counter(o.character for o in speech
                       if o.character != ln.character and abs(o.start - ln.start) <= PITCH_NEAR_S
                       and o.character in usual and abs(usual[o.character] - m) <= PITCH_FIT_ST
                       and gender.get(o.character) and gender[o.character] != gender[ln.character])
        if not near:
            continue
        best = max(near, key=lambda c: (near[c], everywhere[c], -abs(usual[c] - m)))
        job.log(f"line {ln.id} at {ln.start:.1f} s ({ln.source_text[:20]}): pitch {m:.1f} st is "
                f"{prof.characters[best].get('name', best)}'s, not {prof.characters[ln.character].get('name', ln.character)}'s")
        ln.character, ln.voice_note = best, "moved by pitch"
        moved += 1
    job.qc["lines_moved_by_pitch"] = moved
    return moved


def run(job: Job) -> None:
    prof: Profile = job.profile
    lines = L.load(job.path("lines.json"))
    voices = line_voices(job, lines)
    mapping = link_characters(job, prof, lines, voices)
    update_refs(job, prof, mapping, lines, voices)
    move_by_pitch(job, prof, lines)
    assign_voices(job, prof, lines, voices)
    job.set("mixed_characters", mixed_characters(prof, lines, voices, float(job.cfg.get("speakers.same_person_ratio", 0.86))))
    if job.src.name not in prof.data.setdefault("episodes", []):
        prof.data["episodes"].append(job.src.name)
    prof.save()
    L.save(lines, job.path("lines.json"))
    job.set("characters", {s: c for s, c in mapping.items()})
    names = sorted({prof.characters[c]["name"] for c in mapping.values() if c in prof.characters})
    job.status(f"{len(names)} characters: " + ", ".join(names[:12]))
