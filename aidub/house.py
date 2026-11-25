"""House voices: each character spoken in a voice of Ai Dub's own, designed to suit them,
instead of a clone of the actor.

The other way to dub (voice.mode: clone, the default) clones each actor from the film.
With voice.mode: house, nothing of the actor's voice is copied. What the original tells
about each character is turned into a description: man or woman, age range, how high
they speak, how fast, how loud, how lively, the mood they are most often in. A voice is
then designed from that description (Qwen3-TTS VoiceDesign, the model the generic voices
come from) and kept in the show's profile, so the character sounds the same in every
episode. Each line is still voiced with the original line's emotion and timing: only the
timbre is the house voice's.

The mode is chosen per job, per show (the show's Settings) or in config.yaml, like the
review mode. Characters with little speech get one of the generic voices, matched by
gender and age range. Two main characters of one show are never given voices that
sound alike: a design too like an earlier one is made again with another timbre.

Stored in the profile:
  characters[cid]["traits"]  what the original says about the character, from every episode
  characters[cid]["house"]   {file, description, embedding, timbre, made}
  house/<cid>-<n>.wav        the designed voice
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import lines as L
from . import speakers
from .job import Job
from .pitch import Pitch
from .profiles import Profile

MODES = ("clone", "house")
SAMPLE_TEXT = ("I didn't expect to see you here today. Honestly, it has been a long week, "
               "but I think things are finally starting to look up. Come in, sit down, and tell me everything.")
# Timbres tried in turn, starting from one fixed per character, until the voice is distinct.
TIMBRES = {
    "male": ["warm and rounded", "deep and resonant", "slightly husky", "clear and bright", "gravelly and rough",
             "smooth and mellow", "thin and reedy", "rich and full"],
    "female": ["warm and rounded", "clear and bright", "soft and breathy", "slightly husky", "crisp and light",
               "smooth and mellow", "rich and full", "airy and youthful"],
}
TOO_ALIKE = 0.62        # a new house voice this like another main character's is made again
DESIGN_TRIES = 4
EMOTION_WORDS = {"HAPPY": "cheerful", "ANGRY": "angry", "SAD": "sad", "FEARFUL": "nervous",
                 "DISGUSTED": "scornful", "SURPRISED": "excitable"}


def mode(job: Job) -> str:
    """clone (the actor's own voice) or house (a designed voice), per job, show or config."""
    settings = job.profile.settings if job.profile else {}
    for m in (job.options.get("voice"), settings.get("voice"), job.cfg.get("voice.mode", "clone")):
        if m in MODES:
            return m
    return "clone"


# Traits -----------------------------------------------------------------------------------------
def _line_traits(job: Job, lines: list[L.Line]) -> dict[str, dict]:
    """This episode's measures per character (by voice source): pitch, rate, loudness, mood."""
    p = Pitch.load(job)
    speech = [ln for ln in lines if ln.kind == L.SPEECH and ln.character and not ln.overlap and ln.duration >= 0.6]
    prof = job.profile
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ln in speech:
        cid = prof.voice_source(ln.character)
        d = per[cid]
        d["seconds"].append(ln.duration)
        if p is not None:
            m = p.median(ln.start, ln.end)
            if m is not None:
                d["pitch"].append(m)
        if ln.words:
            d["rate"].append(len(ln.words) / ln.duration)
        d["loud"].append(ln.loudness_db)
        if ln.emotion:
            d["emotion"].append(ln.emotion.upper().strip("<|>"))
    # Every measure is taken against the rest of the cast, never against fixed norms: a language's
    # words, a recording's level, a pitch tracker on a 1964 soundtrack and a monster film's
    # shouting all shift the whole cast (in Mothra vs. Godzilla every man measured "high-pitched,
    # very expressive, often angry" by fixed norms). So: pitch against the film's others of the
    # same gender, the rest against everyone's middle, a mood by how much more often than the film's.
    gender = {cid: (prof.characters.get(cid) or {}).get("gender", "") for cid in per}
    all_rate = [r for d in per.values() for r in d["rate"]]
    all_loud = [x for d in per.values() for x in d["loud"]]
    mid_rate = float(np.median(all_rate)) if all_rate else 0.0
    mid_loud = float(np.median(all_loud)) if all_loud else 0.0
    mid_pitch = {g: float(np.median([m for c, d in per.items() if gender[c] == g for m in d["pitch"]]))
                 for g in set(gender.values()) if any(per[c]["pitch"] for c in per if gender[c] == g)}
    spreads = {c: float(np.subtract(*np.percentile(d["pitch"], [75, 25]))) for c, d in per.items() if len(d["pitch"]) >= 4}
    mid_spread = float(np.median(list(spreads.values()))) if spreads else 0.0
    film_emo = Counter(e for d in per.values() for e in d["emotion"])
    n_emo = max(1, sum(film_emo.values()))
    out = {}
    for cid, d in per.items():
        emo = Counter(d["emotion"])
        n = max(1, len(d["emotion"]))
        pitch = float(np.median(d["pitch"])) if d["pitch"] else None
        out[cid] = {
            "seconds": float(sum(d["seconds"])),
            "pitch_st": pitch,
            "pitch_rel": pitch - mid_pitch[gender[cid]] if pitch is not None and gender[cid] in mid_pitch else None,
            "spread_rel": spreads[cid] - mid_spread if cid in spreads else None,
            "rate_rel": (float(np.median(d["rate"])) / mid_rate) if d["rate"] and mid_rate else None,
            "loud_rel": (float(np.median(d["loud"])) - mid_loud) if d["loud"] else None,
            "emotions": {k: v / n for k, v in emo.items()},
            "emotion_lift": {k: (v / n) / (film_emo[k] / n_emo) for k, v in emo.items() if film_emo[k]},
        }
    return out


def _blend(old: dict | None, new: dict) -> dict:
    """Merge an episode's measures into the character's, weighted by seconds of speech."""
    if not old:
        return dict(new)
    w0, w1 = float(old.get("seconds", 0)), float(new.get("seconds", 0))
    out = {"seconds": w0 + w1}
    for k in ("pitch_st", "pitch_rel", "spread_rel", "rate_rel", "loud_rel"):
        a, b = old.get(k), new.get(k)
        out[k] = b if a is None else a if b is None else (a * w0 + b * w1) / max(w0 + w1, 1e-9)
    for field in ("emotions", "emotion_lift"):
        emo: dict[str, float] = defaultdict(float)
        for src, w in ((old.get(field) or {}, w0), (new.get(field) or {}, w1)):
            for k, v in src.items():
                emo[k] += v * w / max(w0 + w1, 1e-9)
        out[field] = dict(emo)
    return out


def analyse(job: Job, lines: list[L.Line]) -> None:
    """Add this episode's measures to each character's traits (once per episode)."""
    prof: Profile = job.profile
    done = prof.data.setdefault("traits_from", [])
    if job.src.name in done:
        return
    for cid, t in _line_traits(job, lines).items():
        if cid in prof.characters:
            prof.characters[cid]["traits"] = _blend(prof.characters[cid].get("traits"), t)
    done.append(job.src.name)


# Description ------------------------------------------------------------------------------------
def _who(gender: str, age: str) -> str:
    """Who is speaking, in words."""
    g = gender if gender in ("male", "female") else "male"
    if age == "child":
        return "a girl of about ten" if g == "female" else "a boy of about ten"
    person, their = ("woman", "her") if g == "female" else ("man", "his")
    return {"young": f"a young {person} around twenty", "senior": f"an older {person} in {their} late sixties"}.get(
        age, f"a {person} in {their} late thirties")


def describe(ch: dict, timbre: str) -> str:
    """The voice-design brief for a character."""
    t = ch.get("traits") or {}
    who = _who(ch.get("gender", ""), ch.get("age", ""))
    st = t.get("pitch_rel")  # semitones from the cast's others of the same gender
    pitch = ("a medium-pitched" if st is None else "a low-pitched" if st < -2.5 else "a high-pitched" if st > 2.5
             else "a fairly low" if st < -1 else "a fairly high" if st > 1 else "a medium-pitched")
    parts = [f"{who.capitalize()} with {pitch}, {timbre} voice."]
    rate = t.get("rate_rel")
    if rate is not None:
        parts.append("Speaks quickly." if rate > 1.15 else "Speaks slowly and deliberately." if rate < 0.87
                     else "Speaks at a natural, even pace.")
    loud = t.get("loud_rel")
    if loud is not None and loud > 3:
        parts.append("A strong, forceful delivery.")
    elif loud is not None and loud < -3:
        parts.append("Soft-spoken and quiet.")
    spread = t.get("spread_rel")
    if spread is not None:
        parts.append("Very expressive, with a lively intonation." if spread > 1.5 else
                     "Calm and fairly even in intonation." if spread < -1.5 else "")
    shares, lift = t.get("emotions") or {}, t.get("emotion_lift") or {}
    moods = [(lift.get(k, 0), k) for k, v in shares.items() if k in EMOTION_WORDS and v >= 0.2 and lift.get(k, 0) >= 1.5]
    if moods:
        parts.append(f"Often sounds {EMOTION_WORDS[max(moods)[1]]}.")
    parts.append("Clear, natural American English, recorded close to the microphone in a quiet room.")
    return " ".join(p for p in parts if p)


# Casting ----------------------------------------------------------------------------------------
def _generic(job: Job) -> list[dict]:
    from .voicebank import generic_voices
    return generic_voices(job)


def _generic_for(job: Job, prof: Profile, cid: str, gender: str, age: str) -> str:
    gen = _generic(job)
    fits = [v for v in gen if (not gender or v.get("gender") == gender) and (not age or v.get("age") == age)] \
        or [v for v in gen if not gender or v.get("gender") == gender] or gen
    if not fits:
        return ""
    return fits[int(hashlib.sha1(f"{prof.slug}:{cid}".encode()).hexdigest(), 16) % len(fits)]["path"]


def _needs_design(job: Job, prof: Profile, cid: str) -> bool:
    ch = prof.characters.get(cid, {})
    h = ch.get("house") or {}
    if h.get("file") and (prof.root / h["file"]).exists():
        return False
    secs = float((ch.get("traits") or {}).get("seconds", 0)) or prof.clean_seconds(cid)
    return secs >= float(job.cfg.get("voice.house.design_min_s", 10.0))


def cast(job: Job, lines: list[L.Line]) -> None:
    """Give every dubbed line its character's house voice, designing the voices still missing."""
    prof: Profile = job.profile
    todo = [ln for ln in lines if ln.kind == L.SPEECH and ln.dubbed and ln.en]
    cids = sorted({prof.voice_source(ln.character) for ln in todo if ln.character and ln.character in prof.characters},
                  key=lambda c: -float((prof.characters[c].get("traits") or {}).get("seconds", 0)))
    design = [c for c in cids if _needs_design(job, prof, c)]
    from .workers import env_ready
    if design and not env_ready(job.cfg, job.cfg.get("stages.voicedesign.worker", "voicedesign")):
        job.log("house voices: voice design isn't installed (Models page): generic voices for everyone")
        design = []
    if design:
        _design(job, prof, design)
        prof.save()
    dia = (job.get("diarization") or {}).get("speakers") or {}
    made = Counter()
    for ln in todo:
        cid = prof.voice_source(ln.character) if ln.character else ""
        ch = prof.characters.get(cid, {})
        h = ch.get("house") or {}
        if h.get("file") and (prof.root / h["file"]).exists():
            ln.voice, ln.voice_note = str(prof.root / h["file"]), f"house voice for {ch.get('name', cid)}"
            made["designed"] += 1
            continue
        # Little speech, or no character: a generic voice of the right gender and age range.
        g = ch.get("gender") or (dia.get(ln.speaker) or {}).get("gender", "")
        a = ch.get("age") or (dia.get(ln.speaker) or {}).get("age", "")
        path = _generic_for(job, prof, cid or ln.speaker or str(ln.id), g, a)
        if path:
            ln.voice, ln.voice_note = path, f"house voice (generic {g or 'voice'} {a})".replace(" )", ")")
            made["generic"] += 1
        else:
            # No generic voices made yet: an actor's clone would break the mode's promise,
            # so the first designed voice of that gender stands in.
            any_house = next((prof.root / c["house"]["file"] for c in prof.characters.values()
                              if (c.get("house") or {}).get("file") and (not g or c.get("gender") == g)), None)
            if any_house is None:
                raise RuntimeError("house voices: no voice to use (make the generic voices on the Models page)")
            ln.voice, ln.voice_note = str(any_house), "house voice (stand-in)"
            made["stand-in"] += 1
    job.qc["house_voices"] = {"designed_now": len(design), **dict(made)}
    job.log(f"house voices: {len(design)} designed now; lines {dict(made)}")


def _design(job: Job, prof: Profile, cids: list[str]) -> None:
    """Design a voice for each character, distinct from the show's other house voices."""
    root = prof.root / "house"
    root.mkdir(parents=True, exist_ok=True)
    tmp = job.dir("house")
    with job.worker("voicedesign", gpu_stage="voicedesign") as vd:
        vd.call("load", {"model": job.cfg.get("stages.voicedesign.model")})
        job.models["voice_design"] = job.cfg.get("stages.voicedesign.model", "")
        with job.worker("speaker") as sp:
            sp.call("load", {})
            for n, cid in enumerate(cids):
                job.check()
                ch = prof.characters[cid]
                kind = "female" if ch.get("gender") == "female" else "male"
                options = TIMBRES[kind]
                salt = int(ch.get("house_salt", 0))
                start = int(hashlib.sha1(f"{prof.slug}:{cid}".encode()).hexdigest(), 16) % len(options)
                start += salt
                others = [c["house"]["embedding"] for o, c in prof.characters.items()
                          if o != cid and (c.get("house") or {}).get("embedding") and c.get("gender") == ch.get("gender")]
                best = None
                for k in range(DESIGN_TRIES):
                    timbre = options[(start + k) % len(options)]
                    desc = describe(ch, timbre)
                    out = tmp / f"{cid}-{k}.wav"
                    vd.call("design", {"description": desc, "text": SAMPLE_TEXT, "out": str(out)})
                    emb = sp.call("embed_files", {"paths": [str(out)]})["embeddings"][0]
                    alike = max((speakers.cosine(emb, o) for o in others), default=0.0) if emb else 1.0
                    if best is None or alike < best[0]:
                        best = (alike, out, desc, emb, timbre)
                    if alike < TOO_ALIKE:
                        break
                alike, out, desc, emb, timbre = best
                rel = f"house/{cid}-{int(time.time())}.wav"
                (prof.root / rel).write_bytes(out.read_bytes())
                old = (ch.get("house") or {}).get("file")
                if old and old != rel:
                    (prof.root / old).unlink(missing_ok=True)
                ch["house"] = {"file": rel, "description": desc, "embedding": emb, "timbre": timbre,
                               "closest_other": round(alike, 3), "made": time.time()}
                job.status(f"designing house voices {n + 1}/{len(cids)}", progress=(n + 1) / len(cids))
                job.log(f"house voice for {ch.get('name', cid)}: {desc}")


def recast(prof: Profile, cid: str) -> None:
    """Forget a character's house voice: the next dub designs a new one (another timbre first)."""
    ch = prof.characters.get(cid)
    if not ch:
        return
    old = (ch.get("house") or {}).get("file")
    if old:
        (prof.root / old).unlink(missing_ok=True)
    ch.pop("house", None)
    # Start from the next timbre: the slug:cid hash picks the first, so salt it.
    ch["house_salt"] = int(ch.get("house_salt", 0)) + 1


def sample_text(job: Job, voice_path: str) -> str:
    """The words a house voice's file says: the cloning model follows the voice more closely given them."""
    from . import voices
    generic = job.cfg.appdata / "voices" / "generic"
    return voices.SAMPLE_TEXT if Path(voice_path).parent == generic else SAMPLE_TEXT


def house_embedding(prof: Profile, voice_path: str) -> list | None:
    """The voiceprint of the house voice at this path, to check takes against."""
    if not voice_path:
        return None
    for ch in prof.characters.values():
        h = ch.get("house") or {}
        if h.get("file") and str(prof.root / h["file"]) == voice_path:
            return h.get("embedding")
    return None
