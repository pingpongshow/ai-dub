"""The review screen's side of a job.

After the voice bank, a job can pause so the user can check what will keep the
original (songs, openings, credits) and who speaks each line, and change it
before anything is translated or voiced. The setting is per job, per show or in
config.yaml (review.mode): off, unsure (pause only when something needs a look)
or always.

Files, in the job's working folder under review/:
  decisions.json  what the user decided (regions, subtitles, line voices), and
                  "started" once they press Start dubbing
  media.json      whether the preview media is ready
  preview.mp4, voices.m4a, music.m4a, peaks.json, subs.en.vtt, subs.ja.vtt
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import lines as L
from . import media
from .job import Job, Paused, require_subtitles

MODES = ("off", "unsure", "always")
MEDIA = {"preview.mp4": "video/mp4", "voices.m4a": "audio/mp4", "music.m4a": "audio/mp4",
         "peaks.json": "application/json", "subs.en.vtt": "text/vtt", "subs.ja.vtt": "text/vtt"}
REGION_GAP_S = 3.0      # kept lines this close together are one region
FLAG_GAP_S = 15.0       # flags of one kind this close together are one thing to check
QUOTED_RUN = 3          # this many quoted subtitles in a row may be a song's lyrics
PEAK_S = 0.05
EDGE_S = 0.3            # a region edge this far inside a dialogue subtitle cuts through it
_building: set[str] = set()
_lock = threading.Lock()


# Settings and files ------------------------------------------------------------------------
def mode(job) -> str:
    rules = ((job.profile.data.get("rules") or {}) if job.profile else {})
    for m in (job.options.get("review"), rules.get("review"), job.cfg.get("review.mode", "unsure")):
        if m in MODES:
            return m
    return "unsure"


def folder(work: Path) -> Path:
    d = work / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_decisions(work: Path) -> dict:
    f = work / "review" / "decisions.json"
    return json.loads(f.read_text()) if f.exists() else {"regions": [], "cues": {}, "lines": {}}


def save_decisions(work: Path, d: dict) -> None:
    clean = {"regions": [r for r in d.get("regions", []) if isinstance(r, dict) and float(r.get("end", 0)) > float(r.get("start", 0))],
             "cues": {str(k): v for k, v in (d.get("cues") or {}).items() if v in ("keep", "dub")},
             "lines": {str(k): v for k, v in (d.get("lines") or {}).items() if isinstance(v, dict)},
             "started": bool(d.get("started")), "saved": time.time()}
    f = folder(work) / "decisions.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=1))
    tmp.replace(f)


def media_status(work: Path) -> dict:
    f = work / "review" / "media.json"
    st = json.loads(f.read_text()) if f.exists() else {"state": "missing"}
    if st.get("state") == "building" and str(work) not in _building and time.time() - st.get("updated", 0) > 900:
        st = {"state": "missing"}  # a build cut short by a restart
    return st


def _set_media(work: Path, **st) -> None:
    f = folder(work) / "media.json"
    f.write_text(json.dumps({**st, "updated": time.time()}))


# The pipeline step ----------------------------------------------------------------------------
BASE = "base-lines.json"   # the lines as they reached the review, before any decision was applied
KEPT = "kept.json"         # what the working folder held then: kept after the dub, for a redub


def snapshot(work: Path) -> None:
    """Remember the lines and the working files as they reach the review, once, so a
    finished dub can be reviewed and dubbed again from here without redoing the analysis."""
    d = folder(work)
    if (d / BASE).exists():
        return
    shutil.copyfile(work / "lines.json", d / BASE)
    (d / KEPT).write_text(json.dumps(sorted(p.name for p in work.iterdir() if p.name not in ("review", "verify"))))


def kept_names(work: Path) -> set[str] | None:
    f = work / "review" / KEPT
    return set(json.loads(f.read_text())) if f.exists() else None


def can_redub(work: Path) -> bool:
    return (work / "review" / BASE).exists() and (work / "review" / KEPT).exists() and (work / "state.json").exists()


def reset_for_redub(work: Path, live: Path) -> None:
    """Back to the moment the job reached the review: the lines as they were, everything
    made after it deleted, the analysis kept. The original is back at `live` by now."""
    from .pipeline import STEP_NAMES
    keep = kept_names(work) or set()
    for p in work.iterdir():
        if p.name not in keep and p.name != "review":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    shutil.copyfile(work / "review" / BASE, work / "lines.json")
    f = work / "state.json"
    st = json.loads(f.read_text())
    st["done"] = [s for s in st.get("done", []) if s in STEP_NAMES[:STEP_NAMES.index("review")]]
    data = st.setdefault("data", {})
    for k in ("deliver_swap", "delivered"):
        data.pop(k, None)
    if data.get("ingest"):
        data["ingest"]["source"] = str(live)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(f)


def run(job: Job) -> None:
    from . import translate
    trusted, why = translate.subtitles_trusted(job)
    if not trusted:
        require_subtitles(job, f"The English subtitles aren't reliable enough to be the script: {why}")
    snapshot(job.work)
    lines = L.load(job.path("lines.json"))
    d = load_decisions(job.work)
    if d.get("started"):
        n = apply(job, lines, d)
        L.save(lines, job.path("lines.json"))
        job.qc["review"] = {"changes": n, "mode": mode(job)}
        job.status(f"applied your review ({n} lines changed)")
        return
    flags = needs_look(job, lines)
    m = mode(job)
    job.qc["review"] = {"flags": len(flags), "mode": m}
    if m == "always" or (m == "unsure" and flags):
        job.status("preparing the review screen")
        build_media(job.work, job.state["data"], job.status)
        raise Paused(f"waiting for review: {len(flags)} to check" if flags else "waiting for review")
    job.status(f"no review needed ({len(flags)} to check, review {m})")


# What needs a look ---------------------------------------------------------------------------
def _english_cues(job) -> list:
    from .subtitles.step import load_chosen
    try:
        return load_chosen(job, "english")
    except Exception:
        return []


def needs_look(job, lines: list[L.Line]) -> list[dict]:
    """Decisions Ai Dub was unsure of, in time order."""
    from . import align, translate
    from .pitch import Pitch
    from .subtitles.model import DIALOGUE
    out: list[dict] = []
    cues = _english_cues(job)
    talk = [q for q in cues if q.kind == DIALOGUE and q.text.strip()]
    duration = float((job.get("ingest") or {}).get("duration", 0) or 0)
    p = Pitch.load(job) if job.get("pitch") else None
    lang = (job.get("subtitles") or {}).get("source_language", "")

    def add(kind, ln_or_t, title, detail, ids=()):
        t0, t1 = (ln_or_t.start, ln_or_t.end) if isinstance(ln_or_t, L.Line) else ln_or_t
        out.append({"kind": kind, "start": round(t0, 2), "end": round(t1, 2), "title": title, "detail": detail,
                    "lines": list(ids)})

    for ln in lines:
        if align.sung_candidate(ln) and not (p is not None and align.clearly_spoken(ln, p)) \
                and not translate.quoted_lyric(ln, cues) and not (p is not None and align.sure_sung(ln, p, lang)):
            add("lyrics", ln, "Sounds sung, with a subtitle",
                f"Held notes over music ({ln.sung:.2f}). The lyrics check decides unless you choose.", [ln.id])
        elif ln.kind == L.SONG and ln.kept_why.startswith("held notes") and 180 < ln.start < duration - 180:
            add("song", ln, "Kept as singing, with no subtitle",
                f"{ln.kept_why[:1].upper()}{ln.kept_why[1:]}. Nothing to read, so the sound alone decided.", [ln.id])
        if ln.voice_note == "moved by pitch":
            add("voice", ln, "Moved to another voice", "Pitch put this short line far from its speaker's usual voice.",
                [ln.id])
    borrowed: dict[str, L.Line] = {}
    for ln in lines:
        if ln.kind == L.SPEECH and (ln.voice_note.startswith("voice of") or ln.voice_note.startswith("generic voice")):
            borrowed.setdefault(ln.character, ln)
    for ln in borrowed.values():
        add("voice", ln, "Dubbed with a borrowed voice", f"Too little of this speaker to clone: {ln.voice_note}.",
            [x.id for x in lines if x.character == ln.character])
    by_id = {ln.id: ln for ln in lines}
    for m in job.get("mixed_characters") or []:
        mine = [by_id[i] for i in m.get("lines", []) if i in by_id]
        if mine:
            add("voice", mine[0], f"{m['name']} may be more than one person",
                f"{len(mine)} of {m['name']}'s lines sound like someone other than the voice of their other "
                f"{m['main_lines']}. Give them to the right speaker, or a new one.", [ln.id for ln in mine])
    for r in regions(lines, duration):
        for q in talk:
            for edge in (r["start"], r["end"]):
                if q.start + EDGE_S < edge < q.end - EDGE_S:
                    add("edge", (q.start, q.end), "A kept region cuts through a subtitle",
                        f"“{q.text[:60]}” is partly inside a {r['kind']}.", [])
    # Lyrics subtitled in quotation marks (a song sung on screen): flagged, never kept by this alone,
    # since quotes also mark titles and quoted speech.
    run: list = []
    for q in talk + [None]:
        quoted = q is not None and translate.QUOTED.match(q.text or "") is not None
        if quoted and (not run or q.start - run[-1].end <= FLAG_GAP_S):
            run.append(q)
            continue
        if len(run) >= QUOTED_RUN:
            ls = [ln for ln in lines if ln.kind == L.SPEECH and any(_overlap(ln, x.start, x.end) > 0.3 for x in run)]
            if ls:
                add("quoted", (run[0].start, run[-1].end), "Subtitles in quotation marks: maybe a song",
                    f"{len(run)} subtitles in a row are quoted, as lyrics often are. They will be dubbed unless you "
                    "keep them.", [ln.id for ln in ls])
        run = [q] if quoted else []
    sub = (job.get("subtitles") or {}).get("english")
    tried = [c for c in (job.get("subtitles") or {}).get("candidates", []) if c.get("language") == "en"]
    said_ok = getattr(job, "options", {}).get("speech_ok")  # asked already, before dubbing started
    if not sub and tried and lines and not said_ok:
        best = max((c.get("score", {}).get("total", 0) for c in tried), default=0)
        add("subtitles", (0.0, min(5.0, duration or 5.0)), "English subtitles weren't good enough to use",
            f"{len(tried)} English subtitle track{'s' if len(tried) != 1 else ''} found, the best scoring {best:.2f}. "
            "The episode would be translated from the speech instead.", [])
    if sub and lines:
        try:
            ok, why = translate.subtitles_trusted(job)
        except Exception:
            ok, why = True, ""
        if not ok and said_ok:
            pass
        elif not ok:
            add("subtitles", (0.0, min(5.0, duration or 5.0)), "English subtitles won't be the script",
                f"{why}. The episode would be translated from the speech instead.", [])
        elif talk:
            # Speech no subtitle covers: translated from what was heard, so worth a listen
            # (a sound like "Ah!" keeps the original and isn't flagged).
            speech = [ln for ln in lines if ln.kind == L.SPEECH and ln.decided != "keep"]
            under = translate.covered_ids(speech, talk)
            for ln in speech:
                if ln.id not in under and not translate.short_reaction(ln, lang):
                    add("nosub", ln, "Speech with no subtitle",
                        f"“{ln.source_text[:40]}” has no English subtitle, so it will be translated from what was "
                        "heard. Listen to check it is speech worth dubbing.", [ln.id])
    out.sort(key=lambda f: f["start"])
    # One thing to check per stretch: ten lines of one unsubtitled song are one flag.
    merged: list[dict] = []
    for f in out:
        last = merged[-1] if merged else None
        if last and last["kind"] == f["kind"] and f["kind"] in ("song", "lyrics", "voice", "edge", "nosub") \
                and last["title"] == f["title"] and f["start"] - last["end"] <= FLAG_GAP_S:
            last["end"] = max(last["end"], f["end"])
            last["lines"] = sorted(set(last["lines"]) | set(f["lines"]))
            n = len(last["lines"])
            last["detail"] = f["detail"] if n <= 1 else (f"{n} lines, {_t(last['start'])}–{_t(last['end'])}. "
                                                          + re.sub(r" \([0-9.]+\)", "", f["detail"]))
        else:
            merged.append(dict(f))
    return merged


def _t(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# Regions ---------------------------------------------------------------------------------------
def regions(lines: list[L.Line], duration: float) -> list[dict]:
    """What keeps the original, as spans of kept lines with their reasons."""
    kept = sorted((ln for ln in lines if ln.kind == L.SONG), key=lambda l: l.start)
    spans: list[list[L.Line]] = []
    for ln in kept:
        if spans and ln.start - max(x.end for x in spans[-1]) <= REGION_GAP_S:
            spans[-1].append(ln)
        else:
            spans.append([ln])
    out = []
    for group in spans:
        a, b = group[0].start, max(x.end for x in group)
        why = Counter(re.sub(r" \([0-9.]+\)", "", x.kept_why or "song") for x in group)
        text = " ".join(why)
        kind = ("opening" if a < 0.2 * duration and ("opening" in text or "credits" in text or "shared" in text
                                                    or "show rule" in text)
                else "ending" if b > 0.8 * duration and ("credits" in text or "shared" in text or "show rule" in text)
                else "song")
        out.append({"id": f"ai-{a:.1f}", "start": round(a, 2), "end": round(b, 2), "kind": kind, "source": "ai",
                    "reasons": [f"{w} ({n} line{'s' if n > 1 else ''})" for w, n in why.most_common(4)],
                    "lines": [x.id for x in group]})
    return out


# Applying the user's decisions -------------------------------------------------------------------
def _overlap(ln: L.Line, a: float, b: float) -> float:
    return max(0.0, min(ln.end, b) - max(ln.start, a)) / max(ln.duration, 1e-3)


DUBBABLE = {L.SONG, L.WORDLESS, L.BACKGROUND, L.CROWD}


def apply(job, lines: list[L.Line], d: dict | None = None) -> int:
    """Put the review's decisions into the lines (before translation). Returns how many
    lines changed. What each line was before is kept in the job record (qc
    review_changes), to tune the automatic decisions from the user's corrections."""
    from . import recurring
    d = d if d is not None else load_decisions(job.work)
    prof = job.profile
    changed: set[int] = set()
    before = {ln.id: (ln.kind, ln.kept_why, ln.character, ln.sung, ln.music_db, ln.english_ref, ln.source_text)
              for ln in lines}
    for ln in lines:  # speakers merged on the review screen
        if ln.character and prof and ln.character not in prof.characters:
            ln.character = prof.current(ln.character)
    for r in d.get("regions", []):
        a, b, action = float(r["start"]), float(r["end"]), r.get("action", "keep")
        for ln in lines:
            if _overlap(ln, a, b) < 0.5:
                continue
            if action == "keep" and ln.kind != L.SONG:
                ln.kind, ln.kept_why, ln.decided = L.SONG, "kept by you", "keep"
                changed.add(ln.id)
            elif action == "dub" and ln.kind in DUBBABLE:
                ln.kind, ln.kept_why, ln.decided, ln.fallback = L.SPEECH, "", "dub", ""
                changed.add(ln.id)
            elif action == "dub" and ln.kind == L.SPEECH and ln.decided != "dub":
                ln.decided = "dub"  # a short line with no subtitle is dubbed, not kept as a sound
                changed.add(ln.id)
        if action == "keep" and r.get("show_rule") and r.get("kind") in ("opening", "ending", "song") and prof:
            try:
                recurring.save_rule(job, a, b, r["kind"], r.get("label", ""))
            except Exception as exc:
                job.log(f"show rule not saved: {exc}")
    by_key = {cue_key(q): q for q in _english_cues(job)}
    for key, action in (d.get("cues") or {}).items():
        q = by_key.get(key)
        if not q:
            continue
        for ln in lines:
            if _overlap(ln, q.start, q.end) < 0.3:
                continue
            if action == "keep" and ln.kind != L.SONG:
                ln.kind, ln.kept_why, ln.decided = L.SONG, "kept by you", "keep"
                changed.add(ln.id)
            elif action == "dub" and ln.kind in DUBBABLE:
                ln.kind, ln.kept_why, ln.decided, ln.fallback = L.SPEECH, "", "dub", ""
                changed.add(ln.id)
            elif action == "dub" and ln.kind == L.SPEECH and ln.decided != "dub":
                ln.decided = "dub"  # a short line with no subtitle is dubbed, not kept as a sound
                changed.add(ln.id)
    new_chars: dict[str, str] = {}
    by_id = {ln.id: ln for ln in lines}
    for lid, choice in (d.get("lines") or {}).items():
        ln = by_id.get(int(lid))
        if ln and prof and choose_voice(prof, ln, choice, new_chars):
            changed.add(ln.id)
    if prof:
        prof.save()
    log = []
    for ln in lines:
        if ln.id in changed:
            kind, why, who, sung, music, en, src = before[ln.id]
            log.append({"start": round(ln.start, 2), "end": round(ln.end, 2), "was": kind, "why": why, "now": ln.kind,
                        "character_was": who, "character": ln.character, "sung": sung, "music_db": music,
                        "english": en, "source": src})
    if hasattr(job, "qc"):
        job.qc["review_changes"] = log
    return len(changed)


def cue_key(q) -> str:
    return f"{q.start:.2f}"


# The data for the page -------------------------------------------------------------------------------
def choose_voice(prof, ln: L.Line, choice: dict, new_chars: dict[str, str] | None = None) -> bool:
    """The review's speaker choice for a line. True when it changed something."""
    from . import voicebank
    who = str(choice.get("character", ""))
    if not who:
        return False
    if who.startswith("new:"):
        name = who[4:].strip()
        new_chars = {} if new_chars is None else new_chars
        if name not in new_chars:
            new_chars[name] = prof.new_character(name, gender=str(choice.get("gender", "")))
        who = new_chars[name]
    if who == "own":
        ln.voice, ln.voice_note = "", "own line as reference (your choice)"
    elif who in prof.characters:
        ln.character = who
        voicebank.voice_for(prof, ln)
    else:
        return False
    return True


def user_english(d: dict) -> dict[int, str]:
    """English the user wrote for lines on the review page: used instead of a translation."""
    out = {}
    for lid, choice in (d.get("lines") or {}).items():
        text = str((choice or {}).get("en", "")).strip()
        if text:
            out[int(lid)] = text
    return out


def outcome(ln: L.Line) -> str:
    """What a finished dub did with a line, for the review page's result lane."""
    from .speak import UNCLEAR
    if ln.kind != L.SPEECH:
        return "kept" if ln.kind in (L.SONG, L.ENGLISH) else ""
    if ln.fallback:
        return "japanese"
    return "unclear" if UNCLEAR in ln.check_note else "dubbed"


def line_as_ref(work: Path, prof, ln: L.Line, cid: str) -> str:
    """A line's original voice as a kept reference clip of `cid`; its path in the profile."""
    import hashlib
    import tempfile
    import time

    import soundfile as sf
    own = work / "lines" / f"{ln.id:05d}" / "orig.wav"
    if own.exists():
        data, sr = media.read_wav(own, mono=True)
        data = data[:, 0]
    else:  # a line voiced in another voice has no clip of its own yet: from the review's voices track
        src = work / "review" / "voices.m4a"
        if not src.exists():
            raise ValueError("this dub's audio isn't kept any more")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clip.wav"
            media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{max(0.0, ln.start - 0.05):.3f}",
                       "-t", f"{ln.duration + 0.1:.3f}", "-i", str(src), "-ac", "1", "-ar", "48000", str(out)])
            data, sr = media.read_wav(out, mono=True)
            data = data[:, 0]
    src_id = prof.voice_source(cid)
    rel = f"refs/{src_id}/{hashlib.sha1(f'line:{work.name}:{ln.id}'.encode()).hexdigest()[:10]}.flac"
    path = prof.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(data))) or 1.0
    sf.write(str(path), (data * min(1.0, 0.9 / peak)).astype(np.float32), sr, format="FLAC", subtype="PCM_16")
    refs = prof.characters[src_id].setdefault("refs", [])
    refs[:] = [r for r in refs if r["file"] != rel]
    top = max([float(r.get("score", 0)) for r in refs] + [1.0])
    refs.append({"file": rel, "duration": round(ln.duration, 2), "score": round(top + 1.0, 4), "bleed_db": None,
                 "embedding": [], "episode": work.name, "joined": False, "locked": True, "added": time.time(),
                 "from_line": ln.id})
    return rel


def can_redo_lines(work: Path) -> bool:
    return can_redub(work) and (work / "lines").is_dir() and (work / "lines.json").exists()


def other_voice(prof, ln: L.Line) -> None:
    """Swap a line between its own clip and its character's usual voice. By ear, nearly every
    time exactly one of the two is right (23 of 24 lines), and nothing measured says which."""
    refs = prof.refs(ln.character)
    if ln.voice or not refs:
        ln.voice, ln.voice_note = "", "own line as reference (your choice)"
    else:
        ln.voice, ln.voice_note = str(prof.root / refs[0]["file"]), "usual voice (your choice)"


def reset_for_lines(work: Path, live: Path, changes: dict[int, dict], prof) -> list[int]:
    """Back to just before voicing, with every take kept but those of the lines to redo;
    their English and speaker as the user set them. The original is back at `live` by now."""
    from .pipeline import STEP_NAMES
    lines = L.load(work / "lines.json")
    by_id = {ln.id: ln for ln in lines}
    done = []
    for lid, ch in changes.items():
        ln = by_id.get(int(lid))
        if not ln:
            continue
        if ch.get("en"):
            ln.en, ln.en_short, ln.en_from = str(ch["en"]).strip(), "", "user"
        if prof:
            choose_voice(prof, ln, ch)
            if ch.get("other_voice"):
                other_voice(prof, ln)
        if ln.kind != L.SPEECH:
            ln.kind, ln.kept_why = L.SPEECH, ""
        ln.decided, ln.fallback, ln.take, ln.take_scores, ln.fit, ln.check_note = "dub", "", "", {}, {}, ""
        shutil.rmtree(work / "lines" / f"{ln.id:05d}", ignore_errors=True)
        done.append(ln.id)
    if prof:
        prof.save()
    L.save(lines, work / "lines.json")
    keep = (kept_names(work) or set()) | {"review", "lines", "lines.json", "state.json", "job.log"}
    for p in work.iterdir():
        if p.name not in keep:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    f = work / "state.json"
    st = json.loads(f.read_text())
    st["done"] = [x for x in st.get("done", []) if x in STEP_NAMES[:STEP_NAMES.index("speak")]]
    data = st.setdefault("data", {})
    for k in ("deliver_swap", "delivered"):
        data.pop(k, None)
    # Each redo of a line voices it with new seeds: the same words and seeds would give
    # the same takes back, mangled word and all.
    rounds = data.setdefault("redo_rounds", {})
    for lid in done:
        rounds[str(lid)] = int(rounds.get(str(lid), 0)) + 1
    if data.get("ingest"):
        data["ingest"]["source"] = str(live)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(f)
    return done


def view(cfg, row: dict, prof) -> dict:
    """Everything the review page shows, read from the job's working folder."""
    work = cfg.work / f"job-{row['id']}"
    rec = row.get("record") or {}
    redub = row["status"] == "done" and rec.get("verify") == "pending" and can_redub(work)
    before = row["status"] in ("queued", "running", "waiting") and (row.get("options") or {}).get("review") != "always" \
        and not (row.get("step") or "") in _after_review()
    base = {"job": {k: row.get(k) for k in ("id", "path", "status", "message", "step", "profile")},
            "name": Path(row["path"]).name, "editable": row["status"] == "review" or redub, "redub": redub,
            "redo_lines": redub and can_redo_lines(work),
            "can_hold": before}
    if not (work / "lines.json").exists():
        return {**base, "ready": False, "why": "Ai Dub hasn't worked out the lines of this episode yet."}
    job = Job(cfg, row["id"], Path(row["path"]), prof, row.get("options") or {})
    lines = L.load(work / "lines.json")
    duration = float((job.get("ingest") or {}).get("duration", 0) or 0)
    cues = _english_cues(job)
    sub = (job.get("subtitles") or {})
    speakers = _speakers(prof, lines)
    try:
        flags = needs_look(job, lines)
    except Exception as exc:
        flags = [{"kind": "error", "start": 0, "end": 0, "title": "Couldn't work out what needs a look",
                  "detail": str(exc), "lines": []}]
    return {**base, "ready": True, "duration": duration, "mode": mode(job),
            "subtitles": {"english": sub.get("english"), "source_language": sub.get("source_language", "")},
            "cues": [{"key": cue_key(q), "start": round(q.start, 2), "end": round(q.end, 2),
                      "text": q.text or q.raw.replace("\n", " "), "kind": q.kind, "italic": q.style == "italic"}
                     for q in cues],
            "lines": [{"id": ln.id, "start": round(ln.start, 2), "end": round(ln.end, 2), "kind": ln.kind,
                       "character": ln.character, "english": ln.english_ref, "source": ln.source_text,
                       "sung": ln.sung, "voice_note": ln.voice_note, "kept_why": ln.kept_why, "decided": ln.decided,
                       "fallback": ln.fallback if redub else "", "en": ln.en if redub else "",
                       "outcome": outcome(ln) if redub else "", "unclear": ln.check_note if redub else ""}
                      for ln in lines],
            "regions": regions(lines, duration), "flags": flags, "speakers": speakers,
            "decisions": load_decisions(work), "media": media_status(work),
            "rules": (prof.data.get("rules") or {}) if prof else {}}


def _after_review() -> set[str]:
    from .pipeline import STEP_NAMES
    return set(STEP_NAMES[STEP_NAMES.index("review"):])


def _speakers(prof, lines: list[L.Line]) -> list[dict]:
    per: dict[str, list[L.Line]] = {}
    for ln in lines:
        if ln.character and ln.kind in (L.SPEECH, L.SONG):
            per.setdefault(ln.character, []).append(ln)
    out = []
    for cid, ls in sorted(per.items(), key=lambda kv: -sum(x.duration for x in kv[1])):
        ch = (prof.characters.get(prof.current(cid)) if prof else None) or {}
        refs = prof.refs(cid) if prof and ch else []
        notes = Counter(x.voice_note for x in ls if x.kind == L.SPEECH)
        if any(n.startswith("voice of") or n.startswith("generic voice") for n in notes):
            source = next(n for n in notes if n.startswith("voice of") or n.startswith("generic voice"))
        elif refs:
            source = f"own voice bank ({len(refs)} clip{'s' if len(refs) != 1 else ''})"
        else:
            source = "each line's own audio"
        out.append({"id": cid, "name": ch.get("name", cid), "gender": ch.get("gender", ""), "age": ch.get("age", ""),
                    "lines": len(ls), "seconds": round(sum(x.duration for x in ls), 1), "voice": source,
                    "sample": refs[0]["file"] if refs else "",
                    "moved_here": sum(1 for x in ls if x.voice_note == "moved by pitch")})
    return out


# Preview media ---------------------------------------------------------------------------------------
def build_media(work: Path, data: dict, status=lambda *a, **k: None) -> dict:
    """The preview video, the voice and music tracks, the waveform and the subtitles."""
    with _lock:
        if str(work) in _building:
            return media_status(work)
        _building.add(str(work))
    d = folder(work)
    try:
        _set_media(work, state="building")
        ing = data.get("ingest") or {}
        src = Path(ing.get("source", ""))
        status("review: subtitles and waveform")
        _vtt(work, data)
        _peaks(work)
        stems = json.loads((work / "audio" / "stems.json").read_text()) if (work / "audio" / "stems.json").exists() else {}
        group = stems.get("main") or stems.get("front") or next(iter(stems.values()), {})
        for name, key in (("voices.m4a", "dialogue"), ("music.m4a", "mne")):
            if group.get(key):
                status(f"review: {name}")
                media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", group[key], "-ac", "2", "-c:a", "aac",
                           "-b:a", "96k", "-movflags", "+faststart", str(d / name)])
        status("review: preview video")
        done = work / "verify" / "preview.mp4"  # a finished dub's check already has it
        if done.exists():
            _link(done, d / "preview.mp4")
        else:
            _preview(src, int(ing.get("audio_index", 1)), bool(ing.get("has_video", True)), d / "preview.mp4")
        _set_media(work, state="ready")
    except Exception as exc:
        _set_media(work, state="failed", error=str(exc)[:500])
    finally:
        _building.discard(str(work))
    return media_status(work)


def _link(a: Path, b: Path) -> None:
    b.unlink(missing_ok=True)
    try:
        os.link(a, b)
    except OSError:
        shutil.copyfile(a, b)


def _preview(src: Path, audio_index: int, has_video: bool, out: Path) -> None:
    tmp = out.with_suffix(".tmp.mp4")
    audio = ["-map", f"0:{audio_index}", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    if not has_video:
        media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=854x480:r=5",
                   "-i", str(src), "-map", "0:v", "-map", f"1:{audio_index}", "-shortest", "-c:v", "libx264",
                   "-preset", "veryfast", "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", str(tmp)])
        tmp.replace(out)
        return
    scale = ["-vf", "scale=-2:480,format=yuv420p"]
    for video in (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "30", "-b:v", "0"],
                  ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", *scale, *video,
                            *audio, "-sn", "-dn", "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
            return
    raise RuntimeError(f"preview video failed: {r.stderr.strip()[-300:]}")


def _peaks(work: Path) -> None:
    wav = work / "audio" / "dialogue16k.wav"
    if not wav.exists():
        return
    x, sr = media.read_wav(wav, mono=True)
    n = max(1, int(sr * PEAK_S))
    x = np.abs(x[: len(x) // n * n, 0]).reshape(-1, n).max(axis=1)
    top = float(np.percentile(x, 99.5)) or 1.0
    (folder(work) / "peaks.json").write_text(json.dumps(
        {"step": PEAK_S, "peaks": np.clip(np.round(x / top * 100), 0, 100).astype(int).tolist()}))


def _ts(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _vtt(work: Path, data: dict) -> None:
    from .subtitles.model import Cue
    sub = data.get("subtitles") or {}
    d = folder(work)
    cid = sub.get("english")
    cues = []
    if cid:
        f = work / "subs" / f"{cid.replace(':', '_').replace('/', '_')}.cues.json"
        if f.exists():
            cues = [Cue(**c) for c in json.loads(f.read_text())]
    rows = ["WEBVTT", ""]
    for q in cues:
        text = (q.raw or q.text).strip()
        if text:
            rows += [f"{_ts(q.start)} --> {_ts(q.end)}", text.replace("\\N", "\n"), ""]
    (d / "subs.en.vtt").write_text("\n".join(rows), encoding="utf-8")
    rows = ["WEBVTT", ""]
    if (work / "lines.json").exists():
        for ln in L.load(work / "lines.json"):
            if ln.source_text.strip():
                rows += [f"{_ts(ln.start)} --> {_ts(ln.end)}", ln.source_text.strip(), ""]
    (d / "subs.ja.vtt").write_text("\n".join(rows), encoding="utf-8")


def start_media(work: Path, data: dict) -> dict:
    """Build the preview media in the background (for a job not paused at the review)."""
    st = media_status(work)
    if st.get("state") in ("ready", "building"):
        return st
    threading.Thread(target=build_media, args=(work, data), daemon=True).start()
    return {"state": "building"}


def remove_media(work: Path) -> None:
    for name in MEDIA:
        (work / "review" / name).unlink(missing_ok=True)
    shutil.rmtree(work / "review" / "tmp", ignore_errors=True)
