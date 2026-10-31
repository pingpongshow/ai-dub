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
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import lines as L
from . import media
from .job import Job, Paused

MODES = ("off", "unsure", "always")
MEDIA = {"preview.mp4": "video/mp4", "voices.m4a": "audio/mp4", "music.m4a": "audio/mp4",
         "peaks.json": "application/json", "subs.en.vtt": "text/vtt", "subs.ja.vtt": "text/vtt"}
REGION_GAP_S = 3.0      # kept lines this close together are one region
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
def run(job: Job) -> None:
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
                f"{ln.kept_why}. Nothing to read, so the sound alone decided.", [ln.id])
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
    for r in regions(lines, duration):
        for q in talk:
            for edge in (r["start"], r["end"]):
                if q.start + EDGE_S < edge < q.end - EDGE_S:
                    add("edge", (q.start, q.end), "A kept region cuts through a subtitle",
                        f"“{q.text[:60]}” is partly inside a {r['kind']}.", [])
    sub = (job.get("subtitles") or {}).get("english")
    if sub and lines:
        try:
            ok, why = translate.subtitles_trusted(job)
        except Exception:
            ok, why = True, ""
        if not ok:
            add("subtitles", (0.0, min(5.0, duration or 5.0)), "English subtitles won't be the script",
                f"{why}. The episode would be translated from the speech instead.", [])
    out.sort(key=lambda f: f["start"])
    return out


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
        why = Counter(x.kept_why or "song" for x in group)
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
    lines changed."""
    from . import recurring, voicebank
    d = d if d is not None else load_decisions(job.work)
    prof = job.profile
    changed: set[int] = set()
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
        if action == "keep" and r.get("show_rule") and r.get("kind") in ("opening", "ending") and prof:
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
    new_chars: dict[str, str] = {}
    by_id = {ln.id: ln for ln in lines}
    for lid, choice in (d.get("lines") or {}).items():
        ln = by_id.get(int(lid))
        who = str(choice.get("character", ""))
        if not ln or not who or not prof:
            continue
        if who.startswith("new:"):
            name = who[4:].strip()
            if name not in new_chars:
                new_chars[name] = prof.new_character(name, gender=str(choice.get("gender", "")))
            who = new_chars[name]
        if who == "own":
            ln.voice, ln.voice_note = "", "own line as reference (your choice)"
        elif who in prof.characters:
            ln.character = who
            voicebank.voice_for(prof, ln)
        changed.add(ln.id)
    if prof:
        prof.save()
    return len(changed)


def cue_key(q) -> str:
    return f"{q.start:.2f}"


# The data for the page -------------------------------------------------------------------------------
def view(cfg, row: dict, prof) -> dict:
    """Everything the review page shows, read from the job's working folder."""
    work = cfg.work / f"job-{row['id']}"
    base = {"job": {k: row.get(k) for k in ("id", "path", "status", "message", "step", "profile")},
            "name": Path(row["path"]).name, "editable": row["status"] == "review"}
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
                       "sung": ln.sung, "voice_note": ln.voice_note, "kept_why": ln.kept_why, "decided": ln.decided}
                      for ln in lines],
            "regions": regions(lines, duration), "flags": flags, "speakers": speakers,
            "decisions": load_decisions(work), "media": media_status(work),
            "rules": (prof.data.get("rules") or {}) if prof else {}}


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
        _preview(src, int(ing.get("audio_index", 1)), bool(ing.get("has_video", True)), d / "preview.mp4")
        _set_media(work, state="ready")
    except Exception as exc:
        _set_media(work, state="failed", error=str(exc)[:500])
    finally:
        _building.discard(str(work))
    return media_status(work)


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
