"""Checking a finished dub before its original is deleted.

When a job finishes, what it made after the review step is deleted. Kept: verify/
(a small preview of the episode with the original sound, the dub as its own track,
the English subtitles and the moments worth listening to) and what the job had at
the review, so the review page can open on the finished dub and dub it again with
the user's changes without redoing the analysis. The Queue page lists
these dubs; on the check screen the user either keeps the dub (the set-aside
original and the check files are deleted), puts the original back, or puts it
back and dubs again with a review.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from . import lines as L
from . import corrections, media, originals, review
from .jobs import Store
from .speak import UNCLEAR, VOICE_DOUBT

MEDIA = {"preview.mp4": "video/mp4", "dub.m4a": "audio/mp4", "subs.en.vtt": "text/vtt"}
LATE_S = 0.5
FAST_TEMPO = 1.2


def folder(work: Path) -> Path:
    return work / "verify"


def prepare(job, rec: dict) -> dict:
    """Keep what the check screen needs; delete the rest of the working folder."""
    work = job.work
    d = folder(work)
    d.mkdir(parents=True, exist_ok=True)
    data = job.state["data"]
    ing = data.get("ingest") or {}
    delivered = rec.get("delivered") or {}
    aside = Path(delivered.get("set_aside") or "")
    # The original sound and picture: the set-aside original (or the file itself, whose
    # first tracks are the original's).
    src = aside if aside.name and aside.exists() and delivered.get("set_aside_kind") == "original" else job.src
    pre = work / "review" / "preview.mp4"
    if pre.exists():
        review._link(pre, d / "preview.mp4")
    else:
        review._preview(src, int(ing.get("audio_index", 1)), bool(ing.get("has_video", True)), d / "preview.mp4")
    if (work / "mix" / "dub.wav").exists():
        media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(work / "mix" / "dub.wav"), "-ac", "2",
                   "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(d / "dub.m4a")])
    if not (work / "review" / "subs.en.vtt").exists():
        review._vtt(work, data)
    if (work / "review" / "subs.en.vtt").exists():
        shutil.copyfile(work / "review" / "subs.en.vtt", d / "subs.en.vtt")
    lines = L.load(work / "lines.json") if (work / "lines.json").exists() else []
    meta = {"name": job.src.name, "duration": float(ing.get("duration", 0) or 0), "created": time.time(),
            "original": str(aside) if aside.name else "", "moments": moments(lines, float(ing.get("duration", 0) or 0), job)}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    try:
        corrections.save(job.cfg.appdata, job.src.name, job.id, work, job.qc)
    except Exception as exc:  # worth having, not worth losing a finished dub over
        job.log(f"corrections not kept: {exc}")
    # What the job had when it reached the review stays until the check, so the review
    # page can open on the finished dub and dub it again with changes; the voiced takes
    # stay too, so one line can be redone; mixes and the rest go.
    kept = review.kept_names(work)
    keep = kept | {"review", "verify", "lines"} if kept else {"verify"}  # lines: the takes, to redo a line
    for p in work.iterdir():
        if p.name not in keep:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    return meta


CHECK, SONGS, NOTES = "check", "songs", "notes"  # the check screen's tabs: nothing is left out, the rest is out of the way
TITLE_LIKE = re.compile(r"[^a-z]+")


def _clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def unvoiced_cues(job, lines: list[L.Line]) -> list[dict]:
    """Every dialogue subtitle no dubbed line covers: "left in Japanese" only counts lines,
    and a subtitle can lose its line altogether."""
    from .align import load_chosen
    from .subtitles.model import DIALOGUE
    out = []
    spans = [(ln.start, ln.end) for ln in lines if ln.kind == L.SPEECH and ln.take and not ln.fallback]
    flagged = [(ln.start, ln.end) for ln in lines if ln.kind == L.SPEECH and ln.fallback]
    def covered(a, b, rows):
        return sum(max(0.0, min(b, y) - max(a, x)) for x, y in rows)
    for q in load_chosen(job, "english"):
        if q.kind != DIALOGUE or not q.text.strip():
            continue
        length = max(0.01, q.end - q.start)
        # (A line voiced late or early still counts: speech rarely fills its subtitle.)
        if covered(q.start - 0.3, q.end + 0.3, spans) >= 0.2 * length or covered(q.start, q.end, flagged) >= 0.2 * length:
            continue
        under = max((ln for ln in lines if min(ln.end, q.end) > max(ln.start, q.start)),
                    key=lambda ln: min(ln.end, q.end) - max(ln.start, q.start), default=None)
        title = bool(TITLE_LIKE.fullmatch(q.text)) or q.text.startswith(("[", "("))
        out.append({"t": q.start, "kind": "unvoiced", "group": NOTES if title or (under is not None and under.kind == L.SONG) else CHECK,
                    "title": "Subtitle not voiced", "line": under.id if under is not None else None,
                    "detail": f"“{q.text[:80]}”" + (f" ({under.kept_why or under.kind})" if under is not None else " (no one was heard here)")})
    return out


def two_voices(job, lines: list[L.Line]) -> list[dict]:
    from .align import main_speaker
    turns = ((job.get("diarization") or {}).get("segments", []) if job is not None else [])
    out = []
    for ln in lines:
        if ln.kind != L.SPEECH or not ln.take or ln.fallback:
            continue
        ranked = main_speaker(turns, ln.start, ln.end)[2]
        total = sum(v for _, v in ranked)
        if len(ranked) >= 2 and ranked[1][1] >= 0.4 and ranked[1][1] / total >= 0.25:
            out.append({"t": ln.start, "kind": "voices", "group": CHECK, "title": "Two people may share this line", "line": ln.id,
                        "detail": f"one voice says all of “{ln.en[:60]}”"})
    return out


def moments(lines: list[L.Line], duration: float, job=None) -> list[dict]:
    """Everything worth a listen in a finished dub, each in one of the screen's three groups."""
    out = []
    for r in review.regions(lines, duration):
        label = {"opening": "Opening", "ending": "Ending"}.get(r["kind"], "Song")
        out.append({"t": r["start"], "end": r["end"], "kind": "song", "group": SONGS,
                    "title": f"{label} {_clock(r['start'])} to {_clock(r['end'])}: original kept",
                    "detail": "; ".join(r.get("reasons", []))})
    for ln in lines:
        if ln.kind != L.SPEECH:
            continue
        if ln.fallback:
            out.append({"t": ln.start, "kind": "kept", "group": CHECK, "title": "Left in Japanese", "detail": f"“{ln.en[:60]}”: {ln.fallback}" if ln.en else ln.fallback, "line": ln.id})
        elif UNCLEAR in ln.check_note:
            out.append({"t": ln.start, "kind": "unclear", "group": CHECK, "title": "One word may be unclear", "line": ln.id,
                        "detail": f"“{ln.en[:60]}”: " + ln.check_note.split(UNCLEAR + ": ", 1)[-1]})
        elif ln.voice_note in ("moved by pitch", "your choice") or ln.voice_note.endswith("(your choice)"):
            out.append({"t": ln.start, "kind": "voice", "group": CHECK, "title": "Voice changed", "line": ln.id,
                        "detail": f"{ln.voice_note}: “{ln.en[:60]}”"})
        elif VOICE_DOUBT in ln.check_note:
            out.append({"t": ln.start, "kind": "voice", "group": CHECK, "title": "Voice may not match", "line": ln.id,
                        "detail": f"every word came out right: “{ln.en[:60]}”"})
        if (ln.fit or {}).get("mix_missed"):
            out.append({"t": ln.start, "kind": "masked", "group": CHECK, "title": "May be hard to hear in the mix", "line": ln.id,
                        "detail": f"“{ln.en[:60]}”: heard “{str(ln.fit.get('mix_heard', ''))[:60]}”"})
        placed = (ln.fit or {}).get("placed")
        if placed is not None and placed - ln.start > LATE_S:
            out.append({"t": ln.start, "kind": "late", "group": CHECK, "title": f"Starts {placed - ln.start:.1f} s late", "line": ln.id,
                        "detail": f"“{ln.en[:60]}”"})
        tempo = (ln.fit or {}).get("paced_tempo") or 1.0
        if tempo > FAST_TEMPO:
            out.append({"t": ln.start, "kind": "fast", "group": NOTES, "title": f"Played {tempo:.2f} times as fast", "line": ln.id,
                        "detail": f"“{ln.en[:60]}”"})
        for note in (ln.check_note or "").split("; "):
            if note and UNCLEAR not in note and VOICE_DOUBT not in note and note != ln.fallback:
                out.append({"t": ln.start, "kind": "note", "group": NOTES, "title": note[:1].upper() + note[1:], "line": ln.id,
                            "detail": f"“{ln.en[:60]}”" + (f" under “{ln.source_text[:30]}”" if ln.source_text else "")})
        if ln.voice_note.startswith("own line as reference ("):
            out.append({"t": ln.start, "kind": "note", "group": NOTES, "title": "Voiced from its own clip", "line": ln.id,
                        "detail": f"{ln.voice_note}: “{ln.en[:60]}”"})
    if job is not None:
        out += unvoiced_cues(job, lines) + two_voices(job, lines)
        for a, b in job.qc.get("faces_one_person") or []:
            out.append({"t": 0.0, "kind": "note", "group": NOTES, "title": "Two voices may be one person",
                        "detail": f"{a} and {b} have the same face on screen. To merge them, see Shows → Voices."})
        track = job.qc.get("subtitle_track") or {}
        if track.get("trusted") is False:
            out.append({"t": 0.0, "kind": "subs", "group": CHECK, "title": "Dubbed from the speech, not the subtitles",
                        "detail": str(track.get("why") or "")})
    out.sort(key=lambda m: m["t"])
    return [{**m, "t": round(m["t"], 2)} for m in out]


def pending(store: Store, work_root: Path) -> list[dict]:
    """Finished dubs waiting for the user's check."""
    out = []
    for j in store.list({"done"}):
        rec = j.get("record") or {}
        d = folder(work_root / f"job-{j['id']}")
        if rec.get("verify") != "pending" or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        orig = Path(meta.get("original") or "")
        out.append({"id": j["id"], "name": meta.get("name", Path(j["path"]).name), "finished": j.get("finished"),
                    "moments": len(meta.get("moments", [])),
                    "original_size": orig.stat().st_size if orig.name and orig.exists() else 0,
                    "check_size": sum(f.stat().st_size for f in d.parent.rglob("*") if f.is_file())})
    out.sort(key=lambda x: -(x["finished"] or 0))
    return out


def view(store: Store, work_root: Path, jid: int) -> dict:
    j = store.get(jid)
    if not j:
        raise KeyError(jid)
    d = folder(work_root / f"job-{jid}")
    if not (d / "meta.json").exists():
        return {"ready": False, "id": jid, "name": Path(j["path"]).name, "verify": (j.get("record") or {}).get("verify", "")}
    meta = json.loads((d / "meta.json").read_text())
    orig = Path(meta.get("original") or "")
    rec = j.get("record") or {}
    work = work_root / f"job-{jid}"
    # What every dubbed line says, so any of them can be voiced again from here: the
    # flagged ones, or one you hear going wrong.
    redo = rec.get("verify") == "pending" and review.can_redo_lines(work)
    # (Dubs checked from before the lists were grouped have no group.)
    moments = [{"group": SONGS if m.get("kind") == "song" else CHECK, **m} for m in meta.get("moments", [])]
    spoken = []
    if (work / "lines.json").exists():
        lines = L.load(work / "lines.json")
        en = {ln.id: ln.en for ln in lines}
        moments = [{**m, "en": en.get(m["line"], "")} if m.get("line") is not None else m for m in moments]
        if redo:
            spoken = [[ln.id, round(ln.start, 2), round(ln.end, 2), ln.en] for ln in lines if ln.kind == L.SPEECH and ln.en]
    return {"ready": True, "id": jid, "name": meta.get("name"), "duration": meta.get("duration"), "path": j["path"],
            "moments": moments, "verify": rec.get("verify", ""),
            "original": {"path": str(orig), "exists": orig.exists() if orig.name else False,
                         "size": orig.stat().st_size if orig.name and orig.exists() else 0},
            "qc": {k: (rec.get("qc") or {}).get(k) for k in ("fallback_rate", "dubbed_lines", "free_timing", "readback", "dub_channels")},
            "can_redub": review.can_redub(work),
            "can_redo_lines": redo, "lines": spoken,
            "work_size": sum(f.stat().st_size for f in (work_root / f"job-{jid}").rglob("*") if f.is_file())}


def _entry(store: Store, jid: int, kind: str = "original") -> dict | None:
    return next((o for o in store.originals() if o.get("job_id") == jid and o["kind"] == kind), None)


def put_back(store: Store, jid: int) -> None:
    """The set-aside original back under its own name; the dub is deleted."""
    o = _entry(store, jid)
    if not o:
        raise ValueError("the original isn't set aside any more")
    originals.restore(store, o["id"])
    dub = _entry(store, jid, "dubbed") or next(
        (x for x in store.originals() if x["kind"] == "dubbed" and x["live_path"] == o["live_path"]), None)
    if dub:
        originals.delete(store, [dub["id"]])


def verdict(svc, jid: int, choice: str) -> dict:
    """good: delete the set-aside original and the check files. restore: put the original
    back and delete the dub. redo: as restore, then queue the episode again with a review."""
    store, j = svc.store, svc.store.get(jid)
    if not j:
        raise KeyError(jid)
    rec = j.get("record") or {}
    work = svc.cfg.work / f"job-{jid}"
    msg = ""
    if choice == "good":
        o = _entry(store, jid)
        freed = 0
        if o:
            _, freed = originals.delete(store, [o["id"]])
        shutil.rmtree(work, ignore_errors=True)
        rec["verify"] = "good"
        msg = f"kept the dub; deleted the original ({freed / 1e9:.2f} GB) and the check files"
    elif choice in ("restore", "redo"):
        put_back(store, jid)
        shutil.rmtree(work, ignore_errors=True)
        rec["verify"] = "restored" if choice == "restore" else "redo"
        msg = "put the original back; the dub is deleted"
        if choice == "redo":
            new = svc.add(j["path"], j["profile"], {**(j.get("options") or {}), "review": "always", "redo": False})
            msg += f"; queued again with a review (job {new})"
    else:
        raise ValueError("choose good, restore or redo")
    store.update(jid, record=rec, message=msg)
    return {"ok": True, "message": msg}
