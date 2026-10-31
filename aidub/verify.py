"""Checking a finished dub before its original is deleted.

When a job finishes, everything in its working folder is deleted except verify/:
a small preview of the episode with the original sound, the dub as its own track,
the English subtitles and the moments worth listening to. The Queue page lists
these dubs; on the check screen the user either keeps the dub (the set-aside
original and the check files are deleted), puts the original back, or puts it
back and dubs again with a review.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import lines as L
from . import media, originals, review
from .jobs import Store

MEDIA = {"preview.mp4": "video/mp4", "dub.m4a": "audio/mp4", "subs.en.vtt": "text/vtt"}
LATE_S = 1.0


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
        shutil.move(str(pre), d / "preview.mp4")
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
            "original": str(aside) if aside.name else "", "moments": moments(lines, float(ing.get("duration", 0) or 0))}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    for p in work.iterdir():
        if p.name != "verify":
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    return meta


def moments(lines: list[L.Line], duration: float) -> list[dict]:
    """The places worth listening to in a finished dub."""
    out = []
    for r in review.regions(lines, duration):
        label = {"opening": "Opening", "ending": "Ending"}.get(r["kind"], "Song")
        out.append({"t": r["start"], "kind": "song", "title": f"{label} starts: original kept",
                    "detail": "; ".join(r.get("reasons", []))})
        out.append({"t": r["end"], "kind": "song", "title": f"{label} ends: dub resumes", "detail": ""})
    for ln in lines:
        if ln.kind != L.SPEECH:
            continue
        if ln.fallback:
            out.append({"t": ln.start, "kind": "kept", "title": "Left in Japanese", "detail": ln.fallback})
        elif ln.voice_note in ("moved by pitch", "your choice") or ln.voice_note.endswith("(your choice)"):
            out.append({"t": ln.start, "kind": "voice", "title": "Voice changed",
                        "detail": f"{ln.voice_note}: “{ln.en[:60]}”"})
        placed = (ln.fit or {}).get("placed")
        if placed is not None and placed - ln.start > LATE_S:
            out.append({"t": ln.start, "kind": "late", "title": f"Starts {placed - ln.start:.1f} s late",
                        "detail": f"“{ln.en[:60]}”"})
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
                    "check_size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file())})
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
    return {"ready": True, "id": jid, "name": meta.get("name"), "duration": meta.get("duration"), "path": j["path"],
            "moments": meta.get("moments", []), "verify": rec.get("verify", ""),
            "original": {"path": str(orig), "exists": orig.exists() if orig.name else False,
                         "size": orig.stat().st_size if orig.name and orig.exists() else 0},
            "qc": {k: (rec.get("qc") or {}).get(k) for k in ("fallback_rate", "dubbed_lines", "free_timing")}}


def _entry(store: Store, jid: int, kind: str = "original") -> dict | None:
    return next((o for o in store.originals() if o.get("job_id") == jid and o["kind"] == kind), None)


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
        o = _entry(store, jid)
        if not o:
            raise ValueError("the original isn't set aside any more")
        originals.restore(store, o["id"])
        dub = _entry(store, jid, "dubbed") or next(
            (x for x in store.originals() if x["kind"] == "dubbed" and x["live_path"] == o["live_path"]), None)
        if dub:
            originals.delete(store, [dub["id"]])
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
