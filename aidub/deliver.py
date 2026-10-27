"""Step 10: write the new file with the dub (and subtitle) tracks, safely.

1. write the new file next to the original under a hidden temporary name with
   a non-video extension (".<name>.aidub-partial"), copying every existing
   track, chapter and attachment bit for bit (mkvmerge for MKV, ffmpeg for MP4);
2. verify it against the original;
3. set the original aside as "<name>.<ext>.aidub-original";
4. give the new file the original's name, owner and permissions.
Any failure undoes the steps: the temporary file is removed and the original
keeps (or gets back) its name.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape

from . import __version__
from . import lines as L
from . import media, originals
from .job import Job
from .subtitles.model import Cue
from .subtitles.sources import write_srt

STEP = "deliver"
LANG_TITLE = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese", "en": "English",
              "es": "Spanish", "fr": "French", "de": "German"}
AI_TITLES = ("(AI dub)", "(AI transcript)", "(AI translation)")


class DeliveryError(RuntimeError):
    pass


def stamp(job: Job, kind: str, text_source: str = "") -> dict:
    return {
        "AIDUB_VERSION": __version__,
        "AIDUB_KIND": kind,
        "AIDUB_MODELS": json.dumps(job.models, ensure_ascii=False, sort_keys=True),
        "AIDUB_TEXT_SOURCE": text_source,
        "AIDUB_PROFILE": job.profile.slug if job.profile else "",
        "AIDUB_DATE": time.strftime("%Y-%m-%d"),
    }


def text_source(job: Job) -> str:
    subs = job.get("subtitles") or {}
    parts = []
    for which in ("source", "english"):
        cid = subs.get(which)
        if cid:
            c = next((c for c in subs.get("candidates", []) if c["id"] == cid), None)
            parts.append(c["label"] if c else cid)
    return "; ".join(parts) or "speech recognition"


def subtitle_tracks(job: Job, lines: list[L.Line]) -> list[dict]:
    """Subtitle tracks to add: only for languages the file has none in (forced
    sign tracks don't count). A downloaded human subtitle beats AI text."""
    if not job.cfg.get("deliver.add_subtitle_tracks", True):
        return []
    subs = job.get("subtitles") or {}
    have = {c["language"] for c in subs.get("candidates", []) if c.get("in_file") and not c.get("forced")}
    src_lang = subs.get("source_language", "")
    out = []
    for lang in (src_lang, "en"):
        if not lang or lang in have or any(t["language"] == lang for t in out):
            continue
        which = "english" if lang == "en" else "source"
        cid = subs.get(which)
        cand = next((c for c in subs.get("candidates", []) if c["id"] == cid), None)
        if cand and cand.get("human") and not cand.get("in_file"):
            from .subtitles.step import load_chosen
            cues = load_chosen(job, which)
            fmt = "ass" if cand["format"] == "ass" else "srt"
            p = job.path("deliver", f"sub_{lang}.{fmt}")
            if fmt == "ass":
                shutil.copyfile(cand["path"], p)  # keeps styling; timing fix applied below
                _shift_ass(p, cand.get("offset", 0.0), cand.get("scale", 1.0))
            else:
                write_srt(cues, p)
            out.append({"language": lang, "path": str(p), "title": LANG_TITLE.get(lang, lang),
                        "format": fmt, "source": cand["label"]})
            continue
        if lang == "en":
            cues = [Cue(l.start, l.end, l.en, raw=l.en) for l in lines if l.kind == L.SPEECH and l.en]
            cues += [Cue(l.start, l.end, l.source_text, raw=l.source_text) for l in lines if l.kind == L.ENGLISH]
            title = "English (AI translation)"
        else:
            cues = [Cue(l.start, l.end, l.source_text, raw=l.source_text) for l in lines
                    if l.kind in (L.SPEECH, L.ENGLISH) and l.source_text]
            title = f"{LANG_TITLE.get(lang, lang)} (AI transcript)"
        if not cues:
            continue
        cues.sort(key=lambda c: c.start)
        p = write_srt(cues, job.path("deliver", f"sub_{lang}.srt"))
        out.append({"language": lang, "path": str(p), "title": title, "format": "srt", "source": "AI"})
    return out


def _shift_ass(path: Path, offset: float, scale: float) -> None:
    if abs(offset) < 1e-3 and abs(scale - 1) < 1e-6:
        return
    import pysubs2
    subs = pysubs2.load(str(path))
    for ev in subs.events:
        ev.start = int(ev.start * scale + offset * 1000)
        ev.end = int(ev.end * scale + offset * 1000)
    subs.save(str(path))


def _tags_xml(path: Path, tags: dict) -> Path:
    simple = "".join(f"<Simple><Name>{escape(k)}</Name><String>{escape(str(v))}</String></Simple>"
                     for k, v in tags.items())
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<Tags><Tag><Targets></Targets>{simple}</Tag></Tags>',
                    encoding="utf-8")
    return path


def _our_tracks_mkvmerge(src: Path) -> list[int]:
    info = json.loads(media.run(["mkvmerge", "-J", str(src)]).stdout)
    return [t["id"] for t in info.get("tracks", [])
            if any(s in (t.get("properties", {}).get("track_name") or "") for s in AI_TITLES)]


def write_mkv(job: Job, src: Path, dub: Path, subs: list[dict], out: Path, drop_ours: bool) -> None:
    cmd = ["mkvmerge", "-q", "-o", str(out)]
    if drop_ours:
        ours = _our_tracks_mkvmerge(src)
        info = json.loads(media.run(["mkvmerge", "-J", str(src)]).stdout)
        kinds = {t["id"]: t["type"] for t in info.get("tracks", [])}
        a = [i for i in ours if kinds.get(i) == "audio"]
        s = [i for i in ours if kinds.get(i) == "subtitles"]
        if a:
            cmd += ["--audio-tracks", "!" + ",".join(map(str, a))]
        if s:
            cmd += ["--subtitle-tracks", "!" + ",".join(map(str, s))]
    cmd.append(str(src))
    tags = _tags_xml(job.path("deliver", "dub_tags.xml"), stamp(job, "dub", text_source(job)))
    cmd += ["--language", "0:eng", "--track-name", f"0:{job.cfg.get('deliver.track_title', 'English (AI dub)')}",
            "--default-track-flag", "0:no", "--forced-display-flag", "0:no", "--tags", f"0:{tags}", str(dub)]
    for i, s in enumerate(subs):
        t = _tags_xml(job.path("deliver", f"sub{i}_tags.xml"), stamp(job, "subtitle", s["source"]))
        cmd += ["--language", f"0:{media.iso3(s['language'])}", "--track-name", f"0:{s['title']}",
                "--default-track-flag", "0:no", "--forced-display-flag", "0:no", "--tags", f"0:{t}",
                "--sub-charset", "0:UTF-8", s["path"]]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode >= 2:  # 1 = warnings
        raise DeliveryError("mkvmerge failed: " + (proc.stdout + proc.stderr)[-800:])


def write_ffmpeg(job: Job, src: Path, dub: Path, subs: list[dict], out: Path, fmt: str, drop: list[int]) -> None:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-i", str(dub)]
    for s in subs:
        cmd += ["-i", s["path"]]
    cmd += ["-map", "0"]
    for idx in drop:
        cmd += ["-map", f"-0:{idx}"]
    cmd += ["-map", "1:a:0"]
    for i in range(len(subs)):
        cmd += ["-map", f"{i + 2}:s:0"]
    cmd += ["-c", "copy"]
    pr = media.probe(src)
    n_audio = sum(1 for s in pr.audio if s.index not in drop)
    n_subs = sum(1 for s in pr.subtitles if s.index not in drop)
    title = job.cfg.get("deliver.track_title", "English (AI dub)")
    a = f"a:{n_audio}"
    cmd += [f"-metadata:s:{a}", "language=eng", f"-metadata:s:{a}", f"title={title}",
            f"-metadata:s:{a}", f"handler_name={title}", f"-disposition:{a}", "0"]
    for k, v in stamp(job, "dub", text_source(job)).items():
        cmd += [f"-metadata:s:{a}", f"{k}={v}"]
    for i, s in enumerate(subs):
        k = f"s:{n_subs + i}"
        if fmt == "mp4":
            cmd += [f"-c:{k}", "mov_text"]
        cmd += [f"-metadata:s:{k}", f"language={media.iso3(s['language'])}", f"-metadata:s:{k}", f"title={s['title']}",
                f"-metadata:s:{k}", f"handler_name={s['title']}", f"-disposition:{k}", "0"]
    if fmt == "mp4":
        cmd += ["-movflags", "+faststart", "-f", "mp4"]
    else:
        cmd += ["-f", "matroska"]
    cmd.append(str(out))
    media.run(cmd)


def verify(src: Path, out: Path, dropped: set[int], added_audio: int, added_subs: int) -> list[str]:
    """Problems found comparing the new file with its source (empty = OK)."""
    a, b = media.probe(src), media.probe(out)
    problems = []
    kept = [s for s in a.streams if s.index not in dropped]
    kept_tracks = [s for s in kept if s.type in ("video", "audio", "subtitle")]
    new_tracks = [s for s in b.streams if s.type in ("video", "audio", "subtitle")]
    if len(new_tracks) != len(kept_tracks) + added_audio + added_subs:
        problems.append(f"expected {len(kept_tracks) + added_audio + added_subs} tracks, found {len(new_tracks)}")
    for s in kept_tracks:
        match = [t for t in new_tracks if t.type == s.type and t.codec == s.codec and t.language == s.language]
        if not match:
            problems.append(f"stream {s.index} ({s.type} {s.codec} {s.language}) missing")
            continue
        if s.duration and not any(abs(t.duration - s.duration) < 1.0 or not t.duration for t in match):
            problems.append(f"stream {s.index} duration changed")
    if len(a.attachments) != len(b.attachments) and a.container == b.container:
        problems.append(f"attachments {len(a.attachments)} -> {len(b.attachments)}")
    if len(a.chapters) != len(b.chapters) and b.container == "mkv":
        problems.append(f"chapters {len(a.chapters)} -> {len(b.chapters)}")
    if a.duration and abs(a.duration - b.duration) > 1.5:
        problems.append(f"duration {a.duration:.1f}s -> {b.duration:.1f}s")
    ours = [s for s in b.audio if any(t in s.title for t in AI_TITLES) or s.is_aidub]
    if len(ours) != 1:
        problems.append("dub track not found in the new file")
    for at in (0.0, max(0.0, b.duration - 8)):
        proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{at:.2f}", "-i", str(out), "-t", "3",
                               "-map", "0:v?", "-map", "0:a", "-f", "null", "-"], capture_output=True, text=True)
        if proc.returncode != 0 or "Error" in proc.stderr:
            problems.append(f"decode check at {at:.0f}s failed: {proc.stderr.strip()[-200:]}")
    return problems


def _copy_owner(src_stat: os.stat_result, dst: Path, owner: dict | None) -> None:
    try:
        os.chmod(dst, src_stat.st_mode & 0o7777)
    except OSError:
        pass
    try:
        os.chown(dst, src_stat.st_uid, src_stat.st_gid)
    except OSError:
        if owner:
            try:
                os.chown(dst, int(owner.get("uid", 99)), int(owner.get("gid", 100)))
            except OSError:
                pass


def run(job: Job) -> None:
    ing = job["ingest"]
    lines = L.load(job.path("lines.json"))
    live = job.src
    source = Path(ing["source"])
    redo_from_aside = source != live
    pr = media.probe(source)

    dub = job.path("deliver", "dub.mka")
    job.status("encoding the dub track")
    media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(job.path("mix", "dub.wav")),
               "-c:a", "aac", "-b:a", str(job.cfg.get("deliver.audio_bitrate", "192k")), "-ar", "48000",
               "-ac", "2", str(dub)])
    subs = subtitle_tracks(job, lines)

    if job.options.get("preview_s"):
        out = Path(job.options.get("output") or job.path("preview.mkv"))
        out.parent.mkdir(parents=True, exist_ok=True)
        if media.have("mkvmerge"):
            write_mkv(job, source, dub, subs, out, drop_ours=True)
        else:
            write_ffmpeg(job, source, dub, subs, out, "mkv", [s.index for s in pr.streams if s.is_aidub])
        job.set("delivered", {"preview": str(out)})
        job.status(f"preview written to {out}")
        return

    container = pr.container
    out_fmt = "mp4" if container == "mp4" else "mkv"
    final = live if container in ("mkv", "mp4") else live.with_suffix(".mkv")
    if final != live and final.exists():
        raise DeliveryError(f"{final.name} already exists next to the original")
    tmp = live.with_name(f".{final.stem}.aidub-partial")
    tmp.unlink(missing_ok=True)
    drop = [s.index for s in pr.streams if s.is_aidub or (s.type in ("audio", "subtitle")
                                                          and any(t in s.title for t in AI_TITLES))]
    st = live.stat()
    if shutil.disk_usage(live.parent).free < source.stat().st_size * 1.1:
        raise DeliveryError("not enough free space next to the file")

    job.status("writing the new file (stream copy)")
    try:
        if out_fmt == "mkv" and media.have("mkvmerge"):
            write_mkv(job, source, dub, subs, tmp, drop_ours=bool(drop))
        else:
            write_ffmpeg(job, source, dub, subs, tmp, out_fmt, drop)
        job.status("verifying the new file")
        problems = verify(source, tmp, set(drop), 1, len(subs))
        if problems:
            raise DeliveryError("verification failed: " + "; ".join(problems))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # Swap: set the original aside, then give the new file its name.
    aside = originals.aside_name(live, "dubbed" if redo_from_aside else "original")
    if aside.exists():
        tmp.unlink(missing_ok=True)
        raise DeliveryError(f"{aside.name} already exists; restore or delete it first")
    os.rename(live, aside)
    try:
        os.rename(tmp, final)
    except Exception:
        os.rename(aside, live)
        tmp.unlink(missing_ok=True)
        raise
    _copy_owner(st, final, job.cfg.get("deliver.owner"))
    job.set("delivered", {
        "file": str(final),
        "set_aside": str(aside),
        "set_aside_kind": "dubbed" if redo_from_aside else "original",
        "subtitle_tracks": [{k: s[k] for k in ("language", "title", "source")} for s in subs],
        "stamp": stamp(job, "dub", text_source(job)),
    })
    job.status(f"done: {final.name}; original set aside as {aside.name}")
