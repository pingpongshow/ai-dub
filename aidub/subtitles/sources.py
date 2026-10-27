"""Collecting subtitle candidates: embedded tracks, sidecar files, Bazarr, OCR."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .. import media
from . import clean
from .model import DIALOGUE, Candidate, Cue

TEXT_EXT = {".srt": "srt", ".ass": "ass", ".ssa": "ass", ".vtt": "vtt"}
BITMAP_EXT = {".sup": "pgs"}
FORCED_WORDS = {"forced", "signs", "songs", "sdh-forced"}
HI_WORDS = {"sdh", "hi", "cc"}


def _lang_from_name(tokens: list[str]) -> str:
    for t in reversed(tokens):
        code = media.lang_code(t)
        if code in {"en", "ja", "ko", "zh", "yue", "es", "fr", "de", "ar"}:
            return code
        if t.lower() in {"jp", "jap"}:
            return "ja"
        if t.lower() in {"kr"}:
            return "ko"
        if t.lower() in {"chs", "cht", "sc", "tc", "cn", "zho", "zh-hans", "zh-hant"}:
            return "zh"
    return ""


def sidecars(video: Path) -> list[tuple[Path, str, str, bool]]:
    """(path, format, language, forced) for subtitle files next to `video`."""
    stem = video.stem
    out = []
    try:
        entries = list(video.parent.iterdir())
    except OSError:
        return out
    for p in entries:
        ext = p.suffix.lower()
        if ext not in TEXT_EXT and ext not in BITMAP_EXT:
            continue
        if not p.name.startswith(stem + ".") and p.stem != stem:
            continue
        extra = p.name[len(stem):-len(ext)].strip(".")
        tokens = [t for t in re.split(r"[.\s_\-\[\]]+", extra.lower()) if t]
        fmt = TEXT_EXT.get(ext) or BITMAP_EXT[ext]
        out.append((p, fmt, _lang_from_name(tokens), bool(set(tokens) & FORCED_WORDS)))
    return sorted(out)


def embedded(job_dir: Path, src: Path, pr: media.Probe) -> list[Candidate]:
    out = []
    for s in pr.subtitles:
        if s.is_aidub:
            continue
        if s.is_text_sub:
            fmt = "ass" if s.codec in {"ass", "ssa"} else "srt"
            dst = job_dir / f"embedded_{s.index}.{fmt}"
            try:
                media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", f"0:{s.index}",
                           "-c:s", fmt if fmt == "ass" else "srt", str(dst)])
            except media.MediaError:
                continue
            out.append(Candidate(id=f"embedded:{s.index}", label=f"embedded track {s.index} ({s.language or '?'}"
                                 f"{', ' + s.title if s.title else ''})", origin="embedded", language=s.language,
                                 format=fmt, path=str(dst), in_file=True, forced=s.forced))
        elif s.codec == "hdmv_pgs_subtitle":
            dst = job_dir / f"embedded_{s.index}.sup"
            try:
                media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", f"0:{s.index}",
                           "-c:s", "copy", str(dst)])
            except media.MediaError:
                continue
            out.append(Candidate(id=f"embedded:{s.index}", label=f"embedded PGS track {s.index} ({s.language or '?'})",
                                 origin="ocr", language=s.language, format="pgs", path=str(dst), in_file=True,
                                 forced=s.forced))
    return out


def from_sidecars(job_dir: Path, video: Path) -> list[Candidate]:
    out = []
    for i, (p, fmt, lang, forced) in enumerate(sidecars(video)):
        dst = job_dir / f"sidecar_{i}{p.suffix.lower()}"
        shutil.copyfile(p, dst)
        out.append(Candidate(id=f"sidecar:{p.name}", label=f"sidecar {p.name}", origin="sidecar" if fmt != "pgs" else "ocr",
                             language=lang, format=fmt, path=str(dst), in_file=True, forced=forced))
    return out


def load_cues(c: Candidate) -> None:
    """Fill `c.cues` for a text candidate and confirm its language from the text."""
    c.cues = clean.load(c.path)
    guessed = clean.script_language([q.text for q in c.cues if q.kind == DIALOGUE])
    if guessed and (not c.language or (guessed != c.language and c.language not in {"yue"})):
        c.language = guessed


def ocr_cues(c: Candidate, worker, lang: str, work: Path) -> None:
    """OCR a PGS candidate into cues using the OCR worker."""
    from PIL import Image  # optional dependency, only for bitmap subtitles

    from . import pgs
    items = []
    img_dir = work / f"ocr_{c.id.replace(':', '_')}"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, (start, end, rgba) in enumerate(pgs.read(c.path)):
        p = img_dir / f"{i:05d}.png"
        Image.fromarray(rgba, "RGBA").save(p)
        items.append((start, end, p))
    res = worker.call("ocr", {"images": [str(p) for _, _, p in items], "language": lang})
    cues = []
    for (start, end, _), text in zip(items, res["texts"]):
        t, name, kind = clean.clean_text(text.replace("\n", " "))
        cues.append(Cue(start=start, end=end, text=t if kind == DIALOGUE else "", raw=text, name=name, kind=kind))
    shutil.rmtree(img_dir, ignore_errors=True)
    srt = Path(c.path).with_suffix(".srt")
    write_srt([q for q in cues if q.raw.strip()], srt)
    c.path, c.format, c.human, c.cues = str(srt), "srt", True, cues
    c.label += " (OCR)"


def write_srt(cues: list[Cue], path: Path, *, use_raw: bool = True) -> Path:
    def ts(t: float) -> str:
        ms = int(round(max(0.0, t) * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    blocks = []
    for i, q in enumerate(cues, 1):
        text = (q.raw if use_raw else q.text).strip()
        if text:
            blocks.append(f"{len(blocks) + 1}\n{ts(q.start)} --> {ts(q.end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path
