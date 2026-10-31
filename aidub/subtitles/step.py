"""Step 3c: gather, sync, score and choose subtitles.

Saves `subtitles`: {"candidates": [...], "english": id|None, "source": id|None,
                    "file_languages": [...]} and the chosen cues as JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import media
from ..job import Job
from . import score, sources
from .bazarr import Bazarr
from .model import DIALOGUE, Candidate, Cue


def _speech_mask(job: Job) -> tuple:
    dur = job["ingest"]["duration"]
    segs = job["diarization"]["segments"]
    return score.mask([(s["start"], s["end"]) for s in segs], dur), dur


def _bazarr(job: Job, cands: list[Candidate], src_lang: str, workdir: Path) -> list[Candidate]:
    pc = job.cfg.get("subtitles.providers.bazarr") or {}
    if not pc.get("enabled"):
        return []
    try:
        bz = Bazarr(pc.get("url", ""), pc.get("api_key", ""), pc.get("path_map"))
    except ValueError as exc:
        job.log(f"bazarr: {exc}")
        return []
    try:
        rec = bz.find(str(job.src))
        if not rec:
            job.status("Bazarr doesn't manage this file; using its own subtitles and recognition")
            return []
        have = {c.language for c in cands if not c.forced}
        if pc.get("search_missing", True):
            for lang in ("en", src_lang):
                if lang and lang not in have:
                    job.status(f"asking Bazarr to search for {lang} subtitles")
                    try:
                        bz.search(rec, lang)
                    except Exception as exc:
                        job.log(f"bazarr search {lang}: {exc}")
        # Whatever Bazarr has is now a sidecar next to the video; mark which.
        known = {c.id for c in cands}
        new = [c for c in sources.from_sidecars(workdir, job.src) if c.id not in known]
        for c in new:
            c.origin = "bazarr"
            c.label = c.label.replace("sidecar", "Bazarr sidecar")
        return new
    except Exception as exc:
        job.status(f"Bazarr unavailable ({type(exc).__name__}); continuing without it")
        job.log(f"bazarr: {exc}")
        return []
    finally:
        bz.close()


def _score_source(c: Candidate, asr: dict, speech, dur: float) -> None:
    off, scale, frac, sharp = score.best_offset(c.cues, speech, dur)
    synced = score.apply_sync(c.cues, off, scale)
    d = score.drift(synced, speech, dur)
    cov = score.coverage(synced, speech, dur)
    content = score.content_source(synced, asr["segments"])
    c.cues, c.offset, c.scale = synced, off, scale
    c.score = {"timing": round(frac, 3), "coverage": round(cov, 3), "content": round(content, 3),
               "drift": round(d, 2), "sharpness": round(sharp, 3),
               "total": score.combine(frac, cov, content, d, sharp)}


def _score_english(c: Candidate, asr: dict, speech, dur: float, embed_worker) -> None:
    off, scale, frac, sharp = score.best_offset(c.cues, speech, dur)
    synced = score.apply_sync(c.cues, off, scale)
    d = score.drift(synced, speech, dur)
    cov = score.coverage(synced, speech, dur)
    pairs = []
    for q in synced:
        if q.kind == DIALOGUE and q.text:
            src = score.asr_text_between(asr["segments"], q.start - 0.4, q.end + 0.4)
            if src:
                pairs.append((q.text, src))
    content = 0.0
    if len(pairs) >= 4 and embed_worker is not None:
        sample = pairs[:: max(1, len(pairs) // 150)]
        k = max(1, len(sample) // 2)
        shifted = [(en, sample[(i + k) % len(sample)][1]) for i, (en, _) in enumerate(sample)]
        res = embed_worker.call("text_similarity", {"pairs": [list(p) for p in sample + shifted]})
        sims = res["similarity"]
        content = score.discrimination(sims[: len(sample)], sims[len(sample):])
    c.cues, c.offset, c.scale = synced, off, scale
    c.score = {"timing": round(frac, 3), "coverage": round(cov, 3), "content": round(content, 3),
               "drift": round(d, 2), "sharpness": round(sharp, 3),
               "total": score.combine(frac, cov, content, d, sharp)}


def _align_source(job: Job, c: Candidate, asr_worker, lang: str) -> None:
    """Force-align each source-language cue's text to the speech for word timings."""
    data, sr = media.read_wav(job.path("audio", "dialogue16k.wav"), mono=True)
    items, idx = [], []
    tmp = job.dir("subs", "align")
    for i, q in enumerate(c.cues):
        if q.kind != DIALOGUE or not q.text:
            continue
        a, b = max(0.0, q.start - 0.5), q.end + 0.5
        if b - a < 0.3:  # a cue moved before the start by the sync offset
            continue
        p = media.write_wav(tmp / f"{i:05d}.wav", data[int(a * sr): int(b * sr)], sr)
        items.append({"path": str(p), "text": q.text, "offset": a})
        idx.append(i)
    aligned = {}
    for k in range(0, len(items), 32):
        job.check()
        part = items[k:k + 32]
        res = asr_worker.call("align", {"items": [{"path": x["path"], "text": x["text"]} for x in part],
                                        "language": lang})
        for x, i, r in zip(part, idx[k:k + 32], res["results"]):
            words = [{"text": w["text"], "start": round(x["offset"] + w["start"], 3),
                      "end": round(x["offset"] + w["end"], 3)} for w in r.get("words", [])]
            if words:
                aligned[i] = words
        job.status(f"aligned {min(k + 32, len(items))}/{len(items)} subtitle lines",
                   progress=min(1.0, (k + 32) / max(1, len(items))))
    for x in items:
        Path(x["path"]).unlink(missing_ok=True)
    for i, words in aligned.items():
        q = c.cues[i]
        if abs(words[0]["start"] - q.start) < 1.5 and abs(words[-1]["end"] - q.end) < 1.5:
            q.start, q.end = words[0]["start"], words[-1]["end"]
    job.set("subtitle_words", {str(i): w for i, w in aligned.items()})


def run(job: Job) -> None:
    workdir = job.dir("subs")
    ing = job["ingest"]
    asr = job["asr"]
    src_lang = asr.get("language") or ing.get("language") or ""
    pr = media.probe(Path(ing["source"]))
    providers = job.cfg.get("subtitles.providers") or {}

    cands: list[Candidate] = []
    if providers.get("embedded", {}).get("enabled", True):
        cands += sources.embedded(workdir, Path(ing["source"]), pr)
    if providers.get("sidecar", {}).get("enabled", True):
        cands += sources.from_sidecars(workdir, job.src)
    cands += _bazarr(job, cands, src_lang, workdir)
    file_languages = sorted({c.language for c in cands if c.in_file and c.language})

    # OCR bitmap candidates in the languages we can use.
    bitmaps = [c for c in cands if c.format == "pgs" and c.language in {"en", src_lang, ""} and not c.forced]
    if bitmaps and job.cfg.get("stages.ocr.worker"):
        with job.worker("ocr") as w:
            w.call("load", {})
            for c in bitmaps:
                job.check()
                job.status(f"reading {c.label} with OCR")
                try:
                    sources.ocr_cues(c, w, c.language or src_lang, workdir)
                    job.models["ocr"] = "PaddleOCR"
                except Exception as exc:
                    job.log(f"OCR failed for {c.label}: {exc}")
                    c.score = {"total": 0.0, "error": str(exc)}
    habits = ((job.profile.data.get("rules") or {}).get("habits") or {}) if job.profile else {}
    for c in cands:
        if c.format != "pgs" and not c.cues:
            try:
                sources.load_cues(c, habits)
            except Exception as exc:
                job.log(f"unreadable subtitle {c.label}: {exc}")

    speech, dur = _speech_mask(job)
    usable = [c for c in cands if c.cues and not c.forced and c.language in {"en", src_lang}]
    english = [c for c in usable if c.language == "en"]
    source = [c for c in usable if c.language == src_lang and src_lang != "en"]

    for c in source:
        _score_source(c, asr, speech, dur)
    if english:
        with job.worker("speaker", gpu_stage="speaker") as w:
            w.call("load", {})
            for c in english:
                _score_english(c, asr, speech, dur, w)

    min_score = float(job.cfg.get("subtitles.min_score", 0.65))
    pick = lambda cs: max((c for c in cs if c.score.get("total", 0) >= min_score),
                          key=lambda c: (c.score["total"], c.human), default=None)
    best_en, best_src = pick(english), pick(source)

    if best_src is not None:
        with job.worker("recognize") as w:
            cfg = job.cfg.get("stages.recognize")
            w.call("load", {"model": cfg.get("model"), "aligner": cfg.get("aligner"), "aligner_only": True})
            _align_source(job, best_src, w, src_lang)

    for c in cands:
        if c.cues:
            Path(workdir / f"{c.id.replace(':', '_').replace('/', '_')}.cues.json").write_text(
                json.dumps([q.to_dict() for q in c.cues], ensure_ascii=False))
    job.set("subtitles", {
        "candidates": [c.to_dict() for c in cands],
        "english": best_en.id if best_en else None,
        "source": best_src.id if best_src else None,
        "file_languages": file_languages,
        "source_language": src_lang,
    })
    parts = []
    if best_en:
        parts.append(f"English: {best_en.label} ({best_en.score['total']:.2f})")
    if best_src:
        parts.append(f"{src_lang}: {best_src.label} ({best_src.score['total']:.2f})")
    job.status("; ".join(parts) if parts else "no usable subtitle; using speech recognition")


def load_chosen(job: Job, which: str) -> list[Cue]:
    cid = job["subtitles"].get(which)
    if not cid:
        return []
    p = job.work / "subs" / f"{cid.replace(':', '_').replace('/', '_')}.cues.json"
    return [Cue(**d) for d in json.loads(p.read_text())] if p.exists() else []


def candidate(job: Job, cid: str | None) -> dict | None:
    for c in job["subtitles"]["candidates"]:
        if c["id"] == cid:
            return c
    return None
