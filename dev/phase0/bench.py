"""Phase 0 model comparisons (development only, not shipped).

Runs inside the Ai Dub container against short excerpts copied from the
library. Every stage is compared on the same inputs, with numbers where a
reference exists and blind listening pages where only ears can judge.

    python bench.py prepare  clips.json      # cut excerpts, run the default pipeline up to the script
    python bench.py separation|recognition|translation|voice|bandwidth|diarization
    python bench.py report                   # results/report.html (+ listening pages)

clips.json: [{"name": "kaiju", "path": "/media/anime/...mkv", "start": 150, "seconds": 180,
              "language": "ja"}, ...]
Everything is written under $PHASE0_ROOT (default /work/phase0); results/ holds
the numbers and the listening pages, which are copied off and then deleted.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from aidub import config, media, pipeline
from aidub import lines as L
from aidub.job import Job
from aidub.profiles import Profiles
from aidub.recognize import speech_chunks
from aidub.speak import wer
from aidub.subtitles import clean
from aidub.subtitles.score import cer, discrimination
from aidub.translate import syllables, target_syllables
from aidub.workers import Worker

ROOT = Path(os.environ.get("PHASE0_ROOT", "/work/phase0"))
RESULTS = ROOT / "results"
SEPARATORS = ["vocals_mel_band_roformer.ckpt", "bs_roformer_vocals_resurrection_unwa.ckpt",
              "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt", "melband_roformer_big_beta6x.ckpt"]
LANG_NAME = {"ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


# Setup -----------------------------------------------------------------------------------------
def cfg():
    c = config.load()
    c.data["paths"]["work"] = str(ROOT / "jobs")
    c.data["paths"]["appdata"] = str(ROOT / "appdata")
    c.data["subtitles"]["providers"]["bazarr"]["enabled"] = False
    return c


def clips() -> list[dict]:
    return json.loads((ROOT / "clips.json").read_text())


def clip_job(c, clip: dict) -> Job:
    prof = Profiles(c.appdata).get_or_create(f"phase0 {clip['name']}")
    return Job(c, f"p0-{clip['name']}", ROOT / "clips" / f"{clip['name']}.mkv", prof, {"language": clip.get("language")})


def save(stage: str, data) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def excerpt(src: Path, dst: Path, start: float, seconds: float) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(src), "-t",
               f"{seconds:.2f}", "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(dst)])
    return dst


def cmd_prepare(path: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, ROOT / "clips.json")
    c = cfg()
    gen = Path("/config/voices")
    if gen.exists() and not (c.appdata / "voices").exists():
        shutil.copytree(gen, c.appdata / "voices")
    for clip in clips():
        dst = ROOT / "clips" / f"{clip['name']}.mkv"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            media.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(clip["start"]), "-i", clip["path"],
                       "-t", str(clip["seconds"]), "-map", "0", "-c", "copy", str(dst)])
        job = clip_job(c, clip)
        job.on_status = lambda d, n=clip["name"]: d.get("message") and log(n, d.get("step"), d["message"])
        pipeline.run(job, until="translate")
        log(clip["name"], "prepared")


# References --------------------------------------------------------------------------------------
def source_reference(clip: dict) -> list:
    """Cues of the clip's own source-language subtitle (not forced), for recognition error rates."""
    src = ROOT / "clips" / f"{clip['name']}.mkv"
    pr = media.probe(src)
    lang = clip.get("language")
    subs = [s for s in pr.subtitles if s.language == lang and s.is_text_sub and not s.forced
            and "forced" not in s.title.lower()]
    subs.sort(key=lambda s: ("sdh" in s.title.lower(), s.index))  # plain before SDH
    if not subs:
        return []
    out = ROOT / "refs" / f"{clip['name']}.srt"
    out.parent.mkdir(parents=True, exist_ok=True)
    media.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", f"0:{subs[0].index}", str(out)])
    return [q for q in clean.load(out) if q.kind == "dialogue" and q.text]


def english_reference(job: Job) -> list:
    from aidub.subtitles.step import load_chosen
    return [q for q in load_chosen(job, "english") if q.kind == "dialogue" and q.text]


def chunk_texts(job: Job, model_worker: str, model: str | None, extra: dict | None = None):
    """Transcribe the job's dialogue stem chunks with a recognizer worker."""
    segs = job["diarization"]["segments"]
    chunks = speech_chunks(segs, job["ingest"]["duration"])
    data, sr = media.read_wav(job.path("audio", "dialogue16k.wav"), mono=True)
    tmp = ROOT / "tmp" / job.id
    tmp.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (a, b) in enumerate(chunks):
        items.append({"path": str(media.write_wav(tmp / f"c{i:04d}.wav", data[int(a * sr): int(b * sr)], sr))})
    out = []
    with Worker(job.cfg, model_worker, log=lambda l: None) as w:
        w.call("load", {"model": model, **(extra or {})})
        t0 = time.time()
        for i in range(0, len(items), 16):
            res = w.call("transcribe", {"items": items[i:i + 16], "language": job["asr"]["language"],
                                        "timestamps": True})["results"]
            out.extend(res)
        took = time.time() - t0
    shutil.rmtree(tmp, ignore_errors=True)
    return chunks, out, took


def text_in(chunks, results, a: float, b: float) -> str:
    parts = []
    for (ca, cb), r in zip(chunks, results):
        if r.get("words"):
            parts += [w["text"] for w in r["words"] if a - 0.3 <= ca + (w["start"] + w["end"]) / 2 <= b + 0.3]
        elif min(cb, b) - max(ca, a) > 0.5 * (b - a):
            parts.append(r.get("text", ""))
    return "".join(parts)


def similarity(c, pairs: list[list[str]]) -> list[float]:
    if not pairs:
        return []
    with Worker(c, "speaker", log=lambda l: None) as w:
        w.call("load", {})
        return w.call("text_similarity", {"pairs": pairs})["similarity"]


def discrimination_score(c, en_cues, chunks, results) -> float | None:
    pairs = [(q.text, text_in(chunks, results, q.start, q.end)) for q in en_cues]
    pairs = [p for p in pairs if p[1]]
    if len(pairs) < 6:
        return None
    k = len(pairs) // 2
    shifted = [(p[0], pairs[(i + k) % len(pairs)][1]) for i, p in enumerate(pairs)]
    sims = similarity(c, [list(p) for p in pairs + shifted])
    return round(discrimination(sims[: len(pairs)], sims[len(pairs):]), 3)


def ref_cer(ref_cues, chunks, results) -> float | None:
    if not ref_cues:
        return None
    hyp = "".join(text_in(chunks, results, q.start, q.end) for q in ref_cues)
    return round(cer("".join(q.text for q in ref_cues), hyp), 3)


# Separation ---------------------------------------------------------------------------------------
def cmd_separation() -> None:
    c = cfg()
    out = {}
    for clip in clips():
        job = clip_job(c, clip)
        src, sr = media.read_wav(job.path("audio", "source.wav"))
        stereo = src[:, :2] if src.shape[1] >= 2 else np.repeat(src, 2, axis=1)
        if src.shape[1] >= 6:  # bench on a standard stereo fold-down of 5.1
            k = 1 / np.sqrt(2)
            stereo = np.stack([src[:, 0] + k * src[:, 2] + k * src[:, 4], src[:, 1] + k * src[:, 2] + k * src[:, 5]], 1)
        inp = media.write_wav(ROOT / "sep" / clip["name"] / "input.wav", stereo, sr)
        refs = source_reference(clip)
        en = english_reference(job)
        rows = {}
        models = SEPARATORS + (["bandit-v2"] if (Path(c.models) / "envs" / "separate-bandit").exists() else [])
        for model in models:
            d = ROOT / "sep" / clip["name"] / model.split(".")[0]
            worker = "separate-bandit" if model == "bandit-v2" else "separate"
            log(clip["name"], "separating with", model)
            with Worker(c, worker, log=lambda l: None) as w:
                w.call("load", {"model": None if model == "bandit-v2" else model})
                t0 = time.time()
                res = w.call("separate", {"input": str(inp), "out_dir": str(d)})
                took = time.time() - t0
            # Recognition on this dialogue stem, and on the M&E stem (voice that leaked into the bed).
            dl, dsr = media.read_wav(res["vocals"], mono=True)
            mn, msr = media.read_wav(res["instrumental"], mono=True)
            media.write_wav(job.path("audio", "dialogue16k.bench.wav"), media.resample(dl, dsr, 16000), 16000)
            orig = job.path("audio", "dialogue16k.wav")
            keep = orig.with_suffix(".orig.wav")
            shutil.copyfile(orig, keep)
            try:
                shutil.copyfile(job.path("audio", "dialogue16k.bench.wav"), orig)
                chunks, res_d, _ = chunk_texts(job, "asr", None)
                media.write_wav(orig, media.resample(mn, msr, 16000), 16000)
                _, res_m, _ = chunk_texts(job, "asr", None)
            finally:
                shutil.copyfile(keep, orig)
                keep.unlink()
            speech_min = sum(b - a for a, b in chunks) / 60
            leaked = sum(len(clean.norm_chars(r.get("text", ""))) for r in res_m)
            rows[model] = {"seconds": round(took, 1), "ref_cer": ref_cer(refs, chunks, res_d),
                           "en_match": discrimination_score(c, en, chunks, res_d),
                           "leaked_chars_per_speech_min": round(leaked / max(speech_min, 0.1), 1)}
            excerpt(Path(res["vocals"]), RESULTS / "listen" / "separation" / clip["name"] / f"{model}-dialogue.m4a", 30, 20)
            excerpt(Path(res["instrumental"]), RESULTS / "listen" / "separation" / clip["name"] / f"{model}-mne.m4a", 30, 20)
            log(clip["name"], model, rows[model])
        out[clip["name"]] = rows
        save("separation", out)


# Recognition ----------------------------------------------------------------------------------------
def recognizers(lang: str) -> list[tuple[str, str, str | None]]:
    out = [("Qwen3-ASR 1.7B", "asr", None)]
    if (Path(cfg().models) / "envs" / "asr-whisper").exists():
        out.append(("Whisper large-v3", "asr-whisper", "openai/whisper-large-v3"))
        if lang == "ja":
            out.append(("anime-whisper", "asr-whisper", "litagin/anime-whisper"))
            out.append(("kotoba-whisper v2.2", "asr-whisper", "kotoba-tech/kotoba-whisper-v2.2"))
    return out


def cmd_recognition() -> None:
    c = cfg()
    out = {}
    for clip in clips():
        job = clip_job(c, clip)
        refs = source_reference(clip)
        en = english_reference(job)
        rows = {}
        for label, worker, model in recognizers(clip.get("language", "")):
            log(clip["name"], "recognising with", label)
            try:
                chunks, res, took = chunk_texts(job, worker, model)
            except Exception as exc:
                rows[label] = {"error": str(exc)[:300]}
                continue
            speech = sum(b - a for a, b in chunks)
            rows[label] = {"ref_cer": ref_cer(refs, chunks, res), "en_match": discrimination_score(c, en, chunks, res),
                           "word_timings": all(r.get("words") for r in res if r.get("text")),
                           "realtime_x": round(speech / max(took, 0.01), 1),
                           "sample": [r.get("text", "")[:80] for r in res[:4]]}
            log(clip["name"], label, {k: v for k, v in rows[label].items() if k != "sample"})
        out[clip["name"]] = {"has_source_reference": bool(refs), "results": rows}
        save("recognition", out)


# Translation ----------------------------------------------------------------------------------------
def cmd_translation() -> None:
    from aidub import translate as T
    c = cfg()
    out = {}
    for clip in clips():
        job = clip_job(c, clip)
        lines = [l for l in L.load(job.path("lines.json")) if l.kind == L.SPEECH]
        with_ref = [l for l in lines if l.english_ref]
        rate = float(c.get("translate.syllables_per_second", 4.3))
        systems = {"Qwen3.8 + subtitle hints": {l.id: l.en for l in lines}}
        # Without hints: what the dub sounds like when a show has no subtitle at all.
        bare = cfg()
        bare.data["paths"]["work"] = str(ROOT / "jobs-bare")
        shutil.copytree(job.work, Path(bare.work) / f"job-p0-{clip['name']}", dirs_exist_ok=True)
        bj = clip_job(bare, clip)
        bl = L.load(bj.path("lines.json"))
        for l in bl:
            l.english_ref = l.english_context = ""
            l.en = l.en_short = ""
        L.save(bl, bj.path("lines.json"))
        bj.path("translate_cache.json").unlink(missing_ok=True)
        t0 = time.time()
        T.run(bj)
        systems["Qwen3.8, no subtitle"] = {l.id: l.en for l in L.load(bj.path("lines.json")) if l.kind == L.SPEECH}
        timings = {"Qwen3.8, no subtitle": round(time.time() - t0, 1)}
        # Hy-MT2 (Tencent's translation model): one line per request, its documented prompt.
        hy = c.get("bench.hymt_model", "tencent/Hy-MT2-7B")
        convs = [[{"role": "user", "content": "Translate the following text into English. Note that you should only "
                                              "output the translated result without any additional explanation:\n\n"
                                              + l.source_text}] for l in lines]
        try:
            with Worker(c, "llm", log=lambda x: None) as w:
                w.call("load", {"model": hy, "gpu_memory_utilization": 0.70, "max_model_len": 4096,
                                "language_model_only": False, "linear_backend": "auto",
                                "attention_backend": "TRITON_ATTN"})
                t0 = time.time()
                outs = w.call("chat", {"conversations": convs, "max_tokens": 256, "temperature": 0.7, "top_p": 0.6,
                                       "top_k": 20, "repetition_penalty": 1.05, "presence_penalty": 0.0})["outputs"]
                timings["Hy-MT2 7B"] = round(time.time() - t0, 1)
            systems["Hy-MT2 7B"] = {l.id: o.strip() for l, o in zip(lines, outs)}
        except Exception as exc:
            log("Hy-MT2 failed:", exc)
        rows = {}
        for sysname, tr in systems.items():
            pairs = [[tr.get(l.id, ""), l.english_ref] for l in with_ref if tr.get(l.id)]
            sims = similarity(c, pairs)
            fit = [abs(syllables(tr.get(l.id, "")) / max(1, target_syllables(l.duration, rate)) - 1) <= 0.3
                   for l in lines if tr.get(l.id)]
            rows[sysname] = {"meaning_vs_subtitle": round(float(np.mean(sims)), 3) if sims else None,
                             "length_on_target": round(float(np.mean(fit)), 3) if fit else None,
                             "translated": sum(1 for l in lines if tr.get(l.id)), "lines": len(lines),
                             "seconds": timings.get(sysname)}
        out[clip["name"]] = {"results": rows, "examples": [
            {"source": l.source_text, "subtitle": l.english_ref or l.english_context,
             **{k: v.get(l.id, "") for k, v in systems.items()}} for l in lines[:40]]}
        log(clip["name"], rows)
        save("translation", out)
        shutil.rmtree(ROOT / "jobs-bare", ignore_errors=True)


# Voice ---------------------------------------------------------------------------------------------
# Higgs Audio v3 is left out: it runs only as a separate server and needs ~40 GB of VRAM.
VOICES = [("IndexTTS-2.5", "tts-indextts", {}), ("IndexTTS-2.5 emo 0.4", "tts-indextts", {"emo_alpha": 0.4}),
          ("IndexTTS-2.5 emo 0.9", "tts-indextts", {"emo_alpha": 0.9}), ("Qwen3-TTS", "tts-qwen3", {}),
          ("Chatterbox", "tts-chatterbox", {}), ("Ming-3.0", "tts-ming", {})]


def voice_lines(job: Job, n: int = 14) -> list[L.Line]:
    lines = [l for l in L.load(job.path("lines.json")) if l.kind == L.SPEECH and l.en and 0.8 <= l.duration <= 6]
    random.Random(7).shuffle(lines)
    by_char: dict[str, list] = {}
    for l in lines:
        by_char.setdefault(l.character, []).append(l)
    picked = []
    while len(picked) < n and any(by_char.values()):
        for ch in list(by_char):
            if by_char[ch] and len(picked) < n:
                picked.append(by_char[ch].pop())
    return sorted(picked, key=lambda l: l.start)


def cmd_voice() -> None:
    c = cfg()
    out = {}
    listen = []
    available = [v for v in VOICES if (Path(c.models) / "envs" / v[1]).exists()]
    for clip in clips():
        job = clip_job(c, clip)
        prof = job.profile
        lines = voice_lines(job)
        dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
        base = ROOT / "voice" / clip["name"]
        files: dict[str, list[tuple[L.Line, str]]] = {}
        timing: dict[str, float] = {}
        for ln in lines:
            orig = base / f"{ln.id:04d}-orig.wav"
            if not orig.exists():
                media.write_wav(orig, dlg[int(ln.start * sr): int(ln.end * sr)], sr)
            if not ln.voice:
                ln.voice = str(orig)
        for label, worker, extra in available:
            reqs = []
            for ln in lines:
                o = base / label.replace(" ", "_") / f"{ln.id:04d}.wav"
                r = {"text": ln.en, "speaker_ref": ln.voice, "emotion_ref": str(base / f"{ln.id:04d}-orig.wav"),
                     "emo_alpha": extra.get("emo_alpha", 0.65), "seed": 7, "out": str(o),
                     "prompt_text": ln.source_text}
                reqs.append(r)
                files.setdefault(label, []).append((ln, str(o)))
            log(clip["name"], "voicing with", label)
            try:
                with Worker(c, worker, log=lambda l: None) as w:
                    w.call("load", {})
                    t0 = time.time()
                    for i in range(0, len(reqs), 8):
                        w.call("synthesize", {"items": reqs[i:i + 8]})
                    timing[label] = time.time() - t0
            except Exception as exc:
                log(label, "failed:", exc)
                files.pop(label, None)
        # Score everything: intelligibility, similarity to the character, naturalness, length.
        from aidub import fit as F
        allf = [(label, ln, p) for label, items in files.items() for ln, p in items if Path(p).exists()]
        for _, _, p in allf:
            F.trim(Path(p))
        with Worker(c, "asr", log=lambda l: None) as w:
            w.call("load", {})
            heard = []
            for i in range(0, len(allf), 16):
                heard += w.call("transcribe", {"items": [{"path": p} for _, _, p in allf[i:i + 16]],
                                               "language": "en", "timestamps": False})["results"]
        with Worker(c, "speaker", log=lambda l: None) as w:
            w.call("load", {})
            embs = w.call("embed_files", {"paths": [p for _, _, p in allf]})["embeddings"]
        mos = [None] * len(allf)
        if (Path(c.models) / "envs" / "mos").exists():
            with Worker(c, "mos", log=lambda l: None) as w:
                w.call("load", {})
                mos = w.call("score", {"paths": [p for _, _, p in allf]})["mos"]
        from aidub.speakers import cosine
        rows: dict[str, dict] = {}
        for (label, ln, p), h, e, m in zip(allf, heard, embs, mos):
            ch = prof.characters.get(prof.voice_source(ln.character), {})
            r = rows.setdefault(label, {"wer": [], "sim": [], "mos": [], "len_ratio": []})
            r["wer"].append(wer(ln.en, h.get("text", "")))
            if ch.get("embedding") and e:
                r["sim"].append(cosine(e, ch["embedding"]))
            if m is not None:
                r["mos"].append(m)
            r["len_ratio"].append(media.duration(p) / max(0.3, ln.duration + 0.25))
        summary = {}
        for label, r in rows.items():
            n = len(r["wer"])
            summary[label] = {"lines": n, "wer": round(float(np.mean(r["wer"])), 3),
                              "similarity": round(float(np.mean(r["sim"])), 3) if r["sim"] else None,
                              "mos": round(float(np.mean(r["mos"])), 2) if r["mos"] else None,
                              "fits_window": round(float(np.mean([x <= 1.12 for x in r["len_ratio"]])), 3),
                              "sec_per_line": round(timing.get(label, 0) / max(1, n), 2)}
        out[clip["name"]] = summary
        log(clip["name"], summary)
        save("voice", out)
        for ln in lines:
            listen.append({"clip": clip["name"], "line": ln.id, "text": ln.en,
                           "orig": str(base / f"{ln.id:04d}-orig.wav"),
                           "takes": {label: p for label, items in files.items() for l2, p in items if l2.id == ln.id
                                     and Path(p).exists()}})
    write_listening("voice", listen)


# Bandwidth ------------------------------------------------------------------------------------------
def high_band(path: str, cutoff: float = 8000) -> float:
    x, sr = media.read_wav(path, mono=True)
    x = media.resample(x, sr, 48000)[:, 0]
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 48000)
    return float(spec[f >= cutoff].sum() / max(spec.sum(), 1e-12))


def cmd_bandwidth() -> None:
    c = cfg()
    out, listen = {}, []
    for clip in clips():
        base = ROOT / "voice" / clip["name"]
        takes = sorted((base / "IndexTTS-2.5").glob("*.wav"))[:10]
        origs = [base / f"{p.stem}-orig.wav" for p in takes]
        variants = {"plain resample": []}
        for p in takes:
            x, sr = media.read_wav(p, mono=True)
            variants["plain resample"].append(str(media.write_wav(base / "bwe-plain" / p.name, media.resample(x, sr, 48000), 48000)))
        for label, worker in (("FlashSR", "bwe-flashsr"), ("FlashSR tiny", "bwe-tiny")):
            if not (Path(c.models) / "envs" / worker).exists():
                continue
            outs = [str(base / f"bwe-{worker}" / p.name) for p in takes]
            for o in outs:
                Path(o).parent.mkdir(parents=True, exist_ok=True)
            with Worker(c, worker, log=lambda l: None) as w:
                w.call("load", {})
                w.call("upsample", {"items": [{"input": str(p), "output": o} for p, o in zip(takes, outs)]})
            variants[label] = outs
        mos = {}
        if (Path(c.models) / "envs" / "mos").exists():
            with Worker(c, "mos", log=lambda l: None) as w:
                w.call("load", {})
                for k, v in variants.items():
                    mos[k] = w.call("score", {"paths": v})["mos"]
        orig_hb = float(np.mean([high_band(str(o)) for o in origs if o.exists()]))
        out[clip["name"]] = {"original_high_band": round(orig_hb, 4), **{
            k: {"high_band": round(float(np.mean([high_band(p) for p in v])), 4),
                "mos": round(float(np.mean(mos[k])), 2) if k in mos else None} for k, v in variants.items()}}
        log(clip["name"], out[clip["name"]])
        save("bandwidth", out)
        for i, p in enumerate(takes):
            listen.append({"clip": clip["name"], "line": p.stem, "text": "", "orig": str(origs[i]),
                           "takes": {k: v[i] for k, v in variants.items()}})
    write_listening("bandwidth", listen)


# Diarization ----------------------------------------------------------------------------------------
def cmd_diarization() -> None:
    c = cfg()
    out = {}
    for clip in clips():
        job = clip_job(c, clip)
        d = job["diarization"]
        res = {"Nemotron 3": {"speakers": len(d["speakers"]), "turns": len(d["segments"]),
                              "overlap_share": round(sum(1 for s in d["segments"] if s.get("overlap")) / max(1, len(d["segments"])), 3)}}
        if (Path(c.models) / "envs" / "diarize-diarizen").exists():
            with Worker(c, "diarize-diarizen", log=lambda l: None) as w:
                w.call("load", {})
                segs = w.call("diarize", {"input": str(job.path("audio", "dialogue16k.wav"))})["segments"]
            res["DiariZen"] = {"speakers": len({s["speaker"] for s in segs}), "turns": len(segs)}
        out[clip["name"]] = res
        log(clip["name"], res)
    save("diarization", out)


# Listening pages and report -------------------------------------------------------------------------
def write_listening(stage: str, items: list[dict]) -> None:
    """Blind A/B page: each line's versions are shuffled under letters; the key is a separate file."""
    d = RESULTS / "listen" / stage
    d.mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    rng = random.Random(11)
    for n, it in enumerate(items):
        def enc(src: str, name: str) -> str:
            dst = d / name
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-c:a", "aac", "-b:a", "160k", str(dst)],
                           check=True)
            return dst.name
        labels = list(it["takes"])
        rng.shuffle(labels)
        cells = [f'<td>original<br><audio controls preload="none" src="{enc(it["orig"], f"{n:03d}-orig.m4a")}"></audio></td>']
        for i, lab in enumerate(labels):
            letter = chr(65 + i)
            key[f"{n:03d}-{letter}"] = lab
            cells.append(f'<td>{letter}<br><audio controls preload="none" src="{enc(it["takes"][lab], f"{n:03d}-{letter}.m4a")}"></audio></td>')
        rows.append(f"<tr><td>{it['clip']} #{it['line']}<br><small>{it['text']}</small></td>{''.join(cells)}</tr>")
    (d / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Listening test: " + stage + "</title>"
        "<style>body{font:14px system-ui;margin:16px}td{padding:6px;border-bottom:1px solid #ccc;vertical-align:top}"
        "audio{height:30px;width:190px}</style><h1>Listening test: " + stage + "</h1>"
        "<p>For each line, which version sounds best (natural, right voice, right emotion)? Letters are shuffled "
        "per line; the answers are in key.json.</p><table>" + "".join(rows) + "</table>")
    (d / "key.json").write_text(json.dumps(key, indent=1))


def cmd_report() -> None:
    parts = ["<!doctype html><meta charset=utf-8><title>Phase 0 results</title><style>body{font:14px system-ui;"
             "margin:16px;max-width:1100px}table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;"
             "padding:4px 8px;text-align:left}</style><h1>Ai Dub: Phase 0 results</h1>"]
    for stage in ("separation", "recognition", "diarization", "translation", "voice", "bandwidth"):
        f = RESULTS / f"{stage}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        parts.append(f"<h2>{stage.title()}</h2>")
        for clip, v in data.items():
            rows = v.get("results", v) if isinstance(v, dict) else v
            rows = {k: x for k, x in rows.items() if isinstance(x, dict)}
            if not rows:
                continue
            cols = sorted({c for x in rows.values() for c in x if c not in ("sample",)})
            parts.append(f"<h3>{clip}</h3><table><tr><th></th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
            for name, x in rows.items():
                parts.append(f"<tr><td>{name}</td>" + "".join(f"<td>{x.get(c, '')}</td>" for c in cols) + "</tr>")
            parts.append("</table>")
        if stage == "translation":
            for clip, v in data.items():
                ex = v.get("examples", [])[:15]
                if ex:
                    cols = list(ex[0])
                    parts.append(f"<h3>{clip}: examples</h3><table><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
                    for e in ex:
                        parts.append("<tr>" + "".join(f"<td>{e.get(c, '')}</td>" for c in cols) + "</tr>")
                    parts.append("</table>")
    for stage in ("voice", "bandwidth"):
        if (RESULTS / "listen" / stage / "index.html").exists():
            parts.append(f'<p><a href="listen/{stage}/index.html">Listening test: {stage}</a></p>')
    if (RESULTS / "listen" / "separation").exists():
        parts.append('<p>Separation excerpts: listen/separation/&lt;clip&gt;/ (dialogue and music &amp; effects stems)</p>')
    (RESULTS / "report.html").write_text("".join(parts))
    log("report written", RESULTS / "report.html")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prepare": lambda: cmd_prepare(sys.argv[2]), "separation": cmd_separation, "recognition": cmd_recognition,
     "translation": cmd_translation, "voice": cmd_voice, "bandwidth": cmd_bandwidth,
     "diarization": cmd_diarization, "report": cmd_report}[cmd]()
