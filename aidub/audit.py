"""Listening to dubs already delivered: `aidub audit`.

Reads the English track of a finished file back against its English subtitles, with the
same recogniser the pipeline uses, whatever version of the app made it: is the track
there all the way through, and how much of what the subtitles say can be heard, in
order? Reads files only; changes nothing.
"""

from __future__ import annotations

import difflib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import media
from .deliver import AI_TITLES, DECODE_MEMORY_KB
from .readback import clock
from .subtitles import clean
from .subtitles.model import DIALOGUE

CHUNK_S = 30.0     # the track is heard in pieces this long, overlapping a little
OVERLAP_S = 1.0
MISSING_RUN = 6    # this many subtitle words in a row never heard is worth listing
QUIET_DB = -70.0


def dub_stream(pr: media.Probe):
    return next((s for s in pr.audio if s.is_aidub or any(t in s.title for t in AI_TITLES)), None)


def extract(path: Path, stream: int, out: Path) -> Path:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", f"0:{stream}", "-vn", "-sn", "-dn",
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(out)]
    subprocess.run(["sh", "-c", f'ulimit -v {DECODE_MEMORY_KB}; exec "$@"', "sh", *cmd], check=True, capture_output=True)
    return out


def english_cues(path: Path, pr: media.Probe, work: Path) -> list:
    """Dialogue subtitles of a file: a sidecar .en file, else an embedded English text track."""
    from .subtitles.sources import sidecars
    for p, _fmt, lang, forced in sidecars(path):
        if lang == "en" and not forced and p.suffix.lower() in (".srt", ".ass", ".ssa", ".vtt"):
            return [c for c in clean.load(p) if c.kind == DIALOGUE and c.text]
    for s in pr.subtitles:
        if s.language == "en" and s.is_text_sub and not s.forced and not s.is_aidub:
            out = work / "en.srt"
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-map", f"0:{s.index}", str(out)]
            subprocess.run(["sh", "-c", f'ulimit -v {DECODE_MEMORY_KB}; exec "$@"', "sh", *cmd], check=True, capture_output=True)
            return [c for c in clean.load(out) if c.kind == DIALOGUE and c.text]
    return []


def _words(text: str) -> list[str]:
    from .speak import _numbers, _words as said
    return _numbers(said(text), False)


def missing_stretches(cues: list, heard: list[str]) -> tuple[float, list[dict]]:
    """How much of what the subtitles say was heard, in order, wherever it is in the track (a
    sidecar's timing can be off by seconds), and the stretches of it never heard."""
    words, owner = [], []
    for c in sorted(cues, key=lambda c: c.start):
        for w in _words(c.text):
            words.append(w)
            owner.append(c)
    if not words:
        return 0.0, []
    sm = difflib.SequenceMatcher(None, words, heard, autojunk=False)
    blocks = sm.get_matching_blocks()
    hit = sum(b.size for b in blocks)
    out, pos = [], 0
    for b in blocks:
        if b.a - pos >= MISSING_RUN:
            out.append({"t": clock(owner[pos].start), "said": " ".join(words[pos: b.a])[:120], "words": b.a - pos})
        pos = b.a + b.size
    return hit / len(words), out


def silent_stretches(data: np.ndarray, sr: int, least: float = 20.0) -> list[list[int]]:
    hop = sr
    db = [10 * np.log10(float(np.mean(data[i: i + hop] ** 2)) + 1e-12) for i in range(0, len(data), hop)]
    out, start = [], None
    for i, d in enumerate(db + [0.0]):
        if d < QUIET_DB and start is None:
            start = i
        elif d >= QUIET_DB and start is not None:
            if i - start >= least:
                out.append([start, i])
            start = None
    return out


def read_back(cfg, path: Path, w) -> dict:
    pr = media.probe(path)
    st = dub_stream(pr)
    rep: dict = {"file": path.name, "duration": round(pr.duration, 1), "dub": bool(st)}
    if not st:
        rep["problem"] = "no English dub track"
        return rep
    rep["channels"] = st.channels
    with tempfile.TemporaryDirectory(dir=cfg.work) as td:
        td = Path(td)
        data, sr = media.read_wav(extract(path, st.index, td / "dub.wav"), mono=True)
        data = data[:, 0]
        rep["track_seconds"] = round(len(data) / sr, 1)
        rep["silent"] = [[clock(a), clock(b)] for a, b in silent_stretches(data, sr)]
        cues = english_cues(path, pr, td)
        rep["subtitle_cues"] = len(cues)
        step = int((CHUNK_S - OVERLAP_S) * sr)
        items = []
        for k, i in enumerate(range(0, len(data), step)):
            items.append(str(media.write_wav(td / f"c{k:05d}.wav", data[i: i + int(CHUNK_S * sr)], sr)))
        heard, per_chunk = [], []
        for i in range(0, len(items), 8):
            for r in w.call("transcribe", {"items": [{"path": p} for p in items[i:i + 8]], "language": "en",
                                           "timestamps": False})["results"]:
                ws = _words(r.get("text", ""))
                heard += ws
                per_chunk.append(len(ws))
    rep["heard_words"] = len(heard)
    # Long stretches with no English heard, whether silent or not.
    quiet, start = [], None
    for k, n in enumerate(per_chunk + [99]):
        if n == 0 and start is None:
            start = k
        elif n and start is not None:
            if k - start >= 4:
                quiet.append([clock(start * (CHUNK_S - OVERLAP_S)), clock(k * (CHUNK_S - OVERLAP_S))])
            start = None
    rep["no_english_heard"] = quiet
    if cues:
        share, missing = missing_stretches(cues, heard)
        rep["heard_share"] = round(share, 3)
        rep["missing"] = missing
    return rep


def run(cfg, paths: list[Path], out: Path | None = None) -> list[dict]:
    from .pronounce import load as load_lexicon
    from .workers import Worker
    load_lexicon(cfg.models / "lexicon" / "cmudict.dict")
    results = []
    stage = cfg.get("stages.recognize")
    with Worker(cfg, stage.get("worker")) as w:
        w.call("load", {"model": stage.get("model")})
        for p in paths:
            try:
                rep = read_back(cfg, p, w)
            except Exception as exc:
                rep = {"file": p.name, "problem": str(exc)[:300]}
            results.append(rep)
            if out:
                out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
            print(f"{rep.get('file')}: silent {rep.get('silent')}, no English heard {rep.get('no_english_heard')}, "
                  f"heard {rep.get('heard_share')} of {rep.get('subtitle_cues')} subtitles' words, "
                  f"{len(rep.get('missing', []))} stretches missing {rep.get('problem', '')}", flush=True)
    return results
