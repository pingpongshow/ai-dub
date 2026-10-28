"""Step 4: build the per-line table.

Lines come from the winning source-language subtitle, or from recognition
(split into sentences) with the English subtitle matched onto them, or from
recognition alone. Each line gets exact times, one speaker, a kind
(speech / english / wordless / song), its scene, emotion, loudness and pan.
"""

from __future__ import annotations

import re

import numpy as np

from . import lines as L
from . import media, scenes
from .job import Job
from .subtitles.model import DIALOGUE, SONG
from .subtitles.score import cer
from .subtitles.step import load_chosen

STEP = "align"
CJK = {"ja", "zh", "yue", "ko"}
INTERJECTION_CHARS = set("あいうえおぁぃぅぇぉっーんアイウエオァィゥェォッーンはひふへほハヒフヘホやゃゎわワ"
                         "きゃキャくっちっ・…!?！？、。~〜 ")
INTERJECTION_WORDS = {
    "ja": {"あ", "あっ", "ああ", "え", "えっ", "えー", "う", "うう", "うん", "ふん", "はぁ", "はあ", "ん", "んっ",
           "お", "おお", "ひっ", "きゃー", "わぁ", "わあ", "ほう", "へえ", "へー", "ちっ", "くっ", "はっ", "ふう", "ふぅ"},
    "ko": {"아", "어", "음", "응", "흠", "헉", "하", "흐", "으", "엇", "앗"},
    "zh": {"啊", "哦", "嗯", "呃", "哈", "唉", "哎", "嘿", "呀", "哼", "噢"},
    "en": {"ah", "oh", "uh", "um", "hmm", "huh", "ha", "eh", "mm", "ugh", "whoa", "wow", "hey"},
}
WORDLESS_EVENTS = {"Laughter", "Cry", "Cough", "Sneeze", "Breath", "laughter", "cry", "cough", "sneeze", "breath"}
MIN_LINE_S = 0.5
SUBTITLE_DISAGREES = 0.6   # character error rate between a source subtitle line and what was heard
SENTENCE_END = re.compile(r"[。！？!?…]$|[.!?]$")


def is_interjection(text: str, lang: str) -> bool:
    t = re.sub(r"[\s、。,.!?！？…~〜ー-]", "", text.lower())
    if not t:
        return True
    words = INTERJECTION_WORDS.get("zh" if lang == "yue" else lang, set())
    if t in {re.sub(r"[ー-]", "", w) for w in words}:
        return True
    if lang == "ja" and len(t) <= 3 and all(ch in INTERJECTION_CHARS for ch in t):
        return True
    if lang in {"zh", "yue", "ko"} and len(t) <= 2 and all(ch in "".join(words) for ch in t):
        return True
    return False


def join_words(words: list[dict], lang: str) -> str:
    sep = "" if lang in CJK and lang != "ko" else " "
    return sep.join(w["text"] for w in words).strip()


def asr_words(asr: dict) -> list[dict]:
    return sorted((w for s in asr["segments"] for w in s.get("words", [])), key=lambda w: w["start"])


def sentences(words: list[dict], turns: list[dict], max_len: float = 8.0, gap: float = 0.45) -> list[list[dict]]:
    """Split recognised words into sentence-sized lines at pauses, punctuation and speaker changes."""
    def speaker_at(t: float) -> str:
        best, ov = "", 0.0
        for s in turns:
            if s["start"] <= t <= s["end"]:
                o = min(s["end"], t + 0.01) - max(s["start"], t)
                if o >= ov:
                    best, ov = s["speaker"], o
        return best

    out: list[list[dict]] = []
    cur: list[dict] = []
    cur_spk = None
    for w in words:
        spk = speaker_at((w["start"] + w["end"]) / 2)
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_len
                    or (spk and cur_spk and spk != cur_spk) or SENTENCE_END.search(cur[-1]["text"])):
            out.append(cur)
            cur = []
        if not cur:
            cur_spk = spk
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def words_in(words: list[dict], a: float, b: float) -> list[dict]:
    return [w for w in words if w["end"] > a and w["start"] < b
            and (min(w["end"], b) - max(w["start"], a)) >= 0.5 * max(0.01, w["end"] - w["start"])]


def main_speaker(turns: list[dict], a: float, b: float) -> tuple[str, bool, list[tuple[str, float]]]:
    share: dict[str, float] = {}
    overlap = False
    for s in turns:
        o = min(s["end"], b) - max(s["start"], a)
        if o > 0:
            share[s["speaker"]] = share.get(s["speaker"], 0.0) + o
            overlap = overlap or s.get("overlap", False)
    ranked = sorted(share.items(), key=lambda kv: -kv[1])
    return (ranked[0][0] if ranked else ""), overlap, ranked


def split_by_speaker(line: L.Line, turns: list[dict], lang: str) -> list[L.Line]:
    """A line spanning a speaker change is split at the word nearest the change."""
    _, _, ranked = main_speaker(turns, line.start, line.end)
    if len(ranked) < 2 or len(line.words) < 2:
        return [line]
    total = sum(v for _, v in ranked)
    if ranked[1][1] < 0.4 or ranked[1][1] / total < 0.25:
        return [line]
    second = ranked[1][0]
    change = None
    for s in turns:
        if s["speaker"] == second and line.start < s["start"] < line.end:
            change = s["start"]
            break
        if s["speaker"] == second and line.start < s["end"] < line.end:
            change = s["end"]
            break
    if change is None:
        return [line]
    k = min(range(1, len(line.words)), key=lambda i: abs(line.words[i].start - change))
    a, b = line.words[:k], line.words[k:]
    out = []
    for part in (a, b):
        ws = [{"text": w.text, "start": w.start, "end": w.end} for w in part]
        out.append(L.Line(id=0, start=part[0].start, end=part[-1].end, source_text=join_words(ws, lang),
                          source_lang=lang, text_origin=line.text_origin, words=list(part)))
    # A subtitle can't be cut between speakers reliably: both halves keep it only as context.
    for o in out:
        o.english_context = line.english_ref or line.english_context
    return out


def _attach(lines: list[L.Line], w: dict, lang: str, reach: float = 0.5) -> bool:
    """Add a recognised word to the line it sits just before or after."""
    best, gap = None, reach
    for ln in lines:
        g = max(ln.start - w["end"], w["start"] - ln.end, 0.0)
        if g < gap:
            best, gap = ln, g
    if best is None:
        return False
    best.words = sorted(best.words + [L.Word(**w)], key=lambda x: x.start)
    best.start, best.end = min(best.start, w["start"]), max(best.end, w["end"])
    if best.text_origin == "asr":
        best.source_text = join_words([{"text": x.text} for x in best.words], lang)
    return True


def _in_ranges(a: float, b: float, ranges: list[list[float]]) -> bool:
    return any(min(b, r[1]) - max(a, r[0]) > 0.5 * (b - a) for r in ranges)


def build(job: Job) -> list[L.Line]:
    asr = job["asr"]
    lang = job["subtitles"].get("source_language") or asr.get("language", "")
    turns = job["diarization"]["segments"]
    words = asr_words(asr)
    src_cues = [q for q in load_chosen(job, "source") if q.kind in (DIALOGUE, SONG)]
    en_cues = [q for q in load_chosen(job, "english") if q.kind in (DIALOGUE, SONG)]
    src_cand = job["subtitles"].get("source")
    en_cand = job["subtitles"].get("english")
    sub_words = job.get("subtitle_words", {})

    raw: list[L.Line] = []
    covered: list[tuple[float, float]] = []
    if src_cues:
        all_src = load_chosen(job, "source")
        for q in src_cues:
            i = all_src.index(q) if q in all_src else -1
            heard = words_in(words, q.start, q.end)
            ws = sub_words.get(str(i)) or heard
            text, origin = q.text, f"subtitle:{src_cand}"
            spoken = join_words(heard, lang)
            # What was actually said wins: a subtitle line that disagrees with the speech
            # (wrong line, different cut, mistranslation in a fan sub) gives way to recognition.
            if len(spoken) >= 3 and cer(q.text, spoken) > SUBTITLE_DISAGREES:
                text, origin, ws = spoken, "asr", heard
            ln = L.Line(id=0, start=q.start, end=q.end, source_text=text, source_lang=lang,
                        text_origin=origin, kind=L.SONG if q.kind == SONG else L.SPEECH,
                        words=[L.Word(**w) for w in ws], name_hint=q.name)
            raw.append(ln)
            covered.append((q.start, q.end))
    elif en_cues:
        for q in en_cues:
            ws = words_in(words, q.start - 0.25, q.end + 0.25)
            a, b = (ws[0]["start"], ws[-1]["end"]) if ws else (q.start, q.end)
            raw.append(L.Line(id=0, start=a, end=b, source_text=join_words(ws, lang), source_lang=lang,
                              text_origin="asr", english_ref=q.text, kind=L.SONG if q.kind == SONG else L.SPEECH,
                              words=[L.Word(**w) for w in ws], name_hint=q.name))
            covered.append((a, b))

    # Recognised speech no subtitle covers (or everything, without subtitles).
    rest = [w for w in words if not any(a - 0.1 <= (w["start"] + w["end"]) / 2 <= b + 0.1 for a, b in covered)]
    if raw:
        # Speech just outside a subtitle line's timing belongs to that line.
        rest = [w for w in rest if not _attach(raw, w, lang)]
    for sent in sentences(rest, turns):
        text = join_words(sent, lang)
        dur = sent[-1]["end"] - sent[0]["start"]
        kind = L.SPEECH if (dur >= MIN_LINE_S and len(text) > 1) else L.WORDLESS  # fragments keep the original
        raw.append(L.Line(id=0, start=sent[0]["start"], end=sent[-1]["end"], source_text=text, kind=kind,
                          source_lang=lang, text_origin="asr", words=[L.Word(**w) for w in sent]))

    # English subtitle as the meaning reference for source-subtitle lines.
    if src_cues and en_cues:
        for ln in raw:
            refs = [q.text for q in en_cues if min(q.end, ln.end) - max(q.start, ln.start) > 0.3 * ln.duration]
            ln.english_ref = " ".join(refs)
        job.qc["english_reference"] = en_cand

    # Speaker splits, then speaker/overlap.
    split: list[L.Line] = []
    for ln in sorted(raw, key=lambda l: l.start):
        split.extend(split_by_speaker(ln, turns, lang) if ln.words else [ln])
    for ln in split:
        ln.speaker, ln.overlap, _ = main_speaker(turns, ln.start, ln.end)
    return split


def classify(job: Job, lines: list[L.Line]) -> None:
    turns = job["diarization"]["segments"]
    ev = job.get("events") or []
    songs = job["ingest"].get("song_ranges", [])
    lang = job["subtitles"].get("source_language", "")
    for ln in lines:
        tags: list[str] = []
        langs: dict[str, float] = {}
        emotion: dict[str, float] = {}
        for s, e in zip(turns, ev):
            o = min(s["end"], ln.end) - max(s["start"], ln.start)
            if o <= 0:
                continue
            tags.extend(t for t in e.get("events", []) if t not in tags)
            if e.get("language"):
                langs[e["language"]] = langs.get(e["language"], 0) + o
            if e.get("emotion"):
                emotion[e["emotion"]] = emotion.get(e["emotion"], 0) + o
        ln.events = tags
        ln.emotion = max(emotion, key=emotion.get) if emotion else ""
        top_lang = max(langs, key=langs.get) if langs else ""
        if ln.kind == L.SONG or _in_ranges(ln.start, ln.end, songs):
            ln.kind = L.SONG
        elif top_lang == "en" and lang != "en" and ln.duration >= 0.8:
            ln.kind = L.ENGLISH
        elif is_interjection(ln.source_text, lang) or (set(tags) & WORDLESS_EVENTS and len(ln.source_text) <= 3):
            ln.kind = L.WORDLESS


def measure(job: Job, lines: list[L.Line]) -> None:
    """Loudness and stereo position of each line in the dialogue stem."""
    data, sr = media.read_wav(job.path("audio", "dialogue.wav"))
    for ln in lines:
        seg = data[int(ln.start * sr): int(ln.end * sr)]
        if seg.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        ln.loudness_db = round(20 * np.log10(rms), 2)
        if seg.shape[1] >= 2:
            l, r = float(np.sqrt(np.mean(seg[:, 0] ** 2))), float(np.sqrt(np.mean(seg[:, 1] ** 2)))
            ln.pan = round((r - l) / max(1e-9, r + l), 3)


def check_english_refs(job: Job, lines: list[L.Line]) -> None:
    """The spoken audio is authoritative. An English subtitle line is kept as a
    line's meaning only if it matches what was said (cross-lingual similarity
    clearly above that of unrelated pairs); otherwise it becomes a hint."""
    cands = [ln for ln in lines if ln.english_ref and len(ln.source_text) >= 4 and ln.kind == L.SPEECH]
    if len(cands) < 4 or not job.cfg.get("stages.speaker.worker"):
        return
    k = max(1, len(cands) // 2)
    pairs = [[ln.english_ref, ln.source_text] for ln in cands]
    shifted = [[ln.english_ref, cands[(i + k) % len(cands)].source_text] for i, ln in enumerate(cands)]
    with job.worker("speaker", gpu_stage="speaker") as w:
        w.call("load", {})
        sims = w.call("text_similarity", {"pairs": pairs + shifted})["similarity"]
    true, unrelated = sims[: len(cands)], sims[len(cands):]
    floor = float(np.percentile(unrelated, 75)) if unrelated else 0.0
    moved = 0
    for ln, s in zip(cands, true):
        if s <= floor:
            ln.english_context, ln.english_ref = ln.english_ref, ""
            moved += 1
    job.qc["subtitle_lines_disagreeing_with_speech"] = moved
    if moved:
        job.log(f"{moved} English subtitle lines didn't match the speech; used only as hints")


def run(job: Job) -> None:
    lines = build(job)
    check_english_refs(job, lines)
    cuts = job["scenes"]["cuts"]
    windows = job["scenes"]["windows"]
    for i, ln in enumerate(lines):
        ln.id = i + 1
        ln.scene = scenes.scene_of(ln.start, cuts)
        ln.start, ln.end = round(ln.start, 3), round(max(ln.end, ln.start + 0.2), 3)
    classify(job, lines)
    measure(job, lines)
    # Translation context: which window each line belongs to.
    job.set("line_windows", [next((wi for wi, (a, b) in enumerate(windows) if a <= ln.start < b), len(windows) - 1)
                             for ln in lines])
    L.save(lines, job.path("lines.json"))
    kinds: dict[str, int] = {}
    for ln in lines:
        kinds[ln.kind] = kinds.get(ln.kind, 0) + 1
    job.qc["lines"] = kinds
    job.status(", ".join(f"{v} {k}" for k, v in kinds.items()) or "no lines")
