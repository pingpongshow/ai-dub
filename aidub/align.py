"""Step 4: build the per-line table.

Lines come from the winning source-language subtitle, or from recognition
(split into sentences) with the English subtitle matched onto them, or from
recognition alone. Each line gets exact times, one speaker, a kind
(speech / english / wordless / song), its scene, emotion, loudness and pan.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import numpy as np

from . import lines as L
from . import media, scenes
from .job import Job
from .subtitles.model import DIALOGUE, SONG
from .subtitles.clean import norm_chars as clean_norm
from .subtitles.score import cer
from .subtitles.step import load_chosen

STEP = "align"
CJK = {"ja", "zh", "yue", "ko"}
INTERJECTION_CHARS = set("あいうえおぁぃぅぇぉっーんアイウエオァィゥェォッーンはひふへほハヒフヘホやゃゎわワ"
                         "きゃキャくっちっ・…!?！？、。~〜 ")
INTERJECTION_WORDS = {
    "ja": {"あ", "あっ", "ああ", "え", "えっ", "えー", "う", "うう", "うん", "ふん", "はぁ", "はあ", "ん", "んっ",
           "お", "おお", "ひっ", "きゃー", "わぁ", "わあ", "ほう", "へえ", "へー", "ちっ", "くっ", "はっ", "ふう", "ふぅ",
           "うわ", "うわぁ", "おっ", "おや", "ふふ", "へへ", "あはは", "うふふ", "ひぃ", "ぎゃ", "うぅ", "はは"},
    "ko": {"아", "어", "음", "응", "흠", "헉", "하", "흐", "으", "엇", "앗", "아이", "아이고", "아유", "어머", "에이",
           "에휴", "아휴", "어휴", "휴", "흥", "헐", "와", "오", "우와", "쯧", "흑", "허", "후", "하하", "히히", "흐흐"},
    "zh": {"啊", "哦", "嗯", "呃", "哈", "唉", "哎", "嘿", "呀", "哼", "噢"},
    "en": {"ah", "oh", "uh", "um", "hmm", "huh", "ha", "eh", "mm", "ugh", "whoa", "wow", "hey"},
}
SINGING_EVENTS = {"Sing", "sing", "Singing", "singing"}  # songs keep the original audio, untranslated
WORDLESS_EVENTS = {"Laughter", "Cry", "Cough", "Sneeze", "Breath", "laughter", "cry", "cough", "sneeze", "breath"}
MIN_LINE_S = 0.5
MIN_SPLIT_S = 0.6
PHRASE_GAP_S = 0.25   # words closer than this are one phrase when assigning them to subtitle cues
SONG_MUSIC_DB = -6.0  # music this loud under a voice (relative to it), next to a song, is part of the song
SONG_REACH_S = 3.0
BACKGROUND_DB = 6.0  # music & effects this far above the voice: the speech is in the background track
CROWD_MAX_S = 2.0     # longer lines over other voices (a TV, a crowd) are still one person's line
CROWD_SHARE = 0.6     # share of a line with overlapping speakers that keeps the original
SONG_STRETCH_S = 15.0  # song lines this close together: what lies between is the same song
COVERED_SHARE = 0.5   # a wordless line this much inside others' speech is a piece of their audio
FRAGMENT_S = 1.2      # a piece of a sentence this short ...
FRAGMENT_GAP_S = 0.35  # ... this close after the same speaker's unfinished line is part of it
ABSORB_GAP_S = 0.4
LATIN = re.compile(r"[A-Za-z]")
NON_LATIN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # kana, CJK, Hangul: not English
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


def assign_words(words: list[dict], cues: list, slack: float = 0.25,
                 turns: list[dict] | None = None) -> list[list[dict]]:
    """Give each recognised word to exactly one subtitle cue: the one it overlaps
    most (cue times widened by `slack`), ties to the nearer cue centre. Overlapping
    cues (a TV playing under the dialogue, two speakers at once) otherwise claim
    the same words and the same speech is voiced twice.

    With diarized `turns` and named cues, a second pass learns which name each
    voice carries and gives a word to a cue of its own speaker's name when there
    is one, so a character talking over a TV keeps his sentence whole."""
    out = _assign(words, cues, slack, {})
    if not turns or not any(getattr(q, "name", "") for q in cues):
        return out
    votes: dict[str, dict[str, int]] = {}
    for k, ws in enumerate(out):
        name = getattr(cues[k], "name", "")
        if not name:
            continue
        for w in ws:
            spk = _speaker_at(turns, (w["start"] + w["end"]) / 2)
            if spk:
                votes.setdefault(spk, {}).setdefault(name, 0)
                votes[spk][name] += 1
    name_of = {spk: max(v, key=v.get) for spk, v in votes.items() if sum(v.values()) >= 3}
    prefer = {}
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        prefer[id(w)] = {name_of[t["speaker"]] for t in turns if t["start"] <= mid <= t["end"] and t["speaker"] in name_of}
    return _assign(words, cues, slack, prefer)


def _speaker_at(turns: list[dict], t: float) -> str:
    best, ov = "", 0.0
    for s in turns:
        if s["start"] <= t <= s["end"]:
            o = s["end"] - s["start"]
            if not best or o < ov:  # the shorter turn is the more specific one
                best, ov = s["speaker"], o
    return best


def _assign(words: list[dict], cues: list, slack: float, prefer: dict[int, set]) -> list[list[dict]]:
    """Words go to cues in phrases (no pause inside, and every word of the phrase could
    belong to the same cue), so a phrase isn't scattered across overlapping cues."""
    def cands(w) -> dict[int, float]:
        out = {}
        for k, q in enumerate(cues):
            o = min(w["end"], q.end + slack) - max(w["start"], q.start - slack)
            if o > 0 and o >= 0.5 * max(0.01, w["end"] - w["start"]):
                out[k] = o
        names = prefer.get(id(w))
        if names:  # not a cue of a character who isn't talking here, while a fitting one is there
            mine = {k: o for k, o in out.items() if getattr(cues[k], "name", "") in names | {""}}
            out = mine or out
        return out

    phrases: list[tuple[list[dict], set[int]]] = []
    for w in sorted(words, key=lambda w: w["start"]):
        c = set(cands(w))
        if phrases:
            ws, common = phrases[-1]
            if w["start"] - ws[-1]["end"] <= PHRASE_GAP_S and (not c or not common or c & common):
                ws.append(w)
                phrases[-1] = (ws, (common & c) if (common and c) else (common or c))
                continue
        phrases.append(([w], c))
    out: list[list[dict]] = [[] for _ in cues]
    for ws, common in phrases:
        if not common:
            continue
        mid = (ws[0]["start"] + ws[-1]["end"]) / 2

        def score(k):
            q = cues[k]
            o = sum(max(0.0, min(w["end"], q.end + slack) - max(w["start"], q.start - slack)) for w in ws)
            return (round(o, 3), -abs(mid - (q.start + q.end) / 2))
        out[max(common, key=score)].extend(ws)
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


def overlap_share(turns: list[dict], a: float, b: float) -> float:
    """Share of [a, b] where two or more diarized speakers talk at once."""
    if b <= a:
        return 0.0
    pts = sorted({a, b, *(t["start"] for t in turns if a < t["start"] < b), *(t["end"] for t in turns if a < t["end"] < b)})
    both = sum(y - x for x, y in zip(pts, pts[1:])
               if sum(1 for t in turns if t["start"] <= (x + y) / 2 <= t["end"]) >= 2)
    return both / (b - a)


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
    if min(a[-1].end - a[0].start, b[-1].end - b[0].start) < MIN_SPLIT_S:
        return [line]  # a sliver of another voice (a TV, a reply) doesn't cut a sentence apart
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
    """Add a recognised word to the line it sits just before or after. Not to a
    subtitle line nothing was heard under: that is a voice recognition missed (a
    child's quiet "六つ"), and the word next to it is someone else's ("うん?")."""
    best, gap = None, reach
    for ln in lines:
        if not ln.words:
            continue
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


def _same_text(a: str, b: str) -> bool:
    x, y = clean_norm(a), clean_norm(b)
    if not x or not y:
        return False
    return x in y or y in x or cer(x, y) <= 0.35


def merge_duplicates(lines: list[L.Line], lang: str) -> list[L.Line]:
    """One stretch of speech must be one line. Subtitles (re-displayed image
    subtitles above all) can repeat a cue, or show a fragment of the next cue at
    the same moment ("サイレン" then "サイレンサー付きの拳銃だと"); those become one
    line with the fuller text, instead of the same audio voiced twice."""
    out: list[L.Line] = []
    for ln in sorted(lines, key=lambda l: (l.start, -l.duration)):
        prev = next((o for o in reversed(out[-3:]) if
                     min(o.end, ln.end) - max(o.start, ln.start) >= 0.5 * max(0.05, min(o.duration, ln.duration))
                     and _same_text(o.source_text, ln.source_text)), None)
        if prev is None:
            out.append(ln)
            continue
        keep, other = (prev, ln) if len(clean_norm(prev.source_text)) >= len(clean_norm(ln.source_text)) else (ln, prev)
        merged = keep
        merged.start, merged.end = min(prev.start, ln.start), max(prev.end, ln.end)
        if not merged.words:
            merged.words = other.words
        merged.name_hint = merged.name_hint or other.name_hint
        if merged.kind != L.SONG and other.kind == L.SONG:
            merged.kind = L.SONG
        refs = [r for r in (keep.english_ref, other.english_ref) if r]
        merged.english_ref = refs[0] if refs else ""
        out[out.index(prev)] = merged
    return out


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
        owned = assign_words(words, src_cues, slack=0.1, turns=turns)
        for k, q in enumerate(src_cues):
            i = all_src.index(q) if q in all_src else -1
            heard = owned[k]
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
        owned = assign_words(words, en_cues, turns=turns)
        for k, q in enumerate(en_cues):
            ws = owned[k]
            a, b = (ws[0]["start"], ws[-1]["end"]) if ws else (q.start, q.end)
            raw.append(L.Line(id=0, start=a, end=b, source_text=join_words(ws, lang), source_lang=lang,
                              text_origin="asr", english_ref=q.text, kind=L.SONG if q.kind == SONG else L.SPEECH,
                              words=[L.Word(**w) for w in ws], name_hint=q.name))
            covered.append((a, b))

    raw = merge_duplicates(raw, lang)
    covered = [(ln.start, ln.end) for ln in raw]

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
        # A sliver of speech with no subtitle of its own (an echo, half a word) is too
        # short to dub well and to tell whose voice it is: it keeps the original.
        if (ln.kind == L.SPEECH and ln.text_origin == "asr" and not ln.english_ref
                and ln.end - ln.start < MIN_LINE_S):
            ln.kind = L.WORDLESS
        ln.speaker, ln.overlap, _ = main_speaker(turns, ln.start, ln.end)
        # How much speech recognition heard here (a subtitle can sit over a sigh). Only
        # measured where recognition ran; elsewhere it's unknown, not silence.
        listened = any(min(c["end"], ln.end) - max(c["start"], ln.start) > 0.5 * ln.duration for c in asr["segments"])
        ln.heard_chars = len(clean_norm(join_words(words_in(words, ln.start, ln.end), lang))) if listened else -1
    return split


def relisten(job: Job, lines: list[L.Line], lang: str, only: list[L.Line] | None = None) -> dict[int, L.Line]:
    """Listen again, one line at a time, wherever a line looks wordless: no text, or
    nothing heard under its subtitle. Chunked recognition can miss a quiet or
    child's voice under music, or a turn diarization didn't mark; a line counts as
    wordless only if this focused pass hears nothing (or only an interjection).
    Returns, for lines whose re-heard words were a neighbour's, that neighbour.
    With `only`, just those of the lines are re-heard."""
    todo = [ln for ln in (lines if only is None else only) if ln.kind in (L.SPEECH, L.WORDLESS) and ln.duration >= 0.3
            and (len(clean_norm(ln.source_text)) <= 1 or ln.heard_chars <= 0)]
    if not todo or not job.cfg.get("stages.recognize.worker"):
        return {}
    data, sr = media.read_wav(job.path("audio", "dialogue16k.wav"), mono=True)
    pad = 0.15
    items = []
    for i, ln in enumerate(todo):
        a, b = max(0.0, ln.start - pad), ln.end + pad
        items.append({"path": str(media.write_wav(job.path("asr", f"relisten{i:04d}.wav"),
                                                   data[int(a * sr): int(b * sr)], sr))})
    cfg = job.cfg.get("stages.recognize")
    with job.worker("recognize") as w:
        w.call("load", {"model": cfg.get("model"), "aligner": cfg.get("aligner")})
        heard = []
        for i in range(0, len(items), 16):
            heard += w.call("transcribe", {"items": items[i:i + 16], "language": lang or None,
                                           "timestamps": False})["results"]
    for it in items:
        Path(it["path"]).unlink(missing_ok=True)
    # First every re-heard text, so a line can be compared with neighbours re-heard too.
    first = {ln.id: (ln.source_text, ln.text_origin) for ln in todo}
    texts = {}
    for ln, r in zip(todo, heard):
        texts[ln.id] = (r.get("text") or "").strip()
        if len(clean_norm(ln.source_text)) <= 1 and clean_norm(texts[ln.id]):
            ln.source_text, ln.text_origin = texts[ln.id], "asr"
    found, shared = 0, {}
    for ln in todo:
        text = texts[ln.id]
        t = clean_norm(text)
        # Speech that belongs to neighbouring lines isn't this line's (unless it is this line's text).
        own = clean_norm(first[ln.id][0])
        mine = bool(own) and (t in own or own in t or cer(own, t) <= 0.5)
        near = [o for o in lines if o is not ln and o.end > ln.start - 0.5 and o.start < ln.end + 0.5
                and not (o.id in first and len(clean_norm(first[o.id][0])) <= 1 and len(own) <= 1)]
        # Theirs where they speak over most of this line, or their own text was re-heard too;
        # otherwise only their words recognised inside this line's window count. A re-heard
        # line between two others may just be what recognition missed there.
        rest, owner = strip_neighbours(t, near)
        if owner is not None and owner.words and _covered(ln, [o for o in near if o.words]) < COVERED_SHARE:
            rest, owner = strip_neighbours(t, near, (ln.start - pad, ln.end + pad))
        if t and not mine and owner is not None and (len(rest) <= 1 or is_interjection(rest, lang)):
            if owner is not None:
                shared[ln.id] = owner
            t, text = "", ""
        ln.relisten = text
        if not t or is_interjection(text, lang):
            ln.source_text, ln.text_origin = first[ln.id]
            ln.heard_chars = 0 if ln.heard_chars <= 0 else ln.heard_chars
            continue
        ln.heard_chars = max(ln.heard_chars, len(t))
        if ln.kind == L.WORDLESS:
            ln.kind = L.SPEECH
        found += 1
    job.qc["relistened" if only is None else "relistened_subtitle_gaps"] = {"lines": len(todo), "speech_found": found}
    job.log(f"re-listened to {len(todo)} lines that looked wordless; speech in {found}")
    return shared


def strip_neighbours(t: str, near: list[L.Line], span: tuple[float, float] | None = None) -> tuple[str, L.Line | None]:
    """What is left of re-heard text `t` once the parts that are neighbouring lines'
    words are taken out, and the neighbour that gave the most of it. With `span`,
    only their words recognised within it (all of a neighbour's text if it has no
    word times)."""
    rest, owner, most = t, None, 0
    for o in near:
        other = clean_norm(o.source_text)
        if span and o.words:
            other = clean_norm("".join(w.text for w in o.words if span[0] <= w.start <= span[1]))
        if not other:
            continue
        taken = 0
        while True:
            m = difflib.SequenceMatcher(None, rest, other, autojunk=False).find_longest_match(0, len(rest), 0, len(other))
            if m.size < 2:
                break
            rest, taken = rest[:m.a] + rest[m.a + m.size:], taken + m.size
        if taken > most:
            owner, most = o, taken
    return rest, owner


def _covered(ln: L.Line, others: list[L.Line]) -> float:
    """Share of a line's time that other lines speak in."""
    pts = sorted({ln.start, ln.end, *(o.start for o in others if ln.start < o.start < ln.end),
                  *(o.end for o in others if ln.start < o.end < ln.end)})
    hit = sum(y - x for x, y in zip(pts, pts[1:]) if any(o.start <= (x + y) / 2 <= o.end for o in others))
    return hit / max(0.01, ln.duration)


def _no_words(ln: L.Line, lang: str) -> bool:
    return ln.heard_chars <= 0 or len(clean_norm(ln.source_text)) <= 1 or is_interjection(ln.source_text, lang)


def drop_pieces(lines: list[L.Line], shared: dict[int, L.Line], lang: str) -> list[L.Line]:
    """Lines that are only a piece of a neighbour's audio go: a wordless line mostly
    inside other people's speech (the tail of a sentence the diarizer gave another
    turn), a short line whose words all sit inside a longer line (the same audio
    cut twice), and a wordless line whose re-listen heard the next line's words
    becomes the start of that line."""
    talk = [o for o in lines if o.kind != L.SONG and not _no_words(o, lang)]
    drop = {id(ln) for ln in lines if ln.kind != L.SONG and _no_words(ln, lang)
            and _covered(ln, [o for o in talk if o is not ln]) >= COVERED_SHARE}
    keep = [ln for ln in lines if id(ln) not in drop]
    for ln in keep:
        if ln.kind == L.SONG or ln.duration > MIN_LINE_S * 2:
            continue
        # A short line inside a longer one: someone's words, or the same wordless audio twice.
        hosts = [o for o in keep if o is not ln and id(o) not in drop and o.kind != L.SONG and o.duration > ln.duration
                 and (_no_words(ln, lang) or not _no_words(o, lang))]
        if ln.words:
            inside = any(all(o.start <= w.start < o.end - 0.05 for w in ln.words) for o in hosts)
        else:
            inside = any(_covered(ln, [o]) >= 0.8 for o in hosts)
        if inside:
            drop.add(id(ln))
    for ln in keep:
        o = shared.get(ln.id)
        if id(ln) in drop or o is None or id(o) in drop or not _no_words(ln, lang):
            continue
        # Someone else's sound just before it (a "うん" before a child's answer) stays theirs.
        if ln.speaker and ln.speaker != o.speaker:
            continue
        if max(o.start - ln.end, ln.start - o.end) <= ABSORB_GAP_S:
            # From the first sound recognition placed in the wordless line.
            a = max(ln.start, ln.words[0].start - 0.1) if ln.words else ln.start
            o.start, o.end = round(min(o.start, a), 3), max(o.end, ln.end)
            drop.add(id(ln))
    return [ln for ln in lines if id(ln) not in drop]


def merge_fragments(lines: list[L.Line], lang: str) -> list[L.Line]:
    """A recognised sentence the speaker turns or English cues cut in two ("それとこいつ"
    | "の身分証の偽造") is one line: same speaker, the first part unfinished, the
    second starting right after, one of them short."""
    out: list[L.Line] = []
    sep = "" if lang in CJK and lang != "ko" else " "
    for ln in sorted(lines, key=lambda l: l.start):
        prev = out[-1] if out else None
        if (prev is not None and prev.speaker and prev.speaker == ln.speaker
                and prev.text_origin == ln.text_origin == "asr"  # a subtitle's own cues are its sentences
                and L.SONG not in (prev.kind, ln.kind)
                # From the last word to the next word (a subtitle can stretch a line's time).
                and -0.1 <= ((ln.words[0].start if ln.words else ln.start)
                             - (prev.words[-1].end if prev.words else prev.end)) <= FRAGMENT_GAP_S
                and not (is_interjection(prev.source_text, lang) or is_interjection(ln.source_text, lang))
                and min(prev.duration, ln.duration) <= FRAGMENT_S and ln.end - prev.start <= 8.0
                and clean_norm(prev.source_text) and clean_norm(ln.source_text)
                and not SENTENCE_END.search(prev.source_text.strip())):
            prev.end = max(prev.end, ln.end)
            prev.words = prev.words + ln.words
            prev.source_text = (prev.source_text.strip() + sep + ln.source_text.strip()).strip()
            prev.english_ref = " ".join(r for r in (prev.english_ref, ln.english_ref) if r)
            prev.heard_chars = max(prev.heard_chars, 0) + max(ln.heard_chars, 0)
            prev.name_hint = prev.name_hint or ln.name_hint
            prev.overlap = prev.overlap or ln.overlap
            if ln.kind == L.SPEECH:
                prev.kind = L.SPEECH
            continue
        out.append(ln)
    return out


def missed_cues(job: Job, lines: list[L.Line], lang: str) -> list[L.Line]:
    """An English subtitle line no line carries (recognition missed a voice talking
    over another, and its words went to the neighbour's line) gets a line of its own
    over the part of its time no other line has, trimmed to the voice heard there.
    Its words are found by re-listening; it goes again if nothing new is heard."""
    refs = " ".join(f"{ln.english_ref} {ln.english_context}" for ln in lines)
    turns = job["diarization"]["segments"]
    cuts = job["scenes"]["cuts"]
    new = []
    for q in load_chosen(job, "english"):
        if q.kind != DIALOGUE or not q.text.strip() or q.text.strip() in refs:
            continue
        if any(ln.kind == L.SONG and ln.end > q.start and ln.start < q.end for ln in lines):
            continue
        a, b = q.start, q.end
        for ln in sorted(lines, key=lambda l: l.start):
            if ln.start <= a < ln.end:
                a = ln.end
        b = min([b] + [ln.start for ln in lines if a < ln.start < b])
        spk, overlap, _ = main_speaker(turns, a, b)
        if not spk:
            continue
        mine = [t for t in turns if t["speaker"] == spk and t["end"] > a and t["start"] < b]
        a, b = max(a, min(t["start"] for t in mine)), min(b, max(t["end"] for t in mine) + 0.1)
        if b - a < MIN_LINE_S:
            continue
        new.append(L.Line(id=0, start=round(a, 3), end=round(b, 3), source_text="", source_lang=lang,
                          text_origin="asr", english_ref=q.text, kind=L.SPEECH, speaker=spk, overlap=overlap,
                          heard_chars=0, scene=scenes.scene_of(a, cuts)))
    return new


def measure_music(job: Job, lines: list[L.Line]) -> None:
    """Level of the music & effects under each line, relative to the voice."""
    data, sr = media.read_wav(job.path("audio", "dialogue16k.wav"), mono=True)
    stems = job.path("audio", "stems.json")
    if not stems.exists():
        return
    mne = None
    for st in json.loads(stems.read_text()).values():
        m, msr = media.read_wav(st["mne"], mono=True)
        m = media.resample(m, msr, sr)[:, 0]
        mne = m if mne is None else mne[: len(m)] + m[: len(mne)]
    for ln in lines:
        a, b = int(ln.start * sr), int(ln.end * sr)
        x = data[a:b, 0]
        if len(x):
            v = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
            m = float(np.sqrt(np.mean(mne[a:b] ** 2)) + 1e-9)
            ln.music_db = round(20 * np.log10(m / v), 1)


def extend_songs(lines: list[L.Line], lang: str) -> int:
    """Song lines the subtitles didn't mark: a voice over music as loud as itself, with
    music tagged, right next to a song line (the edges of an opening, a line of the
    chorus between marked ones), or English words in a non-English show over such
    music (lyrics). Within a song (song lines close on both sides), a line the
    diarizer heard no one talking in is sung too. Spreads until nothing changes."""
    total = 0
    # Voices over music first; the unvoiced lines after, so they don't carry the song on to speech.
    for unvoiced in (False, True):
        changed = True
        while changed:
            changed = False
            songs = [ln for ln in lines if ln.kind == L.SONG]
            for ln in lines:
                if ln.kind == L.SONG:
                    continue
                near = any(max(o.start - ln.end, ln.start - o.end) <= SONG_REACH_S for o in songs)
                if unvoiced:
                    before = [o.end for o in songs if o.end <= ln.start + 0.1]
                    after = [o.start for o in songs if o.start >= ln.end - 0.1]
                    inside = bool(before and after) and min(after) - max(before) <= SONG_STRETCH_S
                    hit = not ln.speaker and (inside or near)
                else:
                    lyrics = (lang != "en" and bool(LATIN.search(ln.source_text))
                              and not re.search(r"[^\x00-\x7f]", ln.source_text))
                    hit = ln.music_db >= SONG_MUSIC_DB and "BGM" in ln.events and (near or lyrics)
                if hit:
                    ln.kind = L.SONG
                    changed, total = True, total + 1
    return total


def classify(job: Job, lines: list[L.Line]) -> None:
    turns = job["diarization"]["segments"]
    ev = job.get("events") or []
    songs = list(job["ingest"].get("song_ranges", []))
    # Any chosen subtitle (source-language or English) that marks a moment as a song.
    for which in ("source", "english"):
        songs += [[q.start, q.end] for q in load_chosen(job, which) if q.kind == SONG]
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
        if ln.kind == L.SONG or _in_ranges(ln.start, ln.end, songs) or set(tags) & SINGING_EVENTS:
            ln.kind = L.SONG
        elif top_lang == "en" and lang != "en" and ln.duration >= 0.8 and not NON_LATIN.search(ln.source_text):
            ln.kind = L.ENGLISH
        elif ln.kind == L.SPEECH and ln.music_db >= BACKGROUND_DB:
            ln.kind = L.BACKGROUND  # the voice is in the music & effects track; a dub can't replace it
        elif (ln.kind == L.SPEECH and ln.duration <= CROWD_MAX_S
              and overlap_share(turns, ln.start, ln.end) >= CROWD_SHARE):
            ln.kind = L.CROWD     # several people at once: one dubbed voice can't stand in for them
        elif ln.heard_chars == 0 and ln.kind == L.SPEECH:
            ln.kind = L.WORDLESS   # nothing spoken here (a sigh or breathing under a subtitle): keep the original
        elif is_interjection(ln.source_text, lang) or (set(tags) & WORDLESS_EVENTS and
                                                       (len(ln.source_text) <= 3 or 0 <= ln.heard_chars <= 2)):
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
    end = float(job["ingest"]["duration"])
    # Subtitles can run past the audio (a preview, a clip): nothing to dub there.
    lines = [ln for ln in build(job) if ln.start < end - 0.2]
    for ln in lines:
        ln.end = min(ln.end, end)
    check_english_refs(job, lines)
    cuts = job["scenes"]["cuts"]
    windows = job["scenes"]["windows"]
    for i, ln in enumerate(lines):
        ln.id = i + 1
        ln.scene = scenes.scene_of(ln.start, cuts)
        ln.start, ln.end = round(ln.start, 3), round(max(ln.end, ln.start + 0.2), 3)
    lang = job["subtitles"].get("source_language") or job["asr"].get("language", "")
    shared = relisten(job, lines, lang)
    n = len(lines)
    lines = merge_fragments(drop_pieces(lines, shared, lang), lang)
    if len(lines) != n:
        job.log(f"{n - len(lines)} lines were pieces of a neighbouring line")
    missed = missed_cues(job, lines, lang)
    if missed:
        lines = sorted(lines + missed, key=lambda l: l.start)
        for i, ln in enumerate(lines):
            ln.id = i + 1
        relisten(job, lines, lang, only=missed)
        gone = {id(ln) for ln in missed if not clean_norm(ln.source_text)}
        lines = [ln for ln in lines if id(ln) not in gone]
        if len(missed) > len(gone):
            job.log(f"{len(missed) - len(gone)} subtitle lines no line carried were heard and added")
    for i, ln in enumerate(lines):
        ln.id = i + 1
    measure_music(job, lines)
    classify(job, lines)
    extend_songs(lines, job["subtitles"].get("source_language", ""))
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
