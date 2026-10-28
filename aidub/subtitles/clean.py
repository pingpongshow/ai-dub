"""Reading subtitle files and setting aside what isn't spoken dialogue.

Hearing-impaired tags, speaker labels, song lyrics, karaoke and on-screen sign
lines are kept out of the dub script. A subtitle added as a track keeps its
own text; this cleaning is only for the script.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pysubs2

from .model import DIALOGUE, HI, SIGN, SONG, Cue

ASS_TAG = re.compile(r"\{[^}]*\}")
HI_BRACKETS = re.compile(r"[\[\(（【][^\]\)）】]{0,60}[\]\)）】]")
SPEAKER_LABEL = re.compile(r"^\s*-?\s*([A-Z][A-Z0-9 .'\-]{0,24})\s*:\s+|^\s*([^\s:：a-z]{1,8})[：]\s*")
JP_SPEAKER = re.compile(r"^\s*[（(]([^）)]{1,12})[）)]\s*")
MUSIC = re.compile(r"[♪♫♬]")
SIGN_STYLES = re.compile(r"sign|title|note|screen|caption|text|card|eyecatch|logo", re.I)
SONG_STYLES = re.compile(r"\b(op|ed|song|karaoke|kfx|lyric|insert|opening|ending)\b|^(op|ed)\d*|romaji|kanji", re.I)
POSITIONED = re.compile(r"\\pos\(|\\move\(|\\an[789]")
KARAOKE = re.compile(r"\\k[fo]?\d", re.I)


def load(path: str | Path, fmt: str | None = None) -> list[Cue]:
    subs = pysubs2.load(str(path), encoding=None, format_=fmt) if fmt else pysubs2.load(str(path))
    is_ass = subs.format in ("ass", "ssa") or str(path).lower().endswith((".ass", ".ssa"))
    out = []
    for ev in subs.events:
        if ev.is_comment or ev.end <= ev.start:
            continue
        raw = ev.text
        kind = DIALOGUE
        if is_ass:
            if SONG_STYLES.search(ev.style or "") or KARAOKE.search(raw):
                kind = SONG
            elif SIGN_STYLES.search(ev.style or "") or (POSITIONED.search(raw) and not (ev.name or "").strip()):
                kind = SIGN
        start, end = ev.start / 1000.0, ev.end / 1000.0
        shown = ASS_TAG.sub("", raw).replace("\\N", "\n").replace("\\n", "\n")
        parts = speaker_turns(shown) if kind == DIALOGUE else []
        if len(parts) >= 2:
            # "- Line one / - Line two": two speakers in one subtitle. Split it, sharing
            # the time by text length; alignment later snaps each half to the speech.
            total = sum(len(p) for p in parts)
            t = start
            for p in parts:
                d = (end - start) * len(p) / total
                text, name, k2 = clean_text(p)
                out.append(Cue(start=round(t, 3), end=round(t + d, 3), text=text if k2 == DIALOGUE else "",
                               raw=p, name=name, style=ev.style or "", kind=k2))
                t += d
            continue
        text, name, k2 = clean_text(raw)
        if k2 != DIALOGUE and kind == DIALOGUE:
            kind = k2
        out.append(Cue(start=start, end=end, text=text if kind == DIALOGUE else "",
                       raw=shown, name=(ev.name or name or "").strip(), style=ev.style or "", kind=kind))
    out.sort(key=lambda c: (c.start, c.end))
    return out


DASH_TURN = re.compile(r"^\s*[-–—‐]\s*")


def speaker_turns(shown: str) -> list[str]:
    """The parts of a two-speaker subtitle ("- Hi.\\n- Hello."), or [] for one speaker."""
    rows = [r for r in re.sub(r"<[^>]+>", "", shown).split("\n") if r.strip()]
    if len(rows) >= 2 and sum(1 for r in rows if DASH_TURN.match(r)) >= 2:
        parts, cur = [], ""
        for r in rows:
            if DASH_TURN.match(r):
                if cur:
                    parts.append(cur)
                cur = DASH_TURN.sub("", r)
            else:
                cur = f"{cur} {r.strip()}".strip()
        if cur:
            parts.append(cur)
        return parts
    one = re.split(r"\s+[-–—]\s+(?=\S)", " ".join(rows))  # "- A. - B." on a single row
    if len(rows) == 1 and DASH_TURN.match(rows[0]) and len(one) >= 2:
        return [DASH_TURN.sub("", x) for x in one]
    return []


def clean_text(raw: str) -> tuple[str, str, str]:
    """Return (spoken text, speaker name if labelled, kind)."""
    t = ASS_TAG.sub("", raw).replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    t = re.sub(r"<[^>]+>", "", t)  # srt/vtt markup
    if MUSIC.search(t):
        return "", "", SONG
    name = ""
    m = JP_SPEAKER.match(t)
    if m:
        name, t = m.group(1), t[m.end():]
    t = HI_BRACKETS.sub(" ", t)
    m = SPEAKER_LABEL.match(t)
    if m:
        label = m.group(1) or m.group(2)
        name = name or (label.title() if m.group(1) else label)
        t = t[m.end():]
    t = re.sub(r"\s+", " ", t).strip(" -–—")
    if not t:
        return "", name, HI
    return t, name, DIALOGUE


def script_language(texts: list[str]) -> str:
    """Guess the language of subtitle text from its writing system."""
    counts = {"kana": 0, "hangul": 0, "han": 0, "latin": 0, "other": 0}
    for ch in "".join(texts)[:20000]:
        if not ch.strip() or not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name or "KATAKANA" in name:
            counts["kana"] += 1
        elif "HANGUL" in name:
            counts["hangul"] += 1
        elif "CJK UNIFIED" in name:
            counts["han"] += 1
        elif "LATIN" in name:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values()) or 1
    if counts["kana"] / total > 0.05:
        return "ja"
    if counts["hangul"] / total > 0.2:
        return "ko"
    if counts["han"] / total > 0.3:
        return "zh"
    if counts["latin"] / total > 0.6:
        return "en"
    return ""


def norm_chars(text: str) -> str:
    """Characters used for comparing texts: no spaces or punctuation, folded width/case."""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in t if ch.isalnum())
