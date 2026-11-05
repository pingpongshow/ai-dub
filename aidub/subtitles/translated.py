"""An English script from subtitles in another language.

A person's subtitle says what was said, with sentences whole and timed to the speech, in
any language; speech recognition garbles film dialogue. So a Japanese subtitle (or, with
nothing better, a Chinese or Korean one) is translated into an English track, line for
line with the same timing, and that track then leads the dub as a human English one would.
With a human English subtitle too, it can guide the translation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher

from ..job import Job
from .model import DIALOGUE, Candidate, Cue

BATCH = 40
BEFORE = 6
NAMES_BATCH = 120
MOST_NAMES = 150    # names from the English subtitles shown with each piece of the name list
NEIGHBOUR_S = 2.0   # lines further apart than this can't be saying one guide line twice
MAX_MISSING = 0.2   # more cues than this untranslated: the track isn't used

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # a word left untranslated ("the 慰劳 trip")

NAMES_SYSTEM = ("You list the names in a film's subtitles. Answer with JSON only. A person's or place's name is written in "
                "the subtitle's language, but it has to be said in English as the film's characters pronounce it, and the "
                "speech recognition transcript beside each line shows how that sounds, though it garbles other words. Read "
                "names by the sound, not by how their characters read in the subtitle's language; a name written in kana reads as "
                "written. Where the transcript doesn't show the name, use its usual reading as a name. Only proper names: not "
                "titles or roles (chairman, boss, section chief, teacher), which are translated, and not common words. "
                "A place's or company's ordinary part stays English (Hirai City, Yonezawa News). Western order: given name first.")
SYSTEM = ("You translate a film's subtitles into natural spoken English, to be read aloud as a dub. Answer with JSON "
          "only. Keep the meaning, tone and register of each line; don't add or explain anything. Japanese and Chinese "
          "often leave out who is speaking or being spoken about: work it out from the lines around it. A sound word "
          "is said in plain English, not spelled out: a siren wails, it doesn't go \"peep-oo\".")


def _prompt(part: list[dict], before: list[str], lang: str, ctx: str, guided: bool, names: dict | None = None) -> list[dict]:
    guide = ("Each line may have a guide: a person's English subtitle shown at about the same time. Use it for names, "
             "who is meant and nuance, but translate what the line itself says. " if guided else "")
    known = (f"Write these names exactly like this: {json.dumps(names, ensure_ascii=False)}. Nothing in the English may stay in "
             "the subtitle's writing system. " if names else "Nothing in the English may stay in the subtitle's writing system. ")
    guide += known
    user = (f"{ctx}\n\nTranslate these {lang} subtitle lines into English, one English line for each, in order. A line "
            "that only continues a sentence stays a continuation: no capital, and no full stop unless the sentence "
            f"ends there. {guide}Lines just before, for context:\n{json.dumps(before, ensure_ascii=False)}\n\n"
            f"Lines:\n{json.dumps(part, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"i": <i>, "en": "<English>"}]} with one entry per i.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _guides(cue: Cue, guide: list[Cue]) -> list[Cue]:
    return [g for g in guide if g.kind == DIALOGUE and g.text
            and min(g.end, cue.end) - max(g.start, cue.start) > 0.3 * min(g.end - g.start, cue.end - cue.start)]


def guide_for(cue: Cue, guide: list[Cue]) -> str:
    return " ".join(g.text for g in _guides(cue, guide))


def english_names(guide: list[Cue]) -> list[str]:
    """The names a person's English subtitles use, most used first: words only ever written
    with a capital ("Hey" is a name by this test only if no line says "hey")."""
    text = " ".join(g.text for g in guide if g.kind == DIALOGUE)
    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text)
    plain = {w for w in words if w.islower()}
    seen = Counter(w for w in words if w[0].isupper() and w[1:].islower() and w.lower() not in plain)
    return [w for w, _ in seen.most_common(MOST_NAMES)]


def heard_between(segments: list[dict], a: float, b: float) -> str:
    return " ".join(s.get("text", "") for s in segments if s["end"] > a and s["start"] < b)[:120]


def name_list(llm, talk: list, segments: list[dict], lang_name: str, log=lambda m: None, guide: list[Cue] | None = None) -> dict:
    """{name as subtitled: name in English}, one spelling for the whole film, from what is heard:
    角田 read from its characters came out "Kakuta" where everyone says "Tsunoda". The whole film,
    a piece at a time (one prompt would be too long for the model). A person's English subtitles
    settle it where they name the same person: 東海林 came out "Tokairin" beside lines saying "Shoji"."""
    from ..translate import parse_json
    known = english_names(guide) if guide else []
    spelled = {w.lower() for w in known}
    note = (f"These names appear in a person's English subtitles for this film: {json.dumps(known, ensure_ascii=False)}. "
            'Where a line\'s "english" subtitle spells its name, write it exactly that way, whatever it sounds like. '
            "Otherwise use one of these only if it is clearly the same person; if not, read the name as usual.\n\n" if known else "")
    convs = []
    for k in range(0, len(talk), NAMES_BATCH):
        rows = [{"line": q.text, "heard": heard_between(segments, q.start - 0.3, q.end + 0.3),
                 **({"english": g} if guide and (g := guide_for(q, guide)) else {})} for _, q in talk[k:k + NAMES_BATCH]]
        user = (f"{note}{lang_name} subtitle lines, with what recognition heard at the time:\n{json.dumps(rows, ensure_ascii=False, indent=0)}\n\n"
                'List every person, place and group named. Return {"names": [{"subtitle": "<only the name, as its \\"line\\" writes it>", '
                '"english": "<romanised as spoken>"}]}.')
        convs.append([{"role": "system", "content": NAMES_SYSTEM}, {"role": "user", "content": user}])
    seen: dict[str, Counter] = {}
    for reply in llm.chat_many(convs, max_tokens=1024):
        try:
            for n in parse_json(reply).get("names", []):
                if n.get("subtitle") and n.get("english"):
                    seen.setdefault(str(n["subtitle"]), Counter())[str(n["english"])] += 1
        except Exception as exc:
            log(f"name list reply unusable: {exc}")  # the lines still translate: names may just vary

    written = " ".join(q.text for _, q in talk)
    seen = {k: v for k, v in seen.items() if k in written}  # (given the English again instead: of no use on these lines)

    def parts(spelling: str) -> set[str]:
        return set(re.findall(r"[^\W\d_]+", spelling.lower()))

    def beside(name: str, spelling: str) -> bool:
        """The English subtitles say it next to a line saying this name."""
        return any(parts(spelling) <= parts(guide_for(q, guide)) for _, q in talk if name in q.text)

    def best(name: str, spellings: Counter) -> str | None:
        """Most votes, but what a person's subtitles say beside the name beats votes, and a name
        from their list with nothing to tie it here is a guess at another person (雅人 came out
        "Ken", someone else in the film): left to the translation, which can read it."""
        ranked = [s for s, _ in spellings.most_common()]
        if not spelled:
            return ranked[0]
        ranked = [s for s in ranked if not parts(s) <= spelled or beside(name, s)]
        return next((s for s in ranked if parts(s) <= spelled), ranked[0] if ranked else None)
    picked = {k: best(k, v) for k, v in seen.items()}
    log(f"names left to the translation: {[k for k, v in picked.items() if v is None]}")
    return {k: v for k, v in picked.items() if v}


def _repeats(a: str, b: str, close: float = 0.75) -> bool:
    """One English line says the other again, whole or nearly."""
    x, y = (re.findall(r"[\w'’]+", t.lower()) for t in (a, b))
    if x and x == y:  # ("You're wrong." / "You're wrong." for お前は間違ってる / 何がだ)
        return True
    short, long = sorted((x, y), key=len)
    if len(short) >= 3 and any(long[k:k + len(short)] == short for k in range(len(long) - len(short) + 1)):
        return True
    return len(short) >= 4 and SequenceMatcher(None, x, y).ratio() >= close


def merge_repeats(talk: list, english: dict[int, str], guide: list[Cue], log=lambda m: None) -> dict[int, tuple[float, float]]:
    """Two neighbouring lines given one English subtitle as their guide can both come back saying
    it. The fuller one stays and takes the time of both; `english` loses the other. Returns each
    line's (start, end). A line the subtitle itself says twice is the film's, and stays."""
    span = {i: (q.start, q.end) for i, q in talk}
    kept: tuple | None = None
    for i, q in talk:
        if i not in english:
            continue
        if kept:
            j, p = kept
            same_guide = {id(g) for g in _guides(p, guide)} & {id(g) for g in _guides(q, guide)}
            if (same_guide and q.start - span[j][1] <= NEIGHBOUR_S and _repeats(english[j], english[i])
                    and SequenceMatcher(None, p.text, q.text).ratio() < 0.6):
                both = (span[j][0], span[i][1])
                drop, keep = (j, i) if len(english[i]) > len(english[j]) else (i, j)
                log(f"said twice: {english[drop]!r} merged into {english[keep]!r}")
                del english[drop]
                span[keep] = both
                kept = (keep, q)  # (the next line is compared with this text, from the cue it follows)
                continue
        kept = (i, q)
    return span


def _replies(reply: str) -> list[tuple[int, str]]:
    """(i, English) from a reply, whatever else is wrong with its JSON: one bad bracket must not cost
    a whole batch."""
    return [(int(i), json.loads(f'"{text}"')) for i, text in re.findall(r'"i":\s*(\d+),\s*"en":\s*"((?:[^"\\]|\\.)*)"', reply)]


def translate_cues(job: Job, cand: Candidate, lang_name: str, guide: list[Cue] | None = None,
                   segments: list[dict] | None = None) -> list[Cue] | None:
    """The candidate's cues in English, same timing; None if too many came back empty. `segments`
    (what recognition heard) tells how names are said, for a subtitle not in the spoken language."""
    from ..translate import LLM, _context_block
    # (a cue of symbols only, "・～", marks music: asked to translate it the model invents a "Mm-hmm.")
    talk = [(i, q) for i, q in enumerate(cand.cues) if q.kind == DIALOGUE and re.search(r"[^\W\d_]", q.text)]
    if not talk:
        return None
    ctx = _context_block(job.profile, set(), cand.language) if job.profile else ""
    parts = [talk[k:k + BATCH] for k in range(0, len(talk), BATCH)]

    def ask(batches):
        convs = []
        for b in batches:
            at = next(k for k, (i, _) in enumerate(talk) if i == b[0][0])
            items = [{"i": i, "text": q.text, **({"guide": g} if guide and (g := guide_for(q, guide)) else {})} for i, q in b]
            convs.append(_prompt(items, [q.text for _, q in talk[max(0, at - BEFORE):at]], lang_name, ctx, bool(guide),
                                 {k: v for k, v in names.items() if any(k in q.text for _, q in b)}))
        return convs

    english: dict[int, str] = {}
    names: dict = {}
    with LLM(job) as llm:
        if segments:
            names = name_list(llm, talk, segments, lang_name, job.log, guide)
            job.log(f"names in the subtitles: {names}")
        todo = parts
        for _ in range(2):
            if not todo:
                break
            job.check()
            job.status(f"translating {lang_name} subtitles ({len(todo)} batches)")
            for reply in llm.chat_many(ask(todo), max_tokens=4096):
                try:
                    for i, en in _replies(reply):
                        text = re.sub(r"\s+", " ", re.sub(r"[<>]", "", en)).strip()  # (<> marks narration)
                        if text and not CJK.search(text):  # (one left in the original goes round again)
                            english[i] = text
                except Exception as exc:
                    job.log(f"subtitle translation reply unusable: {exc}")
            # What's left goes round in smaller pieces: asked the same thing, the model answers the same.
            left = [(i, q) for i, q in talk if i not in english]
            todo = [left[k:k + BATCH // 4] for k in range(0, len(left), BATCH // 4)]
    if len(english) < (1 - MAX_MISSING) * len(talk):
        job.log(f"only {len(english)} of {len(talk)} {lang_name} subtitle lines came back translated: not used")
        return None
    if len(english) < len(talk):
        job.log(f"{len(talk) - len(english)} of {len(talk)} {lang_name} subtitle lines have no translation: they fall to the speech")
    span = merge_repeats(talk, english, guide, job.log) if guide else {}
    out = []
    for i, q in enumerate(cand.cues):
        if q.kind != DIALOGUE or not q.text:
            out.append(q)
        elif i in english:
            start, end = span.get(i, (q.start, q.end))
            out.append(Cue(start=start, end=end, text=english[i], raw=english[i], name=q.name, style=q.style, kind=DIALOGUE))
    return out
