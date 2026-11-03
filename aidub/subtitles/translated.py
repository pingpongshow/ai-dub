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

from ..job import Job
from .model import DIALOGUE, Candidate, Cue

BATCH = 40
BEFORE = 6
NAMES_BATCH = 120
MAX_MISSING = 0.2   # more cues than this untranslated: the track isn't used

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # a word left untranslated ("the 慰劳 trip")

NAMES_SYSTEM = ("You list the names in a film's subtitles. Answer with JSON only. A person's or place's name is written in "
                "the subtitle's language, but it has to be said in English as the film's characters pronounce it, and the "
                "speech recognition transcript beside each line shows how that sounds, though it garbles other words. Read "
                "names by the sound, not by how their characters read in the subtitle's language. Only proper names: not "
                "titles or roles (chairman, boss, section chief, teacher), which are translated, and not common words. "
                "A place's or company's ordinary part stays English (Hirai City, Yonezawa News). Western order: given name first.")
SYSTEM = ("You translate a film's subtitles into natural spoken English, to be read aloud as a dub. Answer with JSON "
          "only. Keep the meaning, tone and register of each line; don't add or explain anything. Japanese and Chinese "
          "often leave out who is speaking or being spoken about: work it out from the lines around it.")


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


def guide_for(cue: Cue, guide: list[Cue]) -> str:
    return " ".join(g.text for g in guide if g.kind == DIALOGUE and g.text
                    and min(g.end, cue.end) - max(g.start, cue.start) > 0.3 * min(g.end - g.start, cue.end - cue.start))


def heard_between(segments: list[dict], a: float, b: float) -> str:
    return " ".join(s.get("text", "") for s in segments if s["end"] > a and s["start"] < b)[:120]


def name_list(llm, talk: list, segments: list[dict], lang_name: str, log=lambda m: None) -> dict:
    """{name as subtitled: name in English}, one spelling for the whole film, from what is heard:
    角田 read from its characters came out "Kakuta" where everyone says "Tsunoda". The whole film,
    a piece at a time (one prompt would be too long for the model)."""
    from ..translate import parse_json
    convs = []
    for k in range(0, len(talk), NAMES_BATCH):
        rows = [{"line": q.text, "heard": heard_between(segments, q.start - 0.3, q.end + 0.3)} for _, q in talk[k:k + NAMES_BATCH]]
        user = (f"{lang_name} subtitle lines, with what recognition heard at the time:\n{json.dumps(rows, ensure_ascii=False, indent=0)}\n\n"
                'List every person, place and group named. Return {"names": [{"subtitle": "<as written>", "english": "<romanised as spoken>"}]}.')
        convs.append([{"role": "system", "content": NAMES_SYSTEM}, {"role": "user", "content": user}])
    seen: dict[str, Counter] = {}
    for reply in llm.chat_many(convs, max_tokens=1024):
        try:
            for n in parse_json(reply).get("names", []):
                if n.get("subtitle") and n.get("english"):
                    seen.setdefault(str(n["subtitle"]), Counter())[str(n["english"])] += 1
        except Exception as exc:
            log(f"name list reply unusable: {exc}")  # the lines still translate: names may just vary
    return {k: v.most_common(1)[0][0] for k, v in seen.items()}


def translate_cues(job: Job, cand: Candidate, lang_name: str, guide: list[Cue] | None = None,
                   segments: list[dict] | None = None) -> list[Cue] | None:
    """The candidate's cues in English, same timing; None if too many came back empty. `segments`
    (what recognition heard) tells how names are said, for a subtitle not in the spoken language."""
    from ..translate import LLM, _context_block, parse_json
    talk = [(i, q) for i, q in enumerate(cand.cues) if q.kind == DIALOGUE and q.text]
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
            names = name_list(llm, talk, segments, lang_name, job.log)
            job.log(f"names in the subtitles: {names}")
        todo = parts
        for _ in range(2):
            if not todo:
                break
            job.check()
            job.status(f"translating {lang_name} subtitles ({len(todo)} batches)")
            for reply in llm.chat_many(ask(todo), max_tokens=4096):
                try:
                    for it in parse_json(reply).get("lines", []):
                        text = re.sub(r"\s+", " ", str(it.get("en", ""))).strip()
                        if text and not CJK.search(text):  # (one left in the original goes round again)
                            english[int(it["i"])] = text
                except Exception as exc:
                    job.log(f"subtitle translation reply unusable: {exc}")
            todo = [[(i, q) for i, q in b if i not in english] for b in parts]
            todo = [b for b in todo if b]
    if len(english) < (1 - MAX_MISSING) * len(talk):
        job.log(f"only {len(english)} of {len(talk)} {lang_name} subtitle lines came back translated: not used")
        return None
    out = []
    for i, q in enumerate(cand.cues):
        if q.kind != DIALOGUE or not q.text:
            out.append(q)
        elif i in english:
            out.append(Cue(start=q.start, end=q.end, text=english[i], raw=english[i], name=q.name, style=q.style, kind=DIALOGUE))
    return out
