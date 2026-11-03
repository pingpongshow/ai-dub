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

from ..job import Job
from .model import DIALOGUE, Candidate, Cue

BATCH = 40
BEFORE = 6
MAX_MISSING = 0.2   # more cues than this untranslated: the track isn't used

SYSTEM = ("You translate a film's subtitles into natural spoken English, to be read aloud as a dub. Answer with JSON "
          "only. Keep the meaning, tone and register of each line; don't add or explain anything. Japanese and Chinese "
          "often leave out who is speaking or being spoken about: work it out from the lines around it.")


def _prompt(part: list[dict], before: list[str], lang: str, ctx: str, guided: bool) -> list[dict]:
    guide = ("Each line may have a guide: a person's English subtitle shown at about the same time. Use it for names, "
             "who is meant and nuance, but translate what the line itself says. " if guided else "")
    user = (f"{ctx}\n\nTranslate these {lang} subtitle lines into English, one English line for each, in order. A line "
            "that only continues a sentence stays a continuation: no capital, and no full stop unless the sentence "
            f"ends there. {guide}Lines just before, for context:\n{json.dumps(before, ensure_ascii=False)}\n\n"
            f"Lines:\n{json.dumps(part, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"i": <i>, "en": "<English>"}]} with one entry per i.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def guide_for(cue: Cue, guide: list[Cue]) -> str:
    return " ".join(g.text for g in guide if g.kind == DIALOGUE and g.text
                    and min(g.end, cue.end) - max(g.start, cue.start) > 0.3 * min(g.end - g.start, cue.end - cue.start))


def translate_cues(job: Job, cand: Candidate, lang_name: str, guide: list[Cue] | None = None) -> list[Cue] | None:
    """The candidate's cues in English, same timing; None if too many came back empty."""
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
            convs.append(_prompt(items, [q.text for _, q in talk[max(0, at - BEFORE):at]], lang_name, ctx, bool(guide)))
        return convs

    english: dict[int, str] = {}
    with LLM(job) as llm:
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
                        if text:
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
