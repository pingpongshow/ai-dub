"""Step 5: the English script.

Free, natural dubbing English sized to each line, using the scene as context,
the characters (names, genders, ages), the show's glossary and honorific
setting. Recipe: translate -> reflect (critique) -> rewrite for speech at the
target length, with a shorter alternative for overruns. Answers are cached for
the life of the job so a retry doesn't run the model again. Afterwards the
show's glossary is updated with names and terms from this episode.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import httpx

from . import lines as L
from .job import Job
from .profiles import Profile

STEP = "translate"
BATCH = 36
LANG_NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Mandarin Chinese", "yue": "Cantonese", "es": "Spanish",
              "fr": "French", "de": "German", "en": "English", "ar": "Arabic"}
THINK = re.compile(r"<think>.*?</think>", re.S)


def syllables(text: str) -> int:
    n = 0
    for word in re.findall(r"[a-zA-Z']+|\d+", text):
        if word.isdigit():
            n += max(1, len(word))  # rough: numbers are read out
            continue
        w = word.lower().strip("'")
        groups = re.findall(r"[aeiouy]+", w)
        c = len(groups)
        if w.endswith("e") and c > 1 and not w.endswith(("le", "ee", "ye")):
            c -= 1
        n += max(1, c)
    return n


def target_syllables(seconds: float, rate: float, overhead: float = 0.25, borrow: float = 0.25) -> int:
    """Syllables that fit a line: the voice model speaks at about `rate`
    syllables a second plus a fixed `overhead` per line, and a line may borrow
    a little following silence."""
    return max(1, round((seconds + borrow - overhead) * rate))


# LLM backends ---------------------------------------------------------------------------
class LLM:
    def __init__(self, job: Job):
        self.job = job
        self.cache_file = job.path("translate_cache.json")
        self.cache: dict = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}
        self.worker = None
        self.endpoint = job.cfg.get("llm_endpoint") or {}

    def __enter__(self):
        if not self.endpoint.get("enabled"):
            self.worker = self.job.worker("translate").__enter__()
            stage = dict(self.job.cfg.get("stages.translate") or {})
            info = self.worker.call("load", stage)
            self.job.models["translation"] = info.get("model", stage.get("model", ""))
        else:
            self.job.models["translation"] = f"{self.endpoint.get('model')} (local endpoint)"
        return self

    def __exit__(self, *exc):
        if self.worker:
            self.worker.__exit__(*exc)

    def chat_many(self, conversations: list[list[dict]], max_tokens: int = 4096) -> list[str]:
        keys = [hashlib.sha256(json.dumps([c, max_tokens], ensure_ascii=False).encode()).hexdigest()
                for c in conversations]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            if self.worker:
                res = self.worker.call("chat", {"conversations": [conversations[i] for i in todo],
                                                "max_tokens": max_tokens, "temperature": 0.3})
                outs = res["outputs"]
            else:
                outs = [self._endpoint_chat(conversations[i], max_tokens) for i in todo]
            for i, text in zip(todo, outs):
                self.cache[keys[i]] = THINK.sub("", text).strip()
            tmp = self.cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False))
            tmp.replace(self.cache_file)
        return [self.cache[k] for k in keys]

    def _endpoint_chat(self, messages: list[dict], max_tokens: int) -> str:
        url = self.endpoint["url"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.endpoint.get('api_key')}"} if self.endpoint.get("api_key") else {}
        r = httpx.post(url, headers=headers, timeout=600, json={
            "model": self.endpoint.get("model"), "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in reply")
    t = t[start:]
    for end in range(len(t), 0, -1):
        if t[end - 1] in "}]":
            try:
                v = json.loads(t[:end])
                return v if isinstance(v, dict) else {"lines": v}
            except json.JSONDecodeError:
                continue
    raise ValueError("unparseable JSON in reply")


# Prompts ---------------------------------------------------------------------------------
def _context_block(prof: Profile, chars_in_batch: set[str], lang: str) -> str:
    settings = prof.settings
    hon = settings.get("honorifics", "keep")
    cast = []
    for cid in sorted(chars_in_batch):
        ch = prof.characters.get(cid)
        if ch:
            bits = [b for b in (ch.get("gender"), ch.get("age")) if b]
            cast.append(f"- {ch['name']}" + (f" ({', '.join(bits)})" if bits else ""))
    gl = [f"- {k} = {v['en']}" for k, v in sorted(prof.glossary.items(), key=lambda kv: -kv[1].get("count", 0))[:120]
          if v.get("en")]
    parts = [f"Source language: {LANG_NAMES.get(lang, lang or 'unknown')}.",
             "Honorifics (-san, -kun, -chan, -senpai, -sama...): " +
             ("keep them attached to names, as anime fans expect." if hon == "keep"
              else "drop them and express the politeness naturally in English.")]
    if cast:
        parts.append("Speakers (use the right pronouns for each):\n" + "\n".join(cast))
    if gl:
        parts.append("Glossary (always use these English renderings):\n" + "\n".join(gl))
    return "\n\n".join(parts)


SYSTEM = (
    "You are an expert dubbing script writer. You turn foreign-language dialogue into English that "
    "voice actors will speak over the original picture. Write free, natural spoken English the way a "
    "professional dub reads: keep meaning, tone, character and intent; do not translate word for word. "
    "Each line must be speakable in the given time, so aim for the target syllable count. "
    "Reply with JSON only."
)


def translate_prompt(batch: list[L.Line], prev: list[L.Line], ctx: str, rate: float, names: dict) -> list[dict]:
    items = []
    for ln in batch:
        d = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
             "seconds": round(ln.duration, 2), "target_syllables": target_syllables(ln.duration, rate)}
        if ln.english_ref:
            d["subtitle_meaning"] = ln.english_ref
        items.append(d)
    before = [{"speaker": names.get(l.character, l.character), "text": l.source_text} for l in prev]
    user = (f"{ctx}\n\nEarlier in the scene:\n{json.dumps(before, ensure_ascii=False)}\n\n"
            "Lines to dub, in order (subtitle_meaning, when present, is a human English subtitle: follow its "
            "meaning and names, but rewrite it as natural speech of the right length):\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<dub line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"}]} with one entry per id.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def reflect_prompt(batch: list[L.Line], ctx: str, rate: float, names: dict, draft: dict) -> list[dict]:
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), "target_syllables": target_syllables(ln.duration, rate),
              "draft_syllables": syllables(draft.get(ln.id, {}).get("en", ""))} for ln in batch]
    user = (f"{ctx}\n\nReview this draft English dub. For each line, note concrete problems: wrong meaning, "
            "unnatural or stiff English, wrong pronoun, inconsistent names, or a length far from "
            "target_syllables. Say 'ok' when a line is fine.\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"notes": [{"id": <id>, "note": "..."}]}.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def rewrite_prompt(batch: list[L.Line], ctx: str, rate: float, names: dict, draft: dict, notes: dict) -> list[dict]:
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), "note": notes.get(ln.id, "ok"),
              "target_syllables": target_syllables(ln.duration, rate)} for ln in batch]
    user = (f"{ctx}\n\nRewrite the dub lines, fixing what each note says, so they sound natural when spoken "
            "and fit target_syllables. Keep lines that are already good.\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<final line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"}]}.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def condense_prompt(items: list[dict], ctx: str) -> list[dict]:
    """Ask for shorter dub lines that keep the meaning.

    items: {"id", "speaker", "text" (source), "subtitle_meaning", "current" (too long),
            "max_syllables"}."""
    user = (f"{ctx}\n\nThese dub lines are too long for the time the character speaks. Rewrite each so it "
            "fits max_syllables when spoken, keeping the meaning and tone. You may drop filler and repeated "
            "words, use contractions and shorter synonyms, and leave out names or details the scene makes "
            "obvious. Never change what is being said. When max_syllables is 1 or 2, use the single word or "
            "short interjection that carries the line's meaning (for example \"What?\", \"No!\", \"Run!\", "
            "\"Thanks.\").\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<line within max_syllables>", "shortest": "<an even '
            'shorter version, about two thirds as long, same meaning>"}]}.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def meaning_check_prompt(items: list[dict], ctx: str) -> list[dict]:
    """Ask whether shortened lines still mean what the original says.

    items: {"id", "text" (source), "meaning" (subtitle or full English), "options": {"a": ..., "b": ...}}"""
    user = (f"{ctx}\n\nFor each line, check every shortened option against the meaning. An option passes only "
            "if a viewer would understand the same thing: same intent, same key information, nothing cut "
            "off mid-thought (\"I\" for \"I know\" fails; \"Got it.\" for \"I understand\" passes).\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "ok": ["a", "b"]}]} listing only the options that pass.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def glossary_prompt(lines: list[L.Line], lang: str, known: list[str]) -> list[dict]:
    pairs = [{"text": l.source_text, "english": l.english_ref or l.en} for l in lines if l.kind == L.SPEECH][:400]
    user = (f"From these {LANG_NAMES.get(lang, lang)} lines and their English, list proper nouns and recurring "
            "special terms (character names, places, organisations, techniques/attacks, invented words) with "
            "the English rendering used. Prefer how a human subtitle rendered them. Skip common words. "
            f"Already known: {json.dumps(known, ensure_ascii=False)}\n"
            f"{json.dumps(pairs, ensure_ascii=False, indent=0)}\n\n"
            'Return {"terms": [{"source": "...", "en": "...", "type": "name|place|term"}]}.')
    return [{"role": "system", "content": "You extract glossaries for translators. Reply with JSON only."},
            {"role": "user", "content": user}]


# Step ------------------------------------------------------------------------------------
def _batches(lines: list[L.Line], windows: list[int]) -> list[list[L.Line]]:
    out: list[list[L.Line]] = []
    cur: list[L.Line] = []
    cur_w = None
    for ln, w in zip(lines, windows):
        if ln.kind != L.SPEECH:
            continue
        if cur and (w != cur_w and len(cur) >= 8 or len(cur) >= BATCH):
            out.append(cur)
            cur = []
        cur.append(ln)
        cur_w = w
    if cur:
        out.append(cur)
    return out


def _collect(reply: str) -> dict[int, dict]:
    data = parse_json(reply)
    out = {}
    for item in data.get("lines", []):
        try:
            out[int(item["id"])] = {"en": str(item.get("en", "")).strip(), "short": str(item.get("short", "")).strip()}
        except (KeyError, ValueError, TypeError):
            continue
    return out


def run(job: Job) -> None:
    prof: Profile = job.profile
    lines = L.load(job.path("lines.json"))
    windows = job.get("line_windows") or [0] * len(lines)
    lang = job["subtitles"].get("source_language", "")
    rate = float(job.cfg.get("translate.syllables_per_second", 4.3))
    reflect = bool(prof.settings.get("reflect", job.cfg.get("translate.reflect", True)))
    names = {cid: ch["name"] for cid, ch in prof.characters.items()}
    batches = _batches(lines, windows)
    by_id = {ln.id: ln for ln in lines}

    with LLM(job) as llm:
        # Pass 1: every scene at once (the engine batches them); each scene sees
        # the end of the previous scene in the original language as context.
        drafts: dict[int, dict] = {}
        ctxs = [_context_block(prof, {l.character for l in b}, lang) for b in batches]
        prevs = [batches[bi - 1][-4:] if bi else [] for bi in range(len(batches))]
        replies = llm.chat_many([translate_prompt(b, p, c, rate, names) for b, p, c in zip(batches, prevs, ctxs)])
        retry = []
        for b, p, c, reply in zip(batches, prevs, ctxs, replies):
            try:
                got = _collect(reply)
            except Exception:
                got = {}
            drafts.update(got)
            missing = [l for l in b if l.id not in got or not got[l.id]["en"]]
            if missing:
                retry.append(translate_prompt(missing, p, c, rate, names))
        for reply in llm.chat_many(retry) if retry else []:  # one retry for lines the model skipped
            try:
                drafts.update({k: v for k, v in _collect(reply).items() if v["en"]})
            except Exception:
                pass
        job.check()
        job.status(f"translated {len(batches)} scenes", progress=0.5 if reflect else 1.0)

        final = dict(drafts)
        if reflect and batches:
            ctxs = [_context_block(prof, {l.character for l in b}, lang) for b in batches]
            notes_replies = llm.chat_many([reflect_prompt(b, c, rate, names, drafts) for b, c in zip(batches, ctxs)])
            job.status("reviewed the draft", progress=0.75)
            rewrite_convs = []
            for b, c, nr in zip(batches, ctxs, notes_replies):
                try:
                    notes = {int(n["id"]): str(n.get("note", "ok")) for n in parse_json(nr).get("notes", [])}
                except Exception:
                    notes = {}
                rewrite_convs.append(rewrite_prompt(b, c, rate, names, drafts, notes))
            for reply in llm.chat_many(rewrite_convs):
                try:
                    for k, v in _collect(reply).items():
                        if v["en"]:
                            final[k] = v
                except Exception as exc:
                    job.log(f"rewrite reply unusable, keeping draft: {exc}")
            job.status("rewrote for speech", progress=0.95)

        for k, v in final.items():
            if k in by_id:
                by_id[k].en = v["en"]
                by_id[k].en_short = v.get("short", "")
        untranslated = [ln for ln in lines if ln.kind == L.SPEECH and not ln.en]
        for ln in untranslated:
            ln.fallback = "no translation"
        job.qc["untranslated"] = len(untranslated)

        # Glossary for the show.
        try:
            known = list(prof.glossary)[:200]
            terms = parse_json(llm.chat_many([glossary_prompt(lines, lang, known)])[0]).get("terms", [])
            human = bool(job["subtitles"].get("english"))
            for t in terms:
                src, en = str(t.get("source", "")).strip(), str(t.get("en", "")).strip()
                if not src or not en or len(src) > 40:
                    continue
                entry = prof.glossary.get(src)
                if entry is None:
                    prof.glossary[src] = {"en": en, "type": t.get("type", "term"), "count": 1,
                                          "from_subtitle": human, "edited": False, "added": time.time()}
                else:
                    entry["count"] = entry.get("count", 0) + 1
                    if not entry.get("edited") and human and not entry.get("from_subtitle"):
                        entry["en"], entry["from_subtitle"] = en, True
            prof.save()
        except Exception as exc:
            job.log(f"glossary update skipped: {exc}")

    L.save(lines, job.path("lines.json"))
    job.status(f"{sum(1 for l in lines if l.en)} lines translated")
