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
def _context_block(prof: Profile, chars_in_batch: set[str], lang: str, script: bool = False) -> str:
    settings = prof.settings
    hon = "subtitle" if script else settings.get("honorifics", "keep")
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
             ("exactly as the English subtitles write them." if hon == "subtitle"
              else "keep them attached to names, as anime fans expect." if hon == "keep"
              else "drop them and express the politeness naturally in English.")]
    if cast:
        parts.append("Speakers (use the right pronouns for each):\n" + "\n".join(cast))
    if gl:
        parts.append("Glossary (always use these English renderings):\n" + "\n".join(gl))
    return "\n\n".join(parts)


PLAIN = ("Keep numbers and units as said (kilometers stay kilometers), and use no slang, jargon, abbreviations "
         "or clipped words the original doesn't (\"klicks\", \"calc\"). ")
SYSTEM = (
    "You are an expert dubbing script writer. You turn foreign-language dialogue into English that "
    "voice actors will speak over the original picture. Write free, natural spoken English the way a "
    "professional dub reads: keep meaning, tone, character and intent; do not translate word for word. "
    "Each line must be speakable in the given time, so aim for the target syllable count. "
    "The source-language text is what was actually spoken and is authoritative. A subtitle hint is "
    "a human translation that may be inaccurate, cut differently or belong to a neighbouring line: "
    "use it for names and nuance only where it agrees with the source text, and for who is meant "
    "(he, she, they) where the source leaves the person unsaid. " + PLAIN +
    "Reply with JSON only."
)


def source_unsure(ln: L.Line) -> bool:
    """Recognition caught this line only when listening again on its own (a child's
    "Daddy!" under music): its text may be misheard (ちち as おじい), so the line's own
    English subtitle carries its meaning."""
    return bool(ln.relisten and ln.english_ref)


UNSURE = {"source_unsure": True}
UNSURE_NOTE = ("source_unsure marks a line recognition barely heard: its source text may be misheard, so "
               "subtitle_hint carries the meaning there. ")

# Subtitle-led translation (translate.lead: subtitle): a line's English subtitle, when the
# translator finds it means what was said, is what the dub says, fitted to the time. The
# speech checks the subtitle instead of being the text translated.
LED = {"subtitle_led": True}
LED_NOTE = ("subtitle_led marks a line written from its subtitle, which was checked to mean what was said: "
            "the dub carries subtitle_hint's meaning, so judge it against that. ")


# Subtitles as the script (translate.lead: subtitle_only): every line the English subtitles
# cover is dubbed from them, the subtitle text of a scene shared out among its spoken lines;
# the speech only checks that each part says a similar thing (a line that doesn't is
# flagged, not replaced) and sets the timing.
SCRIPT_LEAD = "subtitle_only"
SCRIPT = {"subtitle_script": True}
SCRIPT_NOTE = ("subtitle_script marks a line written from the official English subtitles, which are the script: "
               "keep its meaning, and its names, titles, forms of address and honorifics exactly as written; "
               "judge it against the current English, not the source text. ")
SCRIPT_SYSTEM = (
    "You are an expert dubbing script writer. You adapt the official English subtitles of a show into an "
    "English dub that voice actors will speak over the original picture. The subtitles are the script: keep "
    "their meaning, names and terms, and make each line natural spoken English that fits the time the "
    "character speaks (aim for the target syllable count). The original-language text is what was said, from "
    "speech recognition or the original subtitles; use it to check meaning and who says what. " + PLAIN +
    "Reply with JSON only."
)
NO_SUBTITLE = "no subtitle: translated from the speech"
SHORT_KEPT = "no subtitle: a short line keeps the original"


def led(ln: L.Line) -> bool:
    return ln.en_from in ("subtitle", "script")


def flags(ln: L.Line) -> dict:
    """What a prompt is told about where a line's meaning comes from."""
    if ln.en_from == "script":
        return dict(SCRIPT)
    return {**UNSURE, "subtitle_hint": ln.english_ref} if source_unsure(ln) else \
        {**LED, "subtitle_hint": ln.english_ref} if led(ln) else {}


def source_notes(items: list[dict]) -> str:
    return (UNSURE_NOTE + (LED_NOTE if any(it.get("subtitle_led") for it in items) else "")
            + (SCRIPT_NOTE if any(it.get("subtitle_script") for it in items) else ""))


def lead(job: Job) -> str:
    """"audio" (translate what was said; subtitles are hints), "subtitle" (a line's
    subtitle where it means what was said) or "subtitle_only" (the subtitles are the script)."""
    return str(job.profile.settings.get("lead", job.cfg.get("translate.lead", "audio")))


def lead_used(job: Job) -> str:
    """The lead this job's translation actually used (subtitle_only falls back to audio
    when the subtitle track isn't trusted)."""
    return job.get("translate_lead") or lead(job)


def system_for(job: Job) -> str:
    return SCRIPT_SYSTEM if lead_used(job) == SCRIPT_LEAD else SYSTEM


def subtitles_trusted(job: Job) -> tuple[bool, str]:
    """Whether the English subtitle track is good enough to be the script: its meaning
    follows the speech across the episode (content score: true pairs against shuffled
    ones), its timing follows the speech, and nearly all of the lines checked one by
    one against the speech agree. A machine-made, fan or other-cut track fails as a
    whole; then the episode is translated from the speech."""
    from .subtitles.step import candidate
    cid = job["subtitles"].get("english")
    if not cid:
        return False, "no English subtitle"
    sc = (candidate(job, cid) or {}).get("score", {})
    checked = int(job.qc.get("subtitle_lines_checked", 0))
    agree = 1 - int(job.qc.get("subtitle_lines_disagreeing_with_speech", 0)) / checked if checked else 0.0
    need = {"content": 0.8, "total": 0.75, "agree": 0.75}
    need.update(job.cfg.get("translate.script_track") or {})
    got = {"content": float(sc.get("content", 0)), "total": float(sc.get("total", 0)), "agree": agree}
    bad = [f"{k} {got[k]:.2f} below {float(v):.2f}" for k, v in need.items() if got[k] < float(v)]
    return not bad, "; ".join(bad) or (f"meaning {got['content']:.2f}, score {got['total']:.2f}, "
                                       f"{agree:.0%} of checked lines agree")


def covered(ln: L.Line, cues: list) -> bool:
    """A subtitle is shown while the line is spoken (not just touching its edge)."""
    return any(min(q.end, ln.end) - max(q.start, ln.start) >= min(0.2, 0.3 * ln.duration) for q in cues)


def short_reaction(ln: L.Line) -> bool:
    """A short line ("Roger!", a shout) the subtitles left out keeps the original."""
    return ln.duration < 1.2 or len(re.sub(r"[\W_]+", "", ln.source_text)) <= 4


def script_cues(cues: list, batches: list[list[L.Line]]) -> list[list[dict]]:
    """Each subtitle cue goes to the scene whose lines it overlaps most, so no cue is
    voiced in two scenes."""
    out: list[list[dict]] = [[] for _ in batches]
    for q in cues:
        best, most = None, 0.15
        for bi, b in enumerate(batches):
            o = sum(max(0.0, min(q.end, ln.end) - max(q.start, ln.start)) for ln in b)
            if o > most:
                best, most = bi, o
        if best is not None:
            out[best].append({"start": round(q.start, 2), "end": round(q.end, 2), "text": q.text})
    return out


def line_subtitles(batch: list[L.Line], cues: list[dict]) -> dict[int, list[int]]:
    """Which cues each line is spoken under, by time: a cue goes to every line it
    overlaps (by 0.15 s, or 30 % of the shorter of the two), else to the nearest line
    within a second. The translator only splits a cue several lines share; left to
    share out a scene's subtitles itself, it slipped a whole scene by one line."""
    out: dict[int, list[int]] = {ln.id: [] for ln in batch}
    for k, q in enumerate(cues):
        hits = [ln for ln in batch if min(q["end"], ln.end) - max(q["start"], ln.start)
                >= min(0.15, 0.3 * min(ln.duration, q["end"] - q["start"]))]
        if not hits:
            near = min(batch, key=lambda ln: max(ln.start - q["end"], q["start"] - ln.end, 0.0), default=None)
            if near is not None and max(near.start - q["end"], q["start"] - near.end) <= 1.0:
                hits = [near]
        for ln in hits:
            out[ln.id].append(k)
    return out


def script_prompt(batch: list[L.Line], cues: list[dict], prev: list[L.Line], ctx: str, rate: float,
                  names: dict) -> list[dict]:
    under = line_subtitles(batch, cues)
    owners: dict[int, list[int]] = {}
    for lid, ks in under.items():
        for k in ks:
            owners.setdefault(k, []).append(lid)
    items = []
    for ln in batch:
        d = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "seconds": round(ln.duration, 2),
             "text": ln.source_text, "target_syllables": target_syllables(ln.duration, rate)}
        if ln.check_note == NO_SUBTITLE or not under[ln.id]:
            d["no_subtitle"] = True
        else:
            d["subtitle"] = " ".join(cues[k]["text"] for k in under[ln.id])
            shared = sorted({o for k in under[ln.id] for o in owners[k]} - {ln.id})
            if shared:
                d["shared_with"] = shared
        items.append(d)
    before = [{"speaker": names.get(l.character, l.character), "text": l.source_text} for l in prev]
    user = (f"{ctx}\n\nEarlier in the scene:\n{json.dumps(before, ensure_ascii=False)}\n\n"
            "The official English subtitles are the script for this scene. Each line below is one character "
            "speaking, with what they said in the original language (text; it may contain recognition errors) "
            "and subtitle: the official English shown while they speak. Write each line's dub from its own "
            "subtitle, never with words from another line's subtitle. Where lines share a subtitle "
            "(shared_with: the other lines under the same subtitle), split it between them in order, so each "
            "says only its part, never repeating words across lines and never dropping subtitle meaning; "
            "respect each line's target_syllables when you split (a very short line gets a word or two, the "
            "rest goes to the next line). Keep the subtitles' names, titles, numbers, forms of address and "
            "honorifics exactly as "
            "written, and never drop one to save time (a line may run a little fast or long). Make each line "
            "natural spoken English of about target_syllables; you may condense the wording, not change the "
            "meaning. The short alternative follows the same rules. These "
            "lines get \"from_subtitle\": true. Then compare each with its text: \"matches\": true when its "
            "subtitle part expresses a similar idea (the gist, not word for word), false when it clearly says "
            "something else; keep the subtitle either way. A line marked no_subtitle: translate its text, with "
            "\"from_subtitle\": false.\n"
            f"Lines:\n{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<dub line>", "short": "<shorter alternative, about 75% '
            'of the syllables>", "from_subtitle": <true|false>, "matches": <true|false>}]} with one entry per id.')
    return [{"role": "system", "content": SCRIPT_SYSTEM}, {"role": "user", "content": user}]


LEAD_NOTE = ("Lines with \"subtitle\" are subtitle-led: it is the human English subtitle shown at this moment. "
             "First compare its meaning with the source text: the gist, not word for word (subtitles are "
             "condensed and cut differently, and may also cover a neighbouring line). If they say the same "
             "thing, set \"subtitle_ok\": true and write the dub from the subtitle: natural spoken English "
             "of about target_syllables, only the part this line says, keeping its names and who is meant. "
             "If the source text clearly says something else, set \"subtitle_ok\": false and translate the "
             "source text instead. ")


def translate_prompt(batch: list[L.Line], prev: list[L.Line], ctx: str, rate: float, names: dict,
                     lead_by: str = "audio") -> list[dict]:
    items = []
    for ln in batch:
        d = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
             "seconds": round(ln.duration, 2), "target_syllables": target_syllables(ln.duration, rate)}
        if ln.english_ref and lead_by == "subtitle" and not source_unsure(ln):
            d["subtitle"] = ln.english_ref
        elif ln.english_ref:
            d["subtitle_hint"] = ln.english_ref
        elif ln.english_context:
            d["nearby_subtitle"] = ln.english_context
        if source_unsure(ln):
            d.update(UNSURE)
        items.append(d)
    before = [{"speaker": names.get(l.character, l.character), "text": l.source_text} for l in prev]
    user = (f"{ctx}\n\nEarlier in the scene:\n{json.dumps(before, ensure_ascii=False)}\n\n"
            "Lines to dub, in order. Translate what the source text says. subtitle_hint (a human English "
            "subtitle for the line) helps with names and nuance but loses to the source text when they "
            "disagree; take he/she from it when the source doesn't say who; nearby_subtitle may cover this and a neighbouring line, so use only the part that "
            "matches. " + UNSURE_NOTE + (LEAD_NOTE if any("subtitle" in d for d in items) else "") + "Lines:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<dub line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"' + (', "subtitle_ok": <true|false, for lines with subtitle>'
                                     if any("subtitle" in d for d in items) else "") +
            '}]} with one entry per id.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def reflect_prompt(batch: list[L.Line], ctx: str, rate: float, names: dict, draft: dict,
                   system: str = SYSTEM) -> list[dict]:
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), "target_syllables": target_syllables(ln.duration, rate),
              "draft_syllables": syllables(draft.get(ln.id, {}).get("en", "")), **flags(ln)} for ln in batch]
    user = (f"{ctx}\n\nReview this draft English dub. For each line, note concrete problems: wrong meaning, "
            "unnatural or stiff English, wrong pronoun, inconsistent names, or a length far from "
            "target_syllables. Say 'ok' when a line is fine. " + source_notes(items) + "\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"notes": [{"id": <id>, "note": "..."}]}.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rewrite_prompt(batch: list[L.Line], ctx: str, rate: float, names: dict, draft: dict, notes: dict,
                   system: str = SYSTEM) -> list[dict]:
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), "note": notes.get(ln.id, "ok"),
              "target_syllables": target_syllables(ln.duration, rate), **flags(ln)} for ln in batch]
    user = (f"{ctx}\n\nRewrite the dub lines, fixing what each note says, so they sound natural when spoken "
            "and fit target_syllables. Keep lines that are already good. " + source_notes(items) + "\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<final line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"}]}.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def condense_prompt(items: list[dict], ctx: str, system: str = SYSTEM) -> list[dict]:
    """Ask for shorter dub lines that keep the meaning.

    items: {"id", "speaker", "text" (source, authoritative), "subtitle_hint", "current" (too long),
            "max_syllables", optional "voice_could_not_say" (words the voice garbled), "voice_heard_as"
            (what a listener heard from the voice's tries), "source_unsure"}."""
    user = (f"{ctx}\n\nThese dub lines are too long for the time the character speaks, or the voice "
            "couldn't say them clearly. Rewrite each so it "
            "fits max_syllables when spoken, keeping the meaning of the source text (what was actually said) "
            "and its tone. You may drop filler and repeated "
            "words, use contractions and shorter synonyms, and leave out names or details the scene makes "
            "obvious. Never change what is being said. When max_syllables is 1 or 2, use the single word or "
            "short interjection that carries the line's meaning (for example \"What?\", \"No!\", \"Run!\", "
            "\"Thanks.\"). voice_heard_as is what a listener heard every time the voice said the current line: "
            "the voice can't say that wording clearly, so for those lines en and shortest must use different "
            "words from current (never the same words again), plain everyday ones the voice can say clearly; "
            "voice_could_not_say lists the words it garbled. No abbreviations, slang clippings or nonsense "
            "words: a counting rhyme becomes what it means, such as \"Which one, which one...\". Names stay. "
            + source_notes(items) + "\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<line within max_syllables>", "shortest": "<an even '
            'shorter version, about two thirds as long, same meaning>"}]}.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def meaning_check_prompt(items: list[dict], ctx: str, system: str = SYSTEM) -> list[dict]:
    """Ask whether shortened lines still mean what the original says.

    items: {"id", "text" (source, authoritative), "full_english", "options": {"a": ..., "b": ...},
            optional "source_unsure"}"""
    user = (f"{ctx}\n\nFor each line, check every shortened option against what the source text says (the "
            "spoken words are authoritative; full_english is the longer dub line; where source_unsure is "
            "set the source text may be misheard, so check against full_english" +
            ("; so also where subtitle_led is set: the line carries its checked subtitle"
             if any(it.get("subtitle_led") for it in items) else "") +
            ("; where subtitle_script is set, check against full_english only (the subtitles are the script), "
             "and an option that changes or drops a name, title or form of address fails"
             if any(it.get("subtitle_script") for it in items) else "") + "). An option passes only "
            "if a viewer would understand the same thing: same intent, same key information, nothing cut "
            "off mid-thought (\"I\" for \"I know\" fails; \"Got it.\" for \"I understand\" passes).\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "ok": ["a", "b"]}]} listing only the options that pass.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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


def _check_shorts(job: Job, llm: "LLM", lines: list[L.Line], lang: str) -> None:
    """The shorter alternative is used without another look when the line runs long,
    so with the subtitles as the script it must keep their meaning and names first
    ("Dummy Master." shortened to "Idiot." doesn't): the meaning check, or none."""
    prof = job.profile
    convs = []
    for i in range(0, len(lines), 30):
        part = lines[i:i + 30]
        items = [{"id": ln.id, "text": ln.source_text, "full_english": ln.en, "options": {"a": ln.en_short},
                  **flags(ln)} for ln in part]
        convs.append(meaning_check_prompt(items, _context_block(prof, {l.character for l in part}, lang, True),
                                          SCRIPT_SYSTEM))
    ok: set[int] = set()
    for reply in llm.chat_many(convs, max_tokens=1024) if convs else []:
        try:
            ok |= {int(it["id"]) for it in parse_json(reply).get("lines", []) if "a" in it.get("ok", [])}
        except Exception as exc:
            job.log(f"short-line check unusable, dropping those short lines: {exc}")
    dropped = 0
    for ln in lines:
        if ln.id not in ok:
            ln.en_short, dropped = "", dropped + 1
    job.qc["short_lines_dropped"] = dropped
    if dropped:
        job.log(f"{dropped} shorter alternatives changed the meaning and were dropped")


def _collect(reply: str) -> dict[int, dict]:
    data = parse_json(reply)
    out = {}
    for item in data.get("lines", []):
        try:
            out[int(item["id"])] = {"en": str(item.get("en", "")).strip(), "short": str(item.get("short", "")).strip(),
                                    "subtitle_ok": item.get("subtitle_ok") is True,
                                    "from_subtitle": item.get("from_subtitle") is True,
                                    "matches": item.get("matches") is not False}
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
    by_id = {ln.id: ln for ln in lines}
    lead_by = lead(job)
    cues: list = []
    if lead_by == SCRIPT_LEAD:
        from .subtitles.model import DIALOGUE
        from .subtitles.step import load_chosen
        trusted, why = subtitles_trusted(job)
        job.qc["subtitle_track"] = {"trusted": trusted, "why": why}
        if trusted:
            cues = [q for q in load_chosen(job, "english") if q.kind == DIALOGUE and q.text]
            job.log(f"subtitles are the script ({why})")
        else:
            lead_by = "audio"
            job.log(f"subtitle track not trusted as the script ({why}): translating the speech")
        # Speech the subtitles leave out: a short reaction keeps the original, a longer
        # line is translated from the speech and flagged.
        for ln in lines:
            if cues and ln.kind == L.SPEECH and not covered(ln, cues):
                if short_reaction(ln):
                    ln.fallback = ln.check_note = SHORT_KEPT
                else:
                    ln.check_note = NO_SUBTITLE
    job.set("translate_lead", lead_by)
    script = lead_by == SCRIPT_LEAD
    batches = [b for b in ([l for l in b if not l.fallback] for b in _batches(lines, windows)) if b]
    shares = script_cues(cues, batches) if script else [[] for _ in batches]
    system = SCRIPT_SYSTEM if script else SYSTEM

    def prompt(b, p, c, share):
        return script_prompt(b, share, p, c, rate, names) if script else translate_prompt(b, p, c, rate, names, lead_by)

    with LLM(job) as llm:
        # Pass 1: every scene at once (the engine batches them); each scene sees
        # the end of the previous scene in the original language as context.
        drafts: dict[int, dict] = {}
        ctxs = [_context_block(prof, {l.character for l in b}, lang, script) for b in batches]
        prevs = [batches[bi - 1][-4:] if bi else [] for bi in range(len(batches))]
        replies = llm.chat_many([prompt(b, p, c, sh) for b, p, c, sh in zip(batches, prevs, ctxs, shares)])
        retry = []
        for b, p, c, sh, reply in zip(batches, prevs, ctxs, shares, replies):
            try:
                got = _collect(reply)
            except Exception:
                got = {}
            drafts.update(got)
            missing = [l for l in b if l.id not in got or not got[l.id]["en"]]
            if missing:
                retry.append(prompt(missing, p, c, sh))
        for reply in llm.chat_many(retry) if retry else []:  # one retry for lines the model skipped
            try:
                drafts.update({k: v for k, v in _collect(reply).items() if v["en"]})
            except Exception:
                pass
        job.check()
        job.status(f"translated {len(batches)} scenes", progress=0.5 if reflect else 1.0)
        if lead_by == "subtitle":
            offered = [ln for b in batches for ln in b if ln.english_ref and not source_unsure(ln)]
            for ln in offered:
                ln.en_from = "subtitle" if drafts.get(ln.id, {}).get("subtitle_ok") else ""
            took = sum(led(ln) for ln in offered)
            job.qc["subtitle_led"] = {"lines": took, "rejected": len(offered) - took}
            job.log(f"subtitle-led: {took} lines from the subtitle, {len(offered) - took} didn't match the speech")
        if script:
            counts = {"from_subtitles": 0, "differ_from_speech": 0, "from_speech": 0}
            for ln in (l for b in batches for l in b):
                got = drafts.get(ln.id, {})
                if got.get("from_subtitle") and ln.check_note != NO_SUBTITLE:
                    ln.en_from = "script"
                    counts["from_subtitles"] += 1
                    if not got.get("matches", True):
                        ln.check_note = "the subtitle says something different from the speech"
                        counts["differ_from_speech"] += 1
                else:
                    ln.check_note = ln.check_note or "the subtitles didn't cover this line: translated from the speech"
                    counts["from_speech"] += 1
            counts["short_kept"] = sum(ln.check_note == SHORT_KEPT for ln in lines)
            job.qc["subtitle_script"] = counts
            job.log("subtitles as the script: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items()))

        final = dict(drafts)
        # The review judges a draft against the source text, so with the subtitles as the
        # script it would pull their human English back to a literal reading of the
        # original ("Do you have a card?" to "Does the old man not have one?"): only lines
        # translated from the speech are reviewed.
        reviewed = [b for b in ([l for l in b if not (script and l.en_from == "script")] for b in batches) if b]
        if reflect and reviewed:
            ctxs = [_context_block(prof, {l.character for l in b}, lang, script) for b in reviewed]
            notes_replies = llm.chat_many([reflect_prompt(b, c, rate, names, drafts, system)
                                           for b, c in zip(reviewed, ctxs)])
            job.status("reviewed the draft", progress=0.75)
            rewrite_convs = []
            for b, c, nr in zip(reviewed, ctxs, notes_replies):
                try:
                    notes = {int(n["id"]): str(n.get("note", "ok")) for n in parse_json(nr).get("notes", [])}
                except Exception:
                    notes = {}
                rewrite_convs.append(rewrite_prompt(b, c, rate, names, drafts, notes, system))
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
        if script:
            _check_shorts(job, llm, [l for b in batches for l in b if l.en_short and l.en_short != l.en], lang)
        untranslated = [ln for ln in lines if ln.kind == L.SPEECH and not ln.en and not ln.fallback]
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
