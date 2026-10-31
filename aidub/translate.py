"""Step 5: the English script.

Free, natural dubbing English sized to each line, using the scene as context,
the characters (names, genders, ages), the show's glossary and honorific
setting. Recipe: translate -> reflect (critique) -> rewrite for speech at the
target length, with a shorter alternative for overruns. Answers are cached for
the life of the job so a retry doesn't run the model again. Afterwards the
show's glossary is updated with names and terms from this episode.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import time

import httpx

from . import lines as L
from . import media
from .job import Job
from .profiles import Profile

BATCH = 36
LANG_NAMES = {"ja": "Japanese", "ko": "Korean", "zh": "Mandarin Chinese", "yue": "Cantonese", "es": "Spanish",
              "fr": "French", "de": "German", "en": "English", "ar": "Arabic"}
LANG_NAMES.update({c: n.title() for c, (_, n) in media.LANGUAGES.items() if c not in LANG_NAMES})
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


def _length(ln: L.Line, rate: float | None) -> dict:
    """The time a line has and the syllables that fit it; nothing with free timing."""
    return {} if rate is None else {"seconds": round(ln.duration, 2),
                                    "target_syllables": target_syllables(ln.duration, rate)}


FREE_NOTE = ("Length is free: each English line starts with the original and may run longer, so say the line's "
             "full meaning naturally, without squeezing it into the original's time and without padding. ")


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
    """The JSON in a model reply: the first object or list of objects (a list comes back
    as {"lines": ...}), skipping brackets in prose before it ("Notes [1]: {...}")."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1)
    dec = json.JSONDecoder()
    found_list = None
    for i, ch in enumerate(t):
        if ch not in "{[":
            continue
        try:
            v, _ = dec.raw_decode(t, i)
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            return v
        if v and all(isinstance(x, dict) for x in v):
            return {"lines": v}
        if found_list is None:
            found_list = v
    if found_list is not None:
        return {"lines": found_list}
    raise ValueError("no JSON in reply")


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
    if not script and settings.get("style") == "close":
        parts.append("Style: stay close to the original wording and sentence order wherever natural English allows.")
    if cast:
        parts.append("Speakers (use the right pronouns for each):\n" + "\n".join(cast))
    if gl:
        parts.append("Glossary (always use these English renderings):\n" + "\n".join(gl))
    return "\n\n".join(parts)


SYSTEM_FREE_NOTE = "Lines start with the original but may run longer, so keep each line's full meaning. "
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
SYSTEM_FREE = SYSTEM.replace("Each line must be speakable in the given time, so aim for the target syllable count. ",
                             SYSTEM_FREE_NOTE)


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
# translate.lead: subtitle_verbatim goes further: a line says its subtitle word for word;
# lines sharing one divide its words; a line too long is sped up or runs on before any
# word goes, and is then only trimmed (words deleted, never replaced).
VERBATIM_LEAD = "subtitle_verbatim"
SCRIPT_LEADS = (SCRIPT_LEAD, VERBATIM_LEAD)
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
SCRIPT_SYSTEM_FREE = SCRIPT_SYSTEM.replace(
    "that fits the time the character speaks (aim for the target syllable count)",
    "that starts with the character but may run longer, so keep each line's full meaning")
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
    return str(job.profile.settings.get("lead", job.cfg.get("translate.lead", VERBATIM_LEAD)))


def free_timing(job: Job) -> bool:
    from .fit import free_timing as free
    return free(job)


def lead_used(job: Job) -> str:
    """The lead this job's translation actually used (subtitle_only falls back to audio
    when the subtitle track isn't trusted)."""
    return job.get("translate_lead") or lead(job)


def system_for(job: Job) -> str:
    return SCRIPT_SYSTEM if lead_used(job) in SCRIPT_LEADS else SYSTEM


def verbatim(job: Job) -> bool:
    return lead_used(job) == VERBATIM_LEAD


def spoken_form(text: str) -> str:
    """Subtitle text as the voice should read it: "No. 12" read as the word "no" is
    another line ("Number 12")."""
    return re.sub(r"\bNo\.\s*(?=\d)", "Number ", text)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def only_deletes(short: str, full: str) -> bool:
    """`short` is `full` with words left out and nothing added, changed or reordered."""
    it = iter(_tokens(full))
    want = _tokens(short)
    return bool(want) and all(w in it for w in want)


def split_words(text: str, weights: list[float]) -> list[str]:
    """Divide a subtitle's words between lines in order, in proportion to their time,
    cutting after punctuation where one is within two words."""
    toks = text.split()
    n = len(weights)
    if n <= 1:
        return [text]
    total = sum(weights) or n
    cuts, acc, last = [], 0.0, 0
    for w in weights[:-1]:
        acc += w
        target = round(len(toks) * acc / total)
        lo, hi = last + 1, len(toks) - (n - 1 - len(cuts))
        if lo > hi:
            cuts.append(last)
            continue
        target = min(max(target, lo), hi)
        best = min(range(max(lo, target - 2), min(hi, target + 2) + 1),
                   key=lambda i: (not re.search(r"[,.!?;:…]$", toks[i - 1]), abs(i - target)))
        cuts.append(best)
        last = best
    bounds = [0] + cuts + [len(toks)]
    return [" ".join(toks[a:b]) for a, b in zip(bounds, bounds[1:])]


def align_split(whole: str, parts: list[str]) -> list[str] | None:
    """The translator divided a shared subtitle but changed some words ("I'll take care
    of it." / "Mina!"): cut the subtitle's own words where its division falls, matched
    word by word, so each line still says the part meant for it. None when its
    wording is too far from the subtitle to tell."""
    words = whole.split()
    norm = ["".join(_tokens(w)) for w in words]
    mine: list[str] = []
    bounds: list[int] = []
    for p in parts:
        mine += ["".join(_tokens(w)) for w in p.split()]
        bounds.append(len(mine))
    sm = difflib.SequenceMatcher(None, mine, norm, autojunk=False)
    if not mine or sm.ratio() < 0.5:
        return None
    ops = sm.get_opcodes()

    def to_sub(i: int) -> int:
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= i <= i2:
                return j1 + (i - i1) if tag == "equal" else j1 + round((i - i1) * (j2 - j1) / max(1, i2 - i1))
        return len(norm)
    cuts, last = [], 0
    for b in bounds[:-1]:
        c = max(last, min(to_sub(b), len(words)))
        # A cut next to a sentence or comma break moves onto it ("can I pass | through?").
        near = [i for i in range(max(last + 1, c - 2), min(len(words), c + 2) + 1) if re.search(r"[,.!?;:…]$", words[i - 1])]
        last = min(near, key=lambda i: abs(i - c)) if near and not re.search(r"[,.!?;:…]$", words[c - 1] if c else "") else c
        cuts.append(last)
    out = [" ".join(words[a:b]) for a, b in zip([0] + cuts, cuts + [len(words)])]
    return None if any(p.strip() and not o for p, o in zip(parts, out)) else out


def subtitle_groups(batch: list[L.Line], cues: list[dict]) -> list[tuple[list[L.Line], list[int]]]:
    """Lines that share cues, with those cues in time order: a line with a subtitle of
    its own is a group of one."""
    under = line_subtitles(batch, cues)
    groups: list[tuple[list[L.Line], list[int]]] = []
    for ln in batch:
        if not under[ln.id] or ln.check_note == NO_SUBTITLE:
            continue
        mine = set(under[ln.id])
        hit = [g for g in groups if mine & set(g[1])]
        lines_, ks = [ln], set(mine)
        for g in hit:
            groups.remove(g)
            lines_ = g[0] + lines_
            ks |= set(g[1])
        groups.append((sorted(lines_, key=lambda l: l.start), sorted(ks)))
    return groups


def verbatim_lines(batch: list[L.Line], cues: list[dict], drafts: dict[int, dict]) -> int:
    """The subtitles' own words as each line's dub: a line's own subtitle as it is;
    a shared one divided as the translator divided it if that used exactly its words
    in order, else by time. A shorter alternative must only leave words out. Returns
    how many shared subtitles were divided by time."""
    by_time = 0
    for group, ks in subtitle_groups(batch, cues):
        whole = spoken_form(" ".join(cues[k]["text"] for k in ks))
        parts = [spoken_form(drafts.get(ln.id, {}).get("en", "")) for ln in group]
        if len(group) > 1 and _tokens(" ".join(parts)) != _tokens(whole):
            aligned = align_split(whole, parts)
            if aligned is None:
                aligned = split_words(whole, [ln.duration for ln in group])
                by_time += 1
            parts = aligned
        elif len(group) == 1:
            parts = [whole]
        for ln, part in zip(group, parts):
            d = drafts.setdefault(ln.id, {})
            d.update(en=part.strip(), from_subtitle=True)
            short = spoken_form(d.get("short", ""))
            d["short"] = short if short and only_deletes(short, part) and _tokens(short) != _tokens(part) else ""
    return by_time


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
    agree = 1 - int(job.qc.get("subtitle_lines_disagreeing_with_speech", 0)) / checked if checked else 1.0
    need = {"content": 0.8, "total": 0.75, "agree": 0.75}
    need.update(job.cfg.get("translate.script_track") or {})
    if not checked:  # too few lines to check one by one: the track scores decide
        need.pop("agree", None)
    got = {"content": float(sc.get("content", 0)), "total": float(sc.get("total", 0)), "agree": agree}
    bad = [f"{k} {got[k]:.2f} below {float(v):.2f}" for k, v in need.items() if got[k] < float(v)]
    return not bad, "; ".join(bad) or (f"meaning {got['content']:.2f}, score {got['total']:.2f}, " +
                                       (f"{agree:.0%} of checked lines agree" if checked else "no lines checked"))


def covered_ids(lines: list[L.Line], cues: list) -> set[int]:
    """Lines a subtitle is shown for, matched exactly as the script prompt matches
    them (line_subtitles), so a line left out here never gets a cue there."""
    as_dicts = [{"start": q.start, "end": q.end, "text": q.text} for q in cues]
    return {lid for lid, ks in line_subtitles(lines, as_dicts).items() if ks}


def short_reaction(ln: L.Line) -> bool:
    """A short line ("Roger!", a shout) the subtitles left out keeps the original."""
    return ln.duration < 1.2 or len(re.sub(r"[\W_]+", "", ln.source_text)) <= 4


NEAR_CUE_S = 0.3  # a subtitle shown just after its line ("Mina!" 0.04 s after it) still belongs to it


def _sign(text: str) -> bool:
    """On-screen text the subtitles translate ("KAIJU NO. 15"), never spoken."""
    letters = re.sub(r"[^A-Za-z]", "", text)
    return len(letters) >= 4 and letters.isupper()


def _gap(q, ln) -> float:
    return max(ln.start - q.end, q.start - ln.end, 0.0)


def script_cues(cues: list, batches: list[list[L.Line]]) -> list[list[dict]]:
    """Each subtitle cue goes to the scene whose lines it overlaps most (or holds a
    line just next to it), so no cue is voiced in two scenes."""
    out: list[list[dict]] = [[] for _ in batches]
    for q in cues:
        best, most = None, 0.15
        for bi, b in enumerate(batches):
            o = sum(max(0.0, min(q.end, ln.end) - max(q.start, ln.start)) for ln in b)
            if o > most:
                best, most = bi, o
        if best is None and not _sign(q.text):
            near = [(min(_gap(q, ln) for ln in b), bi) for bi, b in enumerate(batches) if b]
            gap, bi = min(near, default=(9.0, None))
            best = bi if gap <= NEAR_CUE_S else None
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
        if not hits and not _sign(q["text"]):
            near = min(batch, key=lambda ln: max(ln.start - q["end"], q["start"] - ln.end, 0.0), default=None)
            if near is not None and max(near.start - q["end"], q["start"] - near.end) <= NEAR_CUE_S:
                hits = [near]
        for ln in hits:
            out[ln.id].append(k)
    return out


VERBATIM_TASK = (
    "The official English subtitles are the script for this scene, word for word. Each line below is one "
    "character speaking, with what they said in the original language (text; it may contain recognition "
    "errors) and subtitle: the official English shown while they speak. Each line's en is its subtitle exactly "
    "as written. Where lines share a subtitle (shared_with: the other lines under the same subtitle), divide "
    "its words between them in order, at the point where the speaker or the thought changes: every word used "
    "once, none added, changed or reordered. short: the same line with some words left out to fit "
    "target_syllables (only leave words out; keep the rest exactly), or the same as en if nothing can go "
    "without losing meaning. Names, titles and numbers are never left out. These lines get "
    "\"from_subtitle\": true. Then compare each with its text: \"matches\": true when its subtitle part "
    "expresses a similar idea (the gist, not word for word), false when it clearly says something else. A "
    "line marked no_subtitle: translate its text into natural spoken English of about target_syllables, with "
    "\"from_subtitle\": false.\n")
VERBATIM_TASK_FREE = (VERBATIM_TASK
                      .replace(" to fit target_syllables", "")
                      .replace("natural spoken English of about target_syllables", "natural spoken English, keeping its full meaning"))


def script_prompt(batch: list[L.Line], cues: list[dict], prev: list[L.Line], ctx: str, rate: float | None,
                  names: dict, word_for_word: bool = False) -> list[dict]:
    under = line_subtitles(batch, cues)
    owners: dict[int, list[int]] = {}
    for lid, ks in under.items():
        for k in ks:
            owners.setdefault(k, []).append(lid)
    items = []
    for ln in batch:
        d = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "seconds": round(ln.duration, 2),
             "text": ln.source_text, **_length(ln, rate)}
        if ln.check_note == NO_SUBTITLE or not under[ln.id]:
            d["no_subtitle"] = True
        else:
            d["subtitle"] = " ".join(cues[k]["text"] for k in under[ln.id])
            shared = sorted({o for k in under[ln.id] for o in owners[k]} - {ln.id})
            if shared:
                d["shared_with"] = shared
        items.append(d)
    before = [{"speaker": names.get(l.character, l.character), "text": l.source_text} for l in prev]
    if word_for_word:
        task = VERBATIM_TASK if rate is not None else VERBATIM_TASK_FREE
        user = (f"{ctx}\n\nEarlier in the scene:\n{json.dumps(before, ensure_ascii=False)}\n\n" + task +
                f"Lines:\n{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
                'Return {"lines": [{"id": <id>, "en": "<dub line>", "short": "<shorter version>", '
                '"from_subtitle": <true|false>, "matches": <true|false>}]} with one entry per id.')
        return [{"role": "system", "content": SCRIPT_SYSTEM if rate is not None else SCRIPT_SYSTEM_FREE},
                {"role": "user", "content": user}]
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


def translate_prompt(batch: list[L.Line], prev: list[L.Line], ctx: str, rate: float | None, names: dict,
                     lead_by: str = "audio") -> list[dict]:
    """rate None: free timing (no length to fit)."""
    items = []
    for ln in batch:
        d = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
             **_length(ln, rate)}
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
            "matches. " + UNSURE_NOTE
            + ((LEAD_NOTE if rate is not None else LEAD_NOTE.replace(" of about target_syllables", ""))
               if any("subtitle" in d for d in items) else "")
            + (FREE_NOTE if rate is None else "") + "Lines:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<dub line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"' + (', "subtitle_ok": <true|false, for lines with subtitle>'
                                     if any("subtitle" in d for d in items) else "") +
            '}]} with one entry per id.')
    return [{"role": "system", "content": SYSTEM if rate is not None else SYSTEM_FREE},
            {"role": "user", "content": user}]


def reflect_prompt(batch: list[L.Line], ctx: str, rate: float | None, names: dict, draft: dict,
                   system: str = SYSTEM) -> list[dict]:
    lengths = (lambda ln: {}) if rate is None else (lambda ln: {
        "target_syllables": target_syllables(ln.duration, rate),
        "draft_syllables": syllables(draft.get(ln.id, {}).get("en", ""))})
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), **lengths(ln), **flags(ln)} for ln in batch]
    user = (f"{ctx}\n\nReview this draft English dub. For each line, note concrete problems: wrong meaning, "
            "unnatural or stiff English, wrong pronoun, inconsistent names"
            + (" (length is free)" if rate is None else ", or a length far from target_syllables")
            + ". Say 'ok' when a line is fine. " + source_notes(items) + "\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"notes": [{"id": <id>, "note": "..."}]}.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rewrite_prompt(batch: list[L.Line], ctx: str, rate: float | None, names: dict, draft: dict, notes: dict,
                   system: str = SYSTEM) -> list[dict]:
    items = [{"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
              "draft": draft.get(ln.id, {}).get("en", ""), "note": notes.get(ln.id, "ok"),
              **({} if rate is None else {"target_syllables": target_syllables(ln.duration, rate)}),
              **flags(ln)} for ln in batch]
    user = (f"{ctx}\n\nRewrite the dub lines, fixing what each note says, so they sound natural when spoken"
            + (" (length is free)" if rate is None else " and fit target_syllables")
            + ". Keep lines that are already good. " + source_notes(items) + "\n"
            f"{json.dumps(items, ensure_ascii=False, indent=0)}\n\n"
            'Return {"lines": [{"id": <id>, "en": "<final line>", "short": "<shorter alternative, about 75% '
            'of the syllables>"}]}.')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


TRIM_NOTE = ("trim_only marks a line that is the official subtitle word for word: shorten it only by leaving "
             "words out of current, keeping every other word exactly as written and in order; never add, replace "
             "or reorder words, and never leave out a name, title or number (this overrides the rewording asked "
             "for above). ")


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
            + source_notes(items) + (TRIM_NOTE if any(it.get("trim_only") for it in items) else "") + "\n"
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


def _str(v) -> str:
    """A reply field as text; a JSON null is no text, never the word "None"."""
    return "" if v is None else str(v).strip()


def _collect(reply: str, ids: set[int] | None = None) -> dict[int, dict]:
    """A reply's lines by id, keeping only the ids that were asked for."""
    data = parse_json(reply)
    out = {}
    for item in data.get("lines", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        try:
            lid = int(item["id"])
            if ids is not None and lid not in ids:
                continue
            out[lid] = {"en": _str(item.get("en")), "short": _str(item.get("short")),
                        "subtitle_ok": item.get("subtitle_ok") is True,
                        "from_subtitle": item.get("from_subtitle") is True,
                        "matches": item.get("matches") is not False}
        except (KeyError, ValueError, TypeError):
            continue
    return out


LYRICS_SYSTEM = "You tell song lyrics from spoken lines in the subtitles of a film or TV episode. Answer with JSON only."
LYRICS_RUN_GAP_S = 20.0


def outside_songs(cues: list, lines: list[L.Line]) -> list:
    """Subtitles that are mostly over song lines are lyrics: none of their words go to a
    spoken line next to the song ("was the stray cat." ran 0.2 s into the announcer's line)."""
    songs = [(ln.start, ln.end) for ln in lines if ln.kind == L.SONG]

    def in_song(q) -> float:
        return sum(max(0.0, min(b, q.end) - max(a, q.start)) for a, b in songs) / max(q.end - q.start, 1e-3)
    return [q for q in cues if in_song(q) < 0.5]


def lyrics_prompt(run: list[L.Line], before: list[L.Line], after: list[L.Line], cues: list) -> list[dict]:
    def english(ln):
        if ln.english_ref:
            return ln.english_ref
        return " ".join(q.text for q in cues if min(q.end, ln.end) - max(q.start, ln.start) > 0.3 * ln.duration)

    def show(ls):
        return "\n".join(f"- {english(x) or '(no subtitle)'} [{x.source_text}]" for x in ls) or "(none)"
    items = [{"id": ln.id, "original": ln.source_text, "english": english(ln)} for ln in run]
    return [{"role": "system", "content": LYRICS_SYSTEM},
            {"role": "user", "content":
             "In this stretch of an episode a voice holds notes over music, which may be singing. For each line, "
             "decide whether it is sung (the lyrics of a song being performed or playing) or spoken (dialogue, "
             "narration, an announcer, a phone call, someone talking over the song). Lyrics read like verse: "
             "imagery, refrains, lines that answer no one. Spoken lines talk to someone or about what is happening; a "
             'line naming or introducing the song, or reacting to it ("She\'s something."), is spoken. '
             "The lines around them are context only.\n\n"
             f"Before:\n{show(before)}\n\nLines:\n{json.dumps(items, ensure_ascii=False, indent=1)}\n\n"
             f"After:\n{show(after)}\n\n"
             'Answer: {"lines": [{"id": <id>, "sung": true or false}]}'}]


def lyrics_check(job: Job, llm: "LLM", lines: list[L.Line], cues: list) -> int:
    """Lines that sound sung (held notes over music, aidub.align.mark_singing) but have
    text to read: the translation model reads them with their neighbours and says
    which are lyrics. Those keep the original voice; a line spoken over a song is
    still dubbed. An answer that can't be read leaves the lines as songs."""
    from .align import sung_candidate, sure_sung
    from .pitch import Pitch
    todo = [ln for ln in lines if sung_candidate(ln)]
    if not todo:
        return 0
    p, lang = Pitch.load(job), job["subtitles"].get("source_language", "")
    sure = [ln for ln in todo if p is not None and sure_sung(ln, p, lang)]
    for ln in sure:
        ln.kind, ln.check_note = L.SONG, "sung (long held notes, far slower than speech)"
    todo = [ln for ln in todo if ln.kind == L.SPEECH]
    if not todo or not job.cfg.get("songs.lyrics_check", True):
        job.qc["lyrics_check"] = {"sure": len(sure), "sung": 0, "spoken": 0}
        return len(sure)
    runs: list[list[L.Line]] = []
    for ln in sorted(todo, key=lambda l: l.start):
        if runs and ln.start - runs[-1][-1].end <= LYRICS_RUN_GAP_S:
            runs[-1].append(ln)
        else:
            runs.append([ln])
    speech = sorted((l for l in lines if l.kind == L.SPEECH), key=lambda l: l.start)
    prompts = []
    for run in runs:
        ids = {l.id for l in run}
        before = [l for l in speech if l.end <= run[0].start and l.id not in ids][-3:]
        after = [l for l in speech if l.start >= run[-1].end and l.id not in ids][:3]
        prompts.append(lyrics_prompt(run, before, after, cues))
    sung = spoken = 0
    for run, reply in zip(runs, llm.chat_many(prompts, max_tokens=1024)):
        try:
            said = {int(x["id"]): bool(x.get("sung")) for x in parse_json(reply).get("lines", []) if isinstance(x, dict)}
        except Exception:
            said = {}
        for ln in run:
            if said.get(ln.id, True):
                ln.kind = L.SONG
                ln.check_note = "lyrics (held notes over music, and the words read as a song)"
                sung += 1
            else:
                spoken += 1
    job.qc["lyrics_check"] = {"sure": len(sure), "sung": sung, "spoken": spoken}
    job.log(f"lyrics check: {len(sure)} lines sung beyond doubt; of the rest {sung} are lyrics, {spoken} spoken over music")
    return len(sure) + sung


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
    if lead_by in SCRIPT_LEADS:
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
        speech = [ln for ln in lines if ln.kind == L.SPEECH]
        under = covered_ids(speech, cues) if cues else set()
        for ln in speech:
            if cues and ln.id not in under:
                if short_reaction(ln):
                    ln.fallback = ln.check_note = SHORT_KEPT
                else:
                    ln.check_note = NO_SUBTITLE
    job.set("translate_lead", lead_by)
    free = free_timing(job)  # (auto timing follows the lead actually used)
    script = lead_by in SCRIPT_LEADS
    word_for_word = lead_by == VERBATIM_LEAD
    system = SCRIPT_SYSTEM if script else SYSTEM

    def prompt(b, p, c, share):
        return (script_prompt(b, share, p, c, None if free else rate, names, word_for_word) if script
                else translate_prompt(b, p, c, None if free else rate, names, lead_by))

    with LLM(job) as llm:
        from .subtitles.step import load_chosen
        lyrics_check(job, llm, lines, cues or [q for q in load_chosen(job, "english") if q.text])
        cues = outside_songs(cues, lines)
        batches = [b for b in ([l for l in b if not l.fallback] for b in _batches(lines, windows)) if b]
        shares = script_cues(cues, batches) if script else [[] for _ in batches]
        # Pass 1: every scene at once (the engine batches them); each scene sees
        # the end of the previous scene in the original language as context.
        drafts: dict[int, dict] = {}
        ctxs = [_context_block(prof, {l.character for l in b}, lang, script) for b in batches]
        prevs = [batches[bi - 1][-4:] if bi else [] for bi in range(len(batches))]
        replies = llm.chat_many([prompt(b, p, c, sh) for b, p, c, sh in zip(batches, prevs, ctxs, shares)])
        retry = []
        for b, p, c, sh, reply in zip(batches, prevs, ctxs, shares, replies):
            try:
                got = _collect(reply, {l.id for l in b})
            except Exception:
                got = {}
            drafts.update(got)
            missing = [l for l in b if l.id not in got or not got[l.id]["en"]]
            if missing:
                # With the subtitles as the script the whole scene is asked again, so a
                # subtitle the missing line shares keeps its shared_with; only the
                # missing lines are taken from the answer.
                retry.append((prompt(b if script else missing, p, c, sh), {l.id for l in missing}))
        for (_, ids), reply in zip(retry, llm.chat_many([r for r, _ in retry]) if retry else []):
            try:
                drafts.update({k: v for k, v in _collect(reply, ids).items() if v["en"]})
            except Exception:
                pass
        job.check()
        job.status(f"translated {len(batches)} scenes", progress=0.5 if reflect else 1.0)
        if word_for_word:
            by_time = sum(verbatim_lines(b, sh, drafts) for b, sh in zip(batches, shares))
            job.qc["shared_subtitles_divided_by_time"] = by_time
            for b, sh in zip(batches, shares):
                under = line_subtitles(b, sh)
                for ln in b:
                    if ln.check_note != NO_SUBTITLE and not drafts.get(ln.id, {}).get("en") and under[ln.id]:
                        # A shared subtitle's words all went to the neighbouring line(s).
                        ln.fallback = ln.check_note = "its shared subtitle is said by the neighbouring line"
        if lead_by == "subtitle":
            offered = [ln for b in batches for ln in b if ln.english_ref and not source_unsure(ln)]
            for ln in offered:
                ln.en_from = "subtitle" if drafts.get(ln.id, {}).get("subtitle_ok") else ""
            took = sum(led(ln) for ln in offered)
            job.qc["subtitle_led"] = {"lines": took, "rejected": len(offered) - took}
            job.log(f"subtitle-led: {took} lines from the subtitle, {len(offered) - took} didn't match the speech")
        if script:
            counts = {"from_subtitles": 0, "differ_from_speech": 0, "from_speech": 0}
            for ln in (l for b in batches for l in b if not l.fallback):
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
            review_system = (SCRIPT_SYSTEM_FREE if script else SYSTEM_FREE) if free else system
            ctxs = [_context_block(prof, {l.character for l in b}, lang, script) for b in reviewed]
            notes_replies = llm.chat_many([reflect_prompt(b, c, None if free else rate, names, drafts,
                                                          review_system)
                                           for b, c in zip(reviewed, ctxs)])
            job.status("reviewed the draft", progress=0.75)
            rewrite_convs = []
            for b, c, nr in zip(reviewed, ctxs, notes_replies):
                try:
                    notes = {int(n["id"]): str(n.get("note", "ok")) for n in parse_json(nr).get("notes", [])}
                except Exception:
                    notes = {}
                rewrite_convs.append(rewrite_prompt(b, c, None if free else rate, names, drafts, notes,
                                                    review_system))
            for b, reply in zip(reviewed, llm.chat_many(rewrite_convs)):
                try:
                    for k, v in _collect(reply, {l.id for l in b}).items():
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
