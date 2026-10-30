"""Steps 7-8: voice-cloned takes, back-check scoring and timing fit.

Each step loads its models once and does all of their work (no switching back
and forth): the voice model voices everything, then the checking models score
everything, then the choice needs no model.

  speak    - 2-4 takes per line at natural speed: character's voice reference,
             the original line as emotion reference (or a detected-emotion vector);
             for lines that miss their window, a speed factor (models that have
             one) and the shorter wording, in the same session
  check    - score takes and refits: recognition back-check (did it say the
             words?) and speaker similarity; keep the best take
  finalize - choose per line in order of quality cost: fitted take, shorter
             wording, time-stretch up to the limit, or the original audio
             (listed in the QC summary)
  condense - lines still in the original language: shorter English, voiced and
             checked again
"""

from __future__ import annotations

import difflib
import queue
import re
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import fit as F
from . import gpu
from . import lines as L
from . import media
from . import pronounce
from .job import Job
from .workers import Worker

EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
SENSEVOICE_TO_VECTOR = {"HAPPY": "happy", "ANGRY": "angry", "SAD": "sad", "FEARFUL": "afraid",
                        "DISGUSTED": "disgusted", "SURPRISED": "surprised", "NEUTRAL": "calm"}


def emo_vector(tag: str, strength: float = 0.7) -> list[float] | None:
    name = SENSEVOICE_TO_VECTOR.get((tag or "").upper().strip("<|>"))
    if not name:
        return None
    v = [0.0] * len(EMOTIONS)
    v[EMOTIONS.index(name)] = strength
    return v


CONTRACTIONS = {"it's": "it is", "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
                "that's": "that is", "what's": "what is", "there's": "there is", "he's": "he is", "she's": "she is",
                "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot", "won't": "will not",
                "isn't": "is not", "aren't": "are not", "wasn't": "was not", "i'll": "i will", "you'll": "you will",
                "we'll": "we will", "i've": "i have", "you've": "you have", "i'd": "i would", "let's": "let us",
                "gonna": "going to", "wanna": "want to", "gotta": "got to", "outta": "out of", "kinda": "kind of",
                "sorta": "sort of", "lemme": "let me", "gimme": "give me", "dunno": "do not know",
                "alright": "all right", "ok": "okay"}


IS_CONTRACTED = {"who", "what", "where", "when", "why", "how", "that", "there", "here", "it", "he", "she", "this"}
SUFFIXES = (("n't", " not"), ("'re", " are"), ("'ll", " will"), ("'ve", " have"), ("'m", " am"))


def said_as(text: str) -> str:
    """Symbols and numbers as words, the way the voice should read them and the
    back-check should compare them: "84%" is "84 percent" (the voice read a bare "84%"
    in Chinese), "9.0" is "9 point 0" (it said "nine point"), "No. 12" is "Number 12"."""
    text = re.sub(r"(?<=\d)\.(?=\d)", " point ", text)
    text = re.sub(r"\s*%", " percent", text)
    return re.sub(r"\bNo\.\s*(?=\d)", "Number ", text)


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9']+", said_as(text).lower().replace("’", "'")):
        if w not in CONTRACTIONS:
            if w.endswith("'s") and w[:-2] in IS_CONTRACTED:  # "who's" = "who is" (not a possessive)
                w = w[:-2] + " is"
            for suf, full in SUFFIXES:
                if w.endswith(suf) and len(w) > len(suf):
                    w = w[: -len(suf)] + full
                    break
        out.extend(CONTRACTIONS.get(w, w).split())
    return out


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> list[str]:
    """212 as said: "two hundred twelve" (an "and" heard after "hundred" is dropped)."""
    if n >= 1000:
        return number_words(n // 1000) + ["thousand"] + (number_words(n % 1000) if n % 1000 else [])
    if n >= 100:
        return [ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    if n >= 20:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    return [ONES[n]]


def _numbers(words: list[str], digitwise: bool) -> list[str]:
    """Numbers in the script as the voice may read them: as a number or digit by digit."""
    out = []
    for w in words:
        if w.isdigit() and len(w) <= 6:
            out.extend([ONES[int(d)] for d in w] if digitwise else number_words(int(w)))
        else:
            out.append(w)
    return out


def wer(ref: str, hyp: str) -> float:
    """Back-check error rate. Short lines (names, one or two words) are compared
    letter by letter, so "Kikoru" heard as "Kikaru" is a small error, not 100%.
    A number counts read either way ("212": "two hundred and twelve", "two one two")."""
    a, b = _words(ref), _words(hyp)
    b = [w for i, w in enumerate(b) if not (w == "and" and i and b[i - 1] in ("hundred", "thousand"))]
    b = ["zero" if w == "oh" and i and b[i - 1] == "point" else w for i, w in enumerate(b)]  # "nine point oh"
    # A hyphenated name heard as one word ("Kikoru-chan" as "Kikorichan") is that name.
    for name in re.findall(r"[A-Za-z']+(?:-[A-Za-z']+)+", ref):
        parts = name.lower().split("-")
        joined = "".join(parts)
        if joined in pronounce.LEXICON:
            continue
        k = next((i for i, h in enumerate(b) if pronounce.same_sound(joined, h) or pronounce.near(joined, h)), None)
        if k is not None:
            b[k:k + 1] = parts
    if any(w.isdigit() for w in a):
        return min(_wer(_numbers(a, d), _numbers(b, d)) for d in (False, True))
    return _wer(a, b)


def _wer(a: list[str], b: list[str]) -> float:
    if not a:
        return 0.0 if not b else 1.0
    # A heard word that sounds like the script's word is that word ("breaks" for "brakes").
    b = [h if h in a else next((w for w in a if pronounce.same_sound(w, h)), h) for h in b]
    # Two short heard words that are pieces of one ("No yen" for "Nguyen") are that word.
    i = 0
    while i + 1 < len(b):
        w = next((w for w in a if w not in b and pronounce.heard_split(w, b[i], b[i + 1])), None)
        if w:
            b[i:i + 2] = [w]
        i += 1
    if len(a) <= 2:
        ca, cb = "".join(a), "".join(b)
        sm = difflib.SequenceMatcher(None, ca, cb, autojunk=False)
        return 1.0 - sum(bl.size for bl in sm.get_matching_blocks()) / max(len(ca), len(cb), 1)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    return 1.0 - matched / max(len(a), len(b))


STOPWORDS = set("""a an the and or but so if then of to in on at by for with from up out off over into onto as is am
are was were be been being do does did have has had will would shall should can could may might must not no yes
i you he she it we they me him her us them my your his its our their this that these those there here what who whom
which when where why how all any some just very too also only own same such than now oh ah uh um huh hey okay ok
well right yeah let go get got come came say said""".split())


def missed_words(text: str, heard: str) -> list[str]:
    """Key words of the line the back-check didn't hear ("shady" heard as "shoddy").
    Names (capitalised mid-sentence) are left out: recognition spells them its own
    way. A word heard with another spelling that sounds the same counts as heard
    (see pronounce)."""
    raw = re.findall(r"[A-Za-z0-9'’]+(?:-[A-Za-z0-9'’]+)*", text)
    # Every part of a hyphenated name ("Ga-yeong", "Ok-hui") is the name.
    names = {p.lower() for i, w in enumerate(raw) if (i or "-" in w) and w[0].isupper() for p in w.split("-")}
    heard_words = _words(heard)
    heard_set = set(heard_words)
    # Made-up words heard as one word: a hyphenated one ("ten-pent" as "tenpence"), or a
    # name and surname run together ("Anya Levski" as "Anielewski").
    lower = [w.lower().replace("’", "'") for w in raw]
    groups = [w.split("-") for w in lower if "-" in w and w.replace("-", "") not in pronounce.LEXICON]
    groups += [[a, b] for a, b in zip(lower, lower[1:]) if not (a in pronounce.LEXICON and b in pronounce.LEXICON)]
    # ... or heard as two words ("ten pents" as "ten pence"), which excuses only the made-up part.
    heard_pairs = [x + y for x, y in zip(heard_words, heard_words[1:])]
    for parts in groups:
        joined = "".join(parts)
        if not pronounce.LEXICON:
            continue
        if any(pronounce.near(joined, h) or pronounce.same_sound(joined, h) for h in heard_words):
            heard_set.update(parts)
        elif any(pronounce.near(joined, h) or pronounce.same_sound(joined, h) for h in heard_pairs):
            heard_set.update(p for p in parts if p not in pronounce.LEXICON)
    out = []
    for w in _words(text):
        base = re.sub(r"'s$", "", w)  # "West's" heard as "West", "how's" as "how is"
        if w.isdigit():  # a number must be heard whole, as a number or digit by digit
            said = [x for i, x in enumerate(heard_words) if not (x == "and" and i and heard_words[i - 1] in ("hundred", "thousand"))]
            said = ["zero" if x == "oh" and i and said[i - 1] == "point" else x for i, x in enumerate(said)]
            joined = f" {' '.join(said)} "
            if not any(f" {' '.join(_numbers([w], d))} " in joined for d in (False, True)) and w not in heard_set:
                out.append(w)
            continue
        if len(w) < 3 or base in STOPWORDS or w in names or base in names or w in heard_set or base in heard_set \
                or pronounce.heard_as(w, heard_words) or pronounce.heard_as(base, heard_words):
            continue
        # Plural, possessive and simple tense endings still count as heard.
        if any(h.rstrip("s") == w.rstrip("s") or h.rstrip("ed") == w.rstrip("ed") for h in heard_set):
            continue
        out.append(w)
    return out


def missed_by_voice(ln: L.Line) -> list[str]:
    """Words of the line the back-check didn't hear in some take."""
    scores = list(ln.take_scores.values()) + list((ln.fit or {}).get("scores", {}).values())
    return sorted({w for s in scores for w in s.get("missed", [])})


def respelled(ln: L.Line) -> str:
    """The line with the words its takes missed spelled as they sound, where their
    spelling doesn't show it ("Nguyen!" as "noo yen!"), and its romanised Japanese
    names ("Kikoru-chan" as "kee koh roo chahn"; the back-check doesn't list names
    as missed, as recognition spells them its own way)."""
    text = ln.en
    names = [w for w in re.findall(r"[A-Za-z]+", text) if pronounce.romaji_respell(w)
             and (w[0].isupper() or w.lower() in ("chan", "kun", "san", "sama", "senpai"))]
    for w in list(dict.fromkeys(missed_by_voice(ln) + names)):
        sound = pronounce.respell(w)
        if sound:
            text = re.sub(rf"(?<![\w']){re.escape(w)}(?![\w'])", sound, text, flags=re.I)
    return text


def _line_dir(job: Job, ln: L.Line) -> Path:
    return job.dir("lines", f"{ln.id:05d}")


def _orig_clip(job: Job, ln: L.Line, dlg: np.ndarray, sr: int) -> Path:
    p = _line_dir(job, ln) / "orig.wav"
    if not p.exists():
        a, b = int(max(0, ln.start - 0.05) * sr), int((ln.end + 0.05) * sr)
        media.write_wav(p, dlg[a:b], sr)
    return p


def spoken(text: str) -> str:
    """The line as the voice model should read it: it garbles hyphenated words
    ("self-esteem" came out as "self in Spain" in 4 of 4 takes, right in 4 of 4 as
    "self esteem"), so hyphens between letters become spaces."""
    return said_as(re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", text))


def _requests(job: Job, ln: L.Line, text: str, takes: int, tag: str, dlg, sr, factor: float | None = None,
              seed_offset: int = 0, voice: str = "") -> list[dict]:
    caps = job.get("tts_caps", {})
    orig = _orig_clip(job, ln, dlg, sr)
    min_emo = float(job.cfg.get("speak.min_emotion_ref_s", 1.0))
    req_base = {"text": spoken(text), "speaker_ref": voice or ln.voice or str(orig)}
    if caps.get("emotion_ref", True) and ln.duration >= min_emo and not ln.overlap:
        req_base["emotion_ref"] = str(orig)
        req_base["emo_alpha"] = float(job.cfg.get("speak.emo_alpha", 0.35))
    else:
        vec = emo_vector(ln.emotion)
        if vec and caps.get("emotion_vector", True):
            req_base["emo_vector"] = vec
    if factor is not None:
        req_base["duration_factor"] = factor
    for key in ("num_beams", "diffusion_steps"):  # voice-model speed settings (unset: the model's own)
        if job.cfg.get(f"speak.{key}"):
            req_base[key] = int(job.cfg.get(f"speak.{key}"))
    out = []
    for t in range(takes):
        out.append({**req_base, "seed": 1000 + ln.id * 7 + t + seed_offset,
                    "out": str(_line_dir(job, ln) / f"{tag}_{t}.wav")})
    return out


def _synthesize(job: Job, w, reqs: list[dict], label: str) -> None:
    """Voice the requests in groups of 8, shared between the voice processes (`w`
    is one worker or a list). Each take has its own seed, so who voices it doesn't
    change it."""
    ws = w if isinstance(w, list) else [w]
    parts: queue.Queue = queue.Queue()
    for i in range(0, len(reqs), 8):
        parts.put(reqs[i:i + 8])
    done, errors, stop = [0], [], threading.Event()

    def run(worker) -> None:
        while not stop.is_set():
            try:
                part = parts.get_nowait()
            except queue.Empty:
                return
            try:
                worker.call("synthesize", {"items": part})
            except Exception as exc:
                errors.append(exc)
                stop.set()
                return
            done[0] += len(part)

    threads = [threading.Thread(target=run, args=(x,), daemon=True) for x in ws]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.5)
        if job.cancel_event.is_set():
            stop.set()
        job.status(f"{label}: {done[0]}/{len(reqs)}", progress=min(1.0, done[0] / max(1, len(reqs))))
    if errors:
        raise errors[0]
    job.check()


@contextmanager
def _voices(job: Job):
    """The voice model, loaded in up to `speak.processes` processes sharing the GPU:
    one voice process leaves most of the card idle (it voices one line at a time).
    Extra processes start only while there is room for them; never waits for one."""
    want = max(1, int(job.cfg.get("speak.processes", 2)))
    need = int((job.cfg.get("gpu.min_free_mb") or {}).get("speak", 4000))
    with ExitStack() as stack:
        ws = [stack.enter_context(job.worker("speak"))]
        _load_tts(job, ws[0])
        for _ in range(want - 1):
            st = gpu.query()
            if st.present and st.free_mb < need:
                break
            w = stack.enter_context(Worker(job.cfg, ws[0].name, log=job.log))
            w.call("load", {"model": job.cfg.get("stages.speak").get("model")})
            ws.append(w)
        if len(ws) > 1:
            job.log(f"voicing with {len(ws)} voice-model processes")
        yield ws


def _load_tts(job: Job, w) -> dict:
    cfg = job.cfg.get("stages.speak")
    info = w.call("load", {"model": cfg.get("model")})
    job.models["voice"] = info.get("model", cfg.get("model", ""))
    caps = {"duration_control": bool(info.get("duration_control")), "emotion_ref": bool(info.get("emotion_ref", True)),
            "emotion_vector": bool(info.get("emotion_vector", True)), "sample_rate": int(info.get("sample_rate", 24000))}
    job.set("tts_caps", caps)
    return caps


# speak --------------------------------------------------------------------------------------
def run_speak(job: Job) -> None:
    """All voicing in one load of the voice model: the takes, then for lines that
    run long or short, speed-factor versions and the shorter wording. The checks
    for all of them come next, in one load of the checking models."""
    lines = L.load(job.path("lines.json"))
    todo = [ln for ln in lines if ln.dubbed and ln.en]
    takes = max(1, int(job.cfg.get("speak.takes", 2)))
    rules = F.FitRules.from_config(job.cfg)
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    with _voices(job) as w:
        caps = job.get("tts_caps", {})
        reqs = []
        for ln in todo:
            reqs += [r for r in _requests(job, ln, ln.en, takes, "take", dlg, sr) if not Path(r["out"]).exists()]
        # Same speaker reference back to back: the model reuses that voice's analysis.
        reqs.sort(key=lambda r: r["speaker_ref"])
        _synthesize(job, w, reqs, "takes")
        plan = _plan_fit(job, lines, rules)
        if F.free_timing(job):  # natural length: nothing to refit
            for ln in plan:
                ln.fit["state"] = "fits"
            plan = []
        if plan:
            _fit(job, w, caps, plan, dlg, sr, rules)
    L.save(lines, job.path("lines.json"))
    job.status(f"{len(todo) * takes} takes for {len(todo)} lines; {len(plan)} refitted")


# check ----------------------------------------------------------------------------------------
def score_files(job: Job, items: list[tuple[L.Line, str, str]]) -> dict[str, dict]:
    """items: (line, text it should say, wav path). Returns path -> {wer, sim, score}."""
    if not items:
        return {}
    prof = job.profile
    out: dict[str, dict] = {}
    pronounce.load(job.cfg.models / "lexicon" / "cmudict.dict")
    asr_cfg = job.cfg.get("stages.recognize")
    with job.worker("recognize") as w:
        w.call("load", {"model": asr_cfg.get("model")})
        for i in range(0, len(items), 16):
            job.check()
            part = items[i:i + 16]
            res = w.call("transcribe", {"items": [{"path": p} for _, _, p in part], "language": "en",
                                        "timestamps": False})
            for (ln, text, p), r in zip(part, res["results"]):
                ws = _words(text)
                sounds = pronounce.only_sounds(ws)  # "Shh.", "Mm-hmm.": nothing recognition can spell
                out[p] = {"wer": 0.0 if sounds else round(wer(text, r.get("text", "")), 3), "heard": r.get("text", ""),
                          "missed": [] if sounds else missed_words(text, r.get("text", "")), "words": len(ws),
                          "interjection": len(ws) == 1 and len(ws[0]) <= 4,
                          "seconds": round(media.duration(p), 2)}
            job.status(f"back-check {min(i + 16, len(items))}/{len(items)}", progress=0.5 * min(1.0, (i + 16) / len(items)))
    with job.worker("speaker") as w:
        w.call("load", {})
        paths = [p for _, _, p in items]
        embs = []
        for i in range(0, len(paths), 64):
            embs += w.call("embed_files", {"paths": paths[i:i + 64]})["embeddings"]
    from .speakers import cosine
    by_char: dict[str, list[float]] = {}
    for (ln, _, p), e in zip(items, embs):
        ch = prof.characters.get(prof.voice_source(ln.character), {})
        target = ch.get("embedding")
        if target and e:
            sim = cosine(e, target)
            by_char.setdefault(ln.character, []).append(sim)
            out[p]["sim"] = round(sim, 3)
        else:
            out[p]["sim"] = None  # no voiceprint to compare with: not held against the take
    for (ln, _, p) in items:
        sims = by_char.get(ln.character)
        out[p]["sim_median"] = round(float(np.median(sims)), 3) if sims else None
        sim = out[p]["sim"] if out[p]["sim"] is not None else (out[p]["sim_median"] or 0.0)
        out[p]["score"] = round(0.6 * (1 - out[p]["wer"]) + 0.4 * sim, 4)
    return out


def short_text(text: str) -> bool:
    return len(_words(text)) <= 2


def _passes(s: dict, cfg) -> bool:
    """Intelligible, and the voice isn't clearly someone else's. Cross-lingual
    similarity runs low (about 0.2-0.4), so a take is only rejected when it falls
    well below that character's own takes or under an absolute floor; clips under
    a second are too short to embed reliably, so only the back-check applies. A single
    interjection ("Huh?", "Hm.") is hard to recognise back, so it gets more slack."""
    max_wer = float(cfg.get("speak.max_wer", 0.35))
    if s.get("interjection"):
        max_wer = max(max_wer, 0.7)
    if s.get("wer", 1) > max_wer:
        return False
    # Every key word of a short line must come out right; a longer line may slip on one.
    if len(s.get("missed", [])) > (0 if s.get("words", 0) <= int(cfg.get("speak.strict_words_max", 12)) else 1):
        return False
    sim, med = s.get("sim"), s.get("sim_median")
    if sim is None or s.get("seconds", 99) < float(cfg.get("speak.min_similarity_seconds", 1.0)):
        return True  # no voiceprint, or too short for a reliable voice embedding
    if sim < float(cfg.get("speak.min_similarity", 0.10)):
        return False
    return med is None or sim >= med - float(cfg.get("speak.max_similarity_drop", 0.25))


FIT_FILES = (("fit1_0.wav", "en"), ("fit0_0.wav", "en"), ("short_0.wav", "en_short"), ("shortfit_0.wav", "en_short"))


def run_check(job: Job) -> None:
    """Back-check and voice-match every take and every fitted version at once."""
    lines = L.load(job.path("lines.json"))
    items, fitted = [], []
    for ln in lines:
        if not (ln.dubbed and ln.en):
            continue
        d = _line_dir(job, ln)
        for p in sorted(d.glob("take_*.wav")):
            F.trim(p)
            items.append((ln, ln.en, str(p)))
        for name, field in FIT_FILES:
            if (d / name).exists():
                fitted.append((ln, getattr(ln, field), str(d / name)))
    scores = score_files(job, items + fitted)
    for ln, _, p in fitted:
        ln.fit.setdefault("scores", {})[p] = scores.get(p, {})
    failed = 0
    for ln in lines:
        mine = {p: s for (l, _, p), s in ((it, scores.get(it[2], {})) for it in items) if l.id == ln.id}
        if not mine:
            continue
        ln.take_scores = mine
        best = max(mine, key=lambda p: mine[p].get("score", 0))
        if _passes(mine[best], job.cfg):
            ln.take, ln.fallback = best, ""  # (a resumed job may carry an earlier run's verdict)
        else:
            ln.take = ""
            ln.fallback = f"every take failed the back-check (best: WER {mine[best].get('wer')}, " \
                          f"similarity {mine[best].get('sim')})"
            failed += 1
    L.save(lines, job.path("lines.json"))
    job.qc["failed_takes"] = failed
    job.status(f"{failed} lines with no usable take" if failed else "all lines have a usable take")


# fit ----------------------------------------------------------------------------------------
def _plan_fit(job: Job, lines: list[L.Line], rules: F.FitRules) -> list[L.Line]:
    """Each line's window and how its first take compares; the lines needing a
    speed factor or shorter wording. The first take is the one a speed-factor
    version re-voices (same seed), so its length sets the factor."""
    win = F.windows([l for l in lines if l.kind == L.SPEECH], job["scenes"]["cuts"], rules, job["ingest"]["duration"])
    job.set("fit_windows", {str(k): v for k, v in win.items()})
    plan = []
    for ln in lines:
        first = _line_dir(job, ln) / "take_0.wav"
        if not (ln.dubbed and ln.en and first.exists()):
            continue
        measured = F.trim(first)
        state = F.needs_fit(measured, win[ln.id], ln.duration, rules)
        ln.fit = {"window": win[ln.id], "measured": round(measured, 3), "state": state or "fits"}
        if state:
            plan.append(ln)
    return plan


def _fit(job: Job, w, caps: dict, plan: list[L.Line], dlg, sr: int, rules: F.FitRules) -> None:
    """Speed-factor versions (measured and corrected once) and the shorter wording
    (fitted too if needed) for the planned lines, in the open voice-model session."""
    if caps["duration_control"]:
        # Up to two factor passes: the factor isn't exact, so measure and correct once.
        for attempt in range(2):
            reqs = []
            for ln in plan:
                measured = ln.fit.get(f"factor{attempt - 1}_measured") if attempt else ln.fit["measured"]
                prev = ln.fit.get(f"factor{attempt - 1}", 1.0) if attempt else 1.0
                if measured is None:
                    continue
                target = ln.fit["window"] if ln.fit["state"] == "over" else max(ln.duration * 0.92, measured)
                if attempt and F.needs_fit(measured, ln.fit["window"], ln.duration, rules) == "":
                    continue
                f = F.factor_for(measured, target - rules.tolerance / 2, rules)
                if f is None:
                    continue
                ln.fit[f"factor{attempt}"] = round(prev * f, 4)
                reqs += _requests(job, ln, ln.en, 1, f"fit{attempt}", dlg, sr, factor=ln.fit[f"factor{attempt}"])
            _synthesize(job, w, reqs, f"speed-factor pass {attempt + 1}")
            for ln in plan:
                p = _line_dir(job, ln) / f"fit{attempt}_0.wav"
                if p.exists() and f"factor{attempt}" in ln.fit:
                    ln.fit[f"factor{attempt}_measured"] = round(F.trim(p), 3)
    # Shorter wording for overruns, at natural speed (and fitted if possible).
    reqs = []
    for ln in plan:
        if ln.fit["state"] == "over" and ln.en_short and ln.en_short != ln.en:
            reqs += _requests(job, ln, ln.en_short, 1, "short", dlg, sr)
    _synthesize(job, w, reqs, "shorter wording")
    if caps["duration_control"]:
        reqs = []
        for ln in plan:
            p = _line_dir(job, ln) / "short_0.wav"
            if p.exists():
                m = F.trim(p)
                ln.fit["short_measured"] = round(m, 3)
                f = F.factor_for(m, ln.fit["window"] - rules.tolerance / 2, rules)
                if m > ln.fit["window"] + rules.tolerance and f is not None:
                    ln.fit["short_factor"] = f
                    reqs += _requests(job, ln, ln.en_short, 1, "shortfit", dlg, sr, factor=f)
        _synthesize(job, w, reqs, "fitting shorter wording")


# finalize ---------------------------------------------------------------------------------------
def choose(options: list[tuple[str, Path]], scores: dict, window: float, rules: F.FitRules,
           cfg) -> tuple[str, Path, float] | None:
    """First option (in order of quality cost) that passed the back-check and fits
    its window, else the first that fits with an allowed time-stretch."""
    usable = [(how, p) for how, p in options
              if p.exists() and not (str(p) in scores and not _passes(scores[str(p)], cfg))]
    for how, p in usable:
        if media.duration(p) <= window + rules.tolerance:
            return how, p, 1.0
    for how, p in usable:
        r = F.stretch_ratio(media.duration(p), window, rules)
        if r is not None:
            return "stretched", p, r
    return None


def _qc_fallbacks(job: Job, lines: list[L.Line]) -> None:
    dubbed = sum(1 for l in lines if l.kind == L.SPEECH)
    fallback = [l for l in lines if l.kind == L.SPEECH and l.fallback]
    job.qc.setdefault("fit", {})["fallback"] = len(fallback)
    job.qc["fallback_rate"] = round(len(fallback) / dubbed, 4) if dubbed else 0.0
    job.qc["fallback_lines"] = [{"id": l.id, "start": l.start, "reason": l.fallback} for l in fallback][:200]


def run_finalize(job: Job) -> None:
    from . import translate as T
    lines = L.load(job.path("lines.json"))
    rules = F.FitRules.from_config(job.cfg)
    word_for_word = T.verbatim(job)
    stretch = float(job.cfg.get("fit.force_stretch_max", 1.3))
    over = float(job.cfg.get("fit.force_overrun_s", 0.8))
    roomy = F.windows([l for l in lines if l.kind == L.SPEECH], job["scenes"]["cuts"],
                      replace(rules, borrow=rules.short_borrow), job["ingest"]["duration"])
    counts = {"fits": 0, "speed_factor": 0, "shorter": 0, "stretched": 0, "fallback": 0, "padded": 0}
    free = F.free_timing(job)
    for ln in lines:
        if not ln.take:
            if ln.kind == L.SPEECH and ln.fallback:
                counts["fallback"] += 1
            continue
        d = _line_dir(job, ln)
        if free:  # the best take as it came out; the mix starts it with the original line
            final = d / "final.wav"
            F.time_stretch(Path(ln.take), final, 1.0)
            ln.take = str(final)
            ln.fit.update({"method": "natural", "tempo": 1.0, "final": round(media.duration(final), 3)})
            counts["fits"] += 1
            continue
        window = ln.fit.get("window", ln.duration + rules.borrow)
        # Candidates in order of quality cost.
        factor = [("speed_factor", d / name) for name in ("fit1_0.wav", "fit0_0.wav")]
        # The best-scoring take first, then any other take that passed the back-check.
        others = sorted((p for p, sc in ln.take_scores.items() if p != ln.take and _passes(sc, job.cfg)),
                        key=lambda p: -ln.take_scores[p].get("score", 0))
        natural = [("fits", Path(ln.take))] + [("fits", Path(p)) for p in others]
        # An underrun prefers the slower regenerated take; otherwise the natural take comes first.
        options: list[tuple[str, Path]] = factor + natural if ln.fit.get("state") == "under" else natural + factor
        whole = list(options)
        options += [("shorter", d / "shortfit_0.wav"), ("shorter", d / "short_0.wav")]
        all_scores = {**ln.take_scores, **ln.fit.get("scores", {})}
        chosen = None
        if word_for_word and ln.en_from == "script":
            # Word for word first: faster or running on into the silence after the line
            # (as a last resort does), before any word of the subtitle is left out.
            chosen = choose(whole, all_scores, window, rules, job.cfg)
            if chosen is None:
                room = max(window, roomy.get(ln.id, window))
                fast = replace(rules, stretch_max=stretch)
                # Faster within the line's time and the silence after it, or running on a little.
                chosen = (choose(whole, all_scores, room, fast, job.cfg)
                          or choose(whole, all_scores, room + over, fast, job.cfg))
                if chosen is not None:
                    late = media.duration(chosen[1]) / chosen[2] - room
                    if late > rules.tolerance:
                        ln.check_note = ln.check_note or f"runs {late:.1f} s past the original line's time"
                    window = room
                    ln.fit["window"] = window
        if chosen is None:
            chosen = choose(options, all_scores, window, rules, job.cfg)
        if chosen is None and short_text(ln.en):  # one- or two-word lines may go a little faster
            chosen = choose(options, all_scores, window, rules.for_short_text(), job.cfg)
        if chosen is None and roomy.get(ln.id, 0) > window:
            # Still too long: use the silence after the line, as for a very short line.
            window = roomy[ln.id]
            ln.fit["window"] = window
            chosen = choose(options, all_scores, window, rules, job.cfg)
        if chosen is None:
            ln.fallback = f"too long for its {window:.2f} s window even after fitting"
            ln.take = ""
            counts["fallback"] += 1
            continue
        how, p, tempo = chosen
        final = d / "final.wav"
        F.time_stretch(p, final, tempo)
        m = media.duration(final)
        if how == "fits" and m < 0.8 * ln.duration:
            counts["padded"] += 1
        counts[how] += 1
        ln.take = str(final)
        ln.fit.update({"method": how, "tempo": tempo, "final": round(m, 3), "source": p.name,
                       "timing_error": round(max(0.0, m - window), 3)})
        if p.name.startswith("short"):
            ln.en = ln.en_short
    L.save(lines, job.path("lines.json"))
    dubbed = sum(1 for l in lines if l.kind == L.SPEECH)
    job.qc["fit"] = counts
    job.qc["stretched_rate"] = round(counts["stretched"] / dubbed, 4) if dubbed else 0.0
    _qc_fallbacks(job, lines)
    job.status(", ".join(f"{k} {v}" for k, v in counts.items() if v))


def _force_fit(job: Job, lines: list[L.Line], texts: dict[str, tuple[L.Line, str]], scores: dict[str, dict],
               rules: F.FitRules) -> int:
    """Subtitles as the script: a subtitled line isn't left in the original only for
    being long. Its shortest take the back-check passed is sped up (at most
    `fit.force_stretch_max`) and may run on past its window, into the silence after it
    or the next line, by up to `fit.force_overrun_s`: saying the line matters more than
    exact timing. A line needing more than that stays in the original."""
    roomy = F.windows([l for l in lines if l.kind == L.SPEECH], job["scenes"]["cuts"],
                      replace(rules, borrow=rules.short_borrow), job["ingest"]["duration"])
    stretch = float(job.cfg.get("fit.force_stretch_max", 1.3))
    over = float(job.cfg.get("fit.force_overrun_s", 0.8))
    forced = 0
    for ln in lines:
        if not (ln.kind == L.SPEECH and ln.fallback.startswith("too long") and ln.en_from == "script"):
            continue
        cands = [(Path(p), ln.fit.get("long_en", ln.en)) for p, sc in ln.take_scores.items() if _passes(sc, job.cfg)]
        cands += [(Path(p), t) for p, (l, t) in texts.items() if l is ln and _passes(scores.get(p, {}), job.cfg)]
        cands = [(p, t) for p, t in cands if p.exists()]
        if not cands:
            continue
        p, text = min(cands, key=lambda c: media.duration(c[0]))
        window = max(roomy.get(ln.id, ln.duration), float(ln.fit.get("window", 0)))
        m = media.duration(p)
        tempo = min(stretch, max(1.0, m / window))
        late = m / tempo - window
        if late > over + rules.tolerance:
            continue
        final = _line_dir(job, ln) / "final.wav"
        F.time_stretch(p, final, tempo)
        ln.en, ln.take, ln.fallback = text, str(final), ""
        ln.fit.update({"window": window, "method": "forced", "tempo": round(tempo, 4), "source": p.name,
                       "final": round(media.duration(final), 3), "timing_error": round(max(0.0, late), 3)})
        if late > rules.tolerance:
            ln.check_note = ln.check_note or f"runs {late:.1f} s past the original line's time"
        forced += 1
    if forced:
        job.log(f"{forced} long subtitled lines sped up or run on instead of kept in the original")
    return forced


# condense ---------------------------------------------------------------------------------------
def _pace(job: Job, ln: L.Line) -> float:
    """This voice's speaking pace (syllables a second) from the line's own takes,
    never faster than the configured pace."""
    from .translate import syllables
    rate = float(job.cfg.get("translate.syllables_per_second", 4.3))
    measured = ln.fit.get("measured")
    if measured and ln.en:
        rate = min(rate, syllables(ln.en) / max(0.3, measured - 0.15))
    return max(2.5, rate)


def run_condense(job: Job) -> None:
    """Lines still in the original language get shorter English that keeps the
    meaning, sized to the time available from the voice's measured pace, then
    are voiced, back-checked and fitted again. Each round loads every model once
    (translation, voice, checks), so a round voices several takes of every option
    rather than coming back for another round. Up to `condense.rounds` rounds."""
    from . import translate as T
    lines = L.load(job.path("lines.json"))
    rules = F.FitRules.from_config(job.cfg)
    rounds = int(job.cfg.get("condense.rounds", 1))
    takes = max(1, int(job.cfg.get("condense.takes", 2)))
    if not job.cfg.get("condense.enabled", True) or rounds <= 0:
        return
    win = F.windows([l for l in lines if l.kind == L.SPEECH], job["scenes"]["cuts"], rules, job["ingest"]["duration"])
    if F.free_timing(job):  # only lines the voice garbled come here; any length will do
        win = {k: 1e6 for k in win}
    dlg, sr = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    prof = job.profile
    names = {cid: ch["name"] for cid, ch in prof.characters.items()}
    lang = job["subtitles"].get("source_language", "")
    overhead = float(job.cfg.get("condense.line_overhead_s", 0.25))
    script = T.lead_used(job) in T.SCRIPT_LEADS
    word_for_word = T.verbatim(job)
    rescued = 0
    seen_texts: dict[str, tuple[L.Line, str]] = {}
    seen_scores: dict[str, dict] = {}
    for rnd in range(rounds):
        todo = [ln for ln in lines if ln.kind == L.SPEECH and ln.fallback and ln.en]
        if not todo:
            break
        job.check()
        items = []
        for ln in todo:
            window = win.get(ln.id, ln.duration + rules.borrow)
            pace = _pace(job, ln) * (0.85 if rnd else 1.0)  # aim a little shorter on the second round
            limit = max(1, int((window * rules.stretch_max - overhead) * pace))
            ln.fit.setdefault("long_en", ln.en)
            item = {"id": ln.id, "speaker": names.get(ln.character, ln.character), "text": ln.source_text,
                    "subtitle_hint": ln.english_ref, "current": ln.en, "max_syllables": limit}
            # Came out garbled ("comm" as "con", "Eeny, meeny" as "Do you have any mini?"): reworded
            # around what the voice can't say.
            if missed_by_voice(ln):
                item["voice_could_not_say"] = missed_by_voice(ln)
            heard = list(dict.fromkeys(s_["heard"] for s_ in ln.take_scores.values() if s_.get("heard")))
            if heard and "back-check" in ln.fallback:
                item["voice_heard_as"] = heard[:3]
            item.update(T.flags(ln))
            if word_for_word and ln.en_from == "script":
                item["trim_only"] = True
            items.append(item)
        job.status(f"round {rnd + 1}: shortening {len(todo)} lines")
        by_id = {ln.id: ln for ln in todo}
        with T.LLM(job) as llm:
            convs = []
            for i in range(0, len(items), 30):
                part = items[i:i + 30]
                ctx = T._context_block(prof, {by_id[it["id"]].character for it in part}, lang, script)
                convs.append(T.condense_prompt(part, ctx, T.system_for(job)))
            short: dict[int, dict] = {}
            for reply in llm.chat_many(convs, max_tokens=2048):
                try:
                    for it in T.parse_json(reply).get("lines", []):
                        short[int(it["id"])] = {"a": str(it.get("en", "")).strip(),
                                                "b": str(it.get("shortest", "")).strip()}
                    if word_for_word:  # a subtitle line may only lose words
                        for i, opts in short.items():
                            if i in by_id and by_id[i].en_from == "script":
                                for k in ("a", "b"):
                                    if opts[k] and not T.only_deletes(opts[k], by_id[i].en):
                                        opts[k] = ""
                except Exception as exc:
                    job.log(f"condense reply unusable: {exc}")
            # Meaning check: a shortened line is used only if it still says the same thing.
            checks = [{"id": i, "text": by_id[i].source_text,
                       "full_english": by_id[i].fit.get("long_en", by_id[i].en),
                       "options": {k: v for k, v in opts.items() if v and v != by_id[i].en},
                       **{k: v for k, v in T.flags(by_id[i]).items() if k != "subtitle_hint"}}
                      for i, opts in short.items() if i in by_id]
            checks = [c for c in checks if c["options"]]
            ok: dict[int, set] = {}
            for i in range(0, len(checks), 30):
                part = checks[i:i + 30]
                ctx = T._context_block(prof, {by_id[c["id"]].character for c in part}, lang, script)
                try:
                    reply = llm.chat_many([T.meaning_check_prompt(part, ctx, T.system_for(job))], max_tokens=1024)[0]
                    for it in T.parse_json(reply).get("lines", []):
                        ok[int(it["id"])] = {str(x) for x in it.get("ok", [])}
                except Exception as exc:
                    job.log(f"meaning check unusable, rejecting those options: {exc}")
            rejected = sum(len(c["options"]) - len(ok.get(c["id"], set()) & set(c["options"])) for c in checks)
            if rejected:
                job.log(f"meaning check rejected {rejected} shortened options")
        texts: dict[str, tuple[L.Line, str]] = {}
        said: dict[str, tuple[str, str]] = {}  # take -> (text the voice read, voice clip) when not the line's own
        pronounce.load(job.cfg.models / "lexicon" / "cmudict.dict")
        with _voices(job) as w:
            caps = job.get("tts_caps", {})
            reqs = []
            for ln in todo:
                alts = [(f"cond{rnd}{k}", short.get(ln.id, {}).get(k)) for k in ("a", "b")]
                alts = [(tag, t, t, "") for tag, t in alts if t and t != ln.en and tag[-1] in ok.get(ln.id, set())]
                # Nothing shorter that keeps the meaning: try the current line again with new seeds.
                alts.append((f"cond{rnd}r", ln.en, ln.en, ""))
                # Voiced from its own short clip, it came out unclear: the character's voice bank too.
                bank = prof.refs(ln.character) if ln.character and not ln.voice else []
                if bank:
                    alts.append((f"cond{rnd}v", ln.en, ln.en, str(prof.root / bank[0]["file"])))
                # A word the voice model can't read ("Nguyen" as "Chen"): spelled as it sounds.
                spelled = respelled(ln)
                if spelled != ln.en:
                    alts.append((f"cond{rnd}p", ln.en, spelled, ""))
                for tag, text, say, voice in alts:
                    for r in _requests(job, ln, say, takes, tag, dlg, sr, seed_offset=100 * (rnd + 1), voice=voice):
                        reqs.append(r)
                        texts[r["out"]] = (ln, text)
                        said[r["out"]] = (say, voice)
            reqs.sort(key=lambda r: r["speaker_ref"])
            _synthesize(job, w, reqs, f"voicing shorter lines (round {rnd + 1})")
            if caps["duration_control"]:  # a speed factor for near misses (more for one- or two-word lines)
                reqs = []
                for out, (ln, text) in list(texts.items()):
                    p = Path(out)
                    if not p.exists():
                        continue
                    m = F.trim(p)
                    window = win.get(ln.id, ln.duration + rules.borrow)
                    r_ = rules.for_short_text() if short_text(text) else rules
                    f = F.factor_for(m, window - rules.tolerance / 2, r_)
                    if m > window + rules.tolerance and f is not None:
                        base, k = p.stem.rsplit("_", 1)
                        say, voice = said.get(out, (text, ""))
                        r = _requests(job, ln, say, 1, f"{base}f{k}", dlg, sr, factor=f,
                                      seed_offset=100 * (rnd + 1) + int(k), voice=voice)[0]
                        reqs.append(r)
                        texts[r["out"]] = (ln, text)
                        said[r["out"]] = (say, voice)
                _synthesize(job, w, reqs, "fitting shorter lines")
        items_to_score = []
        for out, (ln, text) in texts.items():
            p = Path(out)
            if p.exists():
                F.trim(p)
                items_to_score.append((ln, text, out))
        scores = score_files(job, items_to_score)
        seen_texts.update(texts)
        seen_scores.update(scores)
        # A line that ran long: shorter wording first (a, then the shortest b), then the
        # current line retried (as it was, in the voice bank's voice, then spelled as it
        # sounds). A line that had the time but came out unclear keeps its wording if a
        # retry says it clearly. Each option's natural takes before its speed-factor
        # versions (cond0a_1, cond0af1_0).
        prefix = f"cond{rnd}"

        def rank(p: Path, order: str) -> tuple:
            base, k = p.stem.rsplit("_", 1)
            tag = base[len(prefix):]
            return order.index(tag[0]), len(tag) > 1, int(tag[2:] or 0) if len(tag) > 1 else int(k)

        for ln in todo:
            d = _line_dir(job, ln)
            window = win.get(ln.id, ln.duration + rules.borrow)
            order = "abrvp" if ln.fit.get("state") == "over" else "rvpab"
            mine = sorted((Path(o) for o, (l, _) in texts.items() if l is ln and Path(o).name.startswith(prefix)),
                          key=lambda p: rank(p, order))
            opts = [("condensed" if p.name[len(prefix)] in "ab" else "retried", p) for p in mine]
            # Looser fitting only applies to one- or two-word lines.
            chosen = choose(opts, scores, window, rules, job.cfg)
            if chosen is None:
                short_opts = [(h, p) for h, p in opts if short_text(texts[str(p)][1])]
                if short_text(ln.en):  # earlier takes of the same short line count too
                    for p_ in ln.take_scores:
                        short_opts.append(("retried", Path(p_)))
                        texts.setdefault(p_, (ln, ln.en))
                chosen = choose(short_opts, {**ln.take_scores, **scores}, window, rules.for_short_text(), job.cfg)
            if chosen is None:
                continue
            how, p, tempo = chosen
            final = d / "final.wav"
            F.time_stretch(p, final, tempo)
            ln.en = texts[str(p)][1]
            ln.take = str(final)
            ln.fallback = ""
            if said.get(str(p), ("", ""))[1]:
                ln.voice, ln.voice_note = said[str(p)][1], "voice bank (its own clip came out unclear)"
            ln.fit.update({"window": window, "method": "condensed", "tempo": tempo, "source": p.name,
                           "final": round(media.duration(final), 3), "round": rnd + 1})
            rescued += 1
        L.save(lines, job.path("lines.json"))
    counts = job.qc.setdefault("fit", {})
    counts["condensed"] = rescued
    if script:
        counts["forced"] = _force_fit(job, lines, seen_texts, seen_scores, rules)
    _qc_fallbacks(job, lines)
    L.save(lines, job.path("lines.json"))
    job.status(f"shortened {rescued} lines; {counts['fallback']} still in the original language")
