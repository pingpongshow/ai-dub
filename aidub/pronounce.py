"""Do two spellings of a word sound the same? For the back-check, which reads a
take back with speech recognition and compares it with the script: recognition
spells what it hears its own way ("brakes" as "breaks", the name "Loid" as
"Lloyd", a made-up "dalc" as "dalk", "toupee" as "two P"), which is not a
mispronunciation.

Words in the CMU Pronouncing Dictionary must sound the same (so "shady" said
as "shoddy" is still a miss). A word it doesn't know (names, made-up words, odd
spellings) matches on its consonants, as the voice model has to guess those
too. Without the dictionary (models/lexicon/cmudict.dict, fetched with the
speech-recognition model) only exact spellings count, as before.
"""

from __future__ import annotations

import re
from pathlib import Path

LEXICON: dict[str, set[str]] = {}
_loaded: Path | None = None

# Sounds rather than words: recognition can't spell them, so a take isn't checked for them.
NON_WORDS = {"shh", "shhh", "sh", "mm", "mmm", "hmm", "hm", "mhm", "uh", "huh", "um", "umm", "ah", "aah", "oh",
             "ooh", "eh", "ugh", "psst", "tsk", "ha", "haha", "heh", "hah", "whoa", "aw", "aww", "oof", "phew"}


def load(path: Path) -> bool:
    """Read the dictionary once; True if it is available."""
    global _loaded
    if _loaded == path:
        return bool(LEXICON)
    _loaded = path
    LEXICON.clear()
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        LEXICON.setdefault(word, set()).add(" ".join(re.sub(r"\d", "", p) for p in parts[1:]))
    return bool(LEXICON)


def skeleton(word: str) -> str:
    """The consonants as they sound, voiced and unvoiced pairs merged, with any first
    vowel kept as one mark ("Loid" and "Lloyd": "lt"; "eeny" and "any": "an")."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    w = w.replace("x", "ks")
    for a, b in (("ph", "f"), ("ck", "k"), ("gh", ""), ("sch", "sk"), ("sh", "S"), ("ch", "S"), ("th", "0")):
        w = w.replace(a, b)
    w = re.sub(r"mb$", "m", w)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.translate(str.maketrans("cqzdbgv", "kkstpkf"))
    first = "A" if w[0] in "aeiouy" else ""
    rest = re.sub(r"[aeiouyhw]", "", w[1:] if first else w)
    return first + re.sub(r"(.)\1+", r"\1", rest)


def same_sound(word: str, heard: str) -> bool:
    """`heard` (recognition's spelling) sounds like `word` (the script's)."""
    if not LEXICON:
        return False
    a, b = LEXICON.get(word), LEXICON.get(heard)
    if a and b:
        return bool(a & b)
    s = skeleton(word)
    return len(s) >= 2 and s == skeleton(heard)


def heard_as(word: str, heard: list[str]) -> bool:
    """The word, or something sounding the same, is among the heard words; also as
    short pieces recognition split it into ("toupee" heard as "two P")."""
    if any(same_sound(word, h) for h in heard):
        return True
    return any(len(x) <= 3 and len(y) <= 3 and same_sound(word, x + y) for x, y in zip(heard, heard[1:]))


def near(a: str, b: str) -> bool:
    """Skeletons at most one sound apart, for longer made-up compounds ("ten-pent" heard as "tenpence")."""
    x, y = skeleton(a), skeleton(b)
    if min(len(x), len(y)) < 5 or abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        return sum(p != q for p, q in zip(x, y)) <= 1
    longer, shorter = (x, y) if len(x) > len(y) else (y, x)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def only_sounds(words: list[str]) -> bool:
    """A line that is only sounds ("Shh.", "Mm-hmm.")."""
    return bool(words) and all(w in NON_WORDS for w in words)
