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

The dictionary also gives a word the voice model can't read ("Nguyen" comes out
as "Chen") a spelling as it sounds ("noo yen").
"""

from __future__ import annotations

import re
from pathlib import Path

LEXICON: dict[str, list[str]] = {}   # word -> its pronunciations, the usual one first
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
        pron = " ".join(re.sub(r"\d", "", p) for p in parts[1:])
        if pron not in LEXICON.setdefault(word, []):
            LEXICON[word].append(pron)
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
    if not w:  # only silent letters ("gh")
        return ""
    first = "A" if w[0] in "aeiouy" else ""
    rest = re.sub(r"[aeiouyhw]", "", w[1:] if first else w)
    return first + re.sub(r"(.)\1+", r"\1", rest)


def same_sound(word: str, heard: str) -> bool:
    """`heard` (recognition's spelling) sounds like `word` (the script's)."""
    if not LEXICON:
        return False
    a, b = LEXICON.get(word), LEXICON.get(heard)
    if a and b:
        return bool(set(a) & set(b))
    s = skeleton(word)
    return len(s) >= 2 and s == skeleton(heard)


VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def _shape(pron: str) -> tuple:
    return tuple("V" if p in VOWELS else p for p in pron.split())


def split_sound(word: str, x: str, y: str) -> bool:
    """`word` heard as two words with its consonants and as many vowels ("Nguyen",
    said from its spelling "noo yen", heard as "No yen" or "New Yin"). Only for a
    word recognition split in two: one word with another vowel is another word
    ("shady", "shoddy")."""
    a, bx, by = LEXICON.get(word), LEXICON.get(x), LEXICON.get(y)
    if not (a and bx and by):
        return False
    shapes = {_shape(p) for p in a}
    return any(_shape(f"{p} {q}") in shapes for p in bx for q in by)


def heard_split(word: str, x: str, y: str) -> bool:
    """Two short heard words are pieces of `word` ("toupee" heard as "two P")."""
    return len(x) <= 3 and len(y) <= 3 and (same_sound(word, x + y) or split_sound(word, x, y))


def _one_apart(x: str, y: str) -> bool:
    if abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        return sum(p != q for p, q in zip(x, y)) <= 1
    longer, shorter = (x, y) if len(x) > len(y) else (y, x)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def made_up_match(word: str, heard: str) -> bool:
    """A word the dictionary doesn't know (a rhyme's "eeny", an invented name) heard
    with one sound off ("Ifny"): recognition has to guess its spelling from a voice
    that never quite matches. Dictionary words get no such slack."""
    if not LEXICON or word in LEXICON:
        return False
    x, y = skeleton(word), skeleton(heard)
    # Starting with a vowel or not must agree ("eeny" isn't "mini").
    return min(len(x), len(y)) >= 2 and (x[0] == "A") == (y[0] == "A") and _one_apart(x, y)


def heard_as(word: str, heard: list[str]) -> bool:
    """The word, or something sounding the same, is among the heard words; also as
    short pieces recognition split it into."""
    if any(same_sound(word, h) or made_up_match(word, h) for h in heard):
        return True
    return any(heard_split(word, x, y) for x, y in zip(heard, heard[1:]))


SPELL = {"AA": "ah", "AE": "a", "AH": "uh", "AO": "aw", "AW": "ow", "AY": "eye", "EH": "e", "ER": "er", "EY": "ay",
         "IH": "i", "IY": "ee", "OW": "oh", "OY": "oy", "UH": "u", "UW": "oo", "B": "b", "CH": "ch", "D": "d",
         "DH": "th", "F": "f", "G": "g", "HH": "h", "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
         "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th", "V": "v", "W": "w", "Y": "y", "Z": "z",
         "ZH": "zh"}


def respell(word: str) -> str | None:
    """The word spelled as it sounds, a syllable at a time ("nguyen": "noo yen"), when
    its own spelling doesn't show its sound; None when it does or it isn't known."""
    prons = LEXICON.get(word.lower())
    if not prons:
        return None
    ps = prons[0].split()
    chunks: list[list[str]] = [[]]
    for i, p in enumerate(ps):
        # A new syllable at the consonant before a vowel, once this one has its vowel.
        if p not in VOWELS and i + 1 < len(ps) and ps[i + 1] in VOWELS and any(q in VOWELS for q in chunks[-1]):
            chunks.append([])
        chunks[-1].append(p)
    out = " ".join("".join(SPELL[p] for p in c) for c in chunks)
    return out if skeleton(out) != skeleton(word) else None


def near(a: str, b: str) -> bool:
    """Skeletons at most one sound apart, for longer made-up compounds ("ten-pent" heard as "tenpence")."""
    x, y = skeleton(a), skeleton(b)
    return min(len(x), len(y)) >= 5 and _one_apart(x, y)


def only_sounds(words: list[str]) -> bool:
    """A line that is only sounds ("Shh.", "Mm-hmm.")."""
    return bool(words) and all(w in NON_WORDS for w in words)
