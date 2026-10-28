"""The per-line table: the central data structure of a job.

Every step after alignment works line by line, and each line's audio is its own
file until the mix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPEECH, ENGLISH, WORDLESS, SONG = "speech", "english", "wordless", "song"
KEPT_KINDS = {ENGLISH, WORDLESS, SONG}


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Line:
    id: int
    start: float
    end: float
    speaker: str = ""                 # diarization cluster id ("S3")
    character: str = ""               # profile character id
    name_hint: str = ""               # character name from the subtitle, when it gives one
    kind: str = SPEECH
    source_text: str = ""
    source_lang: str = ""
    text_origin: str = "asr"          # "asr", "subtitle:<label>", "subtitle-en:<label>"
    english_ref: str = ""             # English subtitle for this line, when it agrees with the speech
    english_context: str = ""         # nearby subtitle text that may not match this line exactly (hint only)
    words: list[Word] = field(default_factory=list)
    heard_chars: int = -1              # characters speech recognition heard in this span (-1: not measured)
    scene: int = 0
    overlap: bool = False
    emotion: str = ""                 # SenseVoice emotion tag
    events: list[str] = field(default_factory=list)
    loudness_db: float = -30.0
    pan: float = 0.0                  # -1 left .. +1 right
    en: str = ""                      # dubbed English line
    en_short: str = ""                # shorter alternative for overruns
    voice: str = ""                   # reference clip used
    voice_note: str = ""              # minor-speaker substitution, if any
    take: str = ""                    # chosen fitted audio file
    take_scores: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)
    fallback: str = ""                # why the original audio was kept, if it was

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def dubbed(self) -> bool:
        return self.kind == SPEECH and not self.fallback


def save(lines: list[Line], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(l) for l in lines], ensure_ascii=False, indent=1))
    tmp.replace(path)


def load(path: Path) -> list[Line]:
    out = []
    for d in json.loads(path.read_text()):
        d["words"] = [Word(**w) for w in d.get("words", [])]
        out.append(Line(**d))
    return out
