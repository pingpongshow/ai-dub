from __future__ import annotations

from dataclasses import asdict, dataclass, field

DIALOGUE, SIGN, SONG, HI = "dialogue", "sign", "song", "hi"


@dataclass
class Cue:
    start: float
    end: float
    text: str                 # cleaned dialogue text (empty for non-dialogue)
    raw: str = ""             # text as in the file
    name: str = ""            # character name from ASS, when present
    style: str = ""
    kind: str = DIALOGUE

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    id: str
    label: str
    origin: str               # embedded | sidecar | bazarr | ocr | provider:<name>
    language: str
    format: str               # srt | ass | vtt
    path: str                 # copy in the job's working folder
    human: bool = True        # False for AI-made text
    in_file: bool = False     # the file (or a sidecar) already has this track
    forced: bool = False
    cues: list[Cue] = field(default_factory=list)
    offset: float = 0.0
    scale: float = 1.0
    score: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("cues")
        return d
