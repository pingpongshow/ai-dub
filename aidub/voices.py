"""The generic voice library for minor speakers.

Made once, locally, with a voice-design model from text descriptions, so no
real person's voice is used. Stored in <appdata>/voices/generic/.
"""

from __future__ import annotations

import json
from typing import Callable

from .config import Config
from .workers import Worker

GENDERS = ["female", "male"]
AGES = {
    "child": "a child of about ten",
    "young": "a young person in their late teens or early twenties",
    "adult": "an adult in their thirties or forties",
    "senior": "an older person in their sixties or seventies",
}
TIMBRES = ["warm, clear and natural", "slightly husky, relaxed and natural"]
SAMPLE_TEXT = ("I didn't expect to see you here today. Honestly, it has been a long week, "
               "but I think things are finally starting to look up.")


def design_library(cfg: Config, *, force: bool = False, log: Callable[[str], None] = print) -> None:
    root = cfg.appdata / "voices" / "generic"
    root.mkdir(parents=True, exist_ok=True)
    index = root / "voices.json"
    if index.exists() and not force:
        log(f"generic voices already exist ({index}); use --force to redo")
        return
    entries = []
    name = cfg.get("stages.voicedesign.worker", "voicedesign")
    with Worker(cfg, name, log=log) as w:
        w.call("load", {"model": cfg.get("stages.voicedesign.model")})
        for g in GENDERS:
            for age, age_text in AGES.items():
                for k, timbre in enumerate(TIMBRES):
                    who = f"{age_text}, {'girl' if g == 'female' and age == 'child' else 'boy' if age == 'child' else g}"
                    desc = (f"A {timbre} English voice of {who}, speaking at a natural conversational pace "
                            f"with a neutral American accent.")
                    fname = f"{g}-{age}-{k + 1}.wav"
                    w.call("design", {"description": desc, "text": SAMPLE_TEXT, "out": str(root / fname)})
                    entries.append({"file": fname, "gender": g, "age": age, "name": f"{g} {age} {k + 1}",
                                    "description": desc})
                    log(f"designed {fname}")
    index.write_text(json.dumps(entries, indent=1))
