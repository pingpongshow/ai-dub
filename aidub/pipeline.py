"""The ordered steps of a dub, with checkpoints so an interrupted job resumes
from its first unfinished step."""

from __future__ import annotations

import time
from typing import Callable

from . import __version__, align, bwe, deliver, diarize, events, ingest, mix, recognize, scenes, separate, speak, translate, voicebank
from .job import Cancelled, Job, Skipped
from .subtitles import step as subtitles

STEPS: list[tuple[str, str, Callable[[Job], None]]] = [
    ("ingest", "Ingest", ingest.run),
    ("separate", "Separate dialogue", separate.run),
    ("scenes", "Find scenes", scenes.run),
    ("diarize", "Who speaks when", diarize.run),
    ("recognize", "Recognise speech", recognize.run),
    ("events", "Sound events", events.run),
    ("subtitles", "Subtitles", subtitles.run),
    ("align", "Align lines", align.run),
    ("voicebank", "Voice bank", voicebank.run),
    ("translate", "English script", translate.run),
    ("speak", "Speak", speak.run_speak),
    ("check", "Check takes", speak.run_check),
    ("finalize", "Choose takes", speak.run_finalize),
    ("condense", "Shorten long lines", speak.run_condense),
    ("bwe", "Sound quality", bwe.run),
    ("mix", "Mix", mix.run),
    ("deliver", "Deliver", deliver.run),
]
STEP_NAMES = [s for s, _, _ in STEPS]
TITLES = {s: t for s, t, _ in STEPS}


def run(job: Job, *, until: str | None = None) -> dict:
    """Run the remaining steps. Returns the job record. Raises Skipped/Cancelled/other errors."""
    for i, (name, title, fn) in enumerate(STEPS):
        if job.is_done(name):
            if until and name == until:
                break  # already done up to the requested step
            continue
        job.check()
        job.step = name
        job.status(title, progress=0.0, step_index=i, steps=len(STEPS))
        t0 = time.time()
        fn(job)
        job.check()  # a step cut short by a cancel or shutdown is not finished
        job.mark_done(name, time.time() - t0)
        if until and name == until:
            break
    return record(job)


def record(job: Job) -> dict:
    return {
        "version": __version__,
        "models": job.models,
        "qc": job.qc,
        "timings": job.state.get("timings", {}),
        "delivered": job.get("delivered", {}),
        "text_source": deliver.text_source(job) if job.get("subtitles") else "",
        "profile": job.profile.slug if job.profile else "",
        "source_language": (job.get("subtitles") or {}).get("source_language", ""),
    }


__all__ = ["STEPS", "STEP_NAMES", "TITLES", "run", "record", "Skipped", "Cancelled"]
