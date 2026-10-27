"""Step 0: probe the file, apply the skip rules, extract the audio, read chapters."""

from __future__ import annotations

from . import media, originals
from .job import Job, Skipped

STEP = "ingest"


def english_dialogue_tracks(pr: media.Probe) -> list[media.Stream]:
    return [s for s in pr.audio if s.language == "en" and media.is_dialogue_audio(s)]


def classify(pr: media.Probe) -> dict:
    """What the skip rules say about a file, without changing anything."""
    eng = english_dialogue_tracks(pr)
    ours = [s for s in eng if s.is_aidub]
    theirs = [s for s in eng if not s.is_aidub]
    foreign = [s for s in pr.audio if s.language != "en" and media.is_dialogue_audio(s) and not s.is_aidub]
    return {
        "english_audio": bool(theirs),
        "aidub_present": bool(ours),
        "foreign_tracks": [{"index": s.index, "language": s.language, "title": s.title,
                            "channels": s.channels, "default": s.default} for s in foreign],
    }


def pick_audio(pr: media.Probe, wanted: int | None) -> media.Stream:
    candidates = [s for s in pr.audio if not s.is_aidub and s.language != "en" and media.is_dialogue_audio(s)]
    if wanted is not None:
        for s in pr.audio:
            if s.index == int(wanted):
                return s
        raise RuntimeError(f"audio stream {wanted} not found")
    if not candidates:
        raise Skipped("no foreign-language audio track")
    for s in candidates:
        if s.default:
            return s
    return candidates[0]


def song_ranges(pr: media.Probe, names: list[str]) -> list[list[float]]:
    out = []
    for ch in pr.chapters:
        title = (ch.get("tags") or {}).get("title", "").strip().lower()
        if not title:
            continue
        words = title.replace("-", " ").replace("_", " ").split()
        if any(n == title or n in words or title.startswith(n + " ") for n in names):
            out.append([float(ch["start_time"]), float(ch["end_time"])])
    return out


def run(job: Job) -> None:
    pr = media.probe(job.src)
    info = classify(pr)
    redo = bool(job.options.get("redo"))
    source = job.src
    if info["english_audio"] and job.cfg.get("ingest.skip_english_audio", True) and not redo:
        raise Skipped("English audio exists")
    if info["aidub_present"] and not redo:
        raise Skipped("already dubbed by Ai Dub (queue it again with 'redo' to rebuild)")
    if redo:
        aside = originals.aside_name(job.src)
        if aside.exists():
            source = aside
            job.log(f"redo: rebuilding from the set-aside original {aside.name}")
            pr = media.probe(source)

    stream = pick_audio(pr, job.options.get("audio_stream"))
    preview = float(job.options.get("preview_s") or 0)
    dur = min(pr.duration, preview) if preview else pr.duration

    wav = job.path("audio", "source.wav")
    job.status("extracting audio")
    media.extract_audio(source, stream.index, wav, duration=preview or None)

    lang = job.options.get("language") or stream.language or ""
    job.set("ingest", {
        "source": str(source),
        "container": pr.container,
        "duration": dur,
        "full_duration": pr.duration,
        "preview_s": preview,
        "audio_index": stream.index,
        "audio_codec": stream.codec,
        "channels": stream.channels,
        "layout": stream.layout,
        "language": media.lang_code(lang),
        "redo": redo,
        "song_ranges": song_ranges(pr, job.cfg.get("ingest.song_chapters", [])),
        "has_video": any(s.type == "video" and s.codec not in {"mjpeg", "png"} for s in pr.streams),
        "subtitle_languages": sorted({s.language for s in pr.subtitles if s.language and not s.is_aidub}),
    })
    job.status(f"audio track {stream.index} ({stream.language or 'unknown language'}, "
               f"{stream.channels} ch), {dur / 60:.1f} min")
