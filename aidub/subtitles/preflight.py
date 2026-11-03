"""Before any work starts on a file: are there subtitles to dub from, in any language?

A dub from the speech is much worse than one from subtitles, so a file with none waits for
the user's say-so without spending any time on it. This needs no model: the file's own
tracks, the files next to it, and Bazarr (which is asked to search). Whether a subtitle
found here is good enough can only be told once the speech has been analysed.
"""

from __future__ import annotations

from pathlib import Path

from .. import media
from .bazarr import Bazarr
from .sources import sidecars

def _next_to(path: Path) -> bool:
    return any(not forced for _p, _f, _lang, forced in sidecars(path))


def missing_subtitles(cfg, path: Path, pr: media.Probe | None = None) -> str:
    """Why the file has no subtitles to dub from, in any language; empty if it has some."""
    pr = pr or media.probe(path)
    if any(not s.forced and not s.is_aidub for s in pr.subtitles) or _next_to(path):
        return ""
    here = "No subtitles in any language in the file or next to it"
    pc = cfg.get("subtitles.providers.bazarr") or {}
    if not pc.get("enabled"):
        return f"{here}, and Bazarr isn't set up"
    try:
        bz = Bazarr(pc.get("url", ""), pc.get("api_key", ""), pc.get("path_map"))
    except ValueError as exc:
        return f"{here}, and Bazarr isn't set up ({exc})"
    try:
        rec = bz.find(str(path))
        if not rec:
            return f"{here}, and Bazarr doesn't manage this file"
        if pc.get("search_missing", True):
            # English, then the language spoken.
            spoken = next((s.language for s in pr.audio if s.language and s.language != "en" and not s.is_aidub), "")
            for lang in ("en", spoken):
                if lang and not _next_to(path):
                    bz.search(rec, lang)
        return "" if _next_to(path) else f"{here}, and Bazarr has none"
    except Exception as exc:
        return f"{here}, and Bazarr couldn't be asked ({type(exc).__name__})"
    finally:
        bz.close()
