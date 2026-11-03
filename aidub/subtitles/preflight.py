"""Before any work starts on a file: is there an English subtitle to dub from?

A dub from English subtitles is what this app does best. With none, the file waits for the
user without spending any time on it: they can have Bazarr search, choose a subtitle in
another language to translate, or allow a dub from the speech. This needs no model: the
file's own tracks, the files next to it, and Bazarr. Whether an English subtitle found here
is good enough can only be told once the speech has been analysed.
"""

from __future__ import annotations

from pathlib import Path

from .. import media
from .bazarr import Bazarr
from .sources import sidecars

UNTAGGED = ("", "und")  # might be English: the subtitle step reads what it says


def _english(lang: str) -> bool:
    return lang == "en" or lang in UNTAGGED


def _english_next_to(path: Path) -> bool:
    return any(_english(lang) and not forced for _p, _f, lang, forced in sidecars(path))


def choices(path: Path, pr: media.Probe | None = None) -> list[dict]:
    """Every subtitle the file has that the user could choose to dub from."""
    pr = pr or media.probe(path)
    out = [{"id": f"embedded:{s.index}", "language": s.language, "title": s.title, "picture": not s.is_text_sub, "where": "in the file"}
           for s in pr.subtitles if not s.forced and not s.is_aidub]
    out += [{"id": f"sidecar:{p.name}", "language": lang, "title": p.name, "picture": fmt == "pgs", "where": "next to the file"}
            for p, fmt, lang, forced in sidecars(path) if not forced]
    return out


def _bazarr(cfg) -> tuple[Bazarr | None, str]:
    pc = cfg.get("subtitles.providers.bazarr") or {}
    if not pc.get("enabled"):
        return None, "Bazarr isn't set up"
    try:
        return Bazarr(pc.get("url", ""), pc.get("api_key", ""), pc.get("path_map")), ""
    except ValueError as exc:
        return None, f"Bazarr isn't set up ({exc})"


def search(cfg, path: Path, language: str) -> str:
    """Ask Bazarr for a subtitle in this language; empty if it was asked, else why it couldn't be."""
    bz, why = _bazarr(cfg)
    if bz is None:
        return why
    try:
        rec = bz.find(str(path))
        if not rec:
            return "Bazarr doesn't manage this file"
        bz.search(rec, language)
        return ""
    except Exception as exc:
        return f"Bazarr couldn't be asked ({type(exc).__name__})"
    finally:
        bz.close()


def missing_english(cfg, path: Path, pr: media.Probe | None = None) -> str:
    """Why the file has no English subtitle to try; empty if it has one."""
    pr = pr or media.probe(path)
    if any(_english(s.language) and not s.forced and not s.is_aidub for s in pr.subtitles) or _english_next_to(path):
        return ""
    here = "No English subtitles in the file or next to it"
    pc = cfg.get("subtitles.providers.bazarr") or {}
    if not pc.get("enabled"):
        return f"{here}, and Bazarr isn't set up"
    why = search(cfg, path, "en") if pc.get("search_missing", True) else ""
    if _english_next_to(path):
        return ""
    return f"{here}, and {why[0].lower() + why[1:] if why else 'Bazarr has none'}"
