"""What audio and subtitle tracks each video has, for the queue's file browser and its
language filter. ffprobe results are kept in appdata/tracks.json by path, size and
modification time, so each file is read once until it changes."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import media
from .subtitles.sources import BITMAP_EXT, sidecars

READERS = 8          # ffprobe runs at once; it mostly waits on the disk
MAX_ENTRIES = 20000


def language_name(code: str) -> str:
    if not code:
        return "Unknown"
    if code in media.LANGUAGES:
        return media.LANGUAGES[code][1].title()
    return code


def describe(pr: media.Probe) -> dict:
    audio = [{"lang": s.language, "title": s.title, "aidub": s.is_aidub,
              "commentary": not media.is_dialogue_audio(s)} for s in pr.audio]
    subs = [{"lang": s.language, "title": s.title, "forced": s.forced, "picture": not s.is_text_sub,
             "aidub": s.is_aidub} for s in pr.subtitles]
    return {"audio": audio, "subs": subs}


class Tracks:
    def __init__(self, appdata: Path):
        self.file = appdata / "tracks.json"
        self.lock = threading.Lock()
        try:
            self.cache: dict = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.cache = {}
        self.dirty = False

    def _read(self, path: Path) -> tuple[str, dict]:
        key = str(path)
        try:
            st = path.stat()
        except OSError as exc:
            return key, {"error": str(exc)}
        with self.lock:
            hit = self.cache.get(key)
        if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime:
            info = hit["info"]
        else:
            try:
                info = describe(media.probe(path))
            except Exception as exc:
                return key, {"error": str(exc)[:200]}
            with self.lock:
                self.cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "info": info}
                self.dirty = True
        # Subtitle files next to the video come and go without the video changing.
        ext = [{"lang": lang, "forced": forced, "picture": p.suffix.lower() in BITMAP_EXT, "file": p.name}
               for p, _fmt, lang, forced in sidecars(path)]
        return key, {**info, "external": ext}

    def many(self, paths: list[Path]) -> dict[str, dict]:
        if not paths:
            return {}
        with ThreadPoolExecutor(max_workers=min(READERS, len(paths))) as pool:
            out = dict(pool.map(self._read, paths))
        self._save()
        return out

    def _save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            self.dirty = False
            if len(self.cache) > MAX_ENTRIES:  # forget files that are gone, then the oldest
                self.cache = {k: v for k, v in self.cache.items() if Path(k).exists()}
                self.cache = dict(list(self.cache.items())[-MAX_ENTRIES:])
            tmp = self.file.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(self.cache, ensure_ascii=False))
                tmp.replace(self.file)
            except OSError:
                pass  # only a cache


def matches(info: dict, want: set[str], hide: set[str]) -> bool:
    """want: show only files with dialogue audio in one of these languages ("und" = untagged).
    hide: leave out files whose own dialogue audio includes one of these. Tracks Ai Dub added
    don't count either way: a dubbed file shows as dubbed instead."""
    if "error" in info:
        return not want  # unreadable: shown unless a language is required
    langs = {a["lang"] or "und" for a in info.get("audio", []) if not a["aidub"] and not a["commentary"]}
    if want and not langs & want:
        return False
    return not langs & hide
