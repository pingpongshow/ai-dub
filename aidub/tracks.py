"""What audio and subtitle tracks each video has, for the queue's file browser and its
language filter. ffprobe results are kept in appdata/tracks.json by path, size and
modification time, so each file is read once until it changes."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from pathlib import Path

from . import media
from .subtitles.sources import BITMAP_EXT, sidecars

READERS = 8          # ffprobe runs at once; it mostly waits on the disk
MAX_ENTRIES = 50000


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
    """Cached track lists. Files not read yet are read in the background (READERS at a
    time), so a big folder lists at once and fills in as it goes."""

    def __init__(self, appdata: Path):
        self.file = appdata / "tracks.json"
        self.lock = threading.Lock()
        try:
            self.cache: dict = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.cache = {}
        self.dirty = False
        self.pool = ThreadPoolExecutor(max_workers=READERS, thread_name_prefix="tracks")
        self.reading: dict[str, Future] = {}

    def _cached(self, path: Path) -> dict | None:
        """The cached info while the file is unchanged; an error dict if it can't be read."""
        try:
            st = path.stat()
        except OSError as exc:
            return {"error": str(exc)}
        with self.lock:
            hit = self.cache.get(str(path))
        if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime:
            return hit["info"]
        return None

    def _probe(self, path: Path) -> dict:
        try:
            st = path.stat()
            info = describe(media.probe(path))
        except Exception as exc:
            return {"error": str(exc)[:200]}
        with self.lock:
            self.cache[str(path)] = {"size": st.st_size, "mtime": st.st_mtime, "info": info}
            self.dirty = True
        return info

    def _done(self, key: str) -> None:
        with self.lock:
            self.reading.pop(key, None)
            idle = not self.reading
        if idle:
            self._save()

    @staticmethod
    def _external(path: Path) -> list[dict]:
        # Subtitle files next to the video come and go without the video changing.
        return [{"lang": lang, "forced": forced, "picture": p.suffix.lower() in BITMAP_EXT, "file": p.name}
                for p, _fmt, lang, forced in sidecars(path)]

    def lookup(self, paths: list[Path], wait: float | None = None) -> tuple[dict[str, dict], int]:
        """(path -> info, how many are still being read). Waits up to `wait` seconds for
        files not read yet (None: until all are)."""
        out: dict[str, dict] = {}
        todo: dict[str, Future] = {}
        for p in paths:
            key = str(p)
            info = self._cached(p)
            if info is not None:
                out[key] = info if "error" in info else {**info, "external": self._external(p)}
                continue
            with self.lock:
                fut = self.reading.get(key)
                new = fut is None
                if new:
                    fut = self.pool.submit(self._probe, p)
                    self.reading[key] = fut
            if new:  # outside the lock: a finished future runs the callback right here
                fut.add_done_callback(lambda _f, k=key: self._done(k))
            todo[key] = fut
        if todo:
            futures_wait(list(todo.values()), timeout=wait)
        for key, fut in todo.items():
            if fut.done():
                info = fut.result()
                out[key] = info if "error" in info else {**info, "external": self._external(Path(key))}
        self._save()
        return out, len(todo) - sum(f.done() for f in todo.values())

    def many(self, paths: list[Path]) -> dict[str, dict]:
        return self.lookup(paths)[0]

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


def videos_in(folder: Path, depth: int = 2) -> list[Path]:
    """Videos in a folder and the folders below it (a film's folder, a show's seasons)."""
    out = []
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    except OSError:
        return out
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                if depth > 1:
                    out += videos_in(Path(e.path), depth - 1)
            elif Path(e.name).suffix.lower() in media.VIDEO_EXT:
                out.append(Path(e.path))
        except OSError:
            continue
    return out


def matches(info: dict, want: set[str], hide: set[str]) -> bool:
    """want: show only files with their own dialogue audio in one of these languages ("und" =
    untagged). hide: leave out files with audio in one of these, a dub Ai Dub added included.
    Commentary doesn't count either way."""
    if "error" in info:
        return not want  # unreadable: shown unless a language is required
    audio = [a for a in info.get("audio", []) if not a["commentary"]]
    if want and not {a["lang"] or "und" for a in audio if not a["aidub"]} & want:
        return False
    return not {a["lang"] or "und" for a in audio} & hide
