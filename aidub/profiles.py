"""Per-show profiles: characters and their voice references, glossary, settings.

Stored under <appdata>/profiles/<slug>/:
  profile.json      characters, glossary, settings, episodes
  refs/<char>/*.flac voice reference clips
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
from pathlib import Path

from . import speakers

_lock = threading.RLock()
SEASON = re.compile(r"^(season|series|s)\s*\d+$|^specials?$", re.I)
EPISODE_TAIL = re.compile(r"\s*[-–]\s*S\d+E\d+.*$|\s*S\d+E\d+.*$|\s*-\s*\d{1,4}\s*(\[.*)?$", re.I)


def slugify(name: str) -> str:
    t = unicodedata.normalize("NFKD", name)
    t = re.sub(r"[^\w\s()-]", "", t, flags=re.U).strip().lower()
    t = re.sub(r"[\s_]+", "-", t)
    return t[:80] or "untitled"


def guess_name(path: Path, media_roots: list[Path] | None = None) -> str:
    """Show name from Sonarr/Radarr-style paths: Show/Season 01/Show - S01E07.mkv,
    or Movie (Year)/Movie (Year).mkv. Falls back to the file name."""
    parent = path.parent
    roots = {p.resolve() for p in (media_roots or [])}
    if SEASON.match(parent.name):
        parent = parent.parent
    if parent.resolve() in roots or not parent.name:
        stem = EPISODE_TAIL.sub("", path.stem)
        return stem.strip() or path.stem
    return parent.name


class Profile:
    def __init__(self, root: Path, data: dict):
        self.root = root
        self.data = data

    # Basic -------------------------------------------------------------------------
    @property
    def slug(self) -> str:
        return self.root.name

    @property
    def name(self) -> str:
        return self.data.get("name", self.slug)

    @property
    def characters(self) -> dict[str, dict]:
        return self.data.setdefault("characters", {})

    @property
    def glossary(self) -> dict[str, dict]:
        return self.data.setdefault("glossary", {})

    @property
    def settings(self) -> dict:
        return self.data.setdefault("settings", {})

    def save(self) -> None:
        with _lock:
            self.data["updated"] = time.time()
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / "profile.json.tmp"
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))
            tmp.replace(self.root / "profile.json")

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    # Characters --------------------------------------------------------------------
    def new_character(self, name: str = "", **fields) -> str:
        n = 1
        while f"c{n}" in self.characters:
            n += 1
        cid = f"c{n}"
        self.characters[cid] = {"name": name or f"Character {n}", "embedding": [], "embedding_weight": 0.0,
                                "gender": "", "age": "", "refs": [], "voice_of": None, "created": time.time(),
                                **fields}
        return cid

    def match(self, embedding: list[float], threshold: float, exclude: set[str] | None = None) -> tuple[str | None, float]:
        best, score = None, -1.0
        for cid, ch in self.characters.items():
            if exclude and cid in exclude:
                continue
            vec = ch.get("embedding") or (self.characters.get(ch.get("voice_of") or "", {}).get("embedding"))
            if not vec or not embedding:
                continue
            s = speakers.cosine(vec, embedding)
            if s > score:
                best, score = cid, s
        return (best, score) if score >= threshold else (None, score)

    def by_name(self, name: str) -> str | None:
        key = name.strip().lower()
        for cid, ch in self.characters.items():
            if ch.get("name", "").strip().lower() == key or key in [a.lower() for a in ch.get("aliases", [])]:
                return cid
        return None

    def update_embedding(self, cid: str, embedding: list[float], weight: float) -> None:
        ch = self.characters[cid]
        if ch.get("locked") or not embedding:
            return
        old, w0 = ch.get("embedding") or [], float(ch.get("embedding_weight", 0.0))
        if old:
            ch["embedding"] = speakers.mean_embedding([old, embedding], [w0, weight])
        else:
            ch["embedding"] = list(embedding)
        ch["embedding_weight"] = w0 + weight

    def current(self, cid: str) -> str:
        """The character id now standing for `cid` (after merges)."""
        seen = set()
        merged = self.data.get("merged") or {}
        while cid in merged and cid not in seen:
            seen.add(cid)
            cid = merged[cid]
        return cid

    def voice_source(self, cid: str) -> str:
        """The character whose references this character speaks with."""
        seen = set()
        while self.characters.get(cid, {}).get("voice_of") and cid not in seen:
            seen.add(cid)
            cid = self.characters[cid]["voice_of"]
        return cid

    def refs(self, cid: str) -> list[dict]:
        ch = self.characters.get(self.voice_source(cid), {})
        return sorted(ch.get("refs", []), key=lambda r: (not r.get("locked"), -r.get("score", 0)))

    def clean_seconds(self, cid: str) -> float:
        return sum(r.get("duration", 0) for r in self.refs(cid))

    # Corrections from the voice-bank page ---------------------------------------------
    def rename(self, cid: str, name: str) -> None:
        self.characters[cid]["name"] = name.strip() or self.characters[cid]["name"]

    def merge(self, keep: str, other: str) -> None:
        a, b = self.characters[keep], self.characters.pop(other)
        a["refs"] = a.get("refs", []) + b.get("refs", [])
        for r in a["refs"]:
            src = self.root / r["file"]
            if f"/{other}/" in r["file"] and src.exists():
                dst = self.root / "refs" / keep / src.name
                n = 1
                while dst.exists():
                    dst = dst.with_name(f"{src.stem}-{n}{src.suffix}")
                    n += 1
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), dst)
                r["file"] = str(dst.relative_to(self.root))
        if b.get("embedding"):
            self.update_embedding(keep, b["embedding"], float(b.get("embedding_weight", 1.0)))
        a.setdefault("aliases", []).append(b.get("name", other))
        self.data.setdefault("merged", {})[other] = keep  # lines of a job in progress still name `other`
        for ch in self.characters.values():
            if ch.get("voice_of") == other:
                ch["voice_of"] = keep
        shutil.rmtree(self.root / "refs" / other, ignore_errors=True)

    def split(self, cid: str, ref_files: list[str], name: str = "") -> str:
        """Move the given reference clips to a new character."""
        new = self.new_character(name)
        src = self.characters[cid]
        moving = [r for r in src.get("refs", []) if r["file"] in ref_files]
        src["refs"] = [r for r in src.get("refs", []) if r["file"] not in ref_files]
        for r in moving:
            p = self.root / r["file"]
            dst = self.root / "refs" / new / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                shutil.move(str(p), dst)
            r["file"] = str(dst.relative_to(self.root))
        self.characters[new]["refs"] = moving
        embs = [r["embedding"] for r in moving if r.get("embedding")]
        if embs:
            self.characters[new]["embedding"] = speakers.mean_embedding(embs)
            self.characters[new]["embedding_weight"] = float(len(embs))
        for k in ("gender", "age"):
            self.characters[new][k] = src.get(k, "")
        return new

    def remove_ref(self, cid: str, ref_file: str) -> None:
        ch = self.characters[cid]
        if not any(r["file"] == ref_file for r in ch.get("refs", [])):
            return  # only a clip this character actually has
        ch["refs"] = [r for r in ch.get("refs", []) if r["file"] != ref_file]
        target = (self.root / ref_file).resolve()
        if target.is_relative_to(self.root.resolve()):
            target.unlink(missing_ok=True)

    def lock_ref(self, cid: str, ref_file: str, locked: bool = True) -> None:
        for r in self.characters[cid].get("refs", []):
            if r["file"] == ref_file:
                r["locked"] = locked

    def delete_character(self, cid: str) -> None:
        self.characters.pop(cid, None)
        shutil.rmtree(self.root / "refs" / cid, ignore_errors=True)


class Profiles:
    def __init__(self, appdata: Path):
        self.root = appdata / "profiles"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Profile]:
        out = []
        for p in sorted(self.root.iterdir()):
            if (p / "profile.json").exists():
                out.append(Profile(p, json.loads((p / "profile.json").read_text())))
        return out

    def get(self, slug: str) -> Profile | None:
        p = self.root / slug
        if not slug or p.resolve().parent != self.root.resolve() or not (p / "profile.json").exists():
            return None
        return Profile(p, json.loads((p / "profile.json").read_text()))

    def get_or_create(self, name: str, defaults: dict | None = None) -> Profile:
        slug = slugify(name)
        prof = self.get(slug)
        if prof:
            return prof
        prof = Profile(self.root / slug, {"name": name, "created": time.time(), "characters": {},
                                          "glossary": {}, "settings": dict(defaults or {}), "episodes": []})
        prof.save()
        return prof

    def delete(self, slug: str) -> None:
        target = (self.root / slug).resolve()
        if target.parent != self.root.resolve():
            raise ValueError("bad profile")
        shutil.rmtree(target, ignore_errors=True)
