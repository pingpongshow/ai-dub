"""Set-aside files: originals (and dubbed copies after a restore).

Nothing here runs on its own. Restoring and deleting happen only when you ask
from the "Originals pending deletion" page or the CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

from .jobs import Store

ORIGINAL = ".aidub-original"
DUBBED = ".aidub-dubbed"
SUFFIX = {"original": ORIGINAL, "dubbed": DUBBED}


def aside_name(path: Path, kind: str = "original") -> Path:
    return path.with_name(path.name + SUFFIX[kind])


def live_name(aside: Path) -> Path:
    for s in SUFFIX.values():
        if aside.name.endswith(s):
            return aside.with_name(aside.name[: -len(s)])
    raise ValueError(f"not a set-aside file: {aside}")


def kind_of(aside: Path) -> str:
    return "dubbed" if aside.name.endswith(DUBBED) else "original"


def scan(store: Store, roots: list[Path]) -> int:
    """Register set-aside files found on disk that the database doesn't know (e.g. after a reset)."""
    known = {o["path"] for o in store.originals()}
    added = 0
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith((ORIGINAL, DUBBED)):
                    p = Path(dirpath) / f
                    if str(p) not in known:
                        store.add_original(None, str(p), str(live_name(p)), kind_of(p), p.stat().st_size)
                        added += 1
    # Forget entries whose file is gone (deleted outside Ai Dub).
    for o in store.originals():
        if not Path(o["path"]).exists():
            store.drop_original(o["id"])
    return added


def listing(store: Store) -> dict:
    items = []
    for o in store.originals():
        p = Path(o["path"])
        if not p.exists():
            continue
        items.append({**o, "size": p.stat().st_size, "live_exists": Path(o["live_path"]).exists(), "name": p.name})
    return {"items": items, "total": sum(i["size"] for i in items)}


def restore(store: Store, oid: int) -> str:
    """Put a set-aside file back under its own name; the current file is set aside in its place."""
    o = store.original(oid)
    if not o:
        raise KeyError("unknown entry")
    aside, live = Path(o["path"]), Path(o["live_path"])
    if not aside.exists():
        store.drop_original(oid)
        raise FileNotFoundError(f"{aside.name} no longer exists")
    swapped = None
    if live.exists():
        other_kind = "dubbed" if o["kind"] == "original" else "original"
        swapped = aside_name(live, other_kind)
        if swapped.exists():
            raise FileExistsError(f"{swapped.name} already exists; delete or restore it first")
        os.rename(live, swapped)
    try:
        os.rename(aside, live)
    except OSError:
        if swapped:
            os.rename(swapped, live)
        raise
    store.drop_original(oid)
    if swapped:
        store.add_original(o["job_id"], str(swapped), str(live), kind_of(swapped), swapped.stat().st_size)
    return f"restored {live.name}" + (f"; the other copy is now {swapped.name}" if swapped else "")


def delete(store: Store, ids: list[int]) -> tuple[int, int]:
    """Permanently delete set-aside files. Returns (files deleted, bytes freed)."""
    n, freed = 0, 0
    for oid in ids:
        o = store.original(oid)
        if not o:
            continue
        p = Path(o["path"])
        if not p.name.endswith((ORIGINAL, DUBBED)):
            continue  # never delete anything that isn't a set-aside file
        if p.exists():
            size = p.stat().st_size
            p.unlink()
            n, freed = n + 1, freed + size
        store.drop_original(oid)
    return n, freed
