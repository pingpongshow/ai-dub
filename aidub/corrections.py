"""What each review changed, kept after the job's working files are deleted.

A finished dub's review is the only record of what the automatic labels got wrong. One
small file per job in appdata/corrections (no audio): every line's automatic label, its
final label and the signals behind them. Comparing a later run's labels against these
shows whether a rule change helped, on the owner's own corrections.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import lines as L

FIELDS = ("id", "start", "end", "kind", "kept_why", "music_db", "sung", "loudness_db", "heard_chars", "character",
          "source_text", "english_ref", "events")


def folder(appdata: Path) -> Path:
    return Path(appdata) / "corrections"


def _row(ln: dict) -> dict:
    return {k: ln.get(k) for k in FIELDS}


def save(appdata: Path, name: str, job_id: int | str, work: Path, qc: dict) -> Path | None:
    """Keep one job's corrections, if it was reviewed. Called before its working files go."""
    base, final = work / "review" / "base-lines.json", work / "lines.json"
    if not base.exists() or not final.exists():
        return None
    data = {"name": name, "job": job_id, "saved": time.time(),
            "automatic": [_row(x) for x in json.loads(base.read_text())],
            "final": [_row(x) for x in json.loads(final.read_text())],
            "changes": qc.get("review_changes") or []}
    decisions = work / "review" / "decisions.json"
    if decisions.exists():
        data["decisions"] = json.loads(decisions.read_text())
    d = folder(appdata)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"job-{job_id}.json"
    out.write_text(json.dumps(data, ensure_ascii=False))
    return out


def _dubbed(kind: str) -> bool:
    return kind == L.SPEECH


def compare(saved: dict, lines: list[dict]) -> dict:
    """How a run's labels compare with what the review settled on, by whether each moment
    gets dubbed: the choice a viewer hears. `before` is the first automatic label."""
    final = [x for x in saved["final"] if x.get("kind")]

    def label(rows: list[dict], a: float, b: float) -> str | None:
        best, most = None, 0.0
        for x in rows:
            o = min(x["end"], b) - max(x["start"], a)
            if o > most:
                best, most = x, o
        return best["kind"] if best else None

    wrong_before = wrong_now = same = 0
    missed, extra = [], []
    for x in final:
        want = _dubbed(x["kind"])
        was, now = label(saved["automatic"], x["start"], x["end"]), label(lines, x["start"], x["end"])
        if now is None:
            continue
        same += 1
        wrong_before += was is not None and _dubbed(was) != want
        if _dubbed(now) != want:
            wrong_now += 1
            (missed if want else extra).append({"start": x["start"], "text": x.get("source_text", ""), "now": now,
                                                "final": x["kind"], "english": x.get("english_ref", "")})
    return {"name": saved.get("name"), "lines": same, "wrong_before": wrong_before, "wrong_now": wrong_now,
            "should_dub": missed, "should_keep": extra}


def load_all(appdata: Path) -> list[dict]:
    d = folder(appdata)
    return [json.loads(p.read_text()) for p in sorted(d.glob("job-*.json"))] if d.exists() else []


def summary(appdata: Path) -> list[dict]:
    out = []
    for s in load_all(appdata):
        c = compare(s, s["automatic"])
        out.append({"job": s.get("job"), "name": s.get("name"), "lines": c["lines"], "changed": c["wrong_now"],
                    "kept_to_dubbed": len(c["should_dub"]), "dubbed_to_kept": len(c["should_keep"])})
    return out
