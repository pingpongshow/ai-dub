"""Follow-up review page: only the lines that still need the owner's ears.

After a review (a page built by check_page.py, answered with listen-feedback.js),
the fixes are made and the clips re-dubbed. This page shows, per clip:
  - lines marked Poor last time (with the earlier comment),
  - lines whose outcome changed (dubbed vs kept original, English wording, voice
    source) or whose dub audio changed, and lines that are new,
  - lines left unjudged last time.
A line marked Acceptable that is still dubbed keeps exactly what was approved:
its English and its dub audio are put back into the job (--pin), so the owner is
never asked about it again; accepted kept-original lines stay kept. Optional
voice-setting variants (variants.json from variants.py) are extra rows to compare;
the version accepted there is what the line keeps.

    python review_page.py [--pin] /work/check /work/listen/page1,/work/listen/page2 /work/listen/<new-name> clip1 clip2 ...
Earlier pages are given oldest first; a line's answer on the latest page it was on counts.
The app serves the page at http://<server>:8790/listen/<new-name>/.
"""

import difflib
import html
import json
import re
import subprocess
import sys
from pathlib import Path



def enc(src: Path, dst: Path, start=None, dur=None) -> str:
    """Every clip the same way: stereo 48 kHz AAC, level-matched (quiet originals
    were hard to hear), audio only."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.2f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    cmd += ["-vn", "-sn", "-dn", "-ac", "2", "-ar", "48000", "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "160k", str(dst)]
    subprocess.run(cmd, check=True)
    return dst.name


def norm(t: str) -> str:
    return re.sub(r"[\W_]+", "", t or "").lower()


ROW = re.compile(r'<td>([\d.]+)s<br><span class=m>(.*?)</span></td><td>(.*?)<br><b>(.*?)</b><br><span class=m>(.*?)</span>')
PREFER = ["low-emo", "current", "tag-emo", "own-voice", "own-child2", "own-child4", "own-happy"]  # default first


def old_rows(page_dir: Path, rnd: int = 0) -> dict[str, dict]:
    """The rows of an earlier review page with the owner's answers; a line voiced
    several ways (voice-setting versions) is one row holding the version chosen."""
    page = (page_dir / "index.html").read_text(encoding="utf-8")
    fb = json.loads((page_dir / "feedback.json").read_text()) if (page_dir / "feedback.json").exists() else {}
    keys: dict[str, str] = {}
    for k in page_dir.glob("*-variants-key.json"):
        keys.update(json.loads(k.read_text()))
    rows = {}
    for chunk in page.split("<tr")[1:]:
        a = re.search(r'src="([^"]+)-orig\.m4a"', chunk)
        m = ROW.search(chunk)
        if not a or not m or a.group(1) in keys:
            continue
        item = a.group(1)
        t, who, src, en, status = m.groups()
        rows[item] = {"clip": item.rsplit("-", 1)[0], "start": float(t), "who": html.unescape(who),
                      "src": html.unescape(src), "en": html.unescape(en), "status": html.unescape(status),
                      "verdict": fb.get(item, {}).get("verdict", ""), "comment": fb.get(item, {}).get("comment", ""),
                      "dub": page_dir / f"{item}-dub.m4a", "round": rnd}
        # The original clip runs from just before the line's start to just past its end.
        rows[item]["end"] = float(t) + max(0.1, seconds(page_dir / f"{item}-orig.m4a") - 0.25)
    # Voice-setting versions: the accepted one (the default setting if several were), else
    # the one the owner commented on as the best there was.
    variants = json.loads((page_dir / "variants.json").read_text()) if (page_dir / "variants.json").exists() else {}
    for clip, vs in variants.items():
        for v in vs:
            base = f"{clip}-{v['id']:03d}"
            versions = sorted(((item, label, fb.get(item, {})) for item, label in keys.items()
                               if item[:-1] == base), key=lambda x: PREFER.index(x[1]) if x[1] in PREFER else 9)
            ok = [x for x in versions if x[2].get("verdict") == "acceptable"]
            picked = ok or [x for x in versions if x[2].get("verdict") == "poor" and x[2].get("comment")]
            pick = picked[0] if picked else None
            rows[base + "v"] = {"clip": clip, "start": v["start"], "end": v["end"], "who": v.get("who", ""), "src": "",
                                "en": v["en"], "status": f"voice version {pick[1] if pick else 'none'} chosen",
                                "verdict": "acceptable" if pick else ("poor" if any(x[2].get("verdict") for x in versions) else ""),
                                "comment": "" if ok else "; ".join(x[2]["comment"] for x in versions if x[2].get("comment"))
                                or "no version sounded right",
                                "dub": page_dir / f"{pick[0]}-dub.m4a" if pick else page_dir / "-", "round": rnd}
    return rows


def clip_span(ln: dict, lines: list[dict]) -> tuple[float, float]:
    """The original clip: from just before the line to just past it, but not into the
    neighbouring lines (a padded clip caught the start of the next speaker)."""
    a, b = ln["start"] - 0.05, ln["end"] + 0.2
    for o in lines:
        if o is not ln and ln["end"] <= o["start"] < b:
            b = o["start"]
        if o is not ln and a < o["end"] <= ln["start"]:
            a = o["end"]
    return a, b - a


def seconds(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def status_of(ln: dict) -> str:
    if ln["kind"] == "wordless":
        if ln.get("source_text"):
            return "kept original (a sound or interjection)"
        return "kept original (no words spoken here)" if ln.get("heard_chars") == 0 else "kept original (sound, not words)"
    if ln["kind"] == "song":
        return "kept original (song)"
    if ln["kind"] == "english":
        return "kept original (already English)"
    if ln["kind"] == "crowd":
        return "kept original (several people talking at once)"
    if ln["kind"] == "background":
        return "kept original (speech in the background sound, e.g. an announcement)"
    if ln.get("fallback"):
        return "kept original: " + ln["fallback"]
    if not ln.get("take"):
        return "no dub made (no English line)" if not ln.get("en") else "no dub made"
    return ln.get("voice_note") or "character's voice"


def outcome(status: str) -> str:
    return "kept" if status.startswith("kept original") else "dubbed"


def overlap(ln: dict, o: dict) -> float:
    """Shared time as a share of the shorter of the two lines."""
    shared = min(ln["end"], o["end"]) - max(ln["start"], o["start"])
    return max(0.0, shared) / max(0.1, min(ln["end"] - ln["start"], o["end"] - o["start"]))


def text_sim(ln: dict, o: dict) -> float:
    a, b = norm(o["src"]), norm(ln.get("source_text", ""))
    return difflib.SequenceMatcher(None, a, b).ratio() if (a and b) else 1.0


def matches(ln: dict, olds: list[dict]) -> list[dict]:
    """Old rows that are this line (same moment, same words), best first."""
    found = [o for o in olds if (overlap(ln, o) >= 0.4 or abs(o["start"] - ln["start"]) <= 0.4) and text_sim(ln, o) >= 0.4]
    return sorted(found, key=lambda o: (-overlap(ln, o) - text_sim(ln, o), abs(o["start"] - ln["start"])))


def pin(ln: dict, o: dict, job: Path) -> None:
    """Put the approved English and dub audio back on the line."""
    dst = job / "lines" / f"{ln['id']:05d}" / "accepted.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(o["dub"]), "-ac", "1", "-ar", "48000",
                    str(dst)], check=True)
    ln["take"], ln["en"], ln["fallback"] = str(dst), o["en"], ""
    ln.setdefault("fit", {})["method"] = "accepted in review"


PIN = False


def main():
    global PIN
    args = sys.argv[1:]
    if args and args[0] == "--pin":
        PIN, args = True, args[1:]
    root, out = Path(args[0]), Path(args[2])
    old_dirs = [Path(d) for d in args[1].split(",")]  # every earlier page, oldest first
    out.mkdir(parents=True, exist_ok=True)
    olds = {}
    for rnd, d in enumerate(old_dirs):
        olds.update({f"{rnd}:{k}": v for k, v in old_rows(d, rnd).items()})
    last = len(old_dirs) - 1
    variants = json.loads((root / "variants.json").read_text()) if (root / "variants.json").exists() else {}
    (out / "variants.json").write_text(json.dumps(variants, ensure_ascii=False, indent=1))  # this page's versions
    parts = ["<!doctype html><meta charset=utf-8><title>Re-dub review</title><style>body{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;vertical-align:top}"
             "audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}.was{color:#e3a33b;font-size:12px}"
             "h3{margin-top:18px}</style>"
             "<h1>Re-dub review: only what changed</h1><p>Lines you marked Acceptable keep exactly what you "
             "approved and are not shown. Each row says why it's here; your earlier comment is shown in orange. Mark each row Acceptable "
             "or Poor; for Poor, say what is wrong. Answers save as you go.</p>"
             '<script src="/listen-feedback.js" defer></script>']
    summary = []
    for clip in args[3:]:
        job = root / "jobs" / f"job-chk-{clip}"
        lines = json.loads((job / "lines.json").read_text())
        prof = json.loads(next((root / "appdata" / "profiles").glob(f"check-{clip}/profile.json")).read_text())
        names = {cid: ch["name"] for cid, ch in prof["characters"].items()}
        mine = [o | {"item": k} for k, o in olds.items() if o["clip"] == clip]
        used: set[str] = set()
        rows, shown, same = [], 0, 0
        vs = variants.get(clip, [])
        var_ids = {v["id"] for v in vs}
        var_notes: dict[int, str] = {}
        src = root / f"{clip}.mkv"
        pinned = 0
        for ln in lines:
            if ln["kind"] not in ("speech", "wordless", "song", "english", "crowd", "background"):
                continue
            st = status_of(ln)
            every = matches(ln, [x for x in mine if x["item"] not in used])
            for x in every:
                used.add(x["item"])
            # The latest page this line was on holds the owner's latest word on it.
            found = [x for x in every if x["round"] == max(y["round"] for y in every)]
            o, extra = (found[0], found[1:]) if found else (None, [])
            if ln["kind"] == "song" and not any(x["verdict"] in ("poor", "") for x in found):
                continue  # a kept song you didn't flag needs no judging
            if ln["id"] in var_ids:  # judged in the voice-settings section below
                var_notes[ln["id"]] = " / ".join(x["comment"] for x in found if x.get("comment"))
                continue
            reasons, redo_failed = [], False
            if o is None:
                reasons.append("new line")
            else:
                if any(x["verdict"] == "poor" for x in found):
                    reasons.append("you marked it Poor")
                if any(not x["verdict"] for x in found):
                    reasons.append("not judged last time")
                # An approved dub stays when only this run's take failed (the line is still speech).
                redo_failed = (ln["kind"] == "speech" and ln.get("fallback") and o["verdict"] == "acceptable"
                               and outcome(o["status"]) == "dubbed" and o["dub"].exists())
                if outcome(o["status"]) != outcome(st) and not redo_failed:
                    reasons.append(f"now {outcome(st)} (was {outcome(o['status'])})")
                if abs(ln["start"] - o["start"]) > 0.3 and outcome(o["status"]) == "dubbed":
                    reasons.append(f"now starts at {ln['start']:.1f}s (was {o['start']:.1f}s)")
                if extra:
                    reasons.append("merged with the old line(s) at " + ", ".join(f"{x['start']:.1f}s" for x in extra))
            if not reasons:
                if (outcome(st) == "dubbed" or redo_failed) and o["dub"].exists():
                    if PIN:
                        pin(ln, o, job)
                    pinned += 1
                same += 1
                continue
            was = [x for x in ([o] if o else []) + extra if x.get("comment")]
            if not reasons:
                same += 1
                continue
            shown += 1
            n = f"{clip}-{ln['id']:03d}"
            orig = enc(src, out / f"{n}-orig.m4a", *clip_span(ln, lines))
            if ln.get("overlap"):
                st += "; two people talk at once in part of the original"
            dub = ""
            if ln.get("take") and Path(ln["take"]).exists() and outcome(st) == "dubbed":
                dub = f'dub<br><audio controls preload="none" src="{enc(Path(ln["take"]), out / f"{n}-dub.m4a")}"></audio>'
            note = "".join(f'<br><span class=was>before: "{html.escape(x["comment"])}"</span>' for x in was)
            rows.append(f"<tr><td>{ln['start']:.1f}s<br><span class=m>{html.escape(names.get(ln.get('character'), ''))}</span></td>"
                        f"<td>{html.escape(ln.get('source_text', ''))}<br><b>{html.escape(ln.get('en', '') if outcome(st) == 'dubbed' else '')}</b>"
                        f"<br><span class=m>{html.escape(st)}</span><br><span class=m>why shown: {html.escape('; '.join(reasons))}</span>{note}</td>"
                        f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td><td>{dub}</td></tr>')
        # Old Poor lines that no longer exist as a line of their own.
        gone = [x for x in mine if x["item"] not in used and x["verdict"] == "poor" and x["round"] == last]
        for x in gone:
            rows.append(f"<tr><td>{x['start']:.1f}s</td><td>{html.escape(x['src'])}<br><span class=m>this line is gone "
                        f"(merged or removed)</span><br><span class=was>before: \"{html.escape(x['comment'])}\"</span></td>"
                        f"<td colspan=2 class=m>nothing to judge</td></tr>")
        parts.append(f"<h2>{clip}</h2><p class=m>{shown} lines to judge; {same} lines you accepted are kept exactly "
                     f"as you approved them and not shown.</p>")
        parts.append("<table>" + "".join(rows) + "</table>")
        # Voice-setting variants for lines that sounded robotic, accented or unlike the speaker.
        if vs:
            parts.append("<h3>Voice settings: which sounds most natural?</h3><p class=m>Lines you called robotic, "
                         "accented or unlike the speaker, each voiced several ways (A is the version used now). Mark each "
                         "version Acceptable or Poor; the one you like best is kept.</p><table>")
            for v in vs:
                for k, (label, path) in enumerate(v["takes"].items()):
                    n = f"{clip}-{v['id']:03d}{chr(97 + k)}"
                    orig = enc(src, out / f"{n}-orig.m4a", v["start"] - 0.05, v["end"] - v["start"] + 0.25)
                    take = enc(Path(path), out / f"{n}-dub.m4a")
                    parts.append(f"<tr><td>{v['start']:.1f}s<br><span class=m>{html.escape(v['who'])}</span></td>"
                                 f"<td><b>{html.escape(v['en'])}</b><br><span class=m>version {chr(65 + k)}</span>"
                                 + (f'<br><span class=was>before: "{html.escape(var_notes[v["id"]])}"</span>'
                                    if k == 0 and var_notes.get(v["id"]) else "") + "</td>"
                                 f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
                                 f'<td>dub<br><audio controls preload="none" src="{take}"></audio></td></tr>')
            parts.append("</table>")
            (out / f"{clip}-variants-key.json").write_text(json.dumps(
                {f"{clip}-{v['id']:03d}{chr(97 + k)}": label for v in vs for k, label in enumerate(v["takes"])}, indent=1))
        summary.append(f"{clip}: {shown} to judge, {same} accepted and kept ({pinned} dubs kept as approved)")
        if PIN:
            (job / "lines.json").write_text(json.dumps(lines, ensure_ascii=False, indent=1))
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html", "|", "; ".join(summary))


if __name__ == "__main__":
    main()
