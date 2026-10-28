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
voice-setting variants (variants.json from variants.py) are extra rows to compare.

    python review_page.py [--pin] /work/check /work/listen/recheck-flagged /work/listen/<new-name> clip1 clip2 ...
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


def old_rows(page_dir: Path) -> dict[str, dict]:
    page = (page_dir / "index.html").read_text(encoding="utf-8")
    fb = json.loads((page_dir / "feedback.json").read_text()) if (page_dir / "feedback.json").exists() else {}
    rows = {}
    pat = (r'<tr(?: class="flag")?><td>([\d.]+)s<br><span class=m>(.*?)</span></td><td>(.*?)<br><b>(.*?)</b>'
           r'<br><span class=m>(.*?)</span></td><td>original<br><audio[^>]*src="([^"]+)-orig\.m4a"')
    for m in re.finditer(pat, page):
        t, who, src, en, status, item = m.groups()
        clip = item.rsplit("-", 1)[0]
        rows[item] = {"clip": clip, "start": float(t), "who": html.unescape(who), "src": html.unescape(src),
                      "en": html.unescape(en), "status": html.unescape(status), "verdict": fb.get(item, {}).get("verdict", ""),
                      "comment": fb.get(item, {}).get("comment", ""), "dub": page_dir / f"{item}-dub.m4a"}
        # The original clip ran from the line's start to 0.2 s past its end.
        rows[item]["end"] = float(t) + max(0.1, seconds(page_dir / f"{item}-orig.m4a") - 0.2)
    return rows


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
    root, old_dir, out = Path(args[0]), Path(args[1]), Path(args[2])
    out.mkdir(parents=True, exist_ok=True)
    olds = old_rows(old_dir)
    variants = json.loads((root / "variants.json").read_text()) if (root / "variants.json").exists() else {}
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
            found = matches(ln, [x for x in mine if x["item"] not in used])
            o, extra = (found[0], found[1:]) if found else (None, [])
            for x in found:
                used.add(x["item"])
            if ln["kind"] == "song" and not any(x["verdict"] in ("poor", "") for x in found):
                continue  # a kept song you didn't flag needs no judging
            if ln["id"] in var_ids:  # judged in the voice-settings section below
                var_notes[ln["id"]] = " / ".join(x["comment"] for x in found if x.get("comment"))
                continue
            reasons = []
            if o is None:
                reasons.append("new line")
            else:
                if any(x["verdict"] == "poor" for x in found):
                    reasons.append("you marked it Poor")
                if any(not x["verdict"] for x in found):
                    reasons.append("not judged last time")
                if outcome(o["status"]) != outcome(st):
                    reasons.append(f"now {outcome(st)} (was {outcome(o['status'])})")
                if extra:
                    reasons.append("merged with the old line(s) at " + ", ".join(f"{x['start']:.1f}s" for x in extra))
            if not reasons:
                if outcome(st) == "dubbed" and o["dub"].exists():
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
            orig = enc(src, out / f"{n}-orig.m4a", ln["start"] - 0.05, ln["end"] - ln["start"] + 0.25)
            dub = ""
            if ln.get("take") and Path(ln["take"]).exists() and outcome(st) == "dubbed":
                dub = f'dub<br><audio controls preload="none" src="{enc(Path(ln["take"]), out / f"{n}-dub.m4a")}"></audio>'
            note = "".join(f'<br><span class=was>before: "{html.escape(x["comment"])}"</span>' for x in was)
            rows.append(f"<tr><td>{ln['start']:.1f}s<br><span class=m>{html.escape(names.get(ln.get('character'), ''))}</span></td>"
                        f"<td>{html.escape(ln.get('source_text', ''))}<br><b>{html.escape(ln.get('en', '') if outcome(st) == 'dubbed' else '')}</b>"
                        f"<br><span class=m>{html.escape(st)}</span><br><span class=m>why shown: {html.escape('; '.join(reasons))}</span>{note}</td>"
                        f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td><td>{dub}</td></tr>')
        # Old Poor lines that no longer exist as a line of their own.
        gone = [x for x in mine if x["item"] not in used and x["verdict"] == "poor"]
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
                         "accented or unlike the speaker, each voiced four ways (A is the normal setting). Mark each "
                         "version Acceptable or Poor; the versions that do best become the default.</p><table>")
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
