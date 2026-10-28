"""Follow-up review page: only the lines that still need the owner's ears.

After a review (a page built by check_page.py, answered with listen-feedback.js),
the fixes are made and the clips re-dubbed. This page shows, per clip:
  - lines marked Poor last time (with the earlier comment),
  - lines whose outcome changed (dubbed vs kept original, English wording, voice
    source) or whose dub audio changed, and lines that are new,
  - lines left unjudged last time.
Lines marked Acceptable and unchanged are left out. Optional voice-setting
variants (variants.json from variants.py) are shown as extra rows to compare.

    python review_page.py /work/check /work/listen/recheck-flagged /work/listen/<new-name> clip1 clip2 ...
The app serves the page at http://<server>:8790/listen/<new-name>/.
"""

import difflib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


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


def pcm(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def same_audio(a: Path, b: Path) -> bool:
    """True if two clips are the same take (lossy copies of it)."""
    x, y = pcm(a), pcm(b)
    if not len(x) or not len(y) or abs(len(x) - len(y)) > 0.08 * 16000:
        return False
    x, y = x - x.mean(), y - y.mean()
    # Peak of the cross-correlation within +-60 ms (encoder delay shifts lossy copies).
    n = 1 << int(np.ceil(np.log2(len(x) + len(y))))
    xc = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)
    lag = int(0.06 * 16000)
    peak = float(max(np.max(xc[:lag]), np.max(xc[-lag:])))
    return peak / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9) > 0.9


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
    return rows


def status_of(ln: dict) -> str:
    if ln["kind"] == "wordless":
        if ln.get("relisten"):
            return "kept original (only a sound: " + ln["relisten"][:20] + ")"
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


def match(ln: dict, olds: list[dict]) -> dict | None:
    best, key = None, None
    for o in olds:
        dt = abs(o["start"] - ln["start"])
        if dt > 0.6:
            continue
        sim = difflib.SequenceMatcher(None, norm(o["src"]), norm(ln.get("source_text", ""))).ratio() \
            if (o["src"] or ln.get("source_text")) else 1.0
        k = (sim, -dt)
        if sim >= 0.5 and (key is None or k > key):
            best, key = o, k
    return best


def main():
    root, old_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    olds = old_rows(old_dir)
    variants = json.loads((root / "variants.json").read_text()) if (root / "variants.json").exists() else {}
    parts = ["<!doctype html><meta charset=utf-8><title>Re-dub review</title><style>body{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;vertical-align:top}"
             "audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}.was{color:#e3a33b;font-size:12px}"
             "h3{margin-top:18px}</style>"
             "<h1>Re-dub review: only what changed</h1><p>Lines you marked Acceptable that didn't change are not "
             "shown. Each row says why it's here; your earlier comment is shown in orange. Mark each row Acceptable "
             "or Poor; for Poor, say what is wrong. Answers save as you go.</p>"
             '<script src="/listen-feedback.js" defer></script>']
    summary = []
    for clip in sys.argv[4:]:
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
        for ln in lines:
            if ln["kind"] not in ("speech", "wordless", "song", "english", "crowd", "background"):
                continue
            st = status_of(ln)
            o = match(ln, [x for x in mine if x["item"] not in used])
            if ln["id"] in var_ids:  # judged in the voice-settings section below
                if o:
                    used.add(o["item"])
                    var_notes[ln["id"]] = o.get("comment", "")
                continue
            # Old rows this line absorbed (duplicates merged into it).
            extra = [x for x in mine if x is not o and x["item"] not in used and abs(x["start"] - ln["start"]) <= 0.6
                     and norm(x["src"]) and norm(x["src"]) in norm(ln.get("source_text", ""))]
            reasons = []
            if o is None:
                reasons.append("new line")
            else:
                used.add(o["item"])
                if o["verdict"] == "poor":
                    reasons.append("you marked it Poor")
                elif not o["verdict"]:
                    reasons.append("not judged last time")
                if outcome(o["status"]) != outcome(st):
                    reasons.append(f"now {outcome(st)} (was {outcome(o['status'])})")
                elif outcome(st) == "dubbed":
                    if norm(o["en"]) != norm(ln.get("en", "")):
                        reasons.append("English changed")
                    elif o["status"].split(" (")[0] != st.split(" (")[0]:
                        reasons.append("voice source changed")
                    elif o["dub"].exists() and ln.get("take") and Path(ln["take"]).exists() \
                            and not same_audio(o["dub"], Path(ln["take"])):
                        reasons.append("dub audio changed")
            for x in extra:
                used.add(x["item"])
                reasons.append(f"merged duplicate of the old line at {x['start']:.1f}s")
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
        parts.append(f"<h2>{clip}</h2><p class=m>{shown} lines to judge; {same} accepted lines unchanged and not shown.</p>")
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
        summary.append(f"{clip}: {shown} to judge, {same} unchanged")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html", "|", "; ".join(summary))


if __name__ == "__main__":
    main()
