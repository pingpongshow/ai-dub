"""Listening page for a set of preview dubs: every line in time order with the
original audio, the dub, the character and why a line was kept original.

    python check_page.py /work/check /work/listen/<name> clip1 clip2 ...   (jobs in /work/check/jobs/job-chk-<clip>)
The app serves the page at http://<server>:8790/listen/<name>/; delete the folder when the review is over.
Each row gets an Acceptable/Poor choice and a comment box (web/listen-feedback.js),
saved on the server as <page>/feedback.json.
Lines whose English matches one of FLAG (the owner's earlier notes) are highlighted.
"""

import difflib
import html
import json
import subprocess
import sys
from pathlib import Path

FLAG = {"spyfamily": ["I'm Anya. Daddy's daughter.", "This is my house now?"],
        "seoul": ["Let's play dodgeball. - Dodgeball?", "Finish up. I'll be right back.", "Lying about one daughter again?"],
        "squid": ["What's wrong?", "Just tired."]}


def enc(src: Path, dst: Path, start=None, dur=None) -> str:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    subprocess.run(cmd + ["-vn", "-c:a", "aac", "-b:a", "128k", str(dst)], check=True)
    return dst.name


def flagged(clip: str, texts: list[str]) -> bool:
    return any(difflib.SequenceMatcher(None, f.lower(), t.lower()).ratio() > 0.6
               for f in FLAG.get(clip, []) for t in texts if t)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    parts = ["<!doctype html><meta charset=utf-8><title>Voice and line check</title><style>body{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;vertical-align:top}"
             "tr.flag{background:#3a2f12}audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}</style>"
             "<h1>Re-dubbed lines: voice and kept-original check</h1><p>Every line of each clip in time order. "
             "Highlighted rows are the lines you flagged before. \"Kept original\" lines are not dubbed. "
             "Mark each row Acceptable or Poor; for Poor, say what is wrong. Answers save as you go.</p>"
             '<script src="/listen-feedback.js" defer></script>']
    for clip in sys.argv[3:]:
        job = root / "jobs" / f"job-chk-{clip}"
        lines = json.loads((job / "lines.json").read_text())
        prof = json.loads(next((root / "appdata" / "profiles").glob(f"check-{clip}/profile.json")).read_text())
        names = {cid: ch["name"] for cid, ch in prof["characters"].items()}
        parts.append(f"<h2>{clip}</h2>")
        parts.append("<table>")
        src = root / f"{clip}.mkv"
        for ln in lines:
            if ln["kind"] not in ("speech", "wordless"):
                continue
            n = f"{clip}-{ln['id']:03d}"
            orig = enc(src, out / f"{n}-orig.m4a", ln["start"], ln["end"] - ln["start"] + 0.2)
            dub = ""
            if ln.get("take") and Path(ln["take"]).exists():
                dub = f'<audio controls preload="none" src="{enc(Path(ln["take"]), out / f"{n}-dub.m4a")}"></audio>'
            if ln["kind"] == "wordless":
                status = "kept original (no words spoken here)" if ln.get("heard_chars") == 0 else "kept original (sound, not words)"
            elif ln.get("fallback"):
                status = "kept original: " + ln["fallback"]
            else:
                status = html.escape(ln.get("voice_note") or "character's voice")
            cls = ' class="flag"' if flagged(clip, [ln.get("en", ""), ln.get("english_ref", ""), ln.get("english_context", "")]) else ""
            parts.append(f"<tr{cls}><td>{ln['start']:.1f}s<br><span class=m>{html.escape(names.get(ln.get('character'), ''))}</span></td>"
                         f"<td>{html.escape(ln.get('source_text', ''))}<br><b>{html.escape(ln.get('en', '') if ln['kind'] == 'speech' else '')}</b>"
                         f"<br><span class=m>{status}</span></td>"
                         f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
                         f"<td>{'dub<br>' + dub if dub else ''}</td></tr>")
        parts.append("</table>")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html")


if __name__ == "__main__":
    main()
