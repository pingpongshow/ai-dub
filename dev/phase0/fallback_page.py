"""Listening page for lines a run left in the original language because every take
failed the automatic back-check: the original, then the take that came closest,
to tell a too-strict check from a real mispronunciation.

    python fallback_page.py <job dir> <source video> <profile.json> <page dir> [job dir of the
        rule that rescued some, to say which take it now accepts]
"""

import html
import json
import sys
from pathlib import Path

sys.path.append("/work/check/dev")
from review_page import clip_span, enc  # noqa: E402


def main():
    job, src, prof, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    names = {k: v["name"] for k, v in json.loads(prof.read_text())["characters"].items()}
    lines = json.loads((job / "lines.json").read_text())
    parts = ["<!doctype html><meta charset=utf-8><title>Lines left in Japanese</title><style>body{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;vertical-align:top}"
             "audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}</style>"
             "<h1>Lines the automatic check left in Japanese</h1>"
             "<p>SPY x FAMILY episode 1. The check reads each English take back with speech recognition; on these "
             "lines it rejected every take, so the original Japanese stayed. Each row has the original and the take "
             "that came closest. Is the English dub right? Acceptable = it says the line clearly and sounds like "
             "the character; Poor = say what's wrong (mispronounced word, mumbled, wrong voice, too fast...).</p>"
             '<script src="/listen-feedback.js" defer></script><table>']
    n = 0
    for ln in lines:
        if ln["kind"] != "speech" or not ln.get("fallback") or not ln.get("take_scores"):
            continue
        best = min(ln["take_scores"].items(), key=lambda kv: (kv[1].get("wer", 1), len(kv[1].get("missed", []))))
        item = f"spy-e01-{ln['id']:03d}"
        a, dur = clip_span(ln, lines)
        orig = enc(src, out / f"{item}-orig.m4a", a, dur)
        dub = enc(Path(best[0]), out / f"{item}-dub.m4a")
        heard = best[1].get("heard", "")
        parts.append(f"<tr><td>{ln['start']:.1f}s<br><span class=m>{html.escape(names.get(ln['character'], ''))}</span></td>"
                     f"<td>{html.escape(ln['source_text'])}<br><b>{html.escape(ln['en'])}</b>"
                     f"<br><span class=m>the check heard: \"{html.escape(heard)}\"</span></td>"
                     f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
                     f'<td>dub<br><audio controls preload="none" src="{dub}"></audio></td></tr>')
        n += 1
    parts.append("</table>")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html", n, "lines")


if __name__ == "__main__":
    main()
