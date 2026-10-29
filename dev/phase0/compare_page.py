"""Listening page comparing two full-episode dubs of the same episode against its
English subtitles (sub_compare.py output for each): both episodes with the
official subtitles switchable, then every subtitle block with the original and
both dubs, their scripts, and what each dub was heard to say.

    python compare_page.py <source video> <page dir> <title> <grades.json> \
        <label A> <compare A.json> <dubbed A> <label B> <compare B.json> <dubbed B>

grades.json: {"<block index>": ["grade A", "grade B", "note"]}.
"""

import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.append("/work/check/dev")
from episode_page import vtt  # noqa: E402
from review_page import enc  # noqa: E402

CSS = ("body{font:14px system-ui;margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid "
       "#2d333d;vertical-align:top}audio{height:28px;width:150px}.m{color:#9aa3b2;font-size:12px}.k{color:#e0a458}"
       ".g{font-size:12px;font-weight:600}.good{color:#7fc8a9}.minor{color:#e0c458}.wrong,.japanese{color:#f08a7e}"
       "video{max-width:100%;width:640px}.v{display:inline-block;margin-right:12px;vertical-align:top}"
       "th{text-align:left;color:#9aa3b2;font-weight:500}")


def main():
    src, out, title, grades = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], json.loads(Path(sys.argv[4]).read_text())
    runs = [(sys.argv[5], json.loads(Path(sys.argv[6]).read_text()), Path(sys.argv[7])),
            (sys.argv[8], json.loads(Path(sys.argv[9]).read_text()), Path(sys.argv[10]))]
    out.mkdir(parents=True, exist_ok=True)
    subs = vtt([tuple(c) for b in runs[0][1]["blocks"] for c in b["cues"]], out / "official.vtt")
    videos = []
    for n, (label, comp, dubbed) in enumerate(runs):
        video = out / f"episode-{'ab'[n]}.mp4"
        if not video.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dubbed), "-map", "0:v:0",
                            "-map", "0:a:m:language:eng", "-c", "copy", "-movflags", "+faststart", str(video)],
                           check=True)
        script = vtt([(ln["start"], ln["end"], ln["en"]) for b in comp["blocks"] for ln in b["lines"]
                      if ln["kind"] == "speech" and not ln["fallback"]], out / f"script-{'ab'[n]}.vtt")
        videos.append(f'<div class=v><b>{html.escape(label)}</b><br><video controls preload="metadata" '
                      f'src="{video.name}"><track kind="subtitles" srclang="en" label="Official English subtitles" '
                      f'src="{subs}"><track kind="subtitles" srclang="en" label="Dub script" src="{script}"></video></div>')
    a_label, b_label = runs[0][0], runs[1][0]
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{CSS}</style>"
             f"<h1>{html.escape(title)}</h1>"
             f"<p>The same episode dubbed two ways: <b>A</b> = {html.escape(a_label)}, <b>B</b> = "
             f"{html.escape(b_label)}. Use CC in either player for the official English subtitles or the dub "
             "script. Below, every subtitle block: the official subtitle, each dub's script and what each dub "
             "was heard to say, my grade of each against the subtitle, and the audio. Acceptable = B is right "
             "(and at least as good as A); Poor = say what's wrong or which is better.</p>"
             + "".join(videos) + '<script src="/listen-feedback.js" defer></script><table>'
             "<tr><th></th><th>subtitle / A / B</th><th></th><th></th><th></th></tr>"]
    (_, ca, da), (_, cb, db) = runs
    for i, (a, b) in enumerate(zip(ca["blocks"], cb["blocks"])):
        if not (a["script"] or b["script"] or any(ln["kind"] == "speech" for ln in a["lines"])):
            continue  # a sign or credit with no speech
        item = f"blk-{i:03d}"
        s, z = a["start"] - 0.4, a["end"] + 0.6
        orig = enc(src, out / f"{item}-orig.m4a", s, z - s)
        dub_a = enc(da, out / f"{item}-a.m4a", s, z - s)
        dub_b = enc(db, out / f"{item}-b.m4a", s, z - s)
        ga, gb, note = (grades.get(str(i)) or ["", "", ""]) + [""] * (3 - len(grades.get(str(i)) or []))

        def side(tag, blk, g):
            kept = [ln["ja"] for ln in blk["lines"] if ln["kind"] == "speech" and ln["fallback"]]
            return (f"<br><b>{tag}:</b> {html.escape(blk['script'])}"
                    + (f" <span class='g {g.lower()}'>[{g}]</span>" if g else "")
                    + f"<br><span class=m>heard: \"{html.escape(blk['heard'])}\"</span>"
                    + (f"<br><span class=k>left in Japanese: {html.escape('; '.join(kept))}</span>" if kept else ""))
        parts.append(
            f"<tr><td>{a['start']:.1f}s</td><td><i>{html.escape(a['subtitle'])}</i>"
            + side("A", a, ga) + side("B", b, gb)
            + (f"<br><span class=m>{html.escape(note)}</span>" if note else "")
            + f'</td><td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
            f'<td>A<br><audio controls preload="none" src="{dub_a}"></audio></td>'
            f'<td>B<br><audio controls preload="none" src="{dub_b}"></audio></td></tr>')
    parts.append("</table>")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html")


if __name__ == "__main__":
    main()
