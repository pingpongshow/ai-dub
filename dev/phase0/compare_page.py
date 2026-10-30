"""Listening page comparing full-episode dubs of the same episode against its
English subtitles (sub_compare.py output for each): every episode with the
official subtitles switchable, then every subtitle block with the original and
each dub, their scripts, and what each dub was heard to say.

    python compare_page.py <source video> <page dir> <title> <grades.json> \
        <label A> <compare A.json> <dubbed A> [<label B> <compare B.json> <dubbed B> ...]

grades.json: {"<block index>": ["grade A", "grade B", ..., "note"]}.
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
    args = sys.argv[5:]
    runs = [(args[i], json.loads(Path(args[i + 1]).read_text()), Path(args[i + 2])) for i in range(0, len(args), 3)]
    tags = "ABCDEFGH"[:len(runs)]
    out.mkdir(parents=True, exist_ok=True)
    subs = vtt([tuple(c) for b in runs[0][1]["blocks"] for c in b["cues"]], out / "official.vtt")
    videos, tracks = [], []
    for n, (label, comp, dubbed) in enumerate(runs):
        # Clips come from the dub track itself: the file's first audio track is the original.
        track = out / f"dub-{tags[n].lower()}.m4a"
        if not track.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dubbed), "-map", "0:a:m:language:eng",
                            "-c", "copy", str(track)], check=True)
        tracks.append(track)
        video = out / f"episode-{tags[n].lower()}.mp4"
        if not video.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dubbed), "-map", "0:v:0",
                            "-map", "0:a:m:language:eng", "-c", "copy", "-movflags", "+faststart", str(video)],
                           check=True)
        script = vtt([(ln["start"], ln["end"], ln["en"]) for b in comp["blocks"] for ln in b["lines"]
                      if ln["kind"] == "speech" and not ln["fallback"]], out / f"script-{tags[n].lower()}.vtt")
        videos.append(f'<div class=v><b>{tags[n]}: {html.escape(label)}</b><br><video controls preload="metadata" '
                      f'src="{video.name}"><track kind="subtitles" srclang="en" label="Official English subtitles" '
                      f'src="{subs}"><track kind="subtitles" srclang="en" label="Dub script" src="{script}"></video></div>')
    names = ", ".join(f"<b>{t}</b> = {html.escape(r[0])}" for t, r in zip(tags, runs))
    last = tags[-1]
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{CSS}</style>"
             f"<h1>{html.escape(title)}</h1>"
             f"<p>The same episode dubbed {len(runs)} ways: {names}. Use CC in any player for the official "
             "English subtitles or the dub script. Below, every subtitle block: the official subtitle, each "
             "dub's script and what it was heard to say, my grade of each against the subtitle, and the audio. "
             f"Acceptable = {last} is right (and at least as good as the others); Poor = say what's wrong or "
             "which is better.</p>"
             + "".join(videos) + '<script src="/listen-feedback.js" defer></script><table>'
             f"<tr><th></th><th>subtitle / {' / '.join(tags)}</th><th></th>"
             + "<th></th>" * len(runs) + "</tr>"]
    blocks = list(zip(*(r[1]["blocks"] for r in runs)))
    for i, bs in enumerate(blocks):
        a = bs[0]
        if not (any(b["script"] for b in bs) or any(ln["kind"] == "speech" for ln in a["lines"])):
            continue  # a sign or credit with no speech
        item = f"blk-{i:03d}"
        s, z = a["start"] - 0.4, a["end"] + 0.6
        # A version with free timing can say a block's lines later: its clip runs until they end.
        for b in bs:
            for ln in b["lines"]:
                if ln.get("placed") is not None and ln.get("final_s"):
                    z = max(z, ln["placed"] + ln["final_s"] + 0.2)
        orig = enc(src, out / f"{item}-orig.m4a", s, z - s)
        dubs = [enc(tr, out / f"{item}-{t.lower()}.m4a", s, z - s) for t, tr in zip(tags, tracks)]
        g = list(grades.get(str(i)) or [])
        g += [""] * (len(runs) + 1 - len(g))

        def side(tag, blk, grade):
            kept = [ln["ja"] for ln in blk["lines"] if ln["kind"] == "speech" and ln["fallback"]]
            notes = [ln["check_note"] for ln in blk["lines"] if ln.get("check_note")]
            return (f"<br><b>{tag}:</b> {html.escape(blk['script'])}"
                    + (f" <span class='g {grade.lower()}'>[{grade}]</span>" if grade else "")
                    + f"<br><span class=m>heard: \"{html.escape(blk['heard'])}\"</span>"
                    + (f"<br><span class=k>left in Japanese: {html.escape('; '.join(kept))}</span>" if kept else "")
                    + (f"<br><span class=k>flagged: {html.escape('; '.join(notes))}</span>" if notes else ""))
        parts.append(
            f"<tr><td>{a['start']:.1f}s</td><td><i>{html.escape(a['subtitle'])}</i>"
            + "".join(side(t, b, gr) for t, b, gr in zip(tags, bs, g))
            + (f"<br><span class=m>{html.escape(g[len(runs)])}</span>" if g[len(runs)] else "")
            + f'</td><td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
            + "".join(f'<td>{t}<br><audio controls preload="none" src="{d}"></audio></td>' for t, d in zip(tags, dubs))
            + "</tr>")
    parts.append("</table>")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html")


if __name__ == "__main__":
    main()
