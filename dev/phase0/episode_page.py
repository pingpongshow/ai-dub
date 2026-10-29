"""Listening page for a whole dubbed episode checked against its English subtitles
(sub_compare.py output): the episode itself with the dub track and the official
subtitles (switchable), then every subtitle block with the original, the dub,
and what the dub was heard to say.

    python episode_page.py <compare.json> <source video> <dubbed video> <page dir> <title> [notes.json]

notes.json: {"<block start, 1 decimal>": "reviewer's note"} shown beside a block.
"""

import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.append("/work/check/dev")
from review_page import enc  # noqa: E402


def vtt_time(t: float) -> str:
    h, r = divmod(max(0.0, t), 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def vtt(cues: list[tuple[float, float, str]], path: Path) -> str:
    body = "".join(f"{vtt_time(a)} --> {vtt_time(b)}\n{t}\n\n" for a, b, t in cues if t)
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return path.name


def main():
    comp, src, dubbed, out, title = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
                                     sys.argv[5])
    notes = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 else {}
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(comp.read_text())
    blocks = d["blocks"]
    # The episode with only the dub track, playable in a browser.
    video = out / "episode.mp4"
    if not video.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dubbed), "-map", "0:v:0",
                        "-map", "0:a:m:language:eng", "-c", "copy", "-movflags", "+faststart", str(video)], check=True)
    subs = vtt([tuple(c) for b in blocks for c in b["cues"]], out / "official.vtt")
    script = vtt([(ln["start"], ln["end"], ln["en"]) for b in blocks for ln in b["lines"]
                  if ln["kind"] == "speech" and not ln["fallback"]], out / "dub-script.vtt")
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>body{{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;"
             "vertical-align:top}audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}.k{color:#e0a458}"
             ".n{color:#7fc8a9}video{max-width:100%;width:960px}</style>"
             f"<h1>{html.escape(title)}</h1>"
             "<p>The whole episode with the English dub. Use the player's CC button for the official English "
             "subtitles or the dub's own script. Below, every subtitle block: the official subtitle, the dub "
             "script, what the dub track was heard to say, and both audios. Acceptable = the dub says what the "
             "subtitle means, clearly, in time; Poor = say what's wrong.</p>"
             f'<video controls preload="metadata" src="{video.name}"><track kind="subtitles" srclang="en" '
             f'label="Official English subtitles" src="{subs}"><track kind="subtitles" srclang="en" '
             f'label="Dub script" src="{script}"></video>'
             '<script src="/listen-feedback.js" defer></script><table>']
    for i, b in enumerate(blocks):
        item = f"blk-{i:03d}"
        a, z = b["start"] - 0.4, b["end"] + 0.6
        orig = enc(src, out / f"{item}-orig.m4a", a, z - a)
        dub = enc(dubbed, out / f"{item}-dub.m4a", a, z - a) if b["script"] else ""
        kept = [f"{ln['ja']} ({'left in Japanese' if ln['fallback'] else ln['kind']})" for ln in b["lines"]
                if (ln["kind"] != "speech" or ln["fallback"]) and ln["kind"] not in ("wordless", "song")]
        note = notes.get(f"{b['start']:.1f}", "")
        parts.append(
            f"<tr><td>{b['start']:.1f}s</td><td>{html.escape(b['subtitle'])}<br><b>{html.escape(b['script'])}</b>"
            f"<br><span class=m>dub heard as: \"{html.escape(b['heard'])}\"</span>"
            + (f"<br><span class=k>original kept: {html.escape('; '.join(kept))}</span>" if kept else "")
            + (f"<br><span class=n>{html.escape(note)}</span>" if note else "")
            + f'</td><td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
            + (f'<td>dub<br><audio controls preload="none" src="{dub}"></audio></td>' if dub else "<td></td>")
            + "</tr>")
    parts.append("</table>")
    (out / "index.html").write_text("".join(parts))
    print("written", out / "index.html", len(blocks), "blocks")


if __name__ == "__main__":
    main()
