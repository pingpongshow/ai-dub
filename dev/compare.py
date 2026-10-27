"""Blind A/B listening page from preview dubs of the same clip.

    python dev/compare.py dev/listening/out/*.mkv --start 30 --length 20

Extracts the "English (AI dub)" track of each file (same excerpt), shuffles
them under neutral labels, and writes dev/listening/out/index.html with a
key file you open only after listening.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--length", type=float, default=30)
    ap.add_argument("--out", default="dev/listening/out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in a.files]
    random.shuffle(files)
    key = {}
    rows = []
    for i, f in enumerate(files):
        label = chr(65 + i)
        probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(f)],
                                          capture_output=True, text=True, check=True).stdout)
        idx = next(s["index"] for s in probe["streams"]
                   if s["codec_type"] == "audio" and "AI dub" in (s.get("tags") or {}).get("title", ""))
        dst = out / f"{label}.m4a"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(a.start), "-i", str(f), "-t",
                        str(a.length), "-map", f"0:{idx}", "-c:a", "aac", "-b:a", "192k", str(dst)], check=True)
        key[label] = f.name
        rows.append(f'<p><b>{label}</b> <audio controls src="{dst.name}"></audio></p>')
    (out / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Listening test</title>"
                                    "<h1>Which sounds best?</h1>" + "".join(rows))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"open {out / 'index.html'}; answers in {out / 'key.json'}")


if __name__ == "__main__":
    main()
