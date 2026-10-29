"""Compare a finished dub with the episode's own English subtitles: what the script
says, what the dubbed English track actually says (read back with speech
recognition), and when it says it.

    python sub_compare.py <job dir> <source video> <dubbed video> <out json> <profile name>

Subtitle cues are grouped into blocks (cues less than 0.8 s apart); each block's
span of the dubbed track, with a little margin, is read back with word times.
Writes one record per block plus per-line timing, for reading side by side.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/work/check/dev")
from aidub import config, media  # noqa: E402
from aidub import lines as L  # noqa: E402
from aidub import speak as S  # noqa: E402
from aidub.job import Job  # noqa: E402
from aidub.profiles import Profiles  # noqa: E402
from aidub.subtitles import clean  # noqa: E402

MARGIN_BEFORE, MARGIN_AFTER, GAP = 0.6, 0.8, 0.8


def english_track(src: Path, out: Path) -> Path:
    """The first English subtitle stream, as SRT."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:s:m:language:eng?", "-map", "-0:s:m:title:SDH?",
                    "-frames:s", "100000", str(out)], check=True)
    return out


def dub_audio(dubbed: Path, out: Path) -> Path:
    """The dubbed English audio track (tagged eng) as 16 kHz mono."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dubbed), "-map", "0:a:m:language:eng", "-ac", "1",
                    "-ar", "16000", str(out)], check=True)
    return out


def main():
    jobdir, src, dubbed, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    work = jobdir / "compare"
    work.mkdir(exist_ok=True)
    cues = [c for c in clean.load(english_track(src, work / "en.srt")) if c.kind == clean.DIALOGUE and c.text]
    wav = dub_audio(dubbed, work / "dub.wav")
    audio, sr = media.read_wav(wav, mono=True)

    blocks = []
    for c in cues:
        if blocks and c.start - blocks[-1]["end"] < GAP and c.end - blocks[-1]["start"] < 14:
            blocks[-1]["cues"].append(c)
            blocks[-1]["end"] = max(blocks[-1]["end"], c.end)
        else:
            blocks.append({"start": c.start, "end": c.end, "cues": [c]})
    for i, b in enumerate(blocks):
        a, z = max(0.0, b["start"] - MARGIN_BEFORE), b["end"] + MARGIN_AFTER
        p = work / f"b{i:03d}.wav"
        media.write_wav(p, audio[int(a * sr): int(z * sr)], sr)
        b.update(a=a, z=z, path=str(p))

    cfg = config.load("/work/bench/config.yaml")
    job = Job(cfg, jobdir.name.removeprefix("job-"), src, Profiles(cfg.appdata).get_or_create(sys.argv[5]), {})
    with job.worker("recognize") as w:
        w.call("load", {"model": cfg.get("stages.recognize").get("model")})
        for i in range(0, len(blocks), 16):
            part = blocks[i:i + 16]
            res = w.call("transcribe", {"items": [{"path": b["path"]} for b in part], "language": "en", "timestamps": True})
            for b, r in zip(part, res["results"]):
                b["heard"] = r.get("text", "")
                b["words"] = [{**x, "start": round(x["start"] + b["a"], 2), "end": round(x["end"] + b["a"], 2)}
                              for x in r.get("words", [])]

    lines = L.load(jobdir / "lines.json")
    rows = []
    for b in blocks:
        mine = [ln for ln in lines if ln.end > b["a"] and ln.start < b["z"]]
        rows.append({
            "start": round(b["start"], 2), "end": round(b["end"], 2),
            "subtitle": " / ".join(c.text for c in b["cues"]),
            "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in b["cues"]],
            "script": " / ".join(ln.en for ln in mine if ln.dubbed),
            "lines": [{"id": ln.id, "start": round(ln.start, 2), "end": round(ln.end, 2), "kind": ln.kind,
                       "ja": ln.source_text, "en": ln.en, "fallback": ln.fallback, "ref": ln.english_ref,
                       "take_s": round(media.duration(ln.take), 2) if ln.dubbed and ln.take else None}
                      for ln in mine],
            "heard": b["heard"],
            "heard_first": b["words"][0]["start"] if b["words"] else None,
            "heard_last": b["words"][-1]["end"] if b["words"] else None,
            "wer_vs_script": round(S.wer(" ".join(ln.en for ln in mine if ln.dubbed), b["heard"]), 2)
            if any(ln.dubbed for ln in mine) else None,
        })
    # Dubbed lines no subtitle covers: English speech with nothing on screen.
    covered = [(b["a"], b["z"]) for b in blocks]
    extra = [{"id": ln.id, "start": round(ln.start, 2), "end": round(ln.end, 2), "ja": ln.source_text, "en": ln.en}
             for ln in lines if ln.dubbed and not any(ln.end > a and ln.start < z for a, z in covered)]
    out.write_text(json.dumps({"blocks": rows, "uncovered_lines": extra}, ensure_ascii=False, indent=1))
    print("blocks", len(rows), "uncovered dubbed lines", len(extra), "->", out)


if __name__ == "__main__":
    main()
