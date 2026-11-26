"""Watch a finished dub's lines timed two ways: as they are (start) and following the speaker's
mouth (sync). Remixes copies of the jobs (dev/rerun.py, from Mix, the same takes), picks scenes
where the speaker is on screen and the timing changes most, and writes a blind watching page
under /work/listen.

    python dev/sync_test.py /work/dev-sync 45 42
"""

import json
import random
import subprocess
import sys
from pathlib import Path

from aidub import lines as L

VERSIONS = [("start", ["--set", "fit.timing=start"]), ("sync", ["--set", "fit.timing=sync"])]
SCENE_S = 30.0
SCENES = 3


def rerun(*args):
    print("+ rerun", " ".join(args)[:150], flush=True)
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("rerun.py")), *args], capture_output=True, text=True)
    print(p.stdout[-500:], flush=True)
    if p.returncode:
        print(p.stderr[-1500:], flush=True)
    return p.returncode == 0


def heard_end(ln: L.Line) -> float:
    """Where the English ends, as mixed."""
    import soundfile as sf
    f = ln.fit or {}
    length = sf.info(ln.take).duration
    if f.get("sync"):
        t0, t1, at, tempo = f["sync"][-1]
        length = at + (t1 - t0) / tempo - ln.start
    return f.get("placed", ln.start) + length / f.get("paced_tempo", 1.0)


def scenes(live: Path, start: list[L.Line], duration: float) -> list[tuple[float, float]]:
    """Where the speaker is on screen and the current English most misses the mouth."""
    faces = {}
    if (live / "faces.json").exists():
        faces = {r["id"]: max((f["score"] for f in r.get("faces", [])), default=-9) for r in json.loads((live / "faces.json").read_text())}
    miss = []
    for ln in start:
        if ln.kind == L.SPEECH and ln.take and not ln.fallback:
            seen = faces.get(ln.id, -9) > 0
            miss.append((ln.start, (1.0 if seen else 0.3) * abs(heard_end(ln) - ln.end)))
    picked: list[tuple[float, float]] = []
    span = duration * 0.85 / SCENES
    for k in range(SCENES):
        lo = duration * 0.07 + k * span
        best = max((lo + 5 * i for i in range(int((span - SCENE_S) / 5))),
                   key=lambda t: sum(m for s, m in miss if t <= s < t + SCENE_S - 2))
        picked.append((round(best, 1), round(best + SCENE_S, 1)))
    return picked


def clip(video: str, wav: Path, a: float, b: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a}", "-i", video,
                    "-ss", f"{a}", "-t", f"{b - a}", "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-ac", "2", "-c:a", "aac", "-b:a", "160k", "-map_chapters", "-1", "-map_metadata", "-1",  # (Safari: no chapter track)
                    "-movflags", "+faststart", str(out)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "3", "-i", str(out), "-frames:v", "1", "-vf", "scale=640:-2",
                    "-q:v", "4", str(out).replace("-orig.mp4", ".jpg")], check=True)


def main():
    root, jobs = Path(sys.argv[1]), sys.argv[2:]
    page = Path("/work/listen") / "sync-test"
    page.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)
    rows, key, record = [], {}, []
    for k, jid in enumerate(jobs):
        live = Path("/work") / f"job-{jid}"
        made = {}
        for name, flags in VERSIONS:
            r = root / f"{name}-{jid}"
            done = r / "work" / f"job-{jid}" / "mix" / "dub.wav"
            if done.exists() or rerun(str(r), jid, "--from", "mix", "--until", "mix", "--keep-lines", "--no-review", *flags):
                made[name] = done.parent.parent
        if len(made) < len(VERSIONS):
            continue
        state = json.loads((live / "state.json").read_text())["data"]
        video, duration = state["ingest"]["source"], float(state["ingest"]["duration"])
        found = {n: L.load(w / "lines.json") for n, w in made.items()}
        qc = {n: json.loads((w / "state.json").read_text())["data"].get("qc", {}) for n, w in made.items()}
        for n, (a, b) in enumerate(scenes(live, found["start"], duration)):
            order = list(made)
            rnd.shuffle(order)
            for t, name in enumerate(order):
                item = f"f{k}-s{n}-{'ab'[t]}"
                clip(video, made[name] / "mix" / "dub.wav", a, b, page / f"{item}-orig.mp4")
                key[item] = name
                rows.append(f"<tr><td>Film {k + 1}, scene {n + 1}<br>version {'AB'[t]}</td>"
                            f"<td><video controls playsinline preload=metadata poster='{item}.jpg' src='{item}-orig.mp4'></video>"
                            f"<audio hidden preload=none src='{item}-orig.m4a'></audio></td></tr>")
            record.append({"job": jid, "at": a, "lines": [
                {"start": ln.start, "end": ln.end, "en": ln.en,
                 **{v: round(heard_end(next(x for x in found[v] if x.id == ln.id)) - ln.end, 2) for v in made}}
                for ln in found["start"] if ln.kind == L.SPEECH and ln.take and not ln.fallback and a <= ln.start < b]})
        record.append({"job": jid, "qc": {n: {"free_timing": q.get("free_timing"), "sync": q.get("sync")} for n, q in qc.items()}})
    (page / "key.json").write_text(json.dumps(key, indent=1))
    (root / "scenes.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    (page / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Lip timing</title>"
        "<style>body{font:15px system-ui;background:#14171c;color:#e6e9ef;padding:16px}td{padding:8px;border-bottom:1px solid #333;vertical-align:top}"
        "video{width:min(640px,100%);border-radius:6px}</style>"
        "<h2>Does the English follow their mouths?</h2><p>Each scene is here twice, the same English voices timed two ways, in no "
        "particular order. Watch the speakers' mouths: does the English start, pause and stop with them, and does it still sound "
        "natural (not dragged or rushed)? Mark each Acceptable or Poor; more than one can be Acceptable. Comments help: say "
        "what looked or sounded off and roughly when.</p><table>" + "".join(rows) + "</table><script src=/listen-feedback.js></script>")
    print("page", page, len(key), "clips")


if __name__ == "__main__":
    main()
