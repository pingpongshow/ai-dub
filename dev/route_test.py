"""Blind comparison of dub scripts on real films, on copies (dev/rerun.py), for the owner's ears.

Analyses each film once, then makes every version of the same scenes: only those scenes are
voiced, so a version takes minutes, not an hour. Writes a listening page under /work/listen
with the versions in no particular order, and versions.json with every version's English next
to the subtitles, to read.

    python dev/route_test.py /work/dev-tree routes  /media/a.mkv /media/b.mkv
    python dev/route_test.py /work/dev-tree speech  /media/a.mkv /media/b.mkv
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from aidub import lines as L

PLANS = {
    "routes": [("english", ["--route", "english"]), ("japanese", ["--route", "source"]),
               ("guided", ["--route", "source_guided"])],
    "speech": [("window", ["--no-subtitles", "--set", "fit.timing=window", "--set", "translate.repair_speech=false"]),
               ("natural", ["--no-subtitles", "--set", "fit.timing=start", "--set", "translate.repair_speech=false"]),
               ("repaired", ["--no-subtitles", "--set", "fit.timing=start", "--set", "translate.repair_speech=true"])],
    "other": [("translated", []),
              ("speech", ["--no-subtitles", "--set", "fit.timing=start", "--set", "translate.repair_speech=true"])],
}
COMMON = ["--no-review", "--set", "stages.faces.enabled=false", "--set", "subtitles.providers.bazarr.enabled=false",
          "--set", "subtitles.require=false"]
SCENE_S = 45.0
SCENES = 6


def rerun(*args):
    print("+ rerun", " ".join(args)[:150], flush=True)
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("rerun.py")), *args], capture_output=True, text=True)
    print(p.stdout[-700:], flush=True)
    if p.returncode:
        print(p.stderr[-1500:], flush=True)
    return p.returncode == 0


def busiest(asr: dict, duration: float) -> list[tuple[float, float]]:
    """Dialogue-heavy scenes spread through the film."""
    words = np.zeros(int(duration) + 1)
    for s in asr["segments"]:
        words[min(int(s["start"]), len(words) - 1)] += len(s.get("text", ""))
    out = []
    span = duration * 0.8 / SCENES
    for k in range(SCENES):
        a, b = duration * 0.1 + k * span, duration * 0.1 + (k + 1) * span - SCENE_S
        best = max(np.arange(a, max(a + 1, b), 15.0), key=lambda t: words[int(t): int(t + SCENE_S)].sum())
        out.append((round(float(best), 1), round(float(best) + SCENE_S, 1)))
    return out


def clip(wav: Path, a: float, b: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a}", "-t", f"{b - a}", "-i", str(wav), "-ac", "2",
                    "-c:a", "aac", "-b:a", "128k", str(out)], check=True)


def human_english(state: dict, work: Path, a: float, b: float) -> str:
    subs = state["subtitles"]
    for c in subs.get("candidates", []):
        if c["language"] == "en" and c.get("human") and not c.get("forced") and (work / "subs" / f"{c['id'].replace(':', '_').replace('/', '_')}.cues.json").exists():
            rows = json.loads((work / "subs" / f"{c['id'].replace(':', '_').replace('/', '_')}.cues.json").read_text())
            return " / ".join(q["text"] for q in rows if q.get("kind") == "dialogue" and q["end"] > a and q["start"] < b)
    return ""


def main():
    root, plan, films = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    page = Path("/work/listen") / f"script-{plan}"
    page.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(11)
    rows, key, record = [], {}, []
    for k, film in enumerate(films):
        jid = str(900 + k + {"routes": 0, "speech": 50, "other": 80}[plan])
        base = root / f"base-{jid}"
        work = base / "work" / f"job-{jid}"
        state = work / "state.json"
        if not (state.exists() and "subtitles" in json.loads(state.read_text())["done"]):
            if not rerun(str(base), jid, "--file", film, "--until", "subtitles", *COMMON):
                continue
        data = json.loads(state.read_text())["data"]
        scenes = busiest(data["asr"], float(data["ingest"]["duration"]))
        versions = {}
        for name, flags in PLANS[plan]:
            r = root / f"{plan}-{name}-{jid}"
            start = "align" if "--no-subtitles" in flags else "subtitles"
            if rerun(str(r), jid, "--copy-of", str(work), "--from", start, "--until", "listen", "--windows",
                     ",".join(f"{a}-{b}" for a, b in scenes), *flags, *COMMON):
                versions[name] = r / "work" / f"job-{jid}"
        if not versions:
            continue
        wav = work / "audio" / "source.wav"
        for n, (a, b) in enumerate(scenes):
            item = f"f{k}-s{n}"
            clip(wav, a, b, page / f"{item}-source.m4a")
            order = [v for v in versions if (versions[v] / "mix" / "dub.wav").exists()]
            rnd.shuffle(order)
            scene = {"film": Path(film).name, "at": a, "subs_english": "", "versions": {}}
            for t, name in enumerate(order):
                v = f"{item}-{'abc'[t]}"
                clip(versions[name] / "mix" / "dub.wav", a, b, page / f"{v}-orig.m4a")
                key[v] = name
                lines = L.load(versions[name] / "lines.json")
                said = [ln for ln in lines if ln.kind == L.SPEECH and ln.en and ln.end > a and ln.start < b]
                scene["versions"][name] = [ln.en for ln in said]
                scene["source"] = [ln.source_text for ln in said]
                st = json.loads((versions[name] / "state.json").read_text())["data"]
                scene.setdefault("qc", {})[name] = {k: st.get("qc", {}).get(k) for k in ("readback", "free_timing", "transcript_repaired")}
                scene["subs_english"] = scene["subs_english"] or human_english(data, work, a, b)
                scene.setdefault("track", {})[name] = (st.get("qc", {}).get("subtitle_track") or {}).get("why", "")
                rows.append(f"<tr><td>Film {k + 1}, scene {n + 1}: version {'ABC'[t]}</td>"
                            f"<td><audio controls preload=none src='{item}-source.m4a'></audio></td>"
                            f"<td><audio controls preload=none src='{v}-orig.m4a'></audio></td></tr>")
            record.append(scene)
    (page / "key.json").write_text(json.dumps(key, indent=1))
    (root / f"{plan}-versions.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    (page / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Script comparison</title><style>body{font:15px system-ui;background:#14171c;color:#e6e9ef;padding:16px}"
        "td{padding:8px;border-bottom:1px solid #333;vertical-align:top}audio{width:240px}</style>"
        "<h2>Which script works best?</h2><p>The first player is the original scene, the second a dubbed version of it. Each scene has "
        "several versions in no particular order. Mark each: Acceptable if it sounds natural and says the right thing, Poor if not. "
        "A comment helps, especially if a version says something wrong.</p><table>" + "".join(rows) + "</table><script src=/listen-feedback.js></script>")
    print("page", page, len(key), "versions")


if __name__ == "__main__":
    main()
