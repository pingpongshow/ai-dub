"""Blind listening comparison of voice-model speed settings, to pick the fastest
one that sounds as good as the current setting.

Each chosen line is voiced once per setting with the same seed, so only the
setting differs. The page shows the original, then the versions in a random
order per line, each marked Acceptable or Poor (listen-feedback.js). Which
letter is which setting is in `speed-key.json` next to the page, with the
measured voicing time of every setting in `speed-timing.json`.

    python speed_page.py /work/check/config.yaml /work/listen/<page> clip:id,id ... (run outside /work/check/dev
    so the installed aidub is used)
"""

import html
import json
import random
import sys
import time
from pathlib import Path

from aidub import config, media
from aidub import fit as F
from aidub import lines as L
from aidub import speak as S
from aidub.job import Job
from aidub.profiles import Profiles

sys.path.append("/work/check/dev")
from review_page import clip_span, enc  # noqa: E402

SETTINGS = {  # name: (candidates weighed per take, refinement steps); the first is the current setting
    "beams3-steps25": (3, 25),
    "beams1-steps25": (1, 25),
    "beams3-steps12": (3, 12),
    "beams1-steps12": (1, 12),
}


def main():
    cfg = config.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.data["paths"]["work"]).parent
    picks = []
    for arg in sys.argv[3:]:
        clip, ids = arg.split(":")
        prof = Profiles(cfg.appdata).get_or_create(f"check {clip}")
        job = Job(cfg, f"chk-{clip}", root / f"{clip}.mkv", prof, {})
        lines = L.load(job.path("lines.json"))
        by_id = {ln.id: ln for ln in lines}
        names = {cid: ch["name"] for cid, ch in prof.characters.items()}
        for i in (int(x) for x in ids.split(",")):
            picks.append((clip, job, by_id[i], lines, names.get(by_id[i].character, "")))

    # Voice every line once per setting; one process, so the timings compare.
    timing = {}
    takes: dict[tuple[str, int], dict[str, str]] = {}
    job0 = picks[0][1]
    with job0.worker("speak") as w:
        S._load_tts(job0, w)
        warm = S._requests(job0, picks[0][2], picks[0][2].en, 1, "spd-warm", *_dlg(job0))[0]
        w.call("synthesize", {"items": [warm]})
        for name, (beams, steps) in SETTINGS.items():
            reqs = []
            for clip, job, ln, _, _ in picks:
                r = S._requests(job, ln, ln.en, 1, f"spd-{name}", *_dlg(job))[0]
                r.update(num_beams=beams, diffusion_steps=steps)
                reqs.append(r)
                takes.setdefault((clip, ln.id), {})[name] = r["out"]
            t0 = time.time()
            w.call("synthesize", {"items": reqs})
            spent = time.time() - t0
            audio = sum(F.trim(Path(r["out"])) for r in reqs if Path(r["out"]).exists())
            timing[name] = {"seconds": round(spent, 1), "audio_seconds": round(audio, 1),
                            "per_take": round(spent / len(reqs), 2)}
    base = timing[next(iter(SETTINGS))]["seconds"]
    for t in timing.values():
        t["vs_current"] = f"{100 * (1 - t['seconds'] / base):.0f}% less time"
    (out / "speed-timing.json").write_text(json.dumps(timing, indent=1))

    parts = ["<!doctype html><meta charset=utf-8><title>Voice speed comparison</title><style>body{font:14px system-ui;"
             "margin:16px;background:#14171c;color:#e6e9ef}td{padding:6px;border-bottom:1px solid #2d333d;vertical-align:top}"
             "audio{height:28px;width:170px}.m{color:#9aa3b2;font-size:12px}h3{margin:22px 0 4px}</style>"
             "<h1>Voice settings: which versions sound right?</h1>"
             "<p>Each line is voiced four ways. One is the current setting, the others are faster; the order is "
             "shuffled for every line, so the letters don't tell you which is which. Play the original, then mark "
             "<b>every</b> version Acceptable (as good as you'd want in the dub) or Poor (say what's wrong: robotic, "
             "slurred, wrong emotion, glitch, unlike the speaker...). The fastest setting that is Acceptable "
             "wherever the current one is becomes the default.</p>"
             '<script src="/listen-feedback.js" defer></script>']
    key = {}
    for clip, job, ln, lines, who in picks:
        src = root / f"{clip}.mkv"
        order = list(SETTINGS)
        random.Random(f"{clip}-{ln.id}").shuffle(order)
        parts.append(f"<h3>{html.escape(clip)} {ln.start:.1f}s · {html.escape(who)}</h3>"
                     f"<p class=m>{html.escape(ln.source_text)}<br><b>{html.escape(ln.en)}</b></p><table>")
        a, dur = clip_span(ln.__dict__, [o.__dict__ for o in lines])
        for k, name in enumerate(order):
            item = f"{clip}-{ln.id:03d}{chr(97 + k)}"
            key[item] = name
            orig = enc(src, out / f"{item}-orig.m4a", a, dur)
            dub = enc(Path(takes[(clip, ln.id)][name]), out / f"{item}-dub.m4a")
            parts.append(f"<tr><td><b>version {chr(65 + k)}</b></td>"
                         f'<td>original<br><audio controls preload="none" src="{orig}"></audio></td>'
                         f'<td>dub<br><audio controls preload="none" src="{dub}"></audio></td></tr>')
        parts.append("</table>")
    (out / "speed-key.json").write_text(json.dumps(key, indent=1))
    (out / "index.html").write_text("".join(parts))
    print(json.dumps(timing, indent=1))
    print("written", out / "index.html", len(key), "rows")


_DLG: dict[str, tuple] = {}


def _dlg(job):
    if job.id not in _DLG:
        _DLG[job.id] = media.read_wav(job.path("audio", "dialogue.wav"), mono=True)
    return _DLG[job.id]


if __name__ == "__main__":
    main()
