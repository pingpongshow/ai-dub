"""A blind listening page for #10: lines the voice check moved off their character.

Each line the app voiced from its own short clip (because its voice print "didn't match")
is voiced again both ways, from that clip and in the character's usual voice. The page plays the original and, in random
order with no labels, the two versions, each on its own row to mark Acceptable or Poor.
Results stay on the server: feedback.json next to the page, key.json says which was which.
Works on copies; nothing of the job changes.

    python dev/voice_test.py /work/dev-tree 33,36,28 --lines 8
"""

import argparse
import json
import random
import shutil
import subprocess
from html import escape
from pathlib import Path

from aidub import config, media
from aidub import lines as L
from aidub import speak as S
from aidub.job import Job
from aidub.jobs import Store
from aidub.profiles import Profiles, guess_name


def enc(src, dst):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("jobs")
    ap.add_argument("--lines", type=int, default=8)
    args = ap.parse_args()
    cfg = config.load(None)
    live_work = cfg.work
    page = live_work / "listen" / "voice-consistency"
    page.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(10)
    rows, key = [], {}
    root = Path(args.root) / "voice-test"
    for jid in args.jobs.split(","):
        live = live_work / f"job-{jid}"
        state = json.loads((live / "state.json").read_text())
        src = Path(state["data"]["ingest"]["source"])
        row = Store(cfg.appdata / "aidub.db").get(int(jid)) or {}
        prof = Profiles(cfg.appdata).get_or_create(row.get("profile") or guess_name(src))
        lines = L.load(live / "lines.json")
        counts = {}
        for ln in lines:
            counts[ln.character] = counts.get(ln.character, 0) + 1
        main_chars = {c for c, n in counts.items() if c and n >= 40}
        pool = [ln for ln in lines if ln.kind == L.SPEECH and ln.take and not ln.fallback and ln.character in main_chars
                and ln.voice_note.startswith("own line as reference (") and ln.duration >= 0.8 and prof.refs(ln.character)]
        rnd.shuffle(pool)
        pool = pool[: args.lines]
        cfg.data["paths"]["work"] = str(root)
        job = Job(cfg, jid, src, prof, {})
        shutil.copyfile(live / "state.json", job.path("state.json"))
        dlg, sr = media.read_wav(live / "audio" / "dialogue.wav", mono=True)
        reqs = []
        for ln in pool:
            # Both voiced now, the same way: the voice is the only difference.
            voice = str(prof.root / prof.refs(ln.character)[0]["file"])
            own = S._requests(job, ln, ln.en, 1, "own", dlg, sr)[0]
            own["speaker_ref"] = str(S._orig_clip(job, ln, dlg, sr))
            bank = S._requests(job, ln, ln.en, 1, "bank", dlg, sr, voice=voice)[0]
            bank["speaker_ref"] = voice
            ln.fit["own"], ln.fit["bank"] = own["out"], bank["out"]
            reqs += [own, bank]
        if reqs:
            with job.worker("speak") as w:
                S._load_tts(job, w)
                S._synthesize(job, w, reqs, "voice test")
            for r in reqs:
                if Path(r["out"]).exists():
                    S.F.trim(Path(r["out"]))
        cfg.data["paths"]["work"] = str(live_work)
        for ln in pool:
            item = f"j{jid}-{ln.id}"
            orig = dlg[int(ln.start * sr): int(ln.end * sr)]
            media.write_wav(page / "o.wav", orig, sr)
            enc(page / "o.wav", page / f"{item}-source.m4a")
            versions = [("own", Path(ln.fit["own"])), ("bank", Path(ln.fit["bank"]))]
            rnd.shuffle(versions)
            for k, (which, p) in enumerate(versions):
                if not p.exists():
                    continue
                v = f"{item}-{'ab'[k]}"
                enc(p, page / f"{v}-orig.m4a")
                key[v] = which
                rows.append(f"<tr><td>{escape(ln.en)}<br><small>{'ab'[k].upper()} of this line, film {jid}, {ln.voice_note.split('(')[-1].rstrip(')')}</small></td>"
                            f"<td><audio controls preload=none src='{item}-source.m4a'></audio></td>"
                            f"<td><audio controls preload=none src='{v}-orig.m4a'></audio></td></tr>")
    (page / "o.wav").unlink(missing_ok=True)
    (page / "key.json").write_text(json.dumps(key, indent=1))
    (page / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Voice consistency</title><style>body{font:15px system-ui;background:#14171c;color:#e6e9ef;padding:16px}"
        "td{padding:8px;border-bottom:1px solid #333;vertical-align:top}audio{width:240px}</style>"
        "<h2>Does this sound like the same character?</h2><p>The first player is the original Japanese line, the second the English. "
        "Each line appears twice, in two versions, in no particular order. Mark each version: Acceptable if it sounds like that character "
        "in the film, Poor if not. A comment helps.</p><table>" + "".join(rows) + "</table><script src=/listen-feedback.js></script>")
    print("page:", page, len(key), "versions")


if __name__ == "__main__":
    main()
