"""Re-run a real job's later steps on a copy, to judge a code change on real media.

Nothing of the job, its file or the app's data is changed: the job is copied under
<root>/work (big audio by hard link), the appdata profiles under <root>/app, and it stops
before Deliver. With no job folder to copy, --file starts a fresh dub the same way.

    python dev/rerun.py /work/dev-tree 33 --from align --until align
    python dev/rerun.py /work/dev-tree 33 --from translate --until listen
    python dev/rerun.py /work/dev-tree 340 --file /media/movies/x.mkv --until listen

Prints how the labels compare with the job's saved corrections.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from aidub import config, corrections, pipeline
from aidub import lines as L
from aidub.job import Job, Paused
from aidub.jobs import Store
from aidub.profiles import Profiles, guess_name

def link_tree(src: Path, dst: Path, skip: set[str]) -> None:
    """Copy a job folder. Only the analysed audio, which no step from Align on writes, is
    hard-linked: a step overwriting a linked file would change the real job's copy too."""
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if rel.parts[0] in skip:
            continue
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            if (rel.parts[0] == "audio" and p.suffix in (".wav", ".npy")) or p.suffix in (".mp4", ".m4a"):
                os.link(p, q)
            else:
                shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("job")
    ap.add_argument("--from", dest="start", default="align")
    ap.add_argument("--until", default="listen")
    ap.add_argument("--file")
    ap.add_argument("--no-decisions", action="store_true", help="labels only: leave the review's decisions out")
    ap.add_argument("--no-review", action="store_true", help="don't stop for review")
    ap.add_argument("--copy-of", help="copy this job folder, not the live job with this number")
    ap.add_argument("--route", help="subtitles.prefer for this run")
    ap.add_argument("--windows", help="voice only these stretches: 100-145,600-645")
    ap.add_argument("--set", action="append", default=[], help="a setting for this run: fit.timing=start")
    ap.add_argument("--keep-lines", action="store_true", help="from a late step: the copied job's lines as they are, not the review's base")
    ap.add_argument("--no-subtitles", action="store_true", help="as if the file had no subtitles: translated from the speech")
    args = ap.parse_args()

    live = config.load(None)
    live_work, live_app = live.work, live.appdata
    root = Path(args.root)
    app = root / "app"
    if not app.exists():
        shutil.copytree(live_app / "profiles", app / "profiles")
        if (live_app / "voices").exists():
            shutil.copytree(live_app / "voices", app / "voices")
    live.data["paths"]["work"] = str(root / "work")
    live.data["paths"]["appdata"] = str(app)
    corrections_from = corrections.folder(live_app)
    cfg = live
    work = cfg.work / f"job-{args.job}"
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    if args.file:
        src = Path(args.file)
        name = guess_name(src)
    else:
        at = pipeline.STEP_NAMES.index(args.start)
        if at <= pipeline.STEP_NAMES.index("diarize"):
            sys.exit("the audio is shared with the real job: start at Recognise or later, or use --file")
        link_tree(Path(args.copy_of) if args.copy_of else live_work / f"job-{args.job}", work,
                  {"verify", "mix", "deliver"} | ({"lines"} if at <= pipeline.STEP_NAMES.index("speak") else set()))
        state = json.loads((work / "state.json").read_text())
        keep = pipeline.STEP_NAMES[: pipeline.STEP_NAMES.index(args.start)]
        state["done"] = [s for s in state["done"] if s in keep]
        if args.no_subtitles:
            asr = state["data"].get("asr") or {}
            state["data"]["subtitles"] = {"english": None, "source": None, "candidates": [], "file_languages": [],
                                          "source_language": asr.get("language", "")}
            state["done"] = list(dict.fromkeys(state["done"] + ["subtitles"]))
        (work / "state.json").write_text(json.dumps(state, ensure_ascii=False))
        src = Path(state["data"]["ingest"]["source"])
        name = guess_name(src)
        base = work / "review" / "base-lines.json"
        if pipeline.STEP_NAMES.index(args.start) > pipeline.STEP_NAMES.index("align") and base.exists() and not args.keep_lines:
            shutil.copyfile(base, work / "lines.json")
    if args.no_decisions:
        (work / "review" / "decisions.json").unlink(missing_ok=True)
    row = Store(live_app / "aidub.db").get(int(args.job)) or {}
    prof = Profiles(cfg.appdata).get_or_create(row.get("profile") or name)
    job = Job(cfg, args.job, src, prof, {**(row.get("options") or {}), **({"review": "off"} if args.no_review else {}),
                                                **({"speech_ok": True} if args.no_subtitles else {}),
                                                **({"subtitle_route": args.route} if args.route else {})})
    for kv in args.set:
        k, v = kv.split("=", 1)
        at = cfg.data
        for part in k.split(".")[:-1]:
            at = at.setdefault(part, {})
        at[k.split(".")[-1]] = yaml.safe_load(v)
    try:
        if args.windows:
            pipeline.run(job, until="translate")
            spans = [tuple(map(float, w.split("-"))) for w in args.windows.split(",")]
            lines = L.load(job.path("lines.json"))
            for ln in lines:  # only these scenes are voiced: the rest keeps the original
                if ln.kind == L.SPEECH and not any(ln.end > a and ln.start < b for a, b in spans):
                    ln.kind, ln.en, ln.kept_why = L.BACKGROUND, "", "outside the test scenes"
            L.save(lines, job.path("lines.json"))
        pipeline.run(job, until=args.until)
    except Paused as exc:
        print("paused:", exc)
    for f in (corrections_from.glob(f"job-{args.job}.json") if corrections_from.exists() else []):
        saved = json.loads(f.read_text())
        c = corrections.compare(saved, json.loads((work / "lines.json").read_text()))
        print(f"{saved.get('name', '')[:40]}: {c['lines']} lines; differ from the final labels: "
              f"{c['wrong_before']} first time, {c['wrong_now']} now "
              f"({len(c['should_dub'])} should be dubbed, {len(c['should_keep'])} should keep the original)")
        (work / "label-diff.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
    print(json.dumps({k: job.qc.get(k) for k in ("readback", "free_timing", "faces", "sentences_joined", "lines_split_between_speakers",
                                                 "spoken_under_titles", "dub_channels", "quiet_lines_raised", "fit")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
