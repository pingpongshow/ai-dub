"""Command line: `aidub serve`, `aidub setup`, `aidub doctor`, and development tools."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from . import __version__, config, gpu, originals, workers


def _duration(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", text.strip())
    if not m:
        raise argparse.ArgumentTypeError("use e.g. 90s, 5m or 1h")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def cmd_serve(cfg, args) -> None:
    import uvicorn

    from .api import create_app
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(cfg), host=args.host or cfg.get("server.host", "0.0.0.0"),
                port=args.port or int(cfg.get("server.port", 8790)), log_level="info")


def cmd_run(cfg, args) -> None:
    """Dub one file in the foreground (development). With --preview, only the
    first minutes are dubbed and the result is written to --output; the source
    file is never touched."""
    from . import pipeline
    from .job import Cancelled, Job, Skipped
    from .profiles import Profiles, guess_name
    src = Path(args.file).resolve()
    profiles = Profiles(cfg.appdata)
    prof = profiles.get_or_create(args.profile or guess_name(src))
    opts = {"review": "off", "speech_ok": True}  # no screen to stop for on the command line
    if args.preview:
        opts["preview_s"] = args.preview
        opts["output"] = str(Path(args.output or f"{src.stem}.preview.mkv").resolve())
    elif not args.in_place:
        sys.exit("refusing to change the file: pass --preview (with --output) or --in-place")
    if args.audio_stream is not None:
        opts["audio_stream"] = args.audio_stream
    if args.language:
        opts["language"] = args.language
    if args.redo:
        opts["redo"] = True
    job_id = args.job_id or f"cli-{int(time.time())}"

    def on_status(d):
        msg = d.get("message", "")
        if msg:
            p = d.get("progress")
            print(f"[{d.get('step', '')}] {msg}" + (f" ({p * 100:.0f}%)" if p is not None else ""), flush=True)

    job = Job(cfg, job_id, src, prof, opts, on_status=on_status)
    try:
        rec = pipeline.run(job, until=args.until)
    except Skipped as exc:
        print(f"skipped: {exc}")
        job.remove_work()
        return
    except Cancelled:
        print("cancelled")
        return
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    if not args.keep_work and rec.get("delivered"):
        job.remove_work()
    else:
        print(f"working folder kept: {job.work}")


def cmd_setup(cfg, args) -> None:
    names = args.workers or sorted({cfg.get(f"stages.{s}.worker") for s in cfg.get("stages", {})
                                    if cfg.get(f"stages.{s}.worker")})
    failed = []
    for n in names:
        try:
            workers.setup(cfg, n, force=args.force)
        except Exception as exc:
            print(f"{n}: FAILED: {exc}")
            failed.append(n)
    if args.download:
        for n in names:
            if n in failed:
                continue
            stage = next((s for s in cfg.get("stages", {}) if cfg.get(f"stages.{s}.worker") == n), None)
            try:
                res = workers.prefetch(cfg, n, dict(cfg.get(f"stages.{stage}") or {}))
                print(f"{n}: weights ready {res or ''}")
            except Exception as exc:
                print(f"{n}: download FAILED: {exc}")
                failed.append(n)
    if failed:
        sys.exit(f"failed: {', '.join(sorted(set(failed)))}")


def cmd_doctor(cfg, args) -> None:
    """Phase 0 check: is each worker installed, does it start, does it see the GPU."""
    g = gpu.query()
    print(f"Ai Dub {__version__}")
    print(f"GPU: {g.name or 'none'} {g.used_mb}/{g.total_mb} MB used" if g.present else "GPU: none visible")
    names = args.workers or sorted({cfg.get(f"stages.{s}.worker") for s in cfg.get("stages", {})
                                    if cfg.get(f"stages.{s}.worker")})
    ok = True
    for n in names:
        if not workers.env_ready(cfg, n):
            print(f"  {n:16s} not installed")
            ok = False
            continue
        try:
            with workers.Worker(cfg, n, log=(print if args.verbose else None)) as w:
                info = w.call("doctor", {})
            print(f"  {n:16s} ok  {json.dumps(info, ensure_ascii=False)}")
        except Exception as exc:
            print(f"  {n:16s} FAILED  {exc}")
            ok = False
    if not ok:
        sys.exit(1)


def cmd_originals(cfg, args) -> None:
    from .jobs import Store
    store = Store(cfg.appdata / "aidub.db")
    if args.action == "scan":
        print(f"registered {originals.scan(store, [Path(m['path']) for m in cfg.media_roots])} new set-aside files")
    elif args.action == "list":
        info = originals.listing(store)
        for o in info["items"]:
            print(f"{o['id']:5d}  {o['kind']:8s}  {o['size'] / 1e9:7.2f} GB  {o['path']}")
        print(f"total {info['total'] / 1e9:.2f} GB")
    elif args.action == "restore":
        for i in args.ids:
            print(originals.restore(store, i))
    elif args.action == "delete":
        if not args.yes:
            sys.exit("deleting is permanent; pass --yes to confirm")
        n, freed = originals.delete(store, args.ids)
        print(f"deleted {n} files, {freed / 1e9:.2f} GB")


def cmd_audit(cfg, args) -> None:
    """Read back the English track of finished dubs against their subtitles. Changes nothing."""
    from . import audit
    from .jobs import Store
    paths = [Path(p) for p in args.files]
    if args.all:
        store = Store(cfg.appdata / "aidub.db")
        paths += [Path(j["path"]) for j in store.list({"done"}) if (j.get("record") or {}).get("delivered")
                  and (j.get("record") or {}).get("verify") not in ("restored", "redo")]
    paths = [p for p in dict.fromkeys(paths) if p.exists()]
    audit.run(cfg, paths, Path(args.output) if args.output else None)


def cmd_corrections(cfg, args) -> None:
    """What reviews changed (appdata/corrections); with --against, how a job's current labels compare."""
    from . import corrections
    if args.keep_from:
        for d in sorted(Path(args.keep_from).glob("job-*")):
            state = d / "state.json"
            if state.exists() and (d / "review" / "base-lines.json").exists():
                data = json.loads(state.read_text())["data"]
                print(corrections.save(cfg.appdata, Path((data.get("ingest") or {}).get("source", d.name)).name,
                                       d.name.removeprefix("job-"), d, data.get("qc") or {}))
    if args.against:
        lines = json.loads(Path(args.against).read_text())
        saved = next(s for s in corrections.load_all(cfg.appdata) if str(s.get("job")) == str(args.job))
        print(json.dumps(corrections.compare(saved, lines), ensure_ascii=False, indent=1))
        return
    for row in corrections.summary(cfg.appdata):
        print(row)


def cmd_voices(cfg, args) -> None:
    """Design the small library of generic voices used for minor speakers."""
    from .voices import design_library
    design_library(cfg, force=args.force, log=print)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="aidub", description="Local voice-cloned dubbing into English")
    ap.add_argument("--config", help="config file (default /config/config.yaml or $AIDUB_CONFIG)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web UI and queue")
    s.add_argument("--host")
    s.add_argument("--port", type=int)

    s = sub.add_parser("run", help="dub one file in the foreground (development)")
    s.add_argument("file")
    s.add_argument("--profile")
    s.add_argument("--preview", type=_duration, help="dub only the first N (e.g. 5m) into --output")
    s.add_argument("--output", help="where the preview file goes")
    s.add_argument("--in-place", action="store_true", help="really add the track to the file (like the queue)")
    s.add_argument("--audio-stream", type=int)
    s.add_argument("--language")
    s.add_argument("--redo", action="store_true")
    s.add_argument("--until", help="stop after this step")
    s.add_argument("--job-id", help="resume a kept working folder")
    s.add_argument("--keep-work", action="store_true")

    s = sub.add_parser("setup", help="build worker environments (and download weights)")
    s.add_argument("workers", nargs="*")
    s.add_argument("--force", action="store_true")
    s.add_argument("--download", action="store_true")

    s = sub.add_parser("doctor", help="check every worker starts and sees the GPU")
    s.add_argument("workers", nargs="*")
    s.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("originals", help="list, restore or delete set-aside files")
    s.add_argument("action", choices=["list", "scan", "restore", "delete"])
    s.add_argument("ids", nargs="*", type=int)
    s.add_argument("--yes", action="store_true")

    s = sub.add_parser("audit", help="read back the English track of finished dubs (changes nothing)")
    s.add_argument("files", nargs="*")
    s.add_argument("--all", action="store_true", help="every delivered dub in the job history")
    s.add_argument("--output", help="write the report to this JSON file")

    s = sub.add_parser("corrections", help="what reviews changed, kept for testing rule changes")
    s.add_argument("--keep-from", help="save corrections from every job folder under this path")
    s.add_argument("--against", help="compare this lines.json with a job's corrections (needs --job)")
    s.add_argument("--job")

    s = sub.add_parser("voices", help="design the generic minor-speaker voices")
    s.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    cfg = config.load(args.config)
    {"serve": cmd_serve, "run": cmd_run, "setup": cmd_setup, "doctor": cmd_doctor,
     "originals": cmd_originals, "voices": cmd_voices, "audit": cmd_audit, "corrections": cmd_corrections}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
