# Development tools (not shipped)

Used while choosing and tuning models and settings. Sample clips live in `dev/samples/`,
which is git-ignored; nothing here is part of the app.

- `cut_sample.sh` - copy a short excerpt of a file (stream copy, no re-encode)
  from a remote host over SSH, without creating anything on that host.
- `compare.py` - build a blind A/B listening page from several preview dubs of
  the same clip (`aidub run clip.mkv --preview 5m --output out/<variant>.mkv`).
- `rerun.py` - run a real job's later steps on a copy, stopping before Deliver, and compare its
  labels with the review's saved corrections. Options:
  - `--copy-of DIR` copies another job folder instead of the live job;
  - `--from STEP` / `--until STEP` choose the steps;
  - `--windows 100-145,600-645` voices only those stretches, so a version takes minutes;
  - `--set key=value` changes a setting for this run (`--set mix.level_spread=0.5`);
  - `--route guided|english` picks the subtitle route;
  - `--keep-lines` keeps the copied job's lines as they are when starting from a late step
    (to remix the same takes with another setting);
  - `--no-subtitles` dubs as if the file had none.
- `route_test.py` - a blind listening page comparing versions of the same scenes, made by
  `rerun.py` on copies. Each plan in `PLANS` lists the versions: subtitle routes, mix
  settings (`--takes-of` remixes an earlier version's takes, so only the mix differs). It
  analyses each film once, picks dialogue-heavy scenes spread through it, and writes the
  page under `/work/listen`, its answer key, and a JSON file with every version's English
  next to the subtitles. Pass `""` in place of a film to keep the other films' numbers.

      python dev/route_test.py /work/dev-tree pair /media/a.mkv /media/b.mkv

- `voice_test.py` - a blind listening page: lines voiced from their own clip against the
  character's usual voice.
- `phase0/` - the first benchmarks of candidate models and the review pages built from them.

Listening pages save each answer (Acceptable or Poor, and a comment) to `feedback.json`
next to the page; `key.json` says which version was which.

To test code on the server without pushing, copy it into a folder under `/work` and run
it with `PYTHONPATH` pointing there; the workers still come from the image unless
`AIDUB_WORKERS` points elsewhere too.
