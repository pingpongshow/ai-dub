# Development tools (not shipped)

Used while choosing and tuning models. Sample clips live in `dev/samples/`,
which is git-ignored; nothing here is part of the image.

- `cut_sample.sh` - copy a short excerpt of a file (stream copy, no re-encode)
  from a remote host over SSH, without creating anything on that host.
- `compare.py` - build a blind A/B listening page from several preview dubs of
  the same clip (`aidub run clip.mkv --preview 5m --output out/<variant>.mkv`).
