#!/usr/bin/env bash
# Usage: dev/cut_sample.sh user@host "/remote/path/file.mkv" START DURATION out.mkv
# Streams an excerpt over SSH (ffmpeg on the remote side writes to stdout), so
# nothing is written on the remote machine. START/DURATION in seconds or hh:mm:ss.
set -euo pipefail
host="$1"; src="$2"; start="$3"; dur="$4"; out="$5"
mkdir -p "$(dirname "$out")"
ssh "$host" "ffmpeg -nostdin -v error -ss '$start' -i \"$src\" -t '$dur' -map 0 -c copy -f matroska -" > "$out"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$out"
