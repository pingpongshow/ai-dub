#!/usr/bin/env bash
# Runs Ai Dub as PUID:PGID (Unraid: nobody:users = 99:100) so files it writes
# have the usual owner. Only the top of each data folder is chowned.
set -euo pipefail
PUID="${PUID:-99}"
PGID="${PGID:-100}"
mkdir -p /config/home /models /work
if [ ! -f /config/config.yaml ]; then
  cp /app/docker/config.example.yaml /config/config.yaml 2>/dev/null || true
fi
for d in /config /config/home /models /work; do
  chown "$PUID:$PGID" "$d" 2>/dev/null || true
done
if [ "$(id -u)" = "0" ]; then
  exec gosu "$PUID:$PGID" "$@"
fi
exec "$@"
