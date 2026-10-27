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
# Some libraries look the running user up by uid; give PUID/PGID names.
if ! getent group "$PGID" >/dev/null; then echo "aidub:x:$PGID:" >> /etc/group; fi
if ! getent passwd "$PUID" >/dev/null; then echo "aidub:x:$PUID:$PGID:Ai Dub:/config/home:/bin/bash" >> /etc/passwd; fi
for d in /config /config/home /models /work; do
  chown "$PUID:$PGID" "$d" 2>/dev/null || true
done
if [ "$(id -u)" = "0" ]; then
  # gosu takes HOME from the user database (none for 99), so set it explicitly.
  exec gosu "$PUID:$PGID" env HOME=/config/home "$@"
fi
exec "$@"
