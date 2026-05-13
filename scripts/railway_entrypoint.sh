#!/usr/bin/env bash
# Railway boot script: hydrate the persistent volume from the baked seed
# on first boot, then run the verifier server.
#
# Railway mounts a volume at $DATA_ROOT (default /data). The image carries
# /seed/{verifier,pages} from the Docker build. On first boot we copy
# seed -> volume, so the bundle + page PNG corpus is available; on
# subsequent boots we keep whatever the volunteer's edits left behind.

set -euo pipefail

: "${DATA_ROOT:=/data}"
: "${VERIFIER_HOST:=0.0.0.0}"

mkdir -p "$DATA_ROOT/verifier" "$DATA_ROOT/pages"

# Seed bundles if the volume's verifier/ has none yet. Page PNGs are
# seeded independently — they're append-only and don't conflict with
# corrections.json / verified.json the volunteer writes.
if [ -d /seed/verifier ] && [ -z "$(ls -A "$DATA_ROOT/verifier" 2>/dev/null || true)" ]; then
  echo "[entrypoint] seeding $DATA_ROOT/verifier from /seed/verifier"
  cp -r /seed/verifier/. "$DATA_ROOT/verifier/"
fi

if [ -d /seed/pages ] && [ -z "$(ls -A "$DATA_ROOT/pages" 2>/dev/null || true)" ]; then
  echo "[entrypoint] seeding $DATA_ROOT/pages from /seed/pages"
  cp -r /seed/pages/. "$DATA_ROOT/pages/"
fi

export DATA_ROOT VERIFIER_HOST
exec python verifier/serve.py
