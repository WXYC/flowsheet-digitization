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

# Bundle JSONs and page PNGs are deploy-time artifacts — overlay them on
# every boot so geometry/bbox fixes in a new build reach the live volume.
# Volunteer state (.verified.json, .corrections.json) lives at distinct
# filename suffixes in the same directory and is never touched.
if [ -d /seed/verifier ]; then
  bundle_count=$(find /seed/verifier -maxdepth 1 -name '*.bundle.json' | wc -l | tr -d ' ')
  if [ "$bundle_count" -gt 0 ]; then
    echo "[entrypoint] overlaying $bundle_count bundle JSONs onto $DATA_ROOT/verifier"
    cp -f /seed/verifier/*.bundle.json "$DATA_ROOT/verifier/" 2>/dev/null || true
  fi
fi

if [ -d /seed/pages ]; then
  echo "[entrypoint] overlaying /seed/pages onto $DATA_ROOT/pages (newer files only)"
  # -u: copy only when source is newer or destination is missing.
  cp -ru /seed/pages/. "$DATA_ROOT/pages/"
fi

export DATA_ROOT VERIFIER_HOST
exec python verifier/serve.py
