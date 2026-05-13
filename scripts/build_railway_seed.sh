#!/usr/bin/env bash
# Build the Docker image's seed tree: every bundle in data/verifier/ plus
# the specific page PNGs they reference. Excludes the rest of data/pages/
# (~18K PNGs, ~9 GB) so the image stays small.
#
# Output: .seed/{verifier,pages}/  (gitignored; the Dockerfile COPYs from it)
#
# The Railway entrypoint copies /seed -> /data on first boot, populating
# the persistent volume from the baked snapshot.

set -euo pipefail

cd "$(dirname "$0")/.."

SEED=.seed
PYTHON="${PYTHON:-.venv/bin/python}"

rm -rf "$SEED"
mkdir -p "$SEED/verifier" "$SEED/pages"

bundles=(data/verifier/*.bundle.json)
if [ "${bundles[0]}" = "data/verifier/*.bundle.json" ]; then
  echo "no bundles in data/verifier/ — nothing to seed" >&2
  exit 1
fi

cp "${bundles[@]}" "$SEED/verifier/"

# Each bundle's image_path is relative to data/verifier/, e.g.
#   ../pages/1990/April 1990/1990-04apr0106/page-11.png
# Resolve to an absolute path under data/, then mirror into .seed/.
copied=0
for b in "$SEED/verifier"/*.bundle.json; do
  rel_image=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['image_path'])" "$b")
  src=$("$PYTHON" -c "import os,sys; print(os.path.normpath(os.path.join('data/verifier', sys.argv[1])))" "$rel_image")
  if [ ! -f "$src" ]; then
    echo "WARNING: bundle $(basename "$b") references missing image: $src" >&2
    continue
  fi
  # Strip the leading "data/" so the target sits under .seed/.
  dst="$SEED/${src#data/}"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  copied=$((copied + 1))
done

echo "seeded $copied page PNGs + $(ls "$SEED/verifier" | wc -l | tr -d ' ') bundles into $SEED"
du -sh "$SEED"
