FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install in two stages so dep changes don't bust the cache on every
# source edit. `pyproject.toml` packages `core/` (and `cli.py` as a
# py-module), so they must be present when uv runs the install — but the
# verifier/ and scripts/ trees can come later and still be picked up at
# runtime (they're loose files invoked by path, not installed packages).
COPY pyproject.toml uv.lock ./
COPY core/ ./core/
COPY cli.py ./
RUN pip install uv && uv pip install --system --no-cache .

# data/ is NOT copied — the seed under .seed/ is what populates the
# Railway volume on first boot.
COPY scripts/ ./scripts/
COPY verifier/ ./verifier/

# Seed snapshot: bundles + only the page PNGs they reference. Produced
# locally by `scripts/build_railway_seed.sh` before `railway up`.
COPY .seed/ /seed/

# Entrypoint hydrates the volume from /seed on first boot, then runs
# the verifier server.
RUN chmod +x /app/scripts/railway_entrypoint.sh
CMD ["/app/scripts/railway_entrypoint.sh"]
