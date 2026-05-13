FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps directly from pyproject.toml. uv is faster and
# resolves the same lockfile as local dev.
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system --no-cache .

# App code. data/ is NOT copied — the seed under .seed/ is what populates
# the Railway volume on first boot.
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY verifier/ ./verifier/

# Seed snapshot: bundles + only the page PNGs they reference. Produced
# locally by `scripts/build_railway_seed.sh` before `railway up`.
COPY .seed/ /seed/

# Entrypoint hydrates the volume from /seed on first boot, then runs
# the verifier server.
RUN chmod +x /app/scripts/railway_entrypoint.sh
CMD ["/app/scripts/railway_entrypoint.sh"]
