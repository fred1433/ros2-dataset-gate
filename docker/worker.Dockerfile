# Batch worker image: the same entrypoint the local runner executes.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
ARG IMAGE_TAG=dsgate-worker:local
ENV PATH=/app/.venv/bin:$PATH DSGATE_LOCKFILE=/app/uv.lock DSGATE_IMAGE=$IMAGE_TAG
CMD ["python", "-m", "dsgate.worker"]
