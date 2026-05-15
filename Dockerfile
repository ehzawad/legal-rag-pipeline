# syntax=docker/dockerfile:1.7

# --- Frontend build stage ----------------------------------------------------
# Builds the React+Vite operator console. Vite emits its bundle into
# ../src/pipeline/ui/static/dist relative to the frontend/ source dir; in this
# stage that resolves to /src/pipeline/ui/static/dist inside the container,
# which we copy verbatim into the runtime image.
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build && test -f /src/pipeline/ui/static/dist/index.html


# --- Runtime stage -----------------------------------------------------------
FROM python:3.14.5-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/pipeline-docker-entrypoint
RUN chmod +x /usr/local/bin/pipeline-docker-entrypoint
RUN uv sync --frozen --no-dev

COPY datasets ./datasets
COPY eval ./eval
COPY docs ./docs
COPY playbooks ./playbooks
COPY --from=frontend /src/pipeline/ui/static/dist /app/src/pipeline/ui/static/dist

EXPOSE 8000

ENTRYPOINT ["pipeline-docker-entrypoint"]
CMD ["uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
