#!/bin/sh
set -eu

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY_FILE:-}" ]; then
  if [ ! -f "$OPENAI_API_KEY_FILE" ]; then
    echo "OPENAI_API_KEY_FILE is set but the file was not found: $OPENAI_API_KEY_FILE" >&2
    exit 1
  fi
  OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_API_KEY_FILE")"
  export OPENAI_API_KEY
fi

if [ -z "${COHERE_API_KEY:-}" ] && [ -n "${COHERE_API_KEY_FILE:-}" ]; then
  if [ ! -f "$COHERE_API_KEY_FILE" ]; then
    echo "COHERE_API_KEY_FILE is set but the file was not found: $COHERE_API_KEY_FILE" >&2
    exit 1
  fi
  COHERE_API_KEY="$(tr -d '\r\n' < "$COHERE_API_KEY_FILE")"
  export COHERE_API_KEY
fi

if [ -z "${QDRANT_API_KEY:-}" ] && [ -n "${QDRANT_API_KEY_FILE:-}" ]; then
  if [ ! -f "$QDRANT_API_KEY_FILE" ]; then
    echo "QDRANT_API_KEY_FILE is set but the file was not found: $QDRANT_API_KEY_FILE" >&2
    exit 1
  fi
  QDRANT_API_KEY="$(tr -d '\r\n' < "$QDRANT_API_KEY_FILE")"
  export QDRANT_API_KEY
fi

if [ "${PIPELINE_INDEX_BACKEND:-}" = "qdrant" ] && [ -n "${QDRANT_URL:-}" ] && [ "${PIPELINE_WAIT_FOR_QDRANT:-true}" != "false" ]; then
  python - <<'PY'
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

base_url = os.environ["QDRANT_URL"].rstrip("/")
health_url = urllib.parse.urljoin(base_url + "/", "healthz")
deadline = time.monotonic() + float(os.environ.get("PIPELINE_WAIT_FOR_QDRANT_SECONDS", "60"))
headers = {}
api_key = os.environ.get("QDRANT_API_KEY")
if api_key:
    headers["api-key"] = api_key

last_error = ""
while time.monotonic() < deadline:
    try:
        request = urllib.request.Request(health_url, headers=headers)
        with urllib.request.urlopen(request, timeout=2) as response:
            if 200 <= response.status < 300:
                sys.exit(0)
    except (OSError, urllib.error.URLError) as exc:
        last_error = str(exc)
    time.sleep(1)

print(f"Qdrant did not become ready at {health_url}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
fi

exec "$@"
