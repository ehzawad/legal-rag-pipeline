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

exec "$@"
