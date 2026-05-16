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

exec "$@"
