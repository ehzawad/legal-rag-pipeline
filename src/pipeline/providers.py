from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any


class ProviderUnavailable(RuntimeError):
    """Raised when a configured provider is unavailable or returns unusable output."""


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        secret_file = (os.getenv(f"{name}_FILE") or "").strip()
        if secret_file:
            try:
                value = open(secret_file, encoding="utf-8").read().strip()
            except OSError as exc:
                raise ProviderUnavailable(f"{name}_FILE is set but could not be read: {secret_file}") from exc
            if value:
                os.environ[name] = value
    if not value:
        raise ProviderUnavailable(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def _openai_client() -> Any:
    require_env("OPENAI_API_KEY")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ProviderUnavailable("openai SDK is not installed") from exc
    return OpenAI()


def responses_create(
    prompt: str,
    model: str,
    *,
    reasoning_effort: str | None = "low",
    image_inputs: list[dict[str, str]] | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Wrapper around openai.responses.create with reasoning_effort + multimodal inputs.

    image_inputs is a list of {"mime": ..., "base64": ...} dicts already prepared by
    the caller (typically pdf_preprocessing).
    """

    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image in image_inputs or []:
        mime = image.get("mime", "image/png")
        data = image["base64"]
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{data}"})
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    response = _openai_client().responses.create(**kwargs)
    return response.output_text


def embed_with_openai(texts: list[str], model: str) -> list[list[float]]:
    response = _openai_client().embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _extract_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("Provider response was not valid JSON") from exc


def _extract_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned
