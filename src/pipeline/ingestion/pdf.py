"""Local PDF preprocessing.

PyMuPDF-driven. For each page: try the embedded text layer; fall back to
rendering as PNG bytes when the text layer is missing or implausibly short.
The pipeline then sends text pages as plain text and image pages as
multimodal inputs to OpenAI, avoiding the round-trip of the full binary
when extraction is cheap and auditable.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
import re

MAX_PAGES_DEFAULT = 100
RENDER_DPI_DEFAULT = 180
RENDER_DPI_MIN = 96
MIN_TEXT_LAYER_CHARS = 60
TEXT_LAYER_CONFIDENCE = 0.95


@dataclass(slots=True)
class PreprocessedPage:
    page_number: int
    text: str
    image_base64: str
    image_mime: str
    extraction_method: str  # "pdf-text-layer" or "pdf-rendered-png"
    confidence: float
    warnings: list[str] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return bool(self.image_base64)


@dataclass(slots=True)
class PreprocessedDocument:
    pages: list[PreprocessedPage]
    warnings: list[str] = field(default_factory=list)
    page_count_truncated: bool = False

    @property
    def needs_multimodal(self) -> bool:
        return any(page.is_image for page in self.pages)


def preprocess_pdf(
    path: Path,
    *,
    max_pages: int = MAX_PAGES_DEFAULT,
    render_dpi: int = RENDER_DPI_DEFAULT,
) -> PreprocessedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dep missing
        raise RuntimeError("PyMuPDF is required for PDF preprocessing") from exc

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF for preprocessing: {path}") from exc

    try:
        pages: list[PreprocessedPage] = []
        warnings: list[str] = []
        total_pages = doc.page_count
        used = total_pages if total_pages <= max_pages else max_pages
        truncated = total_pages > max_pages
        if truncated:
            warnings.append(
                f"PDF has {total_pages} pages; preprocessing capped at {max_pages}."
            )

        dpi = max(RENDER_DPI_MIN, min(render_dpi, 300))
        scale = dpi / 72.0

        for index in range(used):
            page = doc.load_page(index)
            page_warnings: list[str] = []
            raw_text = _clean_text_layer_artifacts(page.get_text() or "").strip()
            quality_warnings = _text_layer_quality_warnings(raw_text)
            if len(raw_text) >= MIN_TEXT_LAYER_CHARS and not quality_warnings:
                pages.append(
                    PreprocessedPage(
                        page_number=index + 1,
                        text=raw_text,
                        image_base64="",
                        image_mime="",
                        extraction_method="pdf-text-layer",
                        confidence=TEXT_LAYER_CONFIDENCE,
                        warnings=page_warnings,
                    )
                )
                continue

            if raw_text:
                if len(raw_text) < MIN_TEXT_LAYER_CHARS:
                    page_warnings.append(
                        f"Text layer present but only {len(raw_text)} chars; rendering page as image."
                    )
                else:
                    page_warnings.append(
                        "Text layer appears low quality; routing page through image-based extraction: "
                        + "; ".join(quality_warnings)
                    )
            else:
                page_warnings.append("No text layer; routing page through image-based extraction.")

            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pixmap.tobytes("png")
            encoded = base64.b64encode(image_bytes).decode("ascii")

            pages.append(
                PreprocessedPage(
                    page_number=index + 1,
                    text="",
                    image_base64=encoded,
                    image_mime="image/png",
                    extraction_method="pdf-rendered-png",
                    confidence=0.0,
                    warnings=page_warnings,
                )
            )

        return PreprocessedDocument(pages=pages, warnings=warnings, page_count_truncated=truncated)
    finally:
        doc.close()


def render_image_inputs(document: PreprocessedDocument) -> list[dict[str, str]]:
    """Convert image-only pages into the dict format expected by providers.responses_create."""
    return [
        {"mime": page.image_mime, "base64": page.image_base64}
        for page in document.pages
        if page.is_image
    ]


def text_payload_for_prompt(document: PreprocessedDocument) -> str:
    """Render text-layer pages as a labeled block for inclusion in the extraction prompt."""
    blocks: list[str] = []
    for page in document.pages:
        if page.text:
            blocks.append(f"--- PAGE {page.page_number} (extracted from text layer) ---\n{page.text}")
        else:
            blocks.append(
                f"--- PAGE {page.page_number} (no text layer; see the corresponding image input above) ---"
            )
    return "\n\n".join(blocks)


def _clean_text_layer_artifacts(text: str) -> str:
    """Remove common PDF form-widget controls from extracted text-layer text."""

    cleaned = str(text or "").replace("\r\n", "\n")
    cleaned = re.sub(
        r"\n0\.00\s*\nPrint\s*\nSave As\.\.\.\s*(?:\nAdd Attachment\s*)?\nReset\s*",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\nPrint\s*\nSave As\.\.\.\s*(?:\nAdd Attachment\s*)?\nReset\s*",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _text_layer_quality_warnings(text: str) -> list[str]:
    """Return reasons a non-empty PDF text layer should not be trusted alone."""
    if not text:
        return []

    warnings: list[str] = []
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return ["text layer contains only whitespace"]

    printable_ratio = sum(char.isprintable() for char in chars) / len(chars)
    if printable_ratio < 0.92:
        warnings.append(f"printable character ratio is {printable_ratio:.2f}")

    if "\ufffd" in text:
        warnings.append("replacement characters are present")

    symbol_ratio = sum(not char.isalnum() and char not in ".,;:()[]{}'\"-/&$%#@" for char in chars) / len(chars)
    if symbol_ratio > 0.18:
        warnings.append(f"unusual symbol ratio is {symbol_ratio:.2f}")

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&.-]*", text)
    if len(tokens) >= 20:
        one_char_ratio = sum(len(token) == 1 for token in tokens) / len(tokens)
        if one_char_ratio > 0.45:
            warnings.append(f"single-character token ratio is {one_char_ratio:.2f}")

        alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]
        mixed_alnum_ratio = (
            sum(any(char.isalpha() for char in token) and any(char.isdigit() for char in token) for token in alpha_tokens)
            / len(alpha_tokens)
            if alpha_tokens
            else 0.0
        )
        if len(alpha_tokens) >= 10 and mixed_alnum_ratio > 0.35:
            warnings.append(f"mixed letter/digit token ratio is {mixed_alnum_ratio:.2f}")

    return warnings
