"""PDF preprocessing: text-layer quality routing.

These tests cover the pure helper functions and dataclasses that decide
between the text-layer fast path and the rendered-image multimodal path.
They do not depend on a fixture PDF; preprocess_pdf is exercised
end-to-end by the live pipeline run.
"""

from __future__ import annotations

from pipeline.ingestion.pdf import (
    MIN_TEXT_LAYER_CHARS,
    PreprocessedDocument,
    PreprocessedPage,
    _clean_text_layer_artifacts,
    _text_layer_quality_warnings,
    render_image_inputs,
    text_payload_for_prompt,
)


def test_clean_text_layer_strips_pdf_form_widget_chrome():
    raw = "Header\n0.00\nPrint\nSave As...\nReset\nBody line"
    cleaned = _clean_text_layer_artifacts(raw)
    assert "Print" not in cleaned
    assert "Save As" not in cleaned
    assert "Header" in cleaned and "Body line" in cleaned


def test_quality_warnings_flag_replacement_characters():
    text = "Document text � looks fine otherwise"
    warnings = _text_layer_quality_warnings(text)
    assert any("replacement" in w for w in warnings)


def test_quality_warnings_flag_many_single_char_tokens():
    # Single-character noise typical of bad OCR / corrupt text layer.
    text = " ".join(list("abcdefghijklmnopqrstuvwxy")) + " word word word"
    warnings = _text_layer_quality_warnings(text)
    assert any("single-character" in w for w in warnings)


def test_clean_text_returns_empty_string_for_empty_input():
    assert _clean_text_layer_artifacts("") == ""
    assert _text_layer_quality_warnings("") == []


def test_min_text_layer_chars_threshold_is_reasonable():
    # Guard against accidental tuning that would treat e.g. a single line as
    # a full text layer and skip the image path. Keep this tied to a real-world
    # lower bound; if you intentionally lower it, update the test too.
    assert MIN_TEXT_LAYER_CHARS >= 40


def test_render_image_inputs_only_includes_image_pages():
    doc = PreprocessedDocument(
        pages=[
            PreprocessedPage(
                page_number=1,
                text="text layer",
                image_base64="",
                image_mime="",
                extraction_method="pdf-text-layer",
                confidence=0.95,
            ),
            PreprocessedPage(
                page_number=2,
                text="",
                image_base64="ZmFrZQ==",
                image_mime="image/png",
                extraction_method="pdf-rendered-png",
                confidence=0.0,
            ),
        ]
    )
    inputs = render_image_inputs(doc)
    assert len(inputs) == 1
    assert inputs[0]["mime"] == "image/png"
    assert inputs[0]["base64"] == "ZmFrZQ=="


def test_text_payload_for_prompt_labels_image_pages_explicitly():
    doc = PreprocessedDocument(
        pages=[
            PreprocessedPage(
                page_number=1,
                text="text from layer",
                image_base64="",
                image_mime="",
                extraction_method="pdf-text-layer",
                confidence=0.95,
            ),
            PreprocessedPage(
                page_number=2,
                text="",
                image_base64="ZmFrZQ==",
                image_mime="image/png",
                extraction_method="pdf-rendered-png",
                confidence=0.0,
            ),
        ]
    )
    payload = text_payload_for_prompt(doc)
    assert "PAGE 1 (extracted from text layer)" in payload
    assert "text from layer" in payload
    assert "PAGE 2 (no text layer" in payload
    # The image-only page must not include the image bytes inline in the text.
    assert "ZmFrZQ==" not in payload
