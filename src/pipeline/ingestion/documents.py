from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_SUPPORTED_DOC_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".txt",
}

from pipeline.config import ProviderConfig, resolve_provider_name
from pipeline.ingestion.pdf import (
    PreprocessedDocument,
    preprocess_pdf,
    render_image_inputs,
    text_payload_for_prompt,
)
from pipeline.providers import (
    ProviderUnavailable,
    parse_json_response,
    responses_create,
)
from pipeline.schemas import ExtractedField, PageText, ProcessedDocument, SourceDocument


EXTRACTION_PROMPT = """\
You are extracting facts from messy legal-style source documents.

Return only valid JSON with this schema:
{
  "pages": [
    {
      "page_number": 1,
      "text": "verbatim extracted text, preserving unclear markers when text is uncertain",
      "confidence": 0.0,
      "warnings": ["page-level issue such as illegible handwriting, low resolution, cutoff text"]
    }
  ],
  "fields": {
    "document_type": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "parties": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "dates": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "property_or_legal_description": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "recording_references": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "amounts_or_deadlines": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []},
    "unclear_items": {"value": "", "confidence": 0.0, "evidence_page_numbers": [1], "notes": []}
  },
  "warnings": ["document-level processing or reliability issue"]
}

Rules:
- Do not infer missing facts. Use empty strings and warnings instead.
- Keep illegible, handwritten, or low-confidence items visibly qualified.
- Field evidence must point to page numbers that actually contain the support.
- This is a document extraction pipeline. Some pages arrive as embedded PDF
  text-layer text, while image-only or suspect pages arrive as rendered images
  for image-based extraction.
- If a page is sent as an image input, transcribe whatever is readable and mark uncertainty.
- For pages labeled `(extracted from text layer)`, treat the provided text as authoritative; your `pages[].text` for those pages may be omitted or duplicated, as the pipeline will use the supplied text either way.
- Do not describe text-layer pages as image-based extraction. If a text-layer
  page is garbled, call it a text-layer/source-text quality issue.
"""


def process_directory(
    input_dir: Path,
    *,
    provider: str | Any | None = None,
    config: ProviderConfig | None = None,
) -> list[ProcessedDocument]:
    config = config or ProviderConfig.from_env()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    candidates = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    ]
    paths = [path for path in candidates if path.suffix.lower() in _SUPPORTED_DOC_EXTENSIONS]
    skipped = [
        path.relative_to(input_dir).as_posix()
        for path in candidates
        if path.suffix.lower() not in _SUPPORTED_DOC_EXTENSIONS
    ]
    if skipped:
        warnings.warn(
            f"Skipping {len(skipped)} non-document file(s) in {input_dir}: "
            + ", ".join(sorted(skipped))
            + " (allowed extensions: "
            + ", ".join(sorted(_SUPPORTED_DOC_EXTENSIONS))
            + ")",
            stacklevel=2,
        )
    if not paths:
        raise ValueError(f"No source documents with supported extensions found in {input_dir}")

    concurrency = max(1, int(getattr(config, "extraction_concurrency", 1) or 1))
    if concurrency == 1 or len(paths) <= 1:
        return [process_document(path, provider=provider, config=config) for path in paths]

    with ThreadPoolExecutor(max_workers=min(concurrency, len(paths))) as executor:
        futures = [executor.submit(process_document, path, provider=provider, config=config) for path in paths]
        return [future.result() for future in futures]


def process_document(
    path: Path,
    *,
    provider: str | Any | None = None,
    config: ProviderConfig | None = None,
) -> ProcessedDocument:
    config = config or ProviderConfig.from_env()
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Source document does not exist: {path}")

    source = _source_document(path)
    provider_name = resolve_provider_name(provider, fallback=config.extraction_provider)

    if _is_injected_provider(provider):
        payload = provider.extract(path, config)
    elif provider_name == "fixture":
        payload = _fixture_extract(path)
    elif provider_name == "openai":
        payload = _extract_with_openai(path, config)
    else:
        raise ProviderUnavailable(
            "Document processing only supports provider='openai' in this build. "
            "Pass provider='fixture' or an injected provider object in unit tests."
        )

    return _normalize_processed_document(source, _coerce_payload(payload), provider_name, config)


def _extract_with_openai(path: Path, config: ProviderConfig) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        preprocessed = preprocess_pdf(
            path,
            max_pages=config.pdf_max_pages,
            render_dpi=config.pdf_render_dpi,
        )
        prompt_body = EXTRACTION_PROMPT + "\n\nSOURCE DOCUMENT PAGES:\n" + text_payload_for_prompt(preprocessed)
        image_inputs = render_image_inputs(preprocessed)
        raw = responses_create(
            prompt_body,
            config.openai_model,
            reasoning_effort=config.openai_reasoning_effort,
            image_inputs=image_inputs or None,
        )
        try:
            payload = _parse_provider_json(raw)
        except ProviderUnavailable:
            if preprocessed.needs_multimodal:
                raise
            payload = _text_layer_fallback_payload(preprocessed)
        payload = _augment_with_preprocessing(payload, preprocessed)
        return payload

    if suffix in {".txt", ".text"}:
        raw = responses_create(
            EXTRACTION_PROMPT
            + "\n\nSOURCE DOCUMENT TEXT:\n"
            + path.read_text(encoding="utf-8", errors="replace"),
            config.openai_model,
            reasoning_effort=config.openai_reasoning_effort,
        )
        return _parse_provider_json(raw)

    if suffix in {".png", ".jpg", ".jpeg"}:
        encoded_path = path
        import base64 as _b64

        encoded = _b64.b64encode(encoded_path.read_bytes()).decode("ascii")
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        raw = responses_create(
            EXTRACTION_PROMPT,
            config.openai_model,
            reasoning_effort=config.openai_reasoning_effort,
            image_inputs=[{"mime": mime, "base64": encoded}],
        )
        return _parse_provider_json(raw)

    raise ProviderUnavailable(
        f"Unsupported file extension for openai extraction: {suffix}. "
        "Allowed: .pdf, .png, .jpg/.jpeg, .txt."
    )


def _augment_with_preprocessing(
    payload: dict[str, Any],
    preprocessed: PreprocessedDocument,
) -> dict[str, Any]:
    """Build the final page list from `preprocessed.pages` (the authoritative ordering).

    For text-layer pages the local PyMuPDF text is the source of truth. For
    rendered-image pages we look up the model's matching `page_number` and use
    its image-extracted text/confidence/warnings; if the model omitted a
    rendered page entirely, we emit a placeholder so downstream retrieval still sees the
    page slot (with an explanatory warning rather than silent loss).
    """

    model_pages_by_number = {
        p["page_number"]: p
        for p in (payload.get("pages") or [])
        if isinstance(p, dict) and "page_number" in p
    }

    enriched_pages: list[dict[str, Any]] = []
    for pre_page in preprocessed.pages:
        page_number = pre_page.page_number
        model_page = model_pages_by_number.get(page_number)
        preprocessing_warnings = list(pre_page.warnings)

        if pre_page.extraction_method == "pdf-text-layer":
            # Text layer is authoritative; the model's `text` is ignored but its
            # warnings (if any) are still useful signal.
            page_warnings: list[str] = []
            if isinstance(model_page, dict):
                page_warnings.extend(_string_list(model_page.get("warnings")))
            page_warnings.extend(preprocessing_warnings)
            enriched_pages.append(
                {
                    "page_number": page_number,
                    "text": pre_page.text,
                    "confidence": pre_page.confidence,
                    "extraction_method": "pdf-text-layer",
                    "warnings": list(dict.fromkeys(page_warnings)),
                }
            )
            continue

        # pdf-rendered-png: rely on the model's image-based extraction for this page.
        if isinstance(model_page, dict):
            try:
                model_conf = float(model_page.get("confidence", 0.0))
            except (TypeError, ValueError):
                model_conf = 0.0
            page_warnings = _string_list(model_page.get("warnings"))
            page_warnings.extend(preprocessing_warnings)
            enriched_pages.append(
                {
                    "page_number": page_number,
                    "text": str(model_page.get("text") or ""),
                    "confidence": max(0.0, min(1.0, model_conf)),
                    "extraction_method": "pdf-rendered-png",
                    "warnings": list(dict.fromkeys(page_warnings)),
                }
            )
        else:
            placeholder_warnings = [
                "Model omitted this page in its response; downstream retrieval will see empty text.",
            ]
            placeholder_warnings.extend(preprocessing_warnings)
            enriched_pages.append(
                {
                    "page_number": page_number,
                    "text": "",
                    "confidence": 0.0,
                    "extraction_method": "pdf-rendered-png",
                    "warnings": list(dict.fromkeys(placeholder_warnings)),
                }
            )

    payload["pages"] = enriched_pages

    extra_warnings = _string_list(payload.get("warnings"))
    if preprocessed.page_count_truncated:
        extra_warnings.append("PDF preprocessing capped page count; later pages were not sent to the model.")
    for warning in preprocessed.warnings:
        extra_warnings.append(warning)
    if extra_warnings:
        payload["warnings"] = list(dict.fromkeys(extra_warnings))
    return payload


def _text_layer_fallback_payload(preprocessed: PreprocessedDocument) -> dict[str, Any]:
    """Keep clean text-layer PDFs usable when provider field JSON is malformed.

    The local PDF text layer is already authoritative for these pages, so a
    malformed model response should not discard otherwise-readable documents.
    For image-rendered pages we still fail loudly because no local text exists.
    """

    text = "\n\n".join(page.text for page in preprocessed.pages if page.text)
    return {
        "pages": [
            {
                "page_number": page.page_number,
                "text": page.text,
                "confidence": page.confidence,
                "warnings": list(page.warnings),
                "extraction_method": page.extraction_method,
            }
            for page in preprocessed.pages
        ],
        "fields": _heuristic_fields_from_text(text),
        "warnings": [],
    }


def _is_injected_provider(provider: Any) -> bool:
    return provider is not None and not isinstance(provider, str) and callable(getattr(provider, "extract", None))


def _parse_provider_json(text: str) -> dict[str, Any]:
    try:
        return parse_json_response(text)
    except ProviderUnavailable:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return parse_json_response(match.group(0))


def _coerce_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return _parse_provider_json(payload)
    raise ProviderUnavailable("Extraction provider returned an unsupported payload type")


def _normalize_processed_document(
    source: SourceDocument,
    payload: dict[str, Any],
    provider: str,
    config: ProviderConfig,
) -> ProcessedDocument:
    raw_pages = payload.get("pages") or []
    pages: list[PageText] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        page_number = _int_or_default(raw_page.get("page_number"), index)
        warnings = _string_list(raw_page.get("warnings"))
        text = str(raw_page.get("text") or "")
        confidence = _float_or_default(raw_page.get("confidence"), 0.0)
        if not text.strip():
            warnings.append("Provider returned no readable text for this page.")
        if _has_unclear_marker(text):
            warnings.append("Page contains markers suggesting illegible, redacted, or uncertain text.")
        pages.append(
            PageText(
                document_id=source.document_id,
                page_number=page_number,
                text=text,
                extraction_method=raw_page.get("extraction_method") or provider,
                confidence=confidence,
                warnings=warnings,
            )
        )

    if not pages:
        pages.append(
            PageText(
                document_id=source.document_id,
                page_number=1,
                text="",
                extraction_method=provider,
                confidence=0.0,
                warnings=["Provider returned no page text."],
            )
        )

    fields_payload = payload.get("fields") or _heuristic_fields_from_pages(pages)
    fields = _normalize_fields(fields_payload, pages)
    warnings = _string_list(payload.get("warnings"))
    if any(page.confidence < config.extraction_confidence_threshold for page in pages):
        warnings.append("One or more pages have low extraction confidence.")
    for page in pages:
        warnings.extend(f"p.{page.page_number}: {warning}" for warning in page.warnings)
    return ProcessedDocument(source=source, pages=pages, fields=fields, warnings=_dedupe(warnings))


def _normalize_fields(raw_fields: dict[str, Any], pages: list[PageText]) -> dict[str, ExtractedField]:
    result: dict[str, ExtractedField] = {}
    page_map = {page.page_number: page for page in pages}
    for name, raw_value in raw_fields.items():
        if isinstance(raw_value, dict):
            value = str(raw_value.get("value") or "")
            confidence = _float_or_default(raw_value.get("confidence"), 0.0)
            page_numbers = [_int_or_default(item, 0) for item in raw_value.get("evidence_page_numbers") or []]
            notes = _string_list(raw_value.get("notes"))
        else:
            value = str(raw_value or "")
            confidence = 0.5 if value else 0.0
            page_numbers = []
            notes = []
        evidence_ids = []
        for page_number in page_numbers:
            if page_number in page_map:
                evidence_ids.append(f"{page_map[page_number].document_id}:p{page_number}")
        result[str(name)] = ExtractedField(
            name=str(name),
            value=value,
            confidence=confidence,
            evidence_ids=evidence_ids,
            notes=notes,
        )
    return result


def _source_document(path: Path) -> SourceDocument:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SourceDocument(
        path=str(path),
        document_id=digest[:16],
        filename=path.name,
        mime_type=_mime_for_path(path),
        sha256=digest,
    )


def _fixture_extract(path: Path) -> dict[str, Any]:
    """Test-only extraction fixture; production defaults never call this."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "pages": _pages_payload_from_text(text, extraction_method="fixture"),
        "fields": _heuristic_fields_from_text(text),
        "warnings": ["fixture provider used; not valid for production"],
    }


def _pages_payload_from_text(text: str, *, extraction_method: str) -> list[dict[str, Any]]:
    parts = [part.strip() for part in text.split("\f") if part.strip()] or [""]
    pages = []
    for index, part in enumerate(parts, start=1):
        warnings = []
        if _has_unclear_marker(part):
            warnings.append("Page contains markers suggesting illegible, redacted, or uncertain text.")
        pages.append(
            {
                "page_number": index,
                "text": part,
                "confidence": 1.0 if part.strip() else 0.0,
                "warnings": warnings,
                "extraction_method": extraction_method,
            }
        )
    return pages


def _heuristic_fields_from_pages(pages: list[PageText]) -> dict[str, Any]:
    return _heuristic_fields_from_text("\n\n".join(page.text for page in pages))


def _heuristic_fields_from_text(text: str) -> dict[str, Any]:
    return {
        "document_type": _field(_detect_document_type(text), confidence=0.55),
        "parties": _field("; ".join(_detect_parties(text)), confidence=0.55),
        "dates": _field("; ".join(_detect_dates(text)), confidence=0.5),
        "property_or_legal_description": _field(_detect_property_description(text), confidence=0.55),
        "recording_references": _field("; ".join(_detect_recording_references(text)), confidence=0.55),
        "amounts_or_deadlines": _field("; ".join(_detect_amounts_or_deadlines(text)), confidence=0.45),
        "unclear_items": _field(
            "unclear or illegible text markers present" if _has_unclear_marker(text) else "",
            confidence=0.8 if _has_unclear_marker(text) else 0.0,
            notes=["Heuristic warning; verify against the source image."] if _has_unclear_marker(text) else [],
        ),
    }


def _field(value: str | None, *, confidence: float, notes: list[str] | None = None) -> dict[str, Any]:
    return {
        "value": value or "",
        "confidence": confidence if value else 0.0,
        "evidence_page_numbers": [1] if value else [],
        "notes": notes or [],
    }


def _first_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" #")
        if stripped:
            return stripped[:120]
    return ""


def _detect_document_type(text: str) -> str:
    title = _first_title(text)
    candidates = [
        "special warranty deed",
        "warranty deed",
        "quitclaim deed",
        "grant deed",
        "deed of trust",
        "assignment of mortgage",
        "satisfaction of mortgage",
        "notice of default",
        "release of lien",
        "lis pendens",
        "mortgage",
        "assignment",
        "affidavit",
        "easement",
        "lease",
        "agreement",
    ]
    head = "\n".join(text.splitlines()[:40])
    for candidate in sorted(candidates, key=len, reverse=True):
        if re.search(r"\b" + re.escape(candidate).replace(r"\ ", r"\s+") + r"\b", head, flags=re.IGNORECASE):
            if title.isupper() and candidate in title.lower():
                return title
            return candidate.title()
    return title


def _detect_parties(text: str) -> list[str]:
    parties: list[str] = []
    role_pattern = re.compile(
        r"\b(Grantor|Grantee|Borrower|Lender|Trustee|Beneficiary|Assignor|Assignee|Seller|Buyer|Landlord|Tenant)\s*[:\-]\s*([^\n;]{2,160})",
        flags=re.IGNORECASE,
    )
    for match in role_pattern.finditer(text):
        role = match.group(1).title()
        name = _clean_value(match.group(2))
        if name:
            parties.append(f"{role}: {name}")

    between_pattern = re.compile(
        r"\b(?:by\s+and\s+)?between\s+([A-Z][A-Za-z0-9&.,' \-]{2,120}?)\s+and\s+([A-Z][A-Za-z0-9&.,' \-]{2,120}?)(?:[.;,\n]|$)",
        flags=re.IGNORECASE,
    )
    for match in between_pattern.finditer(text):
        for group in (1, 2):
            name = _clean_value(match.group(group))
            if name:
                parties.append(name)
    return _dedupe(parties)


def _detect_dates(text: str) -> list[str]:
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
    ]
    dates: list[str] = []
    for pattern in patterns:
        dates.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return _dedupe(dates)


def _detect_property_description(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    marker = re.compile(
        r"\b(?:legal\s+description|property\s+description|description\s+of\s+property|premises|real\s+property)\b\s*:?\s*(.*)",
        flags=re.IGNORECASE,
    )
    stop = re.compile(
        r"^\s*(?:grantor|grantee|borrower|lender|recorded|recording|instrument|document|dated|signed|notary|state of|county of)\b",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        match = marker.search(line)
        if not match:
            continue
        parts = [_clean_value(match.group(1))]
        for following in lines[index + 1 : index + 8]:
            if not following or stop.search(following):
                break
            parts.append(following)
        return _clean_value(" ".join(part for part in parts if part))

    fallback_patterns = [
        r"\bLot\s+\d+[A-Za-z]?(?:,\s*Block\s+\d+[A-Za-z]?)?[^.\n]{0,220}",
        r"\b(?:Section|Sec\.)\s+\d+[^.\n]{0,220}\b(?:Township|Twp\.|Range|Rge\.)[^.\n]{0,220}",
        r"\bAPN\s*[:#]?\s*[A-Za-z0-9\-\.]{4,40}",
    ]
    for pattern in fallback_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_value(match.group(0))
    return ""


def _detect_recording_references(text: str) -> list[str]:
    patterns = [
        r"\bBook\s+[A-Za-z0-9\-]+,?\s+Page\s+[A-Za-z0-9\-]+\b",
        r"\b(?:Instrument|Document|Recording|Reception)\s+(?:No\.?|Number|#)\s*[:.]?\s*[A-Za-z0-9\-\/\.]+\b",
        r"\b(?:Liber|Lib\.)\s+[A-Za-z0-9\-]+,?\s+(?:Folio|Fol\.)\s+[A-Za-z0-9\-]+\b",
        r"\b(?:recorded|recording)\s+(?:at|in|as)\s+[^.\n;]{3,160}",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(_clean_value(match.group(0)) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return _dedupe(values)


def _detect_amounts_or_deadlines(text: str) -> list[str]:
    patterns = [
        r"\$[0-9][0-9,]*(?:\.[0-9]{2})?",
        r"\b\d+\s+(?:day|days|month|months|year|years)\b",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return _dedupe(values)


def _has_unclear_marker(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in (
            r"\[(?:illegible|unreadable|unclear|blank|redacted)\]",
            r"\b(?:illegible|unreadable|unclear|redacted)\b",
            r"\?{3,}",
            "\ufffd",
        )
    )


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" \t\r\n,;:-")


def _mime_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    try:
        return [str(item) for item in value if str(item).strip()]
    except TypeError:
        text = str(value)
        return [text] if text.strip() else []


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
