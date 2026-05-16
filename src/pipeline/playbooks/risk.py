from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from pipeline.io import read_json, write_json
from pipeline.orchestration.artifacts import draft_from_json, evidence_from_json, processed_documents_from_json
from pipeline.schemas import Draft, EvidenceChunk, ProcessedDocument, now_iso


DEFAULT_PLAYBOOK_PATH = Path(__file__).resolve().parents[3] / "playbooks" / "default_legal_playbook.json"
RISK_REPORT_SCHEMA_VERSION = 2
_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
_SOURCE_SPAN_LIMIT = 5
_EVIDENCE_ID_LIMIT = 8
_SNIPPET_CONTEXT_CHARS = 60
_FIELD_ANCHOR_PAGE_RE = re.compile(r":p(\d+)(?::|$)")

# OCR / PDF text routinely emits ligatures, soft hyphens, NBSP, curly quotes,
# and Unicode dashes. Normalize once before matching so playbook terms written
# with ASCII characters still hit those forms. This intentionally mirrors the
# concept used by retrieval/engine.py:_QUOTE_NORMALIZATION_MAP, with the
# soft-hyphen (U+00AD) deletion that's specific to the matching path.
_TEXT_NORMALIZATION_MAP = str.maketrans(
    {
        "­": "",  # soft hyphen — strip; it's invisible in display but breaks substring matching
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
        "−": "-",  # minus sign
        " ": " ",  # NBSP
        " ": " ",  # figure space
        " ": " ",  # narrow NBSP
        "​": "",  # zero-width space
        "‘": "'",
        "’": "'",
        "‚": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)


def _normalize_for_matching(text: str) -> str:
    """Fold OCR/PDF artifacts into ASCII-shaped text.

    The transform is length-changing (ligatures expand, soft-hyphens delete),
    so snippets surfaced to operators are post-normalization, not byte-perfect
    copies of the source. This matches the contract used by the drafter's
    verbatim-quote validator.
    """

    return text.translate(_TEXT_NORMALIZATION_MAP) if text else ""


@dataclass(slots=True, frozen=True)
class _TermSource:
    term: str
    source_kind: str  # "page" | "field" | "chunk"
    source_id: str
    document_id: str
    filename: str
    page_number: int
    snippet: str
    char_start: int
    char_end: int


def load_playbook(path: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path) if path else DEFAULT_PLAYBOOK_PATH
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError("playbook must be a JSON object")
    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError("playbook must include a positions list")
    for item in positions:
        if not isinstance(item, dict):
            raise ValueError("each playbook position must be an object")
        if not str(item.get("id") or "").strip():
            raise ValueError("each playbook position must include an id")
        if not str(item.get("title") or "").strip():
            raise ValueError("each playbook position must include a title")
    return payload


def score_playbook(
    *,
    processed: list[ProcessedDocument],
    evidence: list[EvidenceChunk],
    draft: Draft,
    playbook: Mapping[str, Any],
) -> dict[str, Any]:
    """Score retrieved/document evidence against a legal-team playbook.

    This is intentionally deterministic. It is a production-safe baseline for
    routing review attention; it is not a legal conclusion and does not replace
    attorney review.
    """

    findings: list[dict[str, Any]] = []
    passed_positions: list[dict[str, Any]] = []

    for raw_position in playbook.get("positions", []):
        position = dict(raw_position)
        position_id = str(position.get("id") or "").strip()
        title = str(position.get("title") or position_id).strip()
        severity = _severity(str(position.get("severity") or "medium"))
        required_terms = _terms(position.get("required_any"))
        preferred_terms = _terms(position.get("preferred_any"))
        escalation_terms = _terms(position.get("escalation_terms"))

        escalation_sources = _gather_term_sources(processed, evidence, escalation_terms)
        required_sources = _gather_term_sources(processed, evidence, required_terms)
        preferred_sources = _gather_term_sources(processed, evidence, preferred_terms)

        escalation_matches = sorted({source.term for source in escalation_sources})
        required_matches = sorted({source.term for source in required_sources})
        preferred_matches = sorted({source.term for source in preferred_sources})

        if escalation_sources:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="escalation_triggered",
                    severity=severity,
                    message=f"Escalation term(s) found: {', '.join(escalation_matches)}.",
                    matched_terms=escalation_matches,
                    source_spans=_select_audit_sources(escalation_sources),
                    evidence_ids=_evidence_ids_from_sources(escalation_sources),
                    source_ids=_source_ids_from_sources(escalation_sources),
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        if required_terms and not required_sources:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="missing_or_unconfirmed",
                    severity=_downgrade(severity),
                    message="Required playbook concept was not found in extracted text.",
                    matched_terms=[],
                    source_spans=[],
                    evidence_ids=[],
                    source_ids=[],
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        if preferred_terms and not preferred_sources:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="review_preferred_position",
                    severity=_downgrade(severity),
                    message="Clause appears present, but preferred playbook language was not found.",
                    matched_terms=required_matches,
                    source_spans=_select_audit_sources(required_sources),
                    evidence_ids=_evidence_ids_from_sources(required_sources),
                    source_ids=_source_ids_from_sources(required_sources),
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        combined_sources = required_sources + preferred_sources
        passed_positions.append(
            {
                "position_id": position_id,
                "title": title,
                "matched_terms": sorted(set(required_matches + preferred_matches)),
                "evidence_ids": _evidence_ids_from_sources(combined_sources),
                "source_ids": _source_ids_from_sources(combined_sources),
            }
        )

    unsupported_sections = [
        {"heading": section.heading, "body": section.body}
        for section in draft.sections
        if section.unsupported
    ]
    if unsupported_sections:
        findings.append(
            _finding(
                position_id="unsupported_draft_sections",
                title="Unsupported Draft Sections",
                status="unsupported_draft_content",
                severity="medium",
                message=f"{len(unsupported_sections)} draft section(s) are marked unsupported.",
                matched_terms=[],
                source_spans=[],
                evidence_ids=[],
                source_ids=[],
                guidance="Route unsupported sections to operator review before relying on the draft.",
            )
        )

    severity_counts = Counter(finding["severity"] for finding in findings)
    summary = {
        "position_count": len(playbook.get("positions", [])),
        "passed_position_count": len(passed_positions),
        "finding_count": len(findings),
        "critical": severity_counts.get("critical", 0),
        "high": severity_counts.get("high", 0),
        "medium": severity_counts.get("medium", 0),
        "low": severity_counts.get("low", 0),
        "info": severity_counts.get("info", 0),
    }
    return {
        "schema_version": RISK_REPORT_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "playbook": {
            "name": str(playbook.get("name") or ""),
            "version": playbook.get("version", 1),
            "description": str(playbook.get("description") or ""),
        },
        "summary": summary,
        "findings": findings,
        "passed_positions": passed_positions,
        "evidence": [
            {
                "evidence_id": chunk.evidence_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "score": chunk.score,
            }
            for chunk in evidence
        ],
        "disclaimer": "Deterministic playbook screen for operator review; not legal advice.",
    }


def score_playbook_run_dir(
    run_dir: str | Path,
    *,
    playbook_path: str | Path | None = None,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    processed = processed_documents_from_json(read_json(run_path / "processed_documents.json"))
    evidence = evidence_from_json(read_json(run_path / "retrieved_evidence.json"))
    draft = draft_from_json(read_json(run_path / "draft.json"))
    report = score_playbook(
        processed=processed,
        evidence=evidence,
        draft=draft,
        playbook=load_playbook(playbook_path),
    )
    if output_path is not None:
        write_risk_report(report, Path(output_path), Path(markdown_path) if markdown_path else None)
    return report


def write_risk_report(
    report: Mapping[str, Any],
    output_path: Path,
    markdown_path: Path | None = None,
) -> None:
    write_json(output_path, dict(report))
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_risk_report_markdown(report) + "\n", encoding="utf-8")


def render_risk_report_markdown(report: Mapping[str, Any]) -> str:
    playbook = report.get("playbook") if isinstance(report.get("playbook"), Mapping) else {}
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    lines = [
        f"# Playbook Risk Report: {playbook.get('name', '')}",
        "",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "## Summary",
        "",
        f"- Positions checked: {summary.get('position_count', 0)}",
        f"- Passed positions: {summary.get('passed_position_count', 0)}",
        f"- Findings: {summary.get('finding_count', 0)}",
        f"- High/Critical: {int(summary.get('high', 0) or 0) + int(summary.get('critical', 0) or 0)}",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No playbook findings.")
    for finding in findings:
        evidence_ids = ", ".join(finding.get("evidence_ids") or []) or "none"
        matched = ", ".join(finding.get("matched_terms") or []) or "none"
        lines.extend(
            [
                f"### {finding.get('title', '')}",
                "",
                f"- Status: `{finding.get('status', '')}`",
                f"- Severity: `{finding.get('severity', '')}`",
                f"- Message: {finding.get('message', '')}",
                f"- Matched terms: {matched}",
                f"- Evidence: {evidence_ids}",
                f"- Guidance: {finding.get('guidance', '')}",
            ]
        )
        spans = finding.get("source_spans") or []
        if spans:
            lines.append("- Source spans:")
            for span in spans:
                page = span.get("page_number") or 0
                page_ref = f"p.{page}" if page else span.get("source_kind", "source")
                filename = span.get("filename") or span.get("document_id") or ""
                term = span.get("term") or ""
                snippet = span.get("snippet") or ""
                location = f"{page_ref} ({filename})" if filename else page_ref
                lines.append(f"  - {location} — `{term}` → \"{snippet}\"")
        if finding.get("audit_warning"):
            lines.append(f"- Audit warning: {finding['audit_warning']}")
        lines.append("")
    lines.extend(["## Disclaimer", "", str(report.get("disclaimer") or "")])
    return "\n".join(lines).rstrip()


def _terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


@lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    """Tight, whitespace/hyphen-tolerant matcher for one playbook term.

    The term is normalized identically to the search text (ligatures, dashes,
    soft-hyphens, NBSP all folded). Internal runs of whitespace and hyphens
    collapse to `[\\s\\-]+`. The boundary uses Unicode-aware `\\w` so non-ASCII
    word characters (accented letters, CJK, etc.) adjacent to a term still
    count as part of a larger word — so a playbook term `term` does not match
    inside `termíno` even though the boundary is not ASCII.

    Caching: playbooks have a small, stable vocabulary; lru_cache avoids
    recompiling the same regex once per (term × page × document × position).
    """

    normalized = _normalize_for_matching(term).strip()
    parts = [part for part in re.split(r"[\s\-]+", normalized) if part]
    if not parts:
        return None
    body = r"[\s\-]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<![\w\-]){body}(?![\w\-])", re.IGNORECASE | re.UNICODE)


def _snippet_around(text: str, start: int, end: int) -> str:
    left = max(0, start - _SNIPPET_CONTEXT_CHARS)
    right = min(len(text), end + _SNIPPET_CONTEXT_CHARS)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + " ".join(text[left:right].split()) + suffix


def _scan_text(
    text: str,
    term: str,
    *,
    source_kind: str,
    source_id: str,
    document_id: str,
    filename: str,
    page_number: int,
) -> list[_TermSource]:
    pattern = _term_pattern(term)
    if not text or pattern is None:
        return []
    normalized = _normalize_for_matching(text)
    return [
        _TermSource(
            term=term,
            source_kind=source_kind,
            source_id=source_id,
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            snippet=_snippet_around(normalized, match.start(), match.end()),
            char_start=match.start(),
            char_end=match.end(),
        )
        for match in pattern.finditer(normalized)
    ]


def _field_anchor_page(field_evidence_ids: list[str]) -> int:
    for evidence_id in field_evidence_ids:
        match = _FIELD_ANCHOR_PAGE_RE.search(str(evidence_id))
        if match:
            return int(match.group(1))
    return 0


def _gather_term_sources(
    processed: list[ProcessedDocument],
    evidence: list[EvidenceChunk],
    terms: list[str],
) -> list[_TermSource]:
    """Collect every source-attributed match across pages, fields, and chunks.

    The same phrase usually appears in multiple sources (a page and the chunk
    derived from it). Callers should dedupe via _select_audit_sources before
    surfacing to operators.
    """

    if not terms:
        return []
    out: list[_TermSource] = []
    for document in processed:
        filename = document.source.filename
        document_id = document.source.document_id
        for page in document.pages:
            for term in terms:
                out.extend(
                    _scan_text(
                        page.text,
                        term,
                        source_kind="page",
                        source_id=f"{page.document_id}:p{page.page_number}",
                        document_id=page.document_id,
                        filename=filename,
                        page_number=page.page_number,
                    )
                )
        for field_name, field in document.fields.items():
            if not field.value:
                continue
            page_number = _field_anchor_page(list(field.evidence_ids))
            for term in terms:
                out.extend(
                    _scan_text(
                        field.value,
                        term,
                        source_kind="field",
                        source_id=f"{document_id}:fields:{field_name}",
                        document_id=document_id,
                        filename=filename,
                        page_number=page_number,
                    )
                )
    for chunk in evidence:
        for term in terms:
            out.extend(
                _scan_text(
                    chunk.text,
                    term,
                    source_kind="chunk",
                    source_id=chunk.evidence_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                )
            )
    return out


def _select_audit_sources(sources: list[_TermSource]) -> list[dict[str, Any]]:
    """Dedupe and rank source matches for inclusion in a finding."""

    rank = {"page": 0, "chunk": 1, "field": 2}
    ordered = sorted(sources, key=lambda s: (rank.get(s.source_kind, 9), s.document_id, s.page_number, s.char_start))
    seen: set[tuple[str, str, int, str]] = set()
    deduped: list[_TermSource] = []
    for source in ordered:
        key = (source.term, source.document_id, source.page_number, source.snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
        if len(deduped) >= _SOURCE_SPAN_LIMIT:
            break
    return [_source_to_dict(source) for source in deduped]


def _source_to_dict(source: _TermSource) -> dict[str, Any]:
    return {
        "term": source.term,
        "source_kind": source.source_kind,
        "source_id": source.source_id,
        "document_id": source.document_id,
        "filename": source.filename,
        "page_number": source.page_number,
        "snippet": source.snippet,
    }


def _evidence_ids_from_sources(sources: list[_TermSource]) -> list[str]:
    """Return retrieval-chunk source IDs only.

    `evidence_ids` historically referred to retrieved-chunk IDs (shape
    `doc_id:pN:cM`). Downstream consumers that resolve evidence IDs against
    the report's top-level `evidence` block depend on that invariant; keep
    `evidence_ids` strictly to chunk IDs and surface page/field IDs through
    `source_ids` instead.
    """

    return list(dict.fromkeys(s.source_id for s in sources if s.source_kind == "chunk"))[:_EVIDENCE_ID_LIMIT]


def _source_ids_from_sources(sources: list[_TermSource]) -> list[str]:
    """Return all source IDs (chunk, page, and field) in match order."""

    chunk_ids = [s.source_id for s in sources if s.source_kind == "chunk"]
    other_ids = [s.source_id for s in sources if s.source_kind != "chunk"]
    return list(dict.fromkeys(chunk_ids + other_ids))[:_EVIDENCE_ID_LIMIT]


def _finding(
    *,
    position_id: str,
    title: str,
    status: str,
    severity: str,
    message: str,
    matched_terms: list[str],
    source_spans: list[dict[str, Any]],
    evidence_ids: list[str],
    source_ids: list[str],
    guidance: str,
) -> dict[str, Any]:
    audit_warning = None
    if not source_spans and status in {"escalation_triggered", "review_preferred_position"}:
        # Invariant: a fired finding always has at least one source span. If we
        # ever reach this branch the matcher and the source-attribution have
        # disagreed; surface that to the operator instead of silently emitting
        # an unauditable high-severity flag.
        audit_warning = "Finding fired without an attributable source span; review playbook terms and extraction."
    payload: dict[str, Any] = {
        "position_id": position_id,
        "title": title,
        "status": status,
        "severity": severity,
        "message": message,
        "matched_terms": sorted(set(matched_terms)),
        "source_spans": source_spans,
        "evidence_ids": evidence_ids,
        "source_ids": source_ids,
        "guidance": guidance,
    }
    if audit_warning:
        payload["audit_warning"] = audit_warning
    return payload


def _severity(value: str) -> str:
    resolved = value.strip().lower()
    return resolved if resolved in _SEVERITY_ORDER else "medium"


def _downgrade(severity: str) -> str:
    order = ["info", "low", "medium", "high", "critical"]
    index = max(0, order.index(_severity(severity)) - 1)
    return order[index]
