from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pipeline.io import read_json, write_json
from pipeline.orchestration.artifacts import draft_from_json, evidence_from_json, processed_documents_from_json
from pipeline.schemas import Draft, EvidenceChunk, ProcessedDocument, now_iso


DEFAULT_PLAYBOOK_PATH = Path(__file__).resolve().parents[3] / "playbooks" / "default_legal_playbook.json"
_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


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

    corpus_text = _combined_document_text(processed)
    evidence_by_id = {chunk.evidence_id: chunk for chunk in evidence}
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

        escalation_matches = _matched_terms(corpus_text, escalation_terms)
        required_matches = _matched_terms(corpus_text, required_terms)
        preferred_matches = _matched_terms(corpus_text, preferred_terms)
        evidence_ids = _matching_evidence_ids(evidence, escalation_matches + required_matches + preferred_matches)

        if escalation_matches:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="escalation_triggered",
                    severity=severity,
                    message=f"Escalation term(s) found: {', '.join(escalation_matches)}.",
                    matched_terms=escalation_matches,
                    evidence_ids=evidence_ids,
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        if required_terms and not required_matches:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="missing_or_unconfirmed",
                    severity=_downgrade(severity),
                    message="Required playbook concept was not found in extracted text.",
                    matched_terms=[],
                    evidence_ids=[],
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        if preferred_terms and not preferred_matches:
            findings.append(
                _finding(
                    position_id=position_id,
                    title=title,
                    status="review_preferred_position",
                    severity=_downgrade(severity),
                    message="Clause appears present, but preferred playbook language was not found.",
                    matched_terms=required_matches,
                    evidence_ids=evidence_ids,
                    guidance=str(position.get("guidance") or ""),
                )
            )
            continue

        passed_positions.append(
            {
                "position_id": position_id,
                "title": title,
                "matched_terms": sorted(set(required_matches + preferred_matches)),
                "evidence_ids": evidence_ids,
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
                evidence_ids=[],
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
                "",
            ]
        )
    lines.extend(["## Disclaimer", "", str(report.get("disclaimer") or "")])
    return "\n".join(lines).rstrip()


def _combined_document_text(processed: list[ProcessedDocument]) -> str:
    parts: list[str] = []
    for document in processed:
        parts.extend(page.text for page in document.pages)
        for field in document.fields.values():
            if field.value:
                parts.append(field.value)
    return "\n".join(parts)


def _terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if _contains_term(text, term)]


def _matching_evidence_ids(evidence: list[EvidenceChunk], terms: list[str]) -> list[str]:
    if not terms:
        return []
    matches: list[str] = []
    for chunk in evidence:
        if any(_contains_term(chunk.text, term) for term in terms):
            matches.append(chunk.evidence_id)
    return matches[:8]


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term.casefold()) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def _finding(
    *,
    position_id: str,
    title: str,
    status: str,
    severity: str,
    message: str,
    matched_terms: list[str],
    evidence_ids: list[str],
    guidance: str,
) -> dict[str, Any]:
    return {
        "position_id": position_id,
        "title": title,
        "status": status,
        "severity": severity,
        "message": message,
        "matched_terms": sorted(set(matched_terms)),
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "guidance": guidance,
    }


def _severity(value: str) -> str:
    resolved = value.strip().lower()
    return resolved if resolved in _SEVERITY_ORDER else "medium"


def _downgrade(severity: str) -> str:
    order = ["info", "low", "medium", "high", "critical"]
    index = max(0, order.index(_severity(severity)) - 1)
    return order[index]
