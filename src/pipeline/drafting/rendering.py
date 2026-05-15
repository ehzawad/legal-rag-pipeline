from __future__ import annotations

from pipeline.drafting.memo import (
    _operator_claim_key,
    _operator_claim_text,
    _operator_language_text,
    render_section_quotes_markdown,
    visible_review_warnings,
)


def render_draft_markdown(draft) -> str:
    if getattr(draft, "case_summary", None) is not None:
        return _render_case_summary_markdown(draft)
    lines = [f"# {draft.title}", "", f"Draft type: {draft.draft_type}", ""]
    warnings = _visible_markdown_warnings(draft)
    if warnings:
        lines.extend(["## Review Flags", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    for section in draft.sections:
        lines.extend([f"## {section.heading}", "", section.body.strip() or "_No supported content found._"])
        if section.evidence_ids:
            lines.append("")
            lines.append("Evidence: " + ", ".join(f"[{item}]" for item in section.evidence_ids))
            lines.extend(render_section_quotes_markdown(section))
        if section.unsupported:
            lines.append("")
            lines.append("Status: unsupported or needs operator review.")
        lines.append("")

    if draft.evidence:
        lines.extend(["## Evidence Index", ""])
        for chunk in draft.evidence:
            excerpt = " ".join(chunk.text.split())
            if len(excerpt) > 280:
                excerpt = excerpt[:277] + "..."
            kind = "derived fields" if chunk.metadata.get("is_field_chunk") else "source page"
            lines.append(
                f"- [{chunk.evidence_id}] {chunk.filename} p.{chunk.page_number} "
                f"({kind}, score {chunk.score:.2f}): {excerpt}"
            )
        lines.append("")
    return "\n".join(lines)


def _render_case_summary_markdown(draft) -> str:
    summary = draft.case_summary
    lines = [f"# {summary.title or draft.title}", "", f"Draft type: {draft.draft_type}", ""]
    warnings = _visible_markdown_warnings(draft, summary)
    if warnings:
        lines.extend(["## Review Flags", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    claims_by_section: dict[str, list[object]] = {}
    for claim in summary.claims:
        claims_by_section.setdefault(claim.section_id, []).append(claim)
    section_order = list(summary.section_order)
    for section_id in claims_by_section:
        if section_id not in section_order:
            section_order.append(section_id)

    for section_id in section_order:
        claims = claims_by_section.get(section_id, [])
        if not claims:
            continue
        lines.extend([f"## {_section_heading(section_id)}", ""])
        seen_lines: set[str] = set()
        for claim in claims:
            citations = " ".join(f"[{citation.evidence_id}]" for citation in claim.citations)
            display_text = _operator_claim_text(section_id, claim.text)
            line_key = _operator_claim_key(display_text)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            line = f"- {display_text}"
            if citations:
                line += f" {citations}"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def _visible_markdown_warnings(draft, summary=None) -> list[str]:
    warnings = list(getattr(draft, "warnings", []) or [])
    if summary is not None:
        warnings.extend(list(getattr(summary, "warnings", []) or []))
    visible = visible_review_warnings(warnings, evidence=getattr(draft, "evidence", []) or [])
    result: list[str] = []
    seen: set[str] = set()
    for warning in visible:
        text = _operator_language_text(warning)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _section_heading(section_id: str) -> str:
    headings = {
        "case_caption": "Case Caption",
        "parties": "Parties",
        "documents_reviewed": "Documents Reviewed",
        "procedural_posture": "Procedural Posture",
        "material_facts": "Material Facts",
        "relief_sought": "Relief Sought / Requested Action",
        "open_questions": "Open Questions",
        "evidence_table": "Evidence Table",
    }
    return headings.get(section_id, section_id.replace("_", " ").title())
