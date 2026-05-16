# Playbook Risk Report: default-legal-review-playbook

Generated: 2026-05-16T18:00:50+00:00

## Summary

- Positions checked: 5
- Passed positions: 2
- Findings: 4
- High/Critical: 1

## Findings

### Limitation of Liability

- Status: `missing_or_unconfirmed`
- Severity: `medium`
- Message: Required playbook concept was not found in extracted text.
- Matched terms: none
- Evidence: none
- Guidance: Escalate uncapped or asymmetric liability positions and confirm whether the cap matches the team's standard.

### Indemnification

- Status: `escalation_triggered`
- Severity: `high`
- Message: Escalation term(s) found: sole discretion.
- Matched terms: sole discretion
- Evidence: none
- Guidance: Flag unilateral or unlimited indemnity and verify whether the obligation is limited to third-party claims.
- Source spans:
  - p.1 (cuad_hosting_agreement.pdf) — `sole discretion` → "…Service except as specified in this Section. Company at its sole discretion may at any time alter or cease providing the Customer Servi…"

### Data Protection

- Status: `missing_or_unconfirmed`
- Severity: `low`
- Message: Required playbook concept was not found in extracted text.
- Matched terms: none
- Evidence: none
- Guidance: Confirm DPA coverage, deletion/return duties, breach notice timing, and subprocessors when personal data appears.

### Unsupported Draft Sections

- Status: `unsupported_draft_content`
- Severity: `medium`
- Message: 1 draft section(s) are marked unsupported.
- Matched terms: none
- Evidence: none
- Guidance: Route unsupported sections to operator review before relying on the draft.

## Disclaimer

Deterministic playbook screen for operator review; not legal advice.
