# A/B Improvement-Loop Evaluation

- Case dir: outputs/e2e_ui_smoke
- Profile: state/operator_profile.json

## Methodology

- Both drafts use the same processed documents and retrieved evidence.
- The with-profile draft receives prompt-time guidance learned from human operator edits.
- The profile should steer structure, tone, and phrasing; grounding/factuality is checked separately.
- AI critique or deterministic warnings are advisory unless accepted as human operator annotations.

Profile guidance:
Use these section labels when relevant: Items Requiring Verification, Operator Directive, Compliance Review Memo, Deposition Renotice & Subpoena — provisional. Avoid these section labels unless specifically requested: Data-Quality Flags for Operator Verification, Deadlines and Dates, Obligations and Amounts, Parties and Documents Identified. Prefer concise phrasing such as: confirm before treating as a deadline; Every factual claim must carry an inline citation; Handwritten and faintly-scanned pages dominate this run; Maintain full-prose, page-and-line citation precision; The fifteen-day cure period is recorded as expired. Use cautious, evidence-bound language; qualify uncertainty with likely, provided records, verify. When consolidating or shortening sections, preserve or increase inline citation density: never drop a citation that supports a kept factual claim, and prefer adding citations to any new factual sentence introduced during consolidation.

Operator-defined knowledge rules (treat as hard constraints):
- Always treat any net-income or aggregate-revenue value with decimal punctuation in a millions context as a transcription error pending source reconciliation.

Past operator-edited memos for similar documents (for tone and structure reference):
Example 1 (category: scanned-pages):
  Operator added sections: Compliance Review Memo, Executive Summary, Findings by Filing, Items Requiring Verification
  Operator removed sections: Data-Quality Flags for Operator Verification, Deadlines and Dates, Evidence Index, First-Pass Internal Memo
  Opening of operator-edited memo: # Compliance Review Memo

## Executive Summary

The retrieved corpus combines six distinct document types: a real-property lien notice, a website services agreement, two tobacco-litigation deposition instruments, an SEC annual audited report (Form X-17A-5 Part III), and an SEC Form F-X appointment o
Adapt structure and tone to match these prior operator edits where appropriate, but never relax the citation and grounding requirements.

## Deltas (with-profile vs. without-profile)

- Section-label adoption delta: 2
- Section-label suppression delta: 0
- Caution-marker delta (body, signed): 20
- Preferred-phrase adoption: 0.00% of top 5 preferred phrases
  - Phrases checked: 'confirm before treating as a deadline.', 'Every factual claim must carry an inline citation.', 'Handwritten and faintly-scanned pages dominate this run', 'Maintain full-prose, page-and-line citation precision.', 'The fifteen-day cure period is recorded as expired'
- Preferred phrases are one signal; section labels and caution tone are scored separately.

## Citation id validity

- Without profile: 100.00%
- With profile:    100.00%

## Sentence grounding score (verbatim quote match)

- Without profile: 100.00%
- With profile:    100.00%

## Non-regression checks

- Claim citation rate without profile: 50.00%
- Claim citation rate with profile:    50.00%
- Unsupported claims without profile: None
- Unsupported claims with profile:    None
- Claim grounding pass rate without profile: not available
- Claim grounding pass rate with profile:    not available
- Raw-page citation share without profile: 50.00%
- Raw-page citation share with profile:    68.75%

## Per-draft summary

Without profile:
- Sections: 8
- Cited sections: 6
- Body length (chars): 12422

With profile:
- Sections: 9
- Cited sections: 6
- Body length (chars): 16635
