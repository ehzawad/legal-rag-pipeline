# A/B Improvement-Loop Evaluation

- Case dir: outputs/all_categories
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

## Deltas (with-profile vs. without-profile)

- Section-label adoption delta: 1
- Section-label suppression delta: 0
- Caution-marker delta (body, signed): 3
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

- Claim citation rate without profile: 100.00%
- Claim citation rate with profile:    100.00%
- Unsupported claims without profile: None
- Unsupported claims with profile:    None
- Claim grounding pass rate without profile: not available
- Claim grounding pass rate with profile:    not available
- Raw-page citation share without profile: 75.00%
- Raw-page citation share with profile:    80.00%

## Edited-reference similarity

- Body token similarity without profile: 10.63%
- Body token similarity with profile:    10.33%
- Section F1 without profile:            0.00%
- Section F1 with profile:               0.00%

## Per-draft summary

Without profile:
- Sections: 21
- Cited sections: 18
- Body length (chars): 29621

With profile:
- Sections: 9
- Cited sections: 6
- Body length (chars): 30271

## Notes

- With-profile draft has fewer cited sections than baseline; review for coverage regression.
- With-profile draft is less similar to the edited reference than baseline.
