# Evaluation Suite

- Run dir: outputs/public_eval
- Citation id validity: 100.00%
- Sentence grounding score: 100.00%
- Claim grounding pass rate: 88.89%
- Unsupported claims: 3
- Eval-points required pass rate: 100.00%
- Simulated edit-improvement pass rate: 100.00%
- Supported section ratio: 25.00%
- Claim citation rate: 100.00%
- Cited field-chunk share: 0.00%
- Cited raw-page share: 100.00%

## Processing

- Documents: 15
- Pages: 40
- Non-empty page rate: 100.00%
- Average page confidence: 0.92
- Low-confidence pages (<0.55): 1
- Extraction methods: openai=3, pdf-rendered-png=12, pdf-text-layer=25

## Retrieval

- Retrieved chunks: 10
- Draft-conditioned chunks: 8
- Retrieved field chunks: 0
- Retrieved raw-page chunks: 10
- Distinct cited files: 3

## Category Coverage

| Category | Docs | Pages | Rendered pages | Text-layer pages | Avg conf |
|---|---:|---:|---:|---:|---:|
| handwritten-notes | 3 | 3 | 0 | 0 | 0.64 |
| inconsistently-formatted-files | 3 | 12 | 0 | 12 | 0.95 |
| low-resolution-pdfs | 3 | 3 | 0 | 3 | 0.95 |
| partially-illegible-records | 3 | 10 | 0 | 10 | 0.95 |
| scanned-pages | 3 | 12 | 12 | 0 | 0.93 |

## Public Manifest

- Manifest: datasets/public_eval_manifest.json
- Items: 15
- Matched processed files: 15
- Missing processed files: 0
- Required categories present: True

## Eval Points

- File: datasets/eval/eval_points.json
- Required pass rate: 100.00%
- Overall pass rate: 100.00%

| Criterion | Required pass | Overall pass | Failed required |
|---|---:|---:|---|
| document_understanding | 100.00% | 100.00% | - |
| grounded_drafting | 100.00% | 100.00% | - |
| edit_improvement | 100.00% | 100.00% | - |

Category point coverage:
- handwritten-notes: 100.00% (1/1)
- inconsistently-formatted-files: 100.00% (2/2)
- low-resolution-pdfs: 100.00% (2/2)
- partially-illegible-records: 100.00% (2/2)
- scanned-pages: 100.00% (1/1)

## Simulated Edit Improvement

- Passed: True
- Pass rate: 100.00%
- State dir: outputs/public_eval/eval/simulated_edits/state
- pass: edit-memory-before-empty
- pass: edit-memory-retrieves-learned-section
- pass: profile-guidance-adopts-operator-section
- pass: knowledge-rule-rendered
- pass: retrieval-feedback-moves-good-evidence-to-top
- pass: grounding-non-regression

## Notes

- Unsupported sections are intentional review hooks, not generation failures.
- Sentence grounding score is quote-substring validity, not semantic entailment.
- Claim grounding pass rate is claim-graph level; section quote metrics remain separate.
- No gold_dir supplied; suite reports artifact integrity and coverage only.
