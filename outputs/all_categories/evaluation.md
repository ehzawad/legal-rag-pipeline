# Evaluation Suite

- Run dir: outputs/all_categories
- Citation id validity: 100.00%
- Sentence grounding score: 100.00%
- Claim grounding pass rate: 100.00%
- Unsupported claims: 0
- Eval-points required pass rate: not available
- Simulated edit-improvement pass rate: not available
- Supported section ratio: 75.00%
- Claim citation rate: 100.00%
- Cited field-chunk share: 18.75%
- Cited raw-page share: 81.25%

## Processing

- Documents: 20
- Pages: 70
- Non-empty page rate: 100.00%
- Average page confidence: 0.89
- Low-confidence pages (<0.55): 2
- Extraction methods: pdf-rendered-png=25, pdf-text-layer=45

## Retrieval

- Retrieved chunks: 10
- Draft-conditioned chunks: 8
- Retrieved field chunks: 1
- Retrieved raw-page chunks: 9
- Distinct cited files: 5

## Category Coverage

| Category | Docs | Pages | Rendered pages | Text-layer pages | Avg conf |
|---|---:|---:|---:|---:|---:|
| handwritten-notes | 3 | 15 | 10 | 5 | 0.72 |
| inconsistently-formatted-files | 3 | 13 | 0 | 13 | 0.95 |
| low-resolution-pdfs | 3 | 5 | 4 | 1 | 0.87 |
| partially-illegible-records | 3 | 15 | 1 | 14 | 0.95 |
| scanned-pages | 3 | 14 | 10 | 4 | 0.94 |
| truepdf | 5 | 8 | 0 | 8 | 0.95 |

## Gold Checks

- Page-text term recall: 80.00%
- Field-value term recall: 90.00%
- Combined term recall: 95.00%
- Documents checked: 5
- Missing gold documents: 0
- Retrieval recall@k: 33.33%
- Retrieval precision@k: 37.50%
- Retrieval MRR: 0.17

## Notes

- Unsupported sections are intentional review hooks, not generation failures.
- Sentence grounding score is quote-substring validity, not semantic entailment.
- Claim grounding pass rate is claim-graph level; section quote metrics remain separate.
