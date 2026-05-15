# Public Evaluation Data

The repo ships two evaluation corpora:

- `datasets/`: curated 20-document corpus — 3 documents per messy category
  plus a 5-doc clean true-PDF control set.
- `eval/public-documents/`: downloaded public bundle, 13 files, ~2-3
  documents per messy category, all capped at 5 pages. The public
  bundle is strictly distinct from `datasets/`: zero filename and zero
  SHA-256 overlap.

The public bundle exists so reviewers can run the pipeline on files that were
not only hand-curated into the repo.

## Manifest Workflow

Download and verify:

```bash
uv run pipeline public-data download \
  --manifest eval/public_eval_manifest.json \
  --output eval/public-documents

uv run pipeline public-data verify \
  --manifest eval/public_eval_manifest.json \
  --dataset eval/public-documents
```

The manifest pins URL, relative filename, byte count, SHA-256, expected page
count, max page count, source, license notes, and category. Verification fails
on hash/byte mismatches or any file above its `max_pages` cap.

## Current Public Bundle

| Category | Files | Source |
|---|---:|---|
| scanned-pages | 2 | SEC EDGAR VPRR image-only paper filing scans |
| low-resolution-pdfs | 3 | Library of Congress Chronicling America microfilm pages |
| handwritten-notes | 3 | National Archives Constitution page images |
| partially-illegible-records | 2 | SEC EDGAR VPRR fax-quality records with broken text layers |
| inconsistently-formatted-files | 3 | GovInfo U.S. Code and CFR legal text PDFs |

All 13 files are verified at five pages or fewer.

## Run And Evaluate

```bash
uv run pipeline run \
  --input eval/public-documents \
  --output outputs/public_eval \
  --case-id public-eval \
  --task "Review the supplied public evaluation documents and produce a grounded case fact summary covering documents reviewed, material facts that appear supported, requested actions, and data-quality flags an operator should verify." \
  --force

uv run pipeline eval-suite \
  --run outputs/public_eval \
  --manifest eval/public_eval_manifest.json \
  --eval-points eval/eval_points.json \
  --simulate-edits \
  --output outputs/public_eval/evaluation.md \
  --json-output outputs/public_eval/evaluation.json
```

The eval suite reports three criteria separately: document understanding,
grounded drafting, and improvement from edits. The simulated edit-improvement
checks are deterministic and use the same `learn_from_files` entry point as real
operator edits.

The historical public-eval artifact under `outputs/public_eval/` was
produced against the prior 15-doc bundle and is left in place as a
sample. To refresh against the current 13-doc bundle, re-run the two
commands above; the one eval point impacted by the dedup
(`sec-illegible-commission-file-term`, `required=false`) does not
affect the `required_pass_rate`.
