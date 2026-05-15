# Evaluation fixtures

This folder is for evaluation pipeline inputs, not uploadable source
documents.

| File | Purpose |
|---|---|
| `harness_cases.json` | Small smoke-test manifest for `pipeline harness run`. |
| `gold_documents.json` | Deterministic document-understanding checks for `pipeline eval-suite`. |
| `gold_retrieval.json` | Deterministic retrieval checks for `pipeline eval-suite`. |
| `eval_points.json` | Grouped public-eval checks for document understanding, grounded drafting, and edit improvement. |
| `public_eval_manifest.json` | Download/verification manifest for `eval/public-documents/`. |
| `public-documents/` | Downloaded public evaluation PDFs/images. These are eval inputs, not the operator sample dataset. |
| `operator-edits/` | Example human edits used by learning and A/B evaluation commands. |

Common commands:

```bash
uv run pipeline harness run \
  --manifest eval/harness_cases.json \
  --output outputs/harness \
  --json

uv run pipeline eval-suite \
  --run outputs/all_categories \
  --gold-dir eval \
  --output outputs/all_categories/evaluation.md \
  --json-output outputs/all_categories/evaluation.json

uv run pipeline public-data verify \
  --manifest eval/public_eval_manifest.json \
  --dataset eval/public-documents
```
