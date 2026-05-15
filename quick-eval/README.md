# Quick-eval drop-folder

Three reviewer-friendly external test PDFs. None of these files appears in
`datasets/` or `eval/public-documents/` — they are SHA-256 verified to not
collide with any tracked corpus file.

| File | Pages | Routing |
|---|---:|---|
| `trump_v_anderson_per_curiam_2024.pdf` | 20 | clean text-layer (`pdf-text-layer` extraction) |
| `nara_constitution_page4_scanned.pdf` | 1 | image-only (`pdf-rendered-png` 180-DPI extraction) |
| `fictional_four_page_legal_case_file.pdf` | 4 | clean text-layer; a fictional case packet generated for workflow smoke testing (clearly labeled "FICTIONAL SAMPLE — not a court record, not legal advice") |

Verify the files are exactly what is committed (full SHA-256 hashes):

```
f3015ab4890996a0cb1f1cb3e943cf27d2d6a58ced20f8dbdc3d06e79c15d07a  trump_v_anderson_per_curiam_2024.pdf
1a24b7e4dc5a4353b429d15087964942be3a7c8678032e2fc48577f4702b9e03  nara_constitution_page4_scanned.pdf
f370f158242f6d28a5f77cb65240d2abaa6df67e3e44b473336daf09dfd8042d  fictional_four_page_legal_case_file.pdf
```

Reproduce with `shasum -a 256 quick-eval/*.pdf`. All three hashes are absent
from `datasets/` and `eval/public-documents/`.

Sources:

- Trump v. Anderson per curiam, March 4 2024 — `https://www.supremecourt.gov/opinions/23pdf/23-719_19m2.pdf`
- NARA U.S. Constitution page 4 (we already track pages 1–3 in
  `eval/public-documents/handwritten-notes/`; page 4 was downloaded fresh
  from `https://www.archives.gov/files/founding-docs/constitution_4_of_4_630.jpg`
  and wrapped as a PDF — pure image, zero text layer).

Prereq: complete the root setup in [`../instructions.md`](../instructions.md)
first (`uv sync --extra dev`, `OPENAI_API_KEY` exported).

## How to test (operator UI)

```bash
uv run pipeline-api                       # leave running
open http://127.0.0.1:8000/ui             # operator console
```

1. Click **Upload documents**.
2. Set a case id (e.g. `quick-eval-clean` or `quick-eval-scanned`).
3. Paste a drafting task — e.g.
   `Review the supplied document and produce a first-pass memo covering parties, obligations, and any data-quality flags.`
4. Drop one of these PDFs in and submit.
5. Watch the **Draft** tab render with clickable `[evidence_id]` chips.

The clean PDF exercises the local text-layer extraction path with no API image
calls. The scanned PDF forces the 180-DPI image render path through the
Responses API.

## How to test (CLI)

```bash
uv run pipeline run \
  --input  quick-eval \
  --output outputs/quick_eval \
  --case-id quick-eval \
  --task   "Review the supplied documents and produce a first-pass memo covering parties, obligations, and any data-quality flags." \
  --force

uv run pipeline eval-suite \
  --run outputs/quick_eval \
  --output outputs/quick_eval/evaluation.md \
  --json-output outputs/quick_eval/evaluation.json

open outputs/quick_eval/draft.md
open outputs/quick_eval/evaluation.md
```

Success criteria (not guaranteed expectations — these PDFs have no
pre-built gold labels here): citation id validity, quote-substring
validity, and claim citation rate should all pin at 100% because
those are validator-enforced contracts at the drafter boundary. Share
metrics, term-recall numbers, and retrieval recall are stochastic
across runs.

## What's not in here

These two files do **not** replace the curated `datasets/` (20 docs covering
the five messy categories plus a clean control set) or the hash-pinned
public bundle in `eval/public-documents/`. They exist solely as a
zero-prep reviewer smoke surface: drag, drop, see grounded output.
