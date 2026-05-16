# Quick-eval drop-folder

Six reviewer-friendly smoke PDFs. None of these files is part of the
curated `datasets/` corpus or the hash-pinned public bundle under
`eval/public-documents/`; they exist for quick manual CLI, Docker, and
operator-UI checks.

`process_directory()` ingests every supported file under the input
directory, so `--input quick-eval` processes all six PDFs. To run only one
legal-dispute PDF, copy it into a temporary one-file input directory first.

| File | Pages | Routing |
|---|---:|---|
| `applevsspotify.pdf` | 22 | clean text-layer (`pdf-text-layer` extraction) |
| `applevsunreal-engine.pdf` | 65 | clean text-layer (`pdf-text-layer` extraction) |
| `openaivselon.pdf` | 21 | clean text-layer (`pdf-text-layer` extraction) |
| `trump_v_anderson_per_curiam_2024.pdf` | 20 | clean text-layer (`pdf-text-layer` extraction) |
| `nara_constitution_page4_scanned.pdf` | 1 | image-only (`pdf-rendered-png` 180-DPI extraction) |
| `fictional_four_page_legal_case_file.pdf` | 4 | clean text-layer; fictional workflow smoke packet |

Verify the files are exactly what is committed:

```bash
shasum -a 256 quick-eval/*.pdf
```

Expected SHA-256 hashes:

```text
b56a4cec78b145803a707ea28f0cd1fa00680e13af87eee97965847fdbdfa924  quick-eval/applevsspotify.pdf
8f012796ec4cd2bd21e065e109ef6957ae231e5a33fbd207acd836ec76c73a78  quick-eval/applevsunreal-engine.pdf
f370f158242f6d28a5f77cb65240d2abaa6df67e3e44b473336daf09dfd8042d  quick-eval/fictional_four_page_legal_case_file.pdf
1a24b7e4dc5a4353b429d15087964942be3a7c8678032e2fc48577f4702b9e03  quick-eval/nara_constitution_page4_scanned.pdf
8b8bf42f436ae3651333943c1dbe3fcedb7658c24a592f5105dcaf930296e0e8  quick-eval/openaivselon.pdf
f3015ab4890996a0cb1f1cb3e943cf27d2d6a58ced20f8dbdc3d06e79c15d07a  quick-eval/trump_v_anderson_per_curiam_2024.pdf
```

Sources:

- Trump v. Anderson per curiam, March 4 2024:
  `https://www.supremecourt.gov/opinions/23pdf/23-719_19m2.pdf`
- NARA U.S. Constitution page 4:
  `https://www.archives.gov/files/founding-docs/constitution_4_of_4_630.jpg`
- Apple/Spotify, Apple/Unreal, and OpenAI/Elon PDFs are local external
  reviewer smoke files added for this project; keep their hashes pinned
  when refreshing them.

Prereq: complete the root setup in [`../instructions.md`](../instructions.md)
first (`uv sync --extra dev`, API keys exported or provided through secrets).

## Operator UI

```bash
uv run pipeline-api
open http://127.0.0.1:8000/ui
```

1. Click **Upload documents**.
2. Set a case id such as `quick-eval-spotify`.
3. Use a task such as:
   `Review the supplied document and produce a grounded case fact summary covering parties, material facts, requested actions, and data-quality flags.`
4. Drop one PDF or a subset of these PDFs and submit.
5. Check the Draft and Grounding tabs for citation chips and grounded claims.

## CLI

Run all six:

```bash
uv run pipeline run \
  --input  quick-eval \
  --output outputs/quick_eval \
  --case-id quick-eval \
  --task   "Review the supplied documents and produce a grounded case fact summary covering parties, material facts, requested actions, and data-quality flags." \
  --force
```

Run only the three current legal-dispute smoke files:

```bash
mkdir -p /tmp/legal-rag-fixed-pdfs
cp quick-eval/applevsspotify.pdf quick-eval/applevsunreal-engine.pdf quick-eval/openaivselon.pdf /tmp/legal-rag-fixed-pdfs/

uv run pipeline run \
  --input  /tmp/legal-rag-fixed-pdfs \
  --output outputs/e2e_fixed_pdfs_local \
  --case-id e2e-fixed-pdfs-local \
  --task   "Review the supplied legal dispute documents and produce a grounded case fact summary." \
  --force
```

Evaluate:

```bash
uv run pipeline eval-suite \
  --run outputs/quick_eval \
  --output outputs/quick_eval/evaluation.md \
  --json-output outputs/quick_eval/evaluation.json
```

## Docker

`quick-eval/` is copied into the image and mounted by Docker Compose at
`/app/quick-eval`, so this works inside the container:

```bash
docker compose exec api pipeline run \
  --input /app/quick-eval \
  --output /app/outputs/quick_eval_docker \
  --case-id quick-eval-docker \
  --task "Review the supplied documents and produce a grounded case fact summary." \
  --force
```

Success criteria: the run writes `processed_documents.json`,
`retrieved_evidence.json`, `draft.json`, `case_fact_summary.json`,
`grounding_report.json`, and `index/retrieval_index.json`; citation id
validity, quote-substring grounding, claim citation coverage, and claim
grounding pass rate should be 100% because those are enforced before
draft artifacts are written.
