# Pipeline

Full-stack, evidence-grounded first-pass drafter for messy legal-style
documents. The AI processes scans/PDFs, retrieves source evidence, and
produces a cited draft; a human operator/reviewer edits that draft through
the React console; captured edits steer future drafts without weakening
the grounding contract.

Stack:

- **Backend** — Python 3.14 / FastAPI / OpenAI Responses + PyMuPDF.
  Hybrid (BM25 + dense) retrieval with optional FAISS, verbatim citation
  grounding, typed-intent edit learning, deterministic evaluation suite.
- **Frontend** — React 18 + TypeScript + Vite + TanStack Query v5 +
  react-router. A multi-page operator console (`frontend/`) talks to
  the FastAPI surface over JSON. The single piece of cross-page UI
  state (the selected-evidence chip) lives in a small React Context
  backed by `useState`; there is no Redux store.
- **Single-binary deploy** — the API serves the built React bundle at
  `/ui`. Docker image is multi-stage (Node build → Python runtime).

Role boundary: the operator is a human legal-ops reviewer, paralegal,
case analyst, or attorney reviewer. AI components assist with document
extraction, retrieval, drafting, critique, and edit classification, but
AI critique is advisory until accepted by the human operator.

## Reviewer map

Every artifact a reviewer needs is in the repo at a stable path:

| Required item | Where to look |
|---|---|
| Source code | `src/pipeline/` (backend) · `frontend/src/` (UI) |
| Quickstart (tiered: local / clean machine / Docker) | [`instructions.md`](instructions.md) |
| Setup and run instructions | [`instructions.md`](instructions.md) · This README §Setup, §Run, §Operator UI walkthrough |
| Two external smoke PDFs (drag-and-drop the operator UI) | [`quick-eval/`](quick-eval/) — clean SCOTUS opinion + image-only NARA scan, both verified distinct from `datasets/` and `eval/` |
| Architecture overview | `docs/architecture.md` |
| Assumptions and tradeoffs | README §Assumptions and tradeoffs · architecture §Honest Limitations |
| Sample inputs | `datasets/` (see `datasets/COVERAGE.md`) |
| Sample outputs | `outputs/all_categories/`, `outputs/public_eval/` (README §Sample Inputs And Outputs) |
| Evaluation approach + results | README §Evaluate · `docs/evaluation.md` · `outputs/*/evaluation.md` |

## Setup

> **In a hurry?** [`instructions.md`](instructions.md) is a tiered quickstart
> (already-set-up local · clean machine · Docker) with copy-pasteable commands
> and expected cost/duration per step.

This is a full-stack project. You need both the Python runtime and the
Node build of the React UI before the operator console renders.

```bash
# Backend
uv sync --extra dev
export OPENAI_API_KEY=...

# Frontend (required for /ui to work under uv run pipeline-api)
cd frontend
npm install
npm run build
cd ..
```

Requirements:

- Python 3.14.5 (pinned in `.python-version`) with `uv`.
- Node 22+ and `npm` for the frontend build.
- An `OPENAI_API_KEY` with access to the configured model.

The frontend build emits `src/pipeline/ui/static/dist/`, which the API
mounts under `/ui` with a SPA fallback. The bundle is committed to the
repo so reviewers using only `uv run pipeline-api` still get the UI; the
Docker image rebuilds it inside the container.

The lockfile is resolved for Python 3.14 only. The OpenAI SDK, PyMuPDF,
Pillow, FastAPI, and uvicorn are declared as required runtime
dependencies in `pyproject.toml` — no other provider SDKs are pulled.
The optional Cohere reranker uses direct HTTPS, so no Cohere SDK
dependency is required.

## Defaults

| | |
|---|---|
| Drafting / extraction model | `gpt-5.5` via Responses API |
| Reranker | off by default; optional Cohere `rerank-v4.0-pro` via direct HTTPS |
| Reasoning effort | `low` (override via `OPENAI_REASONING_EFFORT=low\|medium\|high`) |
| Embedding model | `text-embedding-3-large` |
| Retrieval backend | hybrid dense + BM25 + metadata/retrieval feedback; optional FAISS acceleration and `openai-cached` embeddings |
| PDF preprocessing | local PyMuPDF, cap 100 pages, 180 DPI render fallback |

Override draft/extraction and retrieval knobs via `OPENAI_MODEL`,
`OPENAI_MODEL_DRAFT`, `OPENAI_EMBEDDING_MODEL`,
`PIPELINE_RETRIEVAL_PROVIDER`, `PIPELINE_EMBEDDING_CACHE_DIR`,
`PIPELINE_RETRIEVAL_MODE`, `PIPELINE_HYBRID_DENSE_WEIGHT`,
`PIPELINE_HYBRID_BM25_WEIGHT`,
`PIPELINE_PDF_MAX_PAGES`, `PIPELINE_PDF_RENDER_DPI`, and
`PIPELINE_EXTRACTION_CONFIDENCE_THRESHOLD`. The optional reranker is off
by default. Enable it with `PIPELINE_RERANK_PROVIDER=cohere` plus either
`COHERE_API_KEY` or `CO_API_KEY`. The Cohere model defaults to
`rerank-v4.0-pro` and can be overridden with `COHERE_RERANK_MODEL` or
`PIPELINE_RERANK_MODEL`; the call is made over direct HTTPS, not through
a Cohere SDK dependency.

## Run

**TL;DR — just open it and drag a PDF in:**

```bash
uv run pipeline-api
# open http://127.0.0.1:8000/ui — drag a PDF from quick-eval/ into "Upload documents"
```

**CLI-only smoke (no UI):**

```bash
uv run pipeline run \
  --input quick-eval \
  --output outputs/quick_eval \
  --case-id quick-eval \
  --task "Review the supplied documents and produce a first-pass memo." \
  --force
open outputs/quick_eval/draft.md
```

**Docker:**

```bash
docker compose -f docker-compose.yml -f docker-compose.secrets.yml up --build
# open http://localhost:8000/ui
```

**Full curated 20-doc corpus run** (~3–4 min, ~$3–5):

```bash
uv run pipeline run \
  --input  datasets \
  --output outputs/all_categories \
  --case-id all-categories \
  --task   "Review the supplied documents and produce a first-pass internal memo covering parties, obligations, deadlines, signatures, and any data-quality flags a human operator/reviewer should verify before relying on the draft." \
  --force
```

The deliberate OpenAI drafting/extraction model choice for this build is
`gpt-5.5` at `low` reasoning effort. Both are the configured defaults
(see [`.env.example`](.env.example)), so no env override is needed for
the canonical invocation above. The reranker remains disabled unless
`PIPELINE_RERANK_PROVIDER=cohere` and a Cohere API key are set.

Or run against a single category to exercise one code path in
isolation:

```bash
uv run pipeline run \
  --input  datasets/handwritten-notes \
  --output outputs/handwritten_notes \
  --case-id handwritten-notes \
  --task   "Summarize the obligations and parties named in these handwritten records."
```

Artifacts: `processed_documents.json`, `corpus/manifest.json`,
`retrieved_evidence.json`, `index/retrieval_index.json`, `draft.json`,
`draft.md`, `risk_report.json`, `risk_report.md`, `audit.jsonl`,
`case_run.json`, `workflow_manifest.json`. The operator UI surfaces the
draft, source documents, citations, grounding check, and edit loop; the
CLI/API expose the full artifact shelf for developer and reviewer checks
(see §HTTP API).

## Operator UI walkthrough

After `uv run pipeline-api`:

1. Open `http://127.0.0.1:8000/ui`. The dashboard is a review queue for
   real cases; sample and evaluation outputs are hidden from the
   operator view.
2. Click **Upload documents** to start a review. The form asks for files,
   a case id, and a drafting task, then submits synchronously; expect a
   wait while OpenAI extracts, retrieves, and drafts.
3. Open a case from the review queue. The **Draft** tab parses every
   `[evidence_id]` citation token out of the draft, renders it as a
   clickable chip, and pairs it with an Evidence panel that shows the
   quote, page, and source text. Other operator tabs show source
   documents, citations, and claim-level grounding checks.
4. Click **Edit & teach** to capture an operator edit. The page POSTs
   to `/runs/{caseId}/edits`; the response gives a plain-English note
   about how future drafts will use the edit. Raw learning state,
   audit logs, risk reports, and evaluation artifacts remain available
   through CLI/API surfaces, not the operator UI.

The same loop is exercised by the CLI command `pipeline learn` and by the
`tests/test_api_smoke.py` API smoke test under `tests/`.

- `--resume` reuses checkpoints when the run fingerprint
  (input file hashes, task, enabled state channels, providers, feature
  switches, reasoning effort, PDF preprocessing knobs) matches the prior
  manifest.
- `--force` invalidates workflow checkpoints regardless. It does not
  delete the optional embedding cache; remove `embedding_cache/` or set
  a different `PIPELINE_EMBEDDING_CACHE_DIR` to force fresh embeddings.

### Pipeline feature controls

The run pipeline is split into explicit components:
`DocumentProcessingComponent`, `EvidenceRetrievalComponent`,
`LearningGuidanceComponent`, and `MemoDraftingComponent`. `PipelineFeatures`
lets you remove optional behavior or force a core stage to load an existing
checkpoint.

Core stage disables are checkpoint mode:

```bash
uv run pipeline run \
  --input  ignored-when-processing-disabled \
  --output outputs/checkpointed \
  --disable-processing \
  --disable-retrieval
```

That run expects `outputs/checkpointed/processed_documents.json` and
`outputs/checkpointed/retrieved_evidence.json` to already exist. Missing
checkpoints fail loudly.

Optional feature flags can be removed without checkpoints:

```bash
uv run pipeline run \
  --input datasets \
  --output outputs/no_learning \
  --disable-learning-guidance \
  --disable-retrieval-feedback \
  --disable-field-chunks \
  --disable-claim-support-check \
  --disable-playbook-risk
```

Equivalent env switches include `PIPELINE_LEARNING_GUIDANCE=off`,
`PIPELINE_RETRIEVAL_FEEDBACK=off`, `PIPELINE_FIELD_CHUNKS=off`,
`PIPELINE_CLAIM_SUPPORT_CHECK=off`, `PIPELINE_PLAYBOOK_RISK=off`,
`PIPELINE_MAX_FIELD_CHUNKS=0`, `PIPELINE_MAX_CHUNKS_PER_DOCUMENT=3`, and
`PIPELINE_FIELD_CHUNK_SCORE_PENALTY=0.2`. The HTTP `/runs` endpoint
accepts the same controls in a nested `features` object.

`PipelineFeatures` (defined in `src/pipeline/config.py`) carries the full
list of feature switches. Beyond the catch-all `--disable-learning-guidance`
shown above, there are granular toggles `--disable-operator-profile-guidance`,
`--disable-knowledge-layer-guidance`, `--disable-exemplar-guidance`, and a
core-stage `--disable-drafting` (checkpoint mode like processing/retrieval).
Optional behaviors with defaults: `claim_first_drafting=True` (claim-graph
output for case-fact summaries), `claim_support_check=True`,
`claim_entailment_judge=False` (LLM grounding judge — off by default),
`evidence_pack=True`, `field_chunks=True`, `field_chunk_score_penalty=0.06`.
Every flag has both a CLI form (`--disable-<name>` or `--<knob> <value>`)
and a `PIPELINE_<NAME>` env var; see `src/pipeline/config.py` lines
154–200 for the exhaustive list and defaults.

The retrieval layer enforces two diversity caps: `max_field_chunks=1`
keeps the synthetic structured-field chunk from crowding raw page
evidence, and `max_chunks_per_document=3` keeps one long document from
filling the entire top-k with tightly-similar prose. Both caps fill
remaining top-k slots from overflow so the drafter always sees the
configured `RETRIEVAL_TOP_K` chunks.

### Corpus, index, edit memory, and harness

The CLI and API expose separate resource surfaces for the durable pieces
of the pipeline: a corpus store, a queryable hybrid retrieval index, and
operator edit memory.

```bash
uv run pipeline corpus build \
  --input datasets \
  --output outputs/all_categories/corpus \
  --processed outputs/all_categories/processed_documents.json \
  --json

uv run pipeline corpus settings \
  --input datasets \
  --output outputs/all_categories \
  --json

uv run pipeline index build \
  --corpus outputs/all_categories/corpus \
  --output outputs/all_categories/index/retrieval_index.json \
  --json

uv run pipeline index query \
  --index outputs/all_categories/index/retrieval_index.json \
  --task "Find deadline and signature evidence." \
  --json

uv run pipeline index settings \
  --output outputs/all_categories \
  --state-dir state \
  --json

uv run pipeline edit-memory query \
  --state-dir state \
  --task "Draft a concise operator-review memo." \
  --json

uv run pipeline edit-memory settings \
  --state-dir state \
  --category handwritten-notes \
  --json

uv run pipeline risk \
  --run outputs/all_categories \
  --playbook playbooks/default_legal_playbook.json \
  --output outputs/all_categories/risk_report.json \
  --markdown-output outputs/all_categories/risk_report.md

uv run pipeline harness run \
  --manifest eval/harness_cases.json \
  --output outputs/harness \
  --json
```

`pipeline run` also writes the corpus and retrieval index automatically
under the run output. The standalone commands let you rebuild or query
those resources independently.

Current implementation:

- **Corpus** means originals plus parsed `ProcessedDocument` records under
  `corpus/`, with `manifest.json` as the catalog.
- **Index** means a persisted JSON `RetrievalIndex` containing chunks,
  embeddings, BM25 state, and optional FAISS acceleration rebuilt on load.
- **Edit memory** means `state/edit_memory.json` plus the typed learning
  state under `state/`: `operator_profile.json`, `knowledge_layer.json`,
  `retrieval_feedback.json`, `edits.jsonl`, `learn_suggestions.json`, and
  `exemplars/*.jsonl`.
- **Playbook risk** means deterministic screening against a JSON legal-team
  playbook under `playbooks/`, producing `risk_report.json` and
  `risk_report.md` for operator review.
- **Harness** means manifest-driven deterministic gates around real
  pipeline runs.

The matching HTTP endpoints are `POST /corpus/settings`,
`POST /corpus/build`, `POST /index/settings`, `POST /index/build`,
`POST /index/query`, `POST /edit-memory/settings`,
`POST /edit-memory/query`, and `POST /risk`.

### Operator console (React + Vite)

The operator console is a real SPA — React 18, TypeScript, TanStack
Query v5, react-router — that lives under `frontend/`. Its build emits
into `src/pipeline/ui/static/dist/`, and the FastAPI app serves that
bundle directly under `/ui` with a SPA-friendly fallback for nested
client routes (`/ui/runs/:caseId`, `/ui/runs/new`, etc.). Data fetching
runs through TanStack Query (`useQuery` / `useMutation`); the only
cross-page UI state — the selected-evidence chip — sits in a small
React Context backed by `useState`. The earlier Redux Toolkit / RTK
Query stack was dropped because the app only needs request caching and
one piece of UI state, so the migration trims roughly 100 kB of bundle
without changing behavior.

```bash
# Build the SPA once, then run the API. /ui will serve the React shell.
cd frontend && npm install && npm run build && cd ..
uv run pipeline-api
open http://127.0.0.1:8000/ui
```

For iterative frontend work, run Vite alongside the API:

```bash
# terminal 1 — backend
uv run pipeline-api

# terminal 2 — frontend dev server
cd frontend
npm run dev
# Vite proxies API calls to http://127.0.0.1:8000
# CORS is configured for http://127.0.0.1:5173 by default.
open http://127.0.0.1:5173/
```

`PIPELINE_API_CORS_ORIGINS` is a comma-separated allowlist (default
`http://127.0.0.1:5173,http://localhost:5173`). Setting it to the empty
string disables the CORS middleware entirely (single-origin production
posture).

#### Pages

- **Dashboard** (`/`) — runs the API's `GET /runs` against
  `./outputs/` and lists real case reviews, hiding sample/evaluation
  fixtures.
- **Upload documents** (`/runs/new`) — uploads PDFs/images/text files,
  posts to `POST /runs`, and redirects to the case detail page when the
  draft is ready.
- **Case detail** (`/runs/:caseId`) — tabs over Draft (with clickable
  citation chips bound to an Evidence panel), Source documents,
  Citations, and Grounding (`case_summary.claims` with
  substring/entailment verdicts).
- **Edit & teach** (`/runs/:caseId/edit`) — side-by-side draft viewer
  plus a markdown textarea. Submitting POSTs to
  `POST /runs/{caseId}/edits`, which writes `edited.md` next to the
  draft and runs `learn_from_files` — typed intent routing decides
  which of `operator_profile.json`, `knowledge_layer.json`,
  `retrieval_feedback.json`, the per-category exemplar store, and the
  edit memory get updated. The response summarizes the learning result
  in human language instead of exposing internal JSON.

Developer and reviewer diagnostics such as retrieval-index query,
learning-state inspection, evaluation reports, risk reports, artifact
downloads, and audit logs stay available through CLI/API endpoints. They
are intentionally not surfaced in the operator UI.

Security note: the API and UI remain a trusted local/operator surface.
They accept filesystem paths for runs, corpora, indexes, and state
directories, so do not expose them directly to untrusted networks. By
default API paths are limited to the process working tree and temp
directory; set `PIPELINE_API_ALLOWED_ROOTS` to an `os.pathsep`-separated
allowlist for deployment-specific roots. Put authentication, tenant
isolation, rate limits, and reverse-proxy controls in front before
multi-user deployment. CORS is dev-convenience only and `allow_credentials`
is disabled.

### Docker

`Dockerfile` is multi-stage:

1. `node:22-bookworm-slim` stage runs `npm ci && npm run build` to emit
   the React bundle.
2. `python:3.14.5-slim` stage installs the locked Python deps and copies
   the built bundle into `/app/src/pipeline/ui/static/dist`.

Docker users do **not** need a local Node toolchain — the build happens
inside the image.

Recommended runtime-secret flow:

```bash
cp .env.example .env
mkdir -p secrets
printf 'OpenAI API key: '
read -r -s OPENAI_API_KEY
printf '\n'
printf '%s' "$OPENAI_API_KEY" > secrets/openai_api_key
unset OPENAI_API_KEY
chmod 600 secrets/openai_api_key

docker compose -f docker-compose.yml -f docker-compose.secrets.yml up --build
```

If you enable Cohere reranking and want to load that key from a secret
file too, also write `secrets/cohere_api_key` and add
`-f docker-compose.cohere-secrets.yml` to the compose command.

Leave `OPENAI_API_KEY=` blank in `.env` when using the secret-file override.
`.dockerignore` excludes `.env`, `secrets/`, `outputs/`, and `state/`, so
the OpenAI key and local run artifacts are not copied into the Docker build
context or baked into the image. `docker-compose.secrets.yml` mounts the key
only at runtime as `/run/secrets/openai_api_key`; the entrypoint exports it
for the Python process and does not print it. The optional Cohere secret
overlay follows the same pattern for `/run/secrets/cohere_api_key`.
The base compose file forwards only an explicit allowlist of runtime
settings, and the secret override forces `OPENAI_API_KEY` blank so a host
environment variable is not copied into the container config.

For quick local-only Docker testing, you may put `OPENAI_API_KEY=...` in
`.env` and run `docker compose up --build`, but that places the key in the
container runtime environment. Prefer the secret-file override for sharing,
review, or demos.

`docker-compose.yml` exposes API + UI on `http://localhost:8000`,
mounts `datasets/` read-only, and persists `outputs/` plus `state/` on
the host.

### Cached retrieval backend

Set `PIPELINE_RETRIEVAL_PROVIDER=openai-cached` to reuse embeddings across
runs without changing the downstream evidence contract:

```bash
PIPELINE_RETRIEVAL_PROVIDER=openai-cached \
uv run pipeline run \
  --input datasets \
  --output outputs/all_categories \
  --case-id all-categories \
  --task "Review the supplied documents and produce a first-pass internal memo." \
  --force
```

When run through the CLI pipeline, the cache defaults to
`<output>/embedding_cache`; set `PIPELINE_EMBEDDING_CACHE_DIR` to share a
cache across run directories. Cache entries are keyed by the exact
chunk/query string sent to the embedding provider, the embedding model,
and the embedding backend identity, then validated with cache metadata
so changing `OPENAI_EMBEDDING_MODEL` or test embedding clients does not
silently reuse incompatible vectors. Normalized `chunk_hash` values stay
as stable evidence metadata, not embedding-cache keys; normalization
collapses whitespace, case, common quote/dash variants, and ligatures.
The cache stores hashes and compact base64-encoded vectors, not source
text. Retrieval still returns normal `EvidenceChunk` objects, and
drafting still enforces citation ids and verbatim quote grounding.
Turning cached retrieval on after a prior `openai` run changes the
fingerprint and re-runs the workflow. The cache is additive; delete the
cache directory or point `PIPELINE_EMBEDDING_CACHE_DIR` somewhere else to
reset it.

## PDF preprocessing

`ingestion/pdf.py` opens each PDF with PyMuPDF and routes per page:

- **Text-layer page** (≥60 chars): extract text locally, send as
  `--- PAGE N (extracted from text layer) ---` block in the extraction
  prompt. No image bytes sent to the API.
- **Image-only or near-empty page**: render to PNG at 180 DPI, send as
  multimodal `input_image` to the Responses API.

This keeps token cost down for clean digital PDFs (e.g. SEC EDGAR
contracts, SCOTUS opinions) while still handling true scans (the
synthetic noisy lien notice). Local-text pages get `confidence=0.95` and
`extraction_method="pdf-text-layer"`; rendered pages get the model's
returned confidence and `"pdf-rendered-png"`.

## Verbatim grounding contract

For every cited evidence id in a supported section, the drafter must
emit a verbatim `citation_quotes` entry. A validator
(`drafting._validate_quote_grounding`) checks each quote is a
normalized substring of the cited chunk's text (whitespace folded;
curly quotes/em-dashes/ligatures normalized; case-insensitive). Failed
quotes are stripped from the section's `evidence_ids` and from the
inline `[evidence_id]` body token, and sections that lose all citations
are marked `unsupported`. Each strip is recorded as a draft warning.

## Dataset

The dataset folder is for source documents. It is organized **by the five
spec characteristics**, with 3 representative documents per messy category,
all sliced to ≤6 pages and ≤1.3 MB, plus a clean true-PDF control set:

```
datasets/
├── scanned-pages/                    3 docs  — scanned/fax-style sources, mixed text layers
├── low-resolution-pdfs/              3 docs  — sub-200 DPI sources
├── handwritten-notes/                3 docs  — cursive primary content
├── partially-illegible-records/      3 docs  — visible text-layer mojibake
├── inconsistently-formatted-files/   3 docs  — heterogeneous layouts
└── truepdf/                          5 docs  — clean text-layer court forms
```

Sources are public-domain or CC-licensed: Internet Archive (Lee Co VA
chancery 1863, Marion Co IN probate 1833, Smithsonian Macdougall field
notes 1899, Worcester MA probate), SEC EDGAR VPRR paper filings, UCSF
Industry Documents Library fax scans, LoC Chronicling America
microfilm, CUAD (CC BY 4.0), SCOTUS slip opinion, and one synthetic
noisy lien notice with overlay handwriting.

See [`datasets/README.md`](datasets/README.md) and
[`datasets/COVERAGE.md`](datasets/COVERAGE.md) for the full inventory,
per-document DPI/text-layer measurements, and the cross-category
mapping (most docs exercise more than one characteristic — the folder
placement reflects each doc's *primary* contribution).

Evaluation assets live under `eval/`, not `datasets/`. That includes
`eval/operator-edits/`, the gold labels, harness manifest, eval points, and
the public-eval bundle in `eval/public-documents/` pinned by
`eval/public_eval_manifest.json`.

## Learn from human operator edits

Operator edits are typed events with explicit envelope routing. Five
substantive intents plus an `auto` resolver:

| Intent | Envelope | Channel | What it persists |
|---|---|---|---|
| `preference` | in | `state/operator_profile.json` + `state/exemplars/<category>.jsonl` | section labels, preferred phrases, caution-marker deltas, per-category exemplars |
| `field_correction` | in | `state/knowledge_layer.json` (reusable labels) or advisory | jurisdiction-class field defaults; case-specific corrections stay advisory |
| `knowledge_rule` | out | `state/knowledge_layer.json` | extrinsic hard constraints from the operator note |
| `evidence_dispute` | out | `state/retrieval_feedback.json` + `learn_suggestions.json` | retrieval demotes + candidate gold queries |
| `rejection` | audit | `edits.jsonl` only | wholesale rejection — too coarse to auto-learn |

In-envelope edits change the drafter persona; out-of-envelope edits change
facts/retrieval, not persona; rejections are audit-only. `edits.jsonl` is
appended for every captured event regardless of channel — it is the
source-of-truth event log.

```bash
uv run pipeline learn \
  --draft   outputs/all_categories/draft.md \
  --edited  eval/operator-edits/handwritten_records_edit.md \
  --state-dir state \
  --category handwritten-notes \
  --intent  preference \
  --note    "Use per-document headers; require page-level citations for every deadline and dollar amount." \
  --actor-id paralegal-1
```

`--state-dir state` is the cross-case learning home (recommended default).
The per-case `--profile <path>` invocation is available for isolation.
`--category` is only consulted for preference edits; it keys the exemplar
store so a handwritten-notes edit doesn't bleed into a scanned-pages draft.

Three operator edits ship in `eval/operator-edits/`, each targeting a
different preference axis. Capturing all three plus a knowledge rule
populates `state/`:

| File | Category | Preference axis |
|---|---|---|
| `messy_case_edited_memo.md` | `all-categories` | terse, cautious-tone restructure |
| `handwritten_records_edit.md` | `handwritten-notes` | per-document headers, page-level citations |
| `scanned_filings_edit.md` | `scanned-pages` | full-prose compliance-review tone |
| (knowledge rule via `--intent knowledge_rule`) | n/a | example out-envelope hard constraint |

The captured signals (added/removed section labels, preferred phrases,
caution-marker deltas, Markdown `Label: value` field corrections) are
extracted by `analyze_edit` and routed through `_routing_for` based on the
resolved intent. The drafter sees a combined guidance string composed of
the profile, knowledge layer, and one matching per-category exemplar.

Rerun the same case with state loaded to apply learned guidance:

```bash
uv run pipeline run \
  --input  datasets \
  --output outputs/all_categories_profiled \
  --case-id all-categories-profiled \
  --task   "Review the supplied documents and produce a first-pass internal memo covering parties, obligations, deadlines, signatures, and any data-quality flags a human operator/reviewer should verify before relying on the draft." \
  --state-dir state \
  --force
```

The guidance may steer structure, tone, reusable phrasing, and per-category
shape. It must not relax citation ids, verbatim quote checks,
unsupported-fact routing, or claim-citation reporting — those are enforced
at the drafter boundary (citation-id sanitation, quote-substring
validation, uncited-sentence detection) regardless of what guidance is
loaded.

Export captured edits as DPO-ready section-level preference pairs (for
later TRL-based training; the schema is intentionally `{prompt, chosen,
rejected, metadata}`):

```bash
uv run pipeline export-dpo \
  --edits-log state/edits.jsonl \
  --output    state/dpo_pairs.jsonl
```

Section-level pairs require overlapping `##` headings between original and
edited drafts. Restructure-style edits fall back to a whole-draft pair
that is still citation-preservation-filtered; pairs where the operator
dropped *all* inline citations are excluded because rewarding them would
teach the model to drop grounding. This emits data, not a trained model:
`trl.DPOTrainer` consumes the JSONL directly when you want to graduate
from prompt-time guidance to weight updates.

"Future results" means future pipeline artifacts: later `draft.md` /
`draft.json` files, their cited evidence choices, and their evaluation
scores. It does not mean legal outcomes or autonomous legal decisions.
Preference edits should make future drafts closer to the reviewer-preferred
structure/tone; grounding edits should improve future gold labels,
retrieval/extraction fixes, or case-level annotations.

## Evaluate

Single-draft static evaluation:

```bash
uv run pipeline evaluate --run outputs/all_categories
```

Reports:

- `citation_id_validity` — share of *supported* sections whose declared
  `evidence_ids` are subsets of the retrieved set. Unsupported review
  sections are excluded from the denominator (so a clean draft scores
  100% rather than artificially diluted).
- `sentence_grounding_score` — quote-substring validity: fraction of
  `(evidence_id, quote)` pairs
  where every cited evidence id in supported sections has a verbatim
  quote that is a normalized substring of the cited chunk's text.
  Returns `not available` only when there are no cited supported
  sections to score. This is not semantic entailment.

Expanded deterministic suite:

```bash
uv run pipeline eval-suite \
  --run outputs/all_categories \
  --gold-dir eval \
  --output outputs/all_categories/evaluation.md \
  --json-output outputs/all_categories/evaluation.json
```

The suite reports category coverage, page-vs-field term recall, retrieval
recall/precision/MRR, claim citation rate, and field-chunk vs raw-page citation
share. Public eval-point runs add grouped checks for document understanding,
grounded drafting, and improvement from edits. Current curated-run highlights:

| Metric | Result |
|---|---:|
| Documents / pages | 20 / 70 |
| Rendered pages / text-layer pages | 25 / 45 |
| Citation id validity | 100.00% |
| Quote-substring validity | 100.00% |
| Claim grounding pass rate | 100.00% |
| Claim citation rate | 100.00% |
| Cited field-chunk share | 18.75% |
| Cited raw-page share | 81.25% |
| Page-text term recall | 80.00% |
| Field-value term recall | 90.00% |
| Combined term recall | 95.00% |
| Retrieval recall@k / precision@k | 33.33% / 37.50% |
| Distinct cited files | 5 / 20 |

The run was generated with the in-repo `max_field_chunks=1` and
`max_chunks_per_document=3` caps as defaults, so the curated headline
grounds 81.25% of cited evidence to raw page chunks rather than the
synthetic structured-field chunks. Citation ids, verbatim quotes, and
claim-graph grounding all pass at 100% on this run. Retrieval
recall/precision/MRR are scored against the persisted production
`retrieved_evidence.json` — the same hybrid (BM25 +
`text-embedding-3-large`) result the drafter actually saw, with no
deterministic lexical fallback. 33.33% recall@k reflects the single
multi-document gold query in `eval/gold_retrieval.json` against the
diversity-capped top-k; the gold-doc filename hit-rate (Documents
checked / Missing gold documents) stays at 5/5 in the run report. The
single deliberate stack is `gpt-5.5` for generation/extraction and
`text-embedding-3-large` for retrieval, with the optional Cohere
reranker disabled; the gold gate scores that stack directly. Numbers
above are honest snapshots of this exact run
(`outputs/all_categories/evaluation.md`) and will drift naturally with
each fresh `pipeline run` — the contract metrics (citation id, quote
substring, claim citation) are pinned at 100% by validators; the
share/recall metrics are stochastic.

Public-data smoke:

```bash
uv run pipeline public-data download \
  --manifest eval/public_eval_manifest.json \
  --output eval/public-documents

uv run pipeline public-data verify \
  --manifest eval/public_eval_manifest.json \
  --dataset eval/public-documents
```

`eval/public-documents/` contains a 13-file public bundle: ~2-3 files
per messy category, all verified at five pages or fewer, every file
SHA-256-pinned in `eval/public_eval_manifest.json`. Two files were
removed when the curated `datasets/` and public-eval bundles were made
strictly distinct (no SHA-256 overlap). To download, verify, run, and
evaluate the public bundle:

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

The previous public-eval artifact under `outputs/public_eval/` was
generated against the prior 15-doc bundle and is left in place as a
historical sample. To regenerate it against the current 13-doc bundle,
run the two commands above; the deterministic eval-point gates remain
applicable since the one impacted point (`sec-illegible-commission-file-term`,
covering a SEC VPRR scan that moved out of the public bundle) is
`required=false`.

A/B improvement-loop evaluation:

```bash
uv run pipeline ab-eval \
  --case-dir outputs/all_categories \
  --profile  state/operator_profile.json \
  --state-dir state \
  --edited-reference eval/operator-edits/messy_case_edited_memo.md \
  --draft-output-dir outputs/all_categories/ab_drafts
```

Drafts the same case twice on the same retrieved evidence (without
profile vs. with profile) and reports section-label adoption delta,
suppression delta, caution-marker delta, preferred-phrase adoption, per-side
grounding metrics, claim citation rate, raw-page citation share, and optional
edited-reference body/section similarity.

With `--state-dir state`, the run uses the full state bundle: operator
profile guidance (prose/structure/tone), knowledge-layer guidance (hard
constraints), the per-category exemplar matching the dominant
retrieved-document category, and retrieval feedback boosts/demotes during
evidence selection. Retrieval feedback is not prompt guidance. Current
multi-edit run (3 preference edits across categories + 1 knowledge rule +
1 evidence dispute populate `state/`):

| Metric | Result |
|---|---:|
| Section-label adoption delta | +1 |
| Section-label suppression delta | 0 |
| Caution-marker delta (body, signed) | +3 |
| Preferred-phrase adoption | 0.00% of top 5 (verbatim) |
| Citation id validity without / with | 100.00% / 100.00% |
| Quote-substring validity without / with | 100.00% / 100.00% |
| Claim citation rate without / with | 100.00% / 100.00% |
| Raw-page citation share without / with | 75.00% / 80.00% |
| Body token similarity without / with | 10.63% / 10.33% |
| Section F1 vs messy_case reference, without / with | 0.00% / 0.00% |

Notes on the multi-edit numbers (honest snapshot of this run; metrics
will drift between runs because GPT-5.5 generation is stochastic):

- **Preferred-phrase adoption is 0% on this run.** Section-label
  adoption (+1) and caution-marker shift (+3 signed) are the honest
  non-zero structural/tone signals. An earlier run on the same
  inputs reported 20% verbatim phrase adoption; verbatim phrase
  reuse is the noisiest signal of the three and routinely flips
  between 0% and 20%+ across runs. The `_preferred_phrases`
  citation-token hygiene fix and the `max_chunks_per_document=3`
  diversity cap are still in — they raise the *ceiling* for verbatim
  reuse but don't guarantee a non-zero hit on any single run.
- Section-label adoption (+1) is a modest but real signal: the
  without-profile draft already produces several of the structurally
  relevant section labels on its own, so the gap the profile closes is
  narrow rather than wide. The profile's job is to nudge structure, not
  to fight a weaker baseline.
- Raw-page citation share is 75.00% / 80.00% (without / with profile).
  Both drafts ground predominantly to raw page chunks rather than
  synthetic `:fields` chunks; the with-profile delta is consistent
  with the profile's source-text-citation preference, but a single
  stochastic A/B over the same retrieved evidence does not prove
  causation.
- Citation grounding stays at 100% across all three layers (id validity,
  quote-substring validity, claim citation rate). Sentence-splitter
  agreement between drafting and evaluation is via `drafting.sentence_units`.
- Section F1 against the messy_case_edited_memo reference is 0% / 0%
  on this run; on a prior run it landed at 0% / 13.33%. The reference
  uses very specific section labels and the without-profile baseline
  almost never hits them; the with-profile draft hits them often
  enough to register a non-zero F1 some of the time. Absolute F1 stays
  low by design because the multi-edit profile is intentionally not
  overfit to any single reference.

`outputs/all_categories/ab_drafts/` contains the actual without-profile
and with-profile draft JSON/markdown so reviewers can inspect the
future-draft artifact directly.

## Sample Inputs And Outputs

Open these first when reviewing:

- `datasets/COVERAGE.md` — per-document messy-input matrix.
- `eval/public_eval_manifest.json` — public download manifest with hashes.
- `outputs/all_categories/processed_documents.json` — extracted text, fields,
  confidence, and warnings for the curated 20-doc run.
- `outputs/all_categories/retrieved_evidence.json` — chunks available to the
  drafter.
- `outputs/all_categories/draft.md` — generated first-pass memo.
- `outputs/all_categories/edits.jsonl` — append-only operator edit event log.
- `outputs/all_categories/learn_suggestions.json` — suggested retrieval/gold
  annotation from a captured evidence-dispute edit.
- `outputs/all_categories/evaluation.md` — deterministic evaluator report.
- `outputs/all_categories/risk_report.md` — deterministic playbook-risk report.
- `outputs/all_categories/ab_eval.md` — operator-edit A/B report.
- `outputs/all_categories/ab_drafts/with_profile_draft.md` — inspectable
  future-draft artifact generated with learned profile guidance.
- `outputs/public_eval/evaluation.md` — public bundle evaluation report.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Module map

### Backend (`src/pipeline/`)

- `api.py` / `cli.py` — external entrypoints; thin adapters over the stage packages.
- `config.py` — provider/model config, `PipelineFeatures`, and
  `resolve_provider_name`.
- `providers.py` — OpenAI Responses wrapper, embeddings, and JSON parsing.
- `schemas.py` — shared dataclasses crossing stage boundaries.
- `io.py` — JSON read/write helpers.
- `ingestion/` — PDF routing, document intake, extraction prompt, field
  normalization, and the ingestion stage adapter.
- `corpus/` — durable corpus catalog with originals and parsed document JSON.
- `retrieval/` — chunking, fields-chunk emission, cached OpenAI embeddings,
  BM25 + dense hybrid ranking, metadata/retrieval-feedback boosts, optional
  FAISS acceleration, optional Cohere reranking, and the retrieval stage adapter.
- `drafting/` — memo generation, citation sanitation, verbatim quote
  validation, markdown rendering, and the drafting stage adapter.
- `learning/` — edit capture, profile/knowledge/retrieval-feedback state,
  edit memory, exemplar guidance, and DPO pair export.
- `playbooks/` — JSON legal-team playbook loading, deterministic risk
  scoring, and risk-report rendering.
- `evaluation/` — static eval, expanded eval suite, A/B edit-loop eval, and
  deterministic harness orchestration.
- `public_data/` — public evaluation manifest loader/downloader/verifier.
- `orchestration/` — component registry, artifact paths, run fingerprint,
  audit log, resume/`--force`, retries, workflow manifest persistence, and
  `run_case`.
- `ui/static/dist/` — built React bundle served at `/ui` (regenerated by
  `cd frontend && npm run build`).

### Frontend (`frontend/src/`)

- `main.tsx` / `App.tsx` — router shell, lazy-loaded pages, TanStack
  Query `QueryClientProvider`, and the `SelectedEvidenceProvider`
  context.
- `lib/api.ts` — typed `fetch` wrappers and TanStack Query
  `useQuery` / `useMutation` hooks for the backend routes used by the
  operator UI (run list / summary, draft/source/evidence artifacts,
  uploads, run creation, and edit submission).
- `lib/evidence-selection.tsx` — small React Context backed by
  `useState` that holds the selected-evidence chip; the only piece of
  shared UI state in the app.
- `lib/citations.ts` — regex-based tokenizer that splits draft body text
  into `text` and `citation` spans, plus a helper that distinguishes
  `:fields` derived chunks from raw-page chunks.
- `pages/` — `DashboardPage`, `NewRunPage`, `RunDetailPage` (tabbed),
  `RunEditPage` (operator edit + learning loop), `NotFoundPage`.
- `components/` — `Layout`, `HealthBadge`, `DraftRenderer` (the citation
  chip + grounded section renderer), `EvidencePanel`.
- `types/api.ts` — hand-written TypeScript projections of the Python
  dataclasses crossing the wire (`Draft`, `EvidenceChunk`,
  `ProcessedDocument`, `EditSubmitResponse`, …).
- `styles/` — design tokens + component styles. No CSS framework
  dependency.

## Assumptions and tradeoffs

- OpenAI core providers. Runtime provider settings accept `openai` for
  extraction and generation, and `openai` or `openai-cached` for
  retrieval. Optional reranking is Cohere-only when enabled with
  `PIPELINE_RERANK_PROVIDER=cohere`; test-only fixtures stay isolated
  from production configuration.
- Internal first-pass drafting aid, not a legal-advice system.
- File ingestion allowlist: `.pdf`, `.png`, `.jpg`/`.jpeg`, `.txt`.
  Other files in the input dir are skipped with a warning so docs,
  operator-edit markdown, and helper scripts don't get sent through
  the document extraction pipeline.
- Chunking is fixed-size word windows for page text (default 180 words,
  40 overlap). After page-text chunks, one synthetic "fields chunk" per
  document carries the extracted structured fields so the field schema
  is retrievable in addition to surfacing through page text.
- The optional reranker is off by default (`PIPELINE_RERANK_PROVIDER`).
  Enable it with `PIPELINE_RERANK_PROVIDER=cohere` and either
  `COHERE_API_KEY` or `CO_API_KEY`. It defaults to Cohere
  `rerank-v4.0-pro`, can be overridden with `COHERE_RERANK_MODEL` or
  `PIPELINE_RERANK_MODEL`, and uses direct HTTPS rather than a Cohere
  SDK dependency.
- Grounding is verbatim *substring* match, not semantic entailment. The
  quote being verbatim doesn't prove the surrounding sentence is
  factually entailed by the chunk.
- The improvement loop measures *human preference steering* (section structure,
  caution markers, preferred phrases) plus edited-reference similarity and
  grounding non-regression checks. Moving in the operator's direction is not
  the same as proving a better draft; the current single-case A/B is mixed and
  would need held-out aggregation for a stronger claim.
- Grounding/factuality feedback is a separate loop: deterministic validators
  and human-approved annotations should update eval gold, retrieval/extraction
  fixes, or case-level records. AI critique is advisory unless a human accepts
  it.
- Playbook risk scoring is deterministic term-based screening for operator
  triage. Replace `playbooks/default_legal_playbook.json` with
  organization-specific positions before relying on it for real review.
  It is not a semantic entailment model and does not resolve negation or
  nuanced legal interpretation.
- Retrieval indexes are persisted as portable JSON under `index/`. The
  file includes chunks, embeddings, and BM25 state; FAISS is an optional
  acceleration layer rebuilt from the stored vectors when available. The
  optional `openai-cached` backend persists embedding vectors behind
  exact-input/model/backend cache keys to reduce repeat OpenAI embedding
  calls.
- **Language coverage is partial.** The data path runs on any language
  (PyMuPDF text extraction, GPT-5.5 multimodal OCR, and
  `text-embedding-3-large` are all multilingual, and the verbatim
  quote-substring grounding contract is character-level), so a French or
  Japanese PDF will still produce a grounded, cited draft. But several
  features are English-only and silently degrade on non-English content:
  - `CAUTION_MARKERS` in `src/pipeline/learning/state.py:88` is a
    hardcoded English word list (`"appears", "may", "might", …`); the
    caution-tone A/B delta scores 0 on a non-English draft.
  - The BM25 tokenizer in `src/pipeline/retrieval/engine.py` is
    whitespace-based — fine for Latin scripts, broken for CJK / Thai /
    Arabic. That's half of the hybrid retrieval score.
  - Extraction and drafting prompts are written in English and the
    structured field/section-label vocabulary is English-keyed; the
    model usually mirrors the source language in the body but the schema
    boundary stays English.
  - Preferred-phrase extraction from operator edits uses English
    regex/word-boundary heuristics.
  - Quote-normalization (curly quotes / em-dash / ligatures) is
    Latin-script-centric — harmless for other scripts but doesn't help
    them.

  True multilingual support would need a script-aware tokenizer (e.g.
  ICU word-segmentation), localized caution-marker lists per language,
  and language-aware extraction prompts. None of that is in the build.
- PyMuPDF text-layer extraction has no image-extraction confidence; pages
  reporting ≥60 chars get high local provenance confidence (0.95).
  Documents with hidden garbled text layers can still bypass the image-based
  extraction path if local heuristics pass.

## HTTP API

Start the server:

```bash
uv run pipeline-api
# or: uv run uvicorn pipeline.api:app --host 127.0.0.1 --port 8000
```

The API is the trusted control plane consumed by CLI scripts, developer
checks, and the operator React console. The operator UI uses only a small
review-focused subset. The API accepts direct filesystem paths and should
be treated as trusted infrastructure. It enforces a default path allowlist
of the process working tree plus temp directory; configure
`PIPELINE_API_ALLOWED_ROOTS` for tighter deployments, and deploy behind
auth plus tenant isolation before exposing it to users. CORS is
dev-convenience only; configure `PIPELINE_API_CORS_ORIGINS`.

UI:
- `GET /` redirects to `/ui`.
- `GET /ui` serves the React shell (`dist/index.html`).
- `GET /ui/*` is the SPA fallback. Real static assets win; unknown paths
  return `index.html` so client-side routes resolve.
- `GET /favicon.ico` serves the small ICO under `src/pipeline/ui/static/`.

Pipeline / state surface:

- `POST /uploads/{case_id}` — operator-facing file upload used by the
  "Upload documents" UI page. Accepts multipart `files`, writes them
  under a temp directory, and returns the resulting `input_dir` that
  the subsequent `POST /runs` call should consume.
- `POST /runs` — kick off a pipeline run; accepts nested `features` and
  optional `state_dir`, `profile_path`, `playbook_path`, `resume`,
  `force`.
- `GET /runs?root=outputs` — list run directories under the given root
  with a compact summary (case id, task, section/warning counts, which
  artifacts exist).
- `GET /runs/{case_id}` — raw `case_run.json` for a run.
- `GET /runs/{case_id}/summary` — the same compact summary as the list
  endpoint, scoped to a single run.
- `GET /runs/{case_id}/artifacts` — enumerate every artifact the run
  could produce, with `exists` + `size` + `media_type`.
- `GET /runs/{case_id}/artifacts/{artifact_key}` — fetch one artifact.
  `artifact_key` is an enum (e.g. `draft_md`, `draft_json`,
  `retrieved_evidence`, `processed_documents`, `evaluation_json`,
  `risk_report_md`, `audit_log`, `edits_log`, …) — never a filename.
  JSON files come back as JSON; NDJSON files are parsed into a row array
  (use `?limit=N` to tail); markdown is returned as `text/markdown`.
  Append `?raw=true` to stream the file verbatim with its native
  Content-Type — useful for download links.
- `POST /runs/{case_id}/edits` — operator edit capture. Body:
  `{edited_text, note, intent, category, actor_id, state_dir, output_dir}`.
  The endpoint writes `edited.md` next to the draft, runs
  `learn_from_files`, and returns the updated profile, edit-memory delta,
  routing channel, latest captured event, and latest classification.
- `GET /state/{name}?state_dir=state` — read a typed-learning state file
  from `state_dir`. `name` is one of `operator_profile`,
  `knowledge_layer`, `retrieval_feedback`, `edit_memory`,
  `learn_suggestions`, `edits`, `edit_classifications`, `dpo_pairs`,
  `dpo_pairs_candidates`. JSONL files come back as parsed rows; pass
  `?limit=N` to tail.

Resource surfaces:

- `POST /corpus/settings`, `POST /corpus/build`
- `POST /index/settings`, `POST /index/build`, `POST /index/query`
- `POST /edit-memory/settings`, `POST /edit-memory/query`

Evaluation + learning + ops:

- `POST /learn` — capture an edit from two file paths (CLI-equivalent).
- `GET /evaluate?run_dir=…`
- `POST /eval-suite`
- `POST /risk`
- `POST /ab-eval` (optionally persists A/B drafts with `draft_output_dir`)
- `POST /harness`
- `POST /public-data/download`, `POST /public-data/verify`
- `POST /export-dpo`
- `GET /healthz`

The OpenAPI schema is published at `/openapi.json` and a Swagger UI at
`/docs`.

## Tests

```bash
uv run --extra dev python -m pytest -q
```

A small focused suite lives under `tests/` and runs without any live
OpenAI calls. It covers the verbatim-quote grounding validator, the
field-chunk cap in retrieval, the PDF text-layer / image routing in
ingestion, a FastAPI healthz + OpenAPI shape smoke check, and the
citation-id-validity denominator semantics. It is intentionally narrow:
it does not pretend to be full coverage, only enough to lock down the
properties this codebase actually relies on.

Frontend type-check (requires `npm install` first):

```bash
cd frontend && npm run typecheck
```

## Repository hygiene

Tracked artifacts are intentionally limited to source code, documentation,
sample datasets, manifests, and small inspectable output files. Runtime
environments and caches such as `.venv/`, `.pytest_cache/`, `__pycache__/`,
`embedding_cache/`, and compiled bytecode are ignored and should not be
committed.
