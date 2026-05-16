# Instructions

Two ways to run the pipeline: **without Docker** (local Python +
optional Node) or **with Docker** (single image, no host toolchain).
Pick one.

Prereq for both: an `OPENAI_API_KEY` with access to `gpt-5.5` and
`text-embedding-3-large`. Optional reranking additionally needs
`COHERE_API_KEY`, `CO_API_KEY`, or `COHERE_API_KEY_FILE` when
`PIPELINE_RERANK_PROVIDER=cohere` is enabled.

## Without Docker

Requires Python 3.14.5 (pinned in `.python-version`) and
[`uv`](https://docs.astral.sh/uv/). Node is only needed if you want to
rebuild the React bundle — the committed bundle under
`src/pipeline/ui/static/dist/` is what the API serves at `/ui`.

```bash
git clone git@github.com:ehzawad/legal-rag-pipeline.git
cd legal-rag-pipeline

uv sync --extra dev
export OPENAI_API_KEY=sk-...
export PIPELINE_INDEX_BACKEND=qdrant
export QDRANT_PATH=state/qdrant
export QDRANT_COLLECTION=legal_rag
```

`QDRANT_PATH` uses embedded qdrant-client local storage and is intended
for single-user local development. Docker Compose starts a real Qdrant
server and uses `QDRANT_URL=http://qdrant:6333`, which is the better
default for concurrent API traffic.

### Just open the UI and drag a PDF in

```bash
uv run pipeline-api
# open http://127.0.0.1:8000/ui — drag a PDF from quick-eval/ into "Upload documents"
```

If port 8000 is held by a leftover launchctl agent:

```bash
launchctl bootout gui/$(id -u)/com.ehz.a-rag-pipeline-api 2>/dev/null
```

### CLI runs

```bash
# 33-test suite, no API calls (~0.2 s)
uv run --extra dev python -m pytest -q

# Smoke run against quick-eval/ — 2 external PDFs (~30 s, ~$0.10)
uv run pipeline run \
  --input  quick-eval \
  --output outputs/quick_eval \
  --case-id quick-eval \
  --task   "Review the supplied documents and produce a first-pass case fact summary." \
  --force
open outputs/quick_eval/draft.md

# Full curated 20-doc corpus run (~3-4 min, ~$3-5)
uv run pipeline run \
  --input  datasets \
  --output outputs/all_categories \
  --case-id all-categories \
  --task   "Review the supplied documents and produce a first-pass case fact summary covering parties, obligations, deadlines, signatures, and any data-quality flags a human operator/reviewer should verify before relying on the draft." \
  --force
open outputs/all_categories/draft.md

# Deterministic eval suite (reads disk artifacts; no API calls)
uv run pipeline eval-suite \
  --run outputs/all_categories \
  --gold-dir eval \
  --output outputs/all_categories/evaluation.md \
  --json-output outputs/all_categories/evaluation.json

# A/B improvement-loop eval (drafts twice over same evidence, ~1-2 min, ~$1-2)
uv run pipeline ab-eval \
  --case-dir outputs/all_categories \
  --profile  state/operator_profile.json \
  --state-dir state \
  --edited-reference eval/operator-edits/messy_case_edited_memo.md \
  --draft-output-dir outputs/all_categories/ab_drafts \
  > outputs/all_categories/ab_eval.md
```

### Rebuild the React bundle (optional)

Only needed if you change anything under `frontend/src/`. Requires
Node 22+ and `npm`.

```bash
cd frontend && npm install && npm run build && cd ..
```

## With Docker

The image is multi-stage: `node:22-bookworm-slim` builds the React
bundle, `python:3.14.5-slim` runs the API. Docker users do **not**
need a local Node or Python toolchain.

```bash
git clone git@github.com:ehzawad/legal-rag-pipeline.git
cd legal-rag-pipeline

mkdir -p secrets
printf '%s' "$OPENAI_API_KEY" > secrets/openai_api_key
chmod 600 secrets/openai_api_key

docker compose -f docker-compose.yml -f docker-compose.secrets.yml up --build -d

# If Cohere reranking is enabled, also create secrets/cohere_api_key and
# include -f docker-compose.cohere-secrets.yml in the compose command.
#
# Docker Compose starts a local qdrant/qdrant service by default. For
# Qdrant Cloud, set QDRANT_URL, create secrets/qdrant_api_key, and include
# -f docker-compose.qdrant-secrets.yml in the compose command.

# Wait ~10 s for healthy, then:
curl http://localhost:8000/healthz             # {"status":"ok"}
open  http://localhost:8000/ui                 # operator console

# Run the pipeline inside the container (datasets/ mounted read-only,
# outputs/ read-write).
docker compose exec api pipeline run \
  --input  /app/datasets \
  --output /app/outputs/docker_run \
  --case-id docker-run \
  --draft-type case_fact_summary \
  --task   "Review the supplied documents and produce a first-pass case fact summary." \
  --force

docker compose down
```

The secret-file override keeps the API key out of the image and Compose
environment block. The entrypoint reads the file and exports the key only
inside the runtime process environment before starting Python. For quick
local-only Docker testing you may put `OPENAI_API_KEY=...` in `.env` and
run `docker compose up --build`, but that exposes the key through normal
Compose environment handling; prefer the secret flow for sharing, review,
or demos.

## Quick poke from the UI

1. Open `http://localhost:8000/ui`.
2. **Dashboard** lists draft-bearing real case reviews. Click a completed
   case, or use **Upload documents** to create one from a PDF in `quick-eval/`.
3. **Draft** tab shows the case fact summary with clickable `[evidence_id]` chips;
   clicking one pops the **Evidence panel** with the verbatim quote,
   page number, source filename, and chunk text.
4. **Source documents**, **Citations**, and **Grounding** tabs are
   review aids; **Edit & teach** lets you paste a revised draft and
   the backend captures + routes the edit through the five typed
   channels into `state/`.
5. After capturing an edit, re-run the pipeline (`--force`) or
   `pipeline ab-eval` to see the with-profile delta on a future draft.

## Cost expectations

| Run | Approx duration | Approx cost |
|---|---:|---:|
| `pytest` (no API calls) | <1 s | $0 |
| `eval-suite` (deterministic, reads disk only) | seconds | $0 |
| `quick-eval/` smoke (2 PDFs) | ~30 s | ~$0.10 |
| Full pipeline (20 docs / 70 pages, `gpt-5.5` low effort) | 3-4 min | $3-5 |
| `ab-eval` (drafts twice over same evidence) | 1-2 min | $1-2 |

Re-running with `--force` re-spends embeddings unless
`PIPELINE_RETRIEVAL_PROVIDER=openai-cached` is set (then embeddings are
reused across runs from a JSON cache).

## Single deliberate stack

This build ships one OpenAI core stack, with reranking disabled unless
explicitly enabled:

- Extraction + drafting: `gpt-5.5` (OpenAI Responses API)
- Embeddings: `text-embedding-3-large`
- Reranker (optional, off by default): Cohere `rerank-v4.0-pro`

Enable reranking with `PIPELINE_RERANK_PROVIDER=cohere` and
`COHERE_API_KEY`, `CO_API_KEY`, or `COHERE_API_KEY_FILE`. The model defaults to
`rerank-v4.0-pro` and can be overridden with `COHERE_RERANK_MODEL` or
`PIPELINE_RERANK_MODEL`. The reranker uses direct HTTPS, so no Cohere SDK
dependency is required. When enabled, Cohere orders the whole rerank
candidate pool before local document and field caps are applied, so cap
refills stay in Cohere-ranked order.
