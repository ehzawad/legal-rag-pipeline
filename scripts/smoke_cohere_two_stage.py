"""Two-stage smoke: BM25 retrieve → Cohere Rerank, with concept anchors.

The single-stage smoke (`smoke_cohere_rerank.py`) showed Cohere can't pick out
contract chunks when fed all 70 corpus pages with a bag-of-terms query.
This smoke runs the canonical pattern instead:

    For each playbook position:
        1. Build a concept anchor (a short canonical clause example).
        2. Retrieve top-100 chunks via lexical (BM25-only) retrieval. No OpenAI
           embeddings required — keeps the smoke offline-capable.
        3. Send the anchor + the retrieved chunks to Cohere Rerank.
        4. Print:
            (a) the top-10 BM25 ranking before rerank;
            (b) the top-10 ranking after rerank.

Run:
    uv run python scripts/smoke_cohere_two_stage.py
    uv run python scripts/smoke_cohere_two_stage.py --position term_termination
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.config import ProviderConfig  # noqa: E402
from pipeline.orchestration.artifacts import processed_documents_from_json  # noqa: E402
from pipeline.retrieval.engine import build_index, retrieve  # noqa: E402

DEFAULT_PROCESSED = REPO_ROOT / "outputs" / "all_categories" / "processed_documents.json"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_MODEL = "rerank-v4.0-pro"
RETRIEVE_TOP_K = 100
RERANK_TOP_N = 10

# Short canonical clauses per playbook position. These are the "concept
# anchors" — language a competent contract would use to express the concept.
# Not playbook terms; the anchors do the cross-encoder's heavy lifting.
CONCEPT_ANCHORS: dict[str, str] = {
    "limitation_of_liability": (
        "Each party's aggregate liability under this Agreement shall not exceed the fees paid in the twelve months preceding the claim. Neither party shall be liable for indirect, incidental, or consequential damages."
    ),
    "indemnification": (
        "Each party shall indemnify, defend, and hold harmless the other party from any third-party claims arising from breach of this Agreement, including infringement of intellectual property rights."
    ),
    "data_protection": (
        "The parties shall comply with applicable data protection laws and execute a Data Processing Agreement before any personal data is transferred. Vendor shall notify Customer of any personal data breach within seventy-two hours."
    ),
    "term_termination": (
        "Either party may terminate this Agreement for convenience upon thirty days' written notice. This Agreement automatically renews for successive one-year terms unless either party provides notice of non-renewal."
    ),
    "governing_law": (
        "This Agreement is governed by the laws of the State of Delaware, without regard to its conflicts of law principles. The parties consent to the exclusive jurisdiction of the federal and state courts located in Wilmington, Delaware."
    ),
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json_file(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"error: could not read {label} at {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except json.JSONDecodeError as exc:
        print(f"error: {label} is not valid JSON at {path}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def call_rerank(*, api_key: str, query: str, documents: list[str], model: str, top_n: int) -> list[dict]:
    body = json.dumps({"model": model, "query": query, "documents": documents, "top_n": top_n}).encode("utf-8")
    request = urllib.request.Request(
        COHERE_RERANK_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Cohere returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cohere request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cohere returned invalid JSON: {exc}") from exc
    return payload.get("results") or []


def format_chunk(chunk, score: float) -> str:
    snippet = " ".join(chunk.text.split())[:180]
    return f"      score={score:.4f} ev={chunk.evidence_id} p={chunk.page_number} file={chunk.filename}\n         {snippet!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--position", default="indemnification", choices=sorted(CONCEPT_ANCHORS), help="Playbook position to probe.")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED, help="Processed-documents JSON path.")
    parser.add_argument("--retrieve-k", type=positive_int, default=RETRIEVE_TOP_K, help="How many chunks to retrieve before rerank.")
    parser.add_argument("--rerank-n", type=positive_int, default=RERANK_TOP_N, help="How many top reranked chunks to print.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Cohere rerank model id.")
    args = parser.parse_args()

    load_env_file(REPO_ROOT / ".env")
    if args.model == DEFAULT_MODEL:
        args.model = (
            os.environ.get("PIPELINE_RERANK_MODEL")
            or os.environ.get("COHERE_RERANK_MODEL")
            or DEFAULT_MODEL
        )

    anchor = CONCEPT_ANCHORS[args.position]
    processed_payload = read_json_file(args.processed, "processed documents")
    processed = processed_documents_from_json(processed_payload)

    # Lexical-only retrieval avoids any OpenAI embedding call. Set retrieval_mode
    # via the config override so build_index skips embedding generation.
    os.environ["PIPELINE_RETRIEVAL_MODE"] = "lexical"
    config = ProviderConfig.from_env()
    index = build_index(processed, provider="fixture", config=config)

    candidates = retrieve(
        index,
        anchor,
        top_k=args.retrieve_k,
        provider="fixture",
        config=config,
        dense_weight=0.0,
        lexical_weight=1.0,
        max_chunks_per_document=10,
        reranker_provider="",
    )
    if not candidates:
        print(f"error: BM25 retrieval returned no chunks for position={args.position}", file=sys.stderr)
        return 1

    print(f"position       : {args.position}")
    print(f"anchor         : {anchor[:140]}{'…' if len(anchor) > 140 else ''}")
    print(f"corpus chunks  : {len(index.chunks)}  (retrieve_k={args.retrieve_k}, rerank_n={args.rerank_n})")
    print()
    print(f"Top {min(args.rerank_n, len(candidates))} after BM25 only:")
    for rank, chunk in enumerate(candidates[: args.rerank_n], start=1):
        print(f"  #{rank}\n{format_chunk(chunk, chunk.score)}")

    api_key = (os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY") or "").strip()
    if not api_key:
        print("error: COHERE_API_KEY or CO_API_KEY missing from environment and .env", file=sys.stderr)
        return 2

    rerank_results = call_rerank(
        api_key=api_key,
        query=anchor,
        documents=[chunk.text for chunk in candidates],
        model=args.model,
        top_n=args.rerank_n,
    )
    print()
    print(f"Top {args.rerank_n} after Cohere rerank:")
    for rank, hit in enumerate(rerank_results, start=1):
        idx = hit.get("index")
        score = hit.get("relevance_score", 0.0)
        if isinstance(idx, int) and 0 <= idx < len(candidates):
            print(f"  #{rank}\n{format_chunk(candidates[idx], score)}")

    # Diff: how many of the top-N reranked chunks were ALSO in BM25 top-N?
    bm25_top_ids = {chunk.evidence_id for chunk in candidates[: args.rerank_n]}
    rerank_top_ids = {
        candidates[hit["index"]].evidence_id
        for hit in rerank_results
        if isinstance(hit.get("index"), int) and 0 <= hit["index"] < len(candidates)
    }
    overlap = bm25_top_ids & rerank_top_ids
    moved_in = rerank_top_ids - bm25_top_ids
    moved_out = bm25_top_ids - rerank_top_ids
    print()
    print(f"Top-{args.rerank_n} overlap between BM25 and rerank: {len(overlap)}/{args.rerank_n}")
    if moved_in:
        print(f"  moved IN  (rerank surfaced): {sorted(moved_in)}")
    if moved_out:
        print(f"  moved OUT (rerank demoted):  {sorted(moved_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
