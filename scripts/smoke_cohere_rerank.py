"""Standalone smoke test for Cohere Rerank against this repo's data.

Run:
    uv run python scripts/smoke_cohere_rerank.py

What it does:
- Loads COHERE_API_KEY or CO_API_KEY from .env or the environment.
- Picks one playbook position (default: indemnification) from playbooks/default_legal_playbook.json.
- Loads the retrieved chunks from outputs/all_categories/retrieved_evidence.json.
- Calls Cohere Rerank with the position's concept anchors as the query
  and the chunk texts as documents.
- Prints the reranked top-N with relevance scores, so you can eyeball whether
  the API surfaces the chunks that actually discuss the playbook concept.

No pipeline changes — this is a one-off probe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYBOOK = REPO_ROOT / "playbooks" / "default_legal_playbook.json"
DEFAULT_EVIDENCE = REPO_ROOT / "outputs" / "all_categories" / "retrieved_evidence.json"
DEFAULT_PROCESSED = REPO_ROOT / "outputs" / "all_categories" / "processed_documents.json"
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
DEFAULT_MODEL = "rerank-v4.0-pro"


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_json_file(path: Path, label: str) -> Any:
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


def build_query(position: dict[str, Any]) -> str:
    """Stitch the playbook position into a natural-language probe.

    Cohere Rerank is a cross-encoder — it benefits from a sentence-shaped
    query, not just keywords. So we wrap the title + concepts into a question.
    """
    title = str(position.get("title") or position.get("id") or "").strip()
    required = position.get("required_any") or []
    escalation = position.get("escalation_terms") or []
    parts = [f"Find clauses discussing {title}."]
    if required:
        parts.append("Required concepts: " + ", ".join(required) + ".")
    if escalation:
        parts.append("Escalation triggers: " + ", ".join(escalation) + ".")
    return " ".join(parts)


def call_rerank(*, api_key: str, query: str, documents: list[str], model: str, top_n: int) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        COHERE_RERANK_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Cohere returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cohere request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Cohere returned invalid JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--position", default="indemnification", help="Playbook position id to probe (default: indemnification).")
    parser.add_argument("--playbook", type=Path, default=DEFAULT_PLAYBOOK, help="Playbook JSON path.")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE, help="Retrieved-evidence JSON path.")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED, help="Processed-documents JSON path (used when --source=pages).")
    parser.add_argument("--source", choices=("chunks", "pages"), default="chunks", help="Rerank against retrieved chunks (default) or all processed pages.")
    parser.add_argument("--top-n", type=positive_int, default=5, help="How many reranked results to print.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Cohere rerank model id.")
    args = parser.parse_args()

    load_env_file(REPO_ROOT / ".env")
    if args.model == DEFAULT_MODEL:
        args.model = (
            os.environ.get("PIPELINE_RERANK_MODEL")
            or os.environ.get("COHERE_RERANK_MODEL")
            or DEFAULT_MODEL
        )

    playbook = read_json_file(args.playbook, "playbook")
    if not isinstance(playbook, dict):
        print(f"error: playbook must be a JSON object: {args.playbook}", file=sys.stderr)
        return 2
    positions = playbook.get("positions") or []
    position = next((p for p in positions if p.get("id") == args.position), None)
    if position is None:
        ids = [str(p.get("id")) for p in positions]
        print(f"error: position '{args.position}' not in playbook. Available: {ids}", file=sys.stderr)
        return 2

    if args.source == "chunks":
        evidence = read_json_file(args.evidence, "retrieved evidence")
        if not isinstance(evidence, list) or not evidence:
            print(f"error: no evidence chunks in {args.evidence}", file=sys.stderr)
            return 2
        source_label = args.evidence.name
        items = [
            {
                "text": str(chunk.get("text") or ""),
                "evidence_id": chunk.get("evidence_id"),
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
            }
            for chunk in evidence
        ]
    else:
        processed = read_json_file(args.processed, "processed documents")
        if not isinstance(processed, list) or not processed:
            print(f"error: no processed documents in {args.processed}", file=sys.stderr)
            return 2
        items = []
        for document in processed:
            source = document.get("source") or {}
            for page in document.get("pages") or []:
                text = str(page.get("text") or "").strip()
                if not text:
                    continue
                items.append(
                    {
                        "text": text,
                        "evidence_id": f"{source.get('document_id')}:p{page.get('page_number')}",
                        "filename": source.get("filename"),
                        "page_number": page.get("page_number"),
                    }
                )
        source_label = args.processed.name

    if not items:
        print(f"error: no documents to rerank from {source_label}", file=sys.stderr)
        return 2

    documents = [item["text"] for item in items]
    query = build_query(position)
    print(f"position      : {position.get('id')} ({position.get('title')})")
    print(f"query         : {query}")
    print(f"documents     : {len(documents)} {args.source} from {source_label}")
    print(f"rerank model  : {args.model}")
    print()

    api_key = (os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY") or "").strip()
    if not api_key:
        print("error: COHERE_API_KEY or CO_API_KEY missing from environment and .env", file=sys.stderr)
        return 2

    result = call_rerank(api_key=api_key, query=query, documents=documents, model=args.model, top_n=args.top_n)
    results = result.get("results") or []
    if not results:
        print("Cohere returned no results. Raw payload:")
        print(json.dumps(result, indent=2))
        return 1

    print(f"Top {len(results)} reranked {args.source}:")
    print()
    for rank, hit in enumerate(results, start=1):
        idx = hit.get("index")
        score = hit.get("relevance_score")
        item = items[idx] if isinstance(idx, int) and 0 <= idx < len(items) else {}
        snippet = " ".join(str(item.get("text", "")).split())[:240]
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"  #{rank} score={score_text}  evidence_id={item.get('evidence_id')}  page={item.get('page_number')}  file={item.get('filename')}")
        print(f"      {snippet!r}")
        print()
    billed = result.get("meta", {}).get("billed_units") or {}
    if billed:
        print(f"billed_units  : {billed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
