from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from pipeline.config import PipelineFeatures, ProviderConfig, is_cached_retrieval_provider
from pipeline.orchestration.run import run_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Process messy legal-style documents into grounded operator-editable drafts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run ingestion, retrieval, and drafting. Use --force to invalidate cached checkpoints.",
    )
    run_parser.add_argument("--input", required=True, type=Path, help="Directory of source documents.")
    run_parser.add_argument("--output", required=True, type=Path, help="Directory for processed artifacts.")
    run_parser.add_argument("--case-id", default="sample-case")
    run_parser.add_argument("--task", default="first-pass internal memo")
    run_parser.add_argument("--profile", type=Path, help="Optional operator profile JSON to apply.")
    run_parser.add_argument(
        "--state-dir",
        type=Path,
        help="Cross-case learning state root. Defaults to ./state. Loads operator_profile.json, "
        "knowledge_layer.json, retrieval_feedback.json, and exemplars/ from this directory.",
    )
    run_parser.add_argument("--resume", action="store_true", help="Reuse completed artifact checkpoints when present.")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Invalidate cached checkpoints and rerun every stage regardless of fingerprint match.",
    )
    run_parser.add_argument(
        "--disable-processing",
        action="store_true",
        help="Do not ingest source documents; load processed_documents.json from --output instead.",
    )
    run_parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Do not retrieve evidence; load retrieved_evidence.json from --output instead.",
    )
    run_parser.add_argument(
        "--disable-drafting",
        action="store_true",
        help="Do not generate the memo; load draft.json and draft.md from --output instead.",
    )
    run_parser.add_argument(
        "--disable-learning-guidance",
        dest="disable_learning_guidance",
        action="store_true",
        help="Run without operator profile, knowledge-layer, or exemplar drafting guidance.",
    )
    run_parser.add_argument(
        "--disable-operator-profile-guidance",
        action="store_true",
        help="Exclude operator_profile.json from learned drafting guidance.",
    )
    run_parser.add_argument(
        "--disable-knowledge-layer-guidance",
        action="store_true",
        help="Exclude knowledge_layer.json from learned drafting guidance.",
    )
    run_parser.add_argument(
        "--disable-exemplar-guidance",
        action="store_true",
        help="Exclude per-category exemplars from learned drafting guidance.",
    )
    run_parser.add_argument(
        "--disable-retrieval-feedback",
        dest="disable_retrieval_feedback",
        action="store_true",
        help="Ignore retrieval_feedback.json boosts/demotes.",
    )
    run_parser.add_argument(
        "--disable-field-chunks",
        dest="disable_field_chunks",
        action="store_true",
        help="Do not create synthetic :fields retrieval chunks from extracted fields.",
    )
    run_parser.add_argument(
        "--disable-claim-support-check",
        dest="disable_claim_support_check",
        action="store_true",
        help="Skip the advisory lexical claim-support warning pass while keeping hard citation validation.",
    )
    run_parser.add_argument(
        "--disable-playbook-risk",
        dest="disable_playbook_risk",
        action="store_true",
        help="Skip deterministic playbook risk scoring for this run.",
    )
    run_parser.add_argument(
        "--playbook",
        type=Path,
        help="JSON legal-team playbook for deterministic risk scoring.",
    )
    run_parser.add_argument(
        "--max-field-chunks",
        type=int,
        help="Maximum synthetic :fields chunks allowed in retrieved evidence.",
    )
    run_parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        help="Maximum chunks per source document allowed in retrieved evidence (diversity cap).",
    )
    run_parser.add_argument(
        "--field-chunk-score-penalty",
        type=float,
        help="Score penalty applied to synthetic :fields chunks before ranking.",
    )

    corpus_parser = subparsers.add_parser(
        "corpus",
        help="Build or inspect the persistent corpus store.",
    )
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_subparsers.add_parser(
        "build",
        help="Persist originals plus parsed documents into a corpus store.",
    )
    corpus_build.add_argument("--input", required=True, type=Path)
    corpus_build.add_argument("--output", required=True, type=Path)
    corpus_build.add_argument(
        "--processed",
        type=Path,
        help="Use an existing processed_documents.json instead of running extraction.",
    )
    corpus_build.add_argument("--json", action="store_true")
    corpus_settings = corpus_subparsers.add_parser(
        "settings",
        help="Show effective corpus artifact paths and ingestion settings.",
    )
    corpus_settings.add_argument("--input", type=Path, default=Path("datasets"))
    corpus_settings.add_argument("--output", type=Path, default=Path("outputs/corpus"))
    corpus_settings.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    index_parser = subparsers.add_parser(
        "index",
        help="Build or query the persistent hybrid retrieval index.",
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    index_build = index_subparsers.add_parser(
        "build",
        help="Build a persisted BM25+dense retrieval index from a corpus store.",
    )
    index_build.add_argument("--corpus", required=True, type=Path)
    index_build.add_argument("--output", required=True, type=Path)
    index_build.add_argument("--json", action="store_true")
    index_query = index_subparsers.add_parser("query", help="Query a persisted retrieval index.")
    index_query.add_argument("--index", required=True, type=Path)
    index_query.add_argument("--task", required=True)
    index_query.add_argument("--top-k", type=int, default=5)
    index_query.add_argument("--json", action="store_true")
    index_settings = index_subparsers.add_parser(
        "settings",
        help="Show effective retrieval/index artifact paths and cache settings.",
    )
    index_settings.add_argument("--output", type=Path, default=Path("outputs/index"))
    index_settings.add_argument("--state-dir", type=Path, default=Path("state"))
    index_settings.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    edit_memory_parser = subparsers.add_parser(
        "edit-memory",
        help="Query or inspect operator edit memory.",
    )
    edit_memory_subparsers = edit_memory_parser.add_subparsers(
        dest="edit_memory_command",
        required=True,
    )
    edit_memory_settings = edit_memory_subparsers.add_parser(
        "settings",
        help="Show operator-profile, knowledge-layer, retrieval-feedback, and exemplar paths.",
    )
    edit_memory_settings.add_argument("--state-dir", type=Path, default=Path("state"))
    edit_memory_settings.add_argument("--profile", type=Path)
    edit_memory_settings.add_argument("--category", default="")
    edit_memory_settings.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    edit_memory_query = edit_memory_subparsers.add_parser(
        "query",
        help="Retrieve operator edit-memory suggestions for a task.",
    )
    edit_memory_query.add_argument("--state-dir", type=Path, default=Path("state"))
    edit_memory_query.add_argument("--task", default="")
    edit_memory_query.add_argument("--category", default="")
    edit_memory_query.add_argument("--text", default="")
    edit_memory_query.add_argument("--limit", type=int, default=5)
    edit_memory_query.add_argument("--json", action="store_true")

    learn_parser = subparsers.add_parser("learn", help="Learn reusable drafting preferences from an edit.")
    learn_parser.add_argument("--draft", required=True, type=Path, help="Original draft markdown.")
    learn_parser.add_argument("--edited", required=True, type=Path, help="Operator-edited draft markdown.")
    learn_parser.add_argument(
        "--profile",
        type=Path,
        help="Operator profile JSON to update. Defaults to <state-dir>/operator_profile.json.",
    )
    learn_parser.add_argument(
        "--state-dir",
        type=Path,
        help="Cross-case learning state root. Defaults to ./state. Knowledge rules and "
        "evidence-dispute feedback persist here under knowledge_layer.json and "
        "retrieval_feedback.json. Per-category exemplars live in <state-dir>/exemplars/.",
    )
    learn_parser.add_argument(
        "--category",
        help="Document category for per-category exemplar storage (e.g., handwritten-notes, "
        "scanned-pages). Only preference edits create exemplars. Skip if unknown.",
    )
    learn_parser.add_argument("--note", default="")
    learn_parser.add_argument(
        "--intent",
        choices=[
            "auto",
            "evidence_dispute",
            "field_correction",
            "knowledge_rule",
            "preference",
            "rejection",
        ],
        default="auto",
        help="How to route this edit. preference / field_correction update in-envelope "
        "state; knowledge_rule / evidence_dispute update out-of-envelope state; rejection "
        "is audit-only. `auto` infers from the operator note and diff stats.",
    )
    learn_parser.add_argument(
        "--actor-type",
        choices=["human", "agent"],
        default="human",
        help="Who produced the edit. Agent edits are advisory unless later accepted by a human.",
    )
    learn_parser.add_argument("--actor-id", default="default-operator")
    learn_parser.add_argument(
        "--event-log",
        type=Path,
        help="Append-only JSONL edit log path. Defaults to <profile-dir>/edits.jsonl.",
    )
    learn_parser.add_argument(
        "--suggestions",
        type=Path,
        help="Suggested gold/triage artifact path. Defaults to <profile-dir>/learn_suggestions.json.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a run directory.")
    eval_parser.add_argument("--run", required=True, type=Path, help="Directory containing draft/evidence JSON.")

    suite_parser = subparsers.add_parser(
        "eval-suite",
        help="Run the expanded deterministic evaluation suite over a case directory.",
    )
    suite_parser.add_argument("--run", required=True, type=Path, help="Directory containing run artifacts.")
    suite_parser.add_argument("--manifest", type=Path, help="Optional public eval manifest to compare.")
    suite_parser.add_argument("--gold-dir", type=Path, help="Optional directory with gold_documents/retrieval JSON.")
    suite_parser.add_argument("--eval-points", type=Path, help="Optional eval-points JSON with grouped deterministic checks.")
    suite_parser.add_argument(
        "--simulate-edits",
        action="store_true",
        help="Run deterministic simulated operator edits and include edit-improvement checks.",
    )
    suite_parser.add_argument(
        "--simulated-edits-output",
        type=Path,
        help="Optional directory for simulated edit-improvement artifacts.",
    )
    suite_parser.add_argument("--output", type=Path, help="Optional markdown report path.")
    suite_parser.add_argument("--json-output", type=Path, help="Optional JSON report path.")
    suite_parser.add_argument("--json", action="store_true", help="Emit JSON to stdout instead of markdown.")

    risk_parser = subparsers.add_parser(
        "risk",
        help="Score a completed run against a legal-team playbook.",
    )
    risk_parser.add_argument("--run", required=True, type=Path)
    risk_parser.add_argument("--playbook", type=Path)
    risk_parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    risk_parser.add_argument("--markdown-output", type=Path, help="Optional markdown report path.")
    risk_parser.add_argument("--json", action="store_true")

    harness_parser = subparsers.add_parser(
        "harness",
        help="Run deterministic eval harness cases.",
    )
    harness_subparsers = harness_parser.add_subparsers(dest="harness_command", required=True)
    harness_run = harness_subparsers.add_parser("run", help="Run harness cases from a JSON manifest.")
    harness_run.add_argument("--manifest", required=True, type=Path)
    harness_run.add_argument("--output", required=True, type=Path)
    harness_run.add_argument("--json", action="store_true")

    ab_parser = subparsers.add_parser(
        "ab-eval",
        help="A/B improvement-loop eval: draft once without profile, once with, then compare.",
    )
    ab_parser.add_argument(
        "--case-dir",
        required=True,
        type=Path,
        help="Case directory containing processed_documents.json and retrieved_evidence.json.",
    )
    ab_parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Operator profile JSON to apply on the with-profile draft.",
    )
    ab_parser.add_argument(
        "--task",
        default=None,
        help="Drafting task description fed to the generator. If omitted, read from case_run.json in --case-dir.",
    )
    ab_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown.",
    )
    ab_parser.add_argument(
        "--edited-reference",
        type=Path,
        help="Optional operator-edited reference draft for similarity and section-F1 checks.",
    )
    ab_parser.add_argument(
        "--draft-output-dir",
        type=Path,
        help="Optional directory to persist without/with-profile draft JSON and markdown.",
    )
    ab_parser.add_argument(
        "--state-dir",
        type=Path,
        help="Cross-case state dir. When given, the with-profile draft also sees the "
        "knowledge layer and per-category exemplars, not just the operator profile.",
    )

    public_parser = subparsers.add_parser(
        "public-data",
        help="Download or verify the public evaluation document bundle.",
    )
    public_subparsers = public_parser.add_subparsers(dest="public_command", required=True)
    public_download = public_subparsers.add_parser("download", help="Download manifest items.")
    public_download.add_argument("--manifest", required=True, type=Path)
    public_download.add_argument("--output", required=True, type=Path)
    public_download.add_argument("--force", action="store_true")
    public_download.add_argument("--dry-run", action="store_true")
    public_download.add_argument("--include-optional", action="store_true")
    public_verify = public_subparsers.add_parser("verify", help="Verify downloaded manifest items.")
    public_verify.add_argument("--manifest", required=True, type=Path)
    public_verify.add_argument("--dataset", required=True, type=Path)
    public_verify.add_argument("--include-optional", action="store_true")

    dpo_parser = subparsers.add_parser(
        "export-dpo",
        help="Export captured operator edits as section-level (prompt, chosen, rejected) "
        "preference pairs in DPO-ready JSONL. Filters out pairs where the edited section "
        "lost a citation that the original carried.",
    )
    dpo_parser.add_argument(
        "--edits-log",
        required=True,
        type=Path,
        help="edits.jsonl path (e.g., outputs/<case>/edits.jsonl).",
    )
    dpo_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL path for section-level preference pairs.",
    )
    dpo_parser.add_argument(
        "--include-non-preference",
        action="store_true",
        help="Include non-preference edits (field_correction etc.). Default excludes them.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        features = _features_from_run_args(args)
        run_case(
            args.input,
            args.output,
            case_id=args.case_id,
            task=args.task,
            profile_path=args.profile,
            state_dir=args.state_dir,
            resume=args.resume,
            force=args.force,
            features=features,
            playbook_path=args.playbook,
        )
        print(f"Wrote workflow artifacts to {args.output}")
        return 0
    if args.command == "corpus":
        if args.corpus_command == "build":
            _print_settings(_build_corpus(args.input, args.output, args.processed), as_json=args.json)
            return 0
        if args.corpus_command == "settings":
            _print_settings(_corpus_settings(args.input, args.output), as_json=args.json)
            return 0
    if args.command == "index":
        if args.index_command == "build":
            _print_settings(_build_index_store(args.corpus, args.output), as_json=args.json)
            return 0
        if args.index_command == "query":
            _print_settings(_query_index(args.index, args.task, args.top_k), as_json=args.json)
            return 0
        if args.index_command == "settings":
            _print_settings(_index_settings(args.output, args.state_dir), as_json=args.json)
            return 0
    if args.command == "edit-memory":
        if args.edit_memory_command == "query":
            _print_settings(
                _query_edit_memory(args.state_dir, args.task, args.category, args.text, args.limit),
                as_json=args.json,
            )
            return 0
        if args.edit_memory_command == "settings":
            _print_settings(
                _edit_memory_settings(args.state_dir, args.profile, args.category),
                as_json=args.json,
            )
            return 0
    if args.command == "learn":
        from pipeline.learning import learn_from_files, state_paths

        resolved_state = state_paths(args.state_dir) if args.state_dir is not None else None
        resolved_profile = args.profile or (
            resolved_state["operator_profile"] if resolved_state else None
        )
        if resolved_profile is None:
            parser.error("--profile or --state-dir is required for learn")
        learn_kwargs: dict[str, object] = {
            "operator_note": args.note,
            "intent": args.intent,
            "actor_type": args.actor_type,
            "actor_id": args.actor_id,
            "event_log_path": args.event_log,
            "suggestions_path": args.suggestions,
        }
        if args.state_dir is not None:
            learn_kwargs["state_dir"] = args.state_dir
        if args.category is not None:
            learn_kwargs["category"] = args.category
        profile = learn_from_files(
            args.draft,
            args.edited,
            resolved_profile,
            **learn_kwargs,
        )
        event_log = args.event_log or resolved_profile.parent / "edits.jsonl"
        suggestions = args.suggestions or resolved_profile.parent / "learn_suggestions.json"
        print(f"Processed edit; operator profile at {resolved_profile} ({len(profile.get('examples', []))} examples)")
        print(f"Appended edit event to {event_log}")
        if suggestions.exists():
            print(f"Wrote learning suggestions to {suggestions}")
        if resolved_state is not None:
            if resolved_state["knowledge_layer"].exists():
                print(f"Knowledge layer at {resolved_state['knowledge_layer']}")
            if resolved_state["retrieval_feedback"].exists():
                print(f"Retrieval feedback at {resolved_state['retrieval_feedback']}")
            if args.category and resolved_state["exemplars_dir"].exists():
                print(f"Exemplars under {resolved_state['exemplars_dir']}")
        return 0
    if args.command == "export-dpo":
        from pipeline.learning import export_dpo_pairs

        counts = export_dpo_pairs(
            edits_log=args.edits_log,
            output=args.output,
            include_non_preference=args.include_non_preference,
        )
        candidates_path = args.output.with_name(args.output.stem + ".candidates.jsonl")
        print(
            f"Wrote {counts['strict']} strict preference pairs to {args.output}"
        )
        if counts["candidates"]:
            print(
                f"Wrote {counts['candidates']} unsafe restructure candidates "
                f"(dropped citations) to {candidates_path} — review before training."
            )
        return 0
    if args.command == "evaluate":
        from pipeline.evaluation import evaluate_run

        result = evaluate_run(args.run)
        print(result.to_markdown())
        return 0
    if args.command == "eval-suite":
        from pipeline.evaluation import evaluate_suite

        result = evaluate_suite(
            args.run,
            manifest_path=args.manifest,
            gold_dir=args.gold_dir,
            eval_points_path=args.eval_points,
            simulate_edits=args.simulate_edits,
            simulated_edits_output_dir=args.simulated_edits_output,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result.to_markdown() + "\n", encoding="utf-8")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(result.to_json() + "\n", encoding="utf-8")
        print(result.to_json() if args.json else result.to_markdown())
        return 0
    if args.command == "risk":
        from pipeline.playbooks import render_risk_report_markdown, score_playbook_run_dir, write_risk_report

        result = score_playbook_run_dir(args.run, playbook_path=args.playbook)
        if args.output:
            write_risk_report(result, args.output, args.markdown_output)
        elif args.markdown_output:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(render_risk_report_markdown(result) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else render_risk_report_markdown(result))
        return 0
    if args.command == "harness":
        if args.harness_command == "run":
            result = _run_harness_manifest(args.manifest, args.output)
            print(result.to_json() if args.json else json.dumps(result.summary, indent=2, sort_keys=True))
            return 0 if result.summary.get("passed") else 1
    if args.command == "ab-eval":
        from pipeline.evaluation import evaluate_ab

        task = args.task or _resolve_ab_task(args.case_dir)
        result = evaluate_ab(
            args.case_dir,
            args.profile,
            task=task,
            edited_reference_path=args.edited_reference,
            draft_output_dir=args.draft_output_dir,
            state_dir=args.state_dir,
        )
        print(result.to_json() if args.json else result.to_markdown())
        return 0
    if args.command == "public-data":
        from pipeline.public_data import download_public_eval_set, verify_public_eval_set

        if args.public_command == "download":
            result = download_public_eval_set(
                args.manifest,
                args.output,
                force=args.force,
                dry_run=args.dry_run,
                include_optional=args.include_optional,
            )
        else:
            result = verify_public_eval_set(
                args.manifest,
                args.dataset,
                include_optional=args.include_optional,
            )
        print(json.dumps(result.to_jsonable(), indent=2, ensure_ascii=False, sort_keys=True))
        return 1 if result.failed else 0
    parser.error(f"Unknown command {args.command}")
    return 2


def _resolve_ab_task(case_dir: Path) -> str:
    from pipeline.evaluation import resolve_ab_task

    return resolve_ab_task(case_dir)


def _features_from_run_args(args: argparse.Namespace) -> PipelineFeatures:
    features = PipelineFeatures.from_env()
    updates: dict[str, object] = {}
    if args.disable_processing:
        updates["process_documents"] = False
    if args.disable_retrieval:
        updates["retrieve_evidence"] = False
    if args.disable_drafting:
        updates["generate_draft"] = False
    if args.disable_learning_guidance:
        updates["learning_guidance"] = False
    if args.disable_operator_profile_guidance:
        updates["operator_profile_guidance"] = False
    if args.disable_knowledge_layer_guidance:
        updates["knowledge_layer_guidance"] = False
    if args.disable_exemplar_guidance:
        updates["exemplar_guidance"] = False
    if args.disable_retrieval_feedback:
        updates["retrieval_feedback"] = False
    if args.disable_field_chunks:
        updates["field_chunks"] = False
    if args.disable_claim_support_check:
        updates["claim_support_check"] = False
    if args.disable_playbook_risk:
        updates["playbook_risk"] = False
    if args.max_field_chunks is not None:
        updates["max_field_chunks"] = args.max_field_chunks
    if args.max_chunks_per_document is not None:
        updates["max_chunks_per_document"] = args.max_chunks_per_document
    if args.field_chunk_score_penalty is not None:
        updates["field_chunk_score_penalty"] = args.field_chunk_score_penalty
    if updates:
        features = replace(features, **updates)
        features.validate_runtime()
    return features


def _build_corpus(input_dir: Path, corpus_dir: Path, processed_path: Path | None) -> dict[str, object]:
    from pipeline.corpus import write_corpus_store
    from pipeline.ingestion.documents import process_directory
    from pipeline.orchestration.artifacts import load_processed_documents

    config = ProviderConfig.from_env()
    processed = (
        load_processed_documents(processed_path)
        if processed_path is not None
        else process_directory(input_dir, provider=config.extraction_provider, config=config)
    )
    manifest = write_corpus_store(input_dir=input_dir, processed_documents=processed, corpus_dir=corpus_dir)
    return {
        "surface": "corpus",
        "mode": "built",
        "input_dir": str(input_dir),
        "corpus_dir": str(corpus_dir),
        "manifest": manifest.to_jsonable(),
    }


def _build_index_store(corpus_dir: Path, output: Path) -> dict[str, object]:
    from pipeline.corpus import load_corpus_documents
    from pipeline.retrieval import build_index, save_index

    config = ProviderConfig.from_env()
    documents = load_corpus_documents(corpus_dir)
    index = build_index(documents, provider=config.retrieval_provider, config=config)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, output)
    return {
        "surface": "index",
        "mode": "built",
        "corpus_dir": str(corpus_dir),
        "index": str(output),
        "chunk_count": len(index.chunks),
        "embedding_model": index.embedding_model,
        "vector_backend": getattr(index, "index_backend", "memory"),
        "qdrant_collection": getattr(index, "qdrant_collection", ""),
    }


def _query_index(index_path: Path, task: str, top_k: int) -> dict[str, object]:
    from pipeline.retrieval import load_index, retrieve

    config = ProviderConfig.from_env()
    index = load_index(index_path)
    # Let retrieve() resolve PIPELINE_RETRIEVAL_MODE. Persisted lexical indexes
    # store placeholder embeddings, so forcing hybrid weights here can make the
    # query path call dense embeddings and then fail on vector dimensions.
    evidence = retrieve(
        index,
        task,
        top_k=top_k,
        provider=config.retrieval_provider,
        config=config,
    )
    return {
        "surface": "index",
        "mode": "query",
        "index": str(index_path),
        "task": task,
        "settings": {
            "retrieval_provider": config.retrieval_provider,
            "reranker_provider": config.reranker_provider,
            "reranker_model": config.cohere_rerank_model,
        },
        "results": [
            {
                "evidence_id": chunk.evidence_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "score": chunk.score,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in evidence
        ],
    }


def _query_edit_memory(state_dir: Path, task: str, category: str, text: str, limit: int) -> dict[str, object]:
    from pipeline.learning import load_edit_memory, render_edit_memory_guidance, retrieve_edit_memory, state_paths

    paths = state_paths(state_dir)
    matches = retrieve_edit_memory(
        load_edit_memory(paths["edit_memory"]),
        task=task,
        category=category,
        text=text,
        limit=limit,
    )
    return {
        "surface": "edit-memory",
        "mode": "query",
        "state_dir": str(state_dir),
        "matches": matches,
        "guidance": render_edit_memory_guidance(matches),
    }


def _run_harness_manifest(manifest_path: Path, output_dir: Path):
    from pipeline.evaluation import HarnessCase, evaluate_run, run_harness

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    cases = [
        HarnessCase(
            case_id=str(item["case_id"]),
            input_dir=str(item["input_dir"]),
            task=str(item["task"]),
            deterministic_checks=dict(item.get("deterministic_checks") or {}),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw_cases
    ]

    def runner(case: HarnessCase, case_dir: Path) -> dict[str, object]:
        run_dir = case_dir / "run"
        run_case(Path(case.input_dir), run_dir, case_id=case.case_id, task=case.task, force=True)
        static = evaluate_run(run_dir)
        return {
            "case_id": case.case_id,
            "output_dir": str(run_dir),
            "static": asdict(static),
            "artifacts": {
                "case_run": str(run_dir / "case_run.json"),
                "draft": str(run_dir / "draft.md"),
                "retrieval_index": str(run_dir / "index" / "retrieval_index.json"),
                "corpus_manifest": str(run_dir / "corpus" / "manifest.json"),
            },
        }

    return run_harness(cases, output_dir, runner=runner)


def _corpus_settings(input_dir: Path, output_dir: Path) -> dict[str, object]:
    config = ProviderConfig.from_env()
    features = PipelineFeatures.from_env()
    return {
        "surface": "corpus",
        "mode": "settings",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "artifacts": {
            "processed_documents": str(output_dir / "processed_documents.json"),
            "corpus_manifest": str(output_dir / "corpus" / "manifest.json"),
            "workflow_manifest": str(output_dir / "workflow_manifest.json"),
        },
        "settings": {
            "process_documents": features.process_documents,
            "extraction_provider": config.extraction_provider,
            "pdf_max_pages": config.pdf_max_pages,
            "pdf_render_dpi": config.pdf_render_dpi,
            "extraction_concurrency": config.extraction_concurrency,
        },
        "commands": {
            "ingest_with_run": (
                f"pipeline run --input {input_dir} --output {output_dir} --disable-retrieval --disable-drafting"
            ),
            "full_run": f"pipeline run --input {input_dir} --output {output_dir}",
        },
        "notes": [
            "Use `pipeline corpus build` to persist originals plus parsed documents.",
            "A normal `pipeline run` also writes output/corpus/ automatically.",
        ],
    }


def _index_settings(output_dir: Path, state_dir: Path) -> dict[str, object]:
    config = ProviderConfig.from_env()
    features = PipelineFeatures.from_env()
    effective_cache_dir = ""
    if is_cached_retrieval_provider(config.retrieval_provider):
        effective_cache_dir = config.embedding_cache_dir or str(output_dir / "embedding_cache")
    return {
        "surface": "index",
        "mode": "settings",
        "output_dir": str(output_dir),
        "artifacts": {
            "retrieved_evidence": str(output_dir / "retrieved_evidence.json"),
            "retrieval_index": str(output_dir / "index" / "retrieval_index.json"),
            "embedding_cache_dir": effective_cache_dir,
            "workflow_manifest": str(output_dir / "workflow_manifest.json"),
        },
        "settings": {
            "retrieve_evidence": features.retrieve_evidence,
            "retrieval_provider": config.retrieval_provider,
            "retrieval_mode": config.retrieval_mode,
            "index_backend": config.index_backend,
            "qdrant_url": config.qdrant_url,
            "qdrant_path": config.qdrant_path,
            "qdrant_collection": config.qdrant_collection,
            "qdrant_prefer_grpc": config.qdrant_prefer_grpc,
            "embedding_model": config.openai_embedding_model,
            "hybrid_dense_weight": config.hybrid_dense_weight,
            "hybrid_bm25_weight": config.hybrid_bm25_weight,
            "retrieval_top_k": config.retrieval_top_k,
            "reranker_provider": config.reranker_provider,
            "reranker_model": config.cohere_rerank_model,
            "field_chunks": features.field_chunks,
            "max_field_chunks": features.max_field_chunks,
            "max_chunks_per_document": features.max_chunks_per_document,
            "field_chunk_score_penalty": features.field_chunk_score_penalty,
            "retrieval_feedback": features.retrieval_feedback,
            "retrieval_feedback_path": str(state_dir / "retrieval_feedback.json"),
        },
        "commands": {
            "retrieve_with_run": (
                f"pipeline run --input <corpus-dir> --output {output_dir} --disable-processing --disable-drafting"
            ),
            "full_run": f"pipeline run --input <corpus-dir> --output {output_dir}",
        },
        "notes": [
            "Use `pipeline index build` to persist a queryable BM25+dense index.",
            "Set PIPELINE_INDEX_BACKEND=qdrant to store dense vectors in Qdrant while keeping retrieval_index.json portable.",
            "A normal `pipeline run` also writes output/index/retrieval_index.json automatically.",
        ],
    }


def _edit_memory_settings(
    state_dir: Path,
    profile_path: Path | None,
    category: str,
) -> dict[str, object]:
    from pipeline.learning import state_paths

    features = PipelineFeatures.from_env()
    paths = state_paths(state_dir)
    resolved_profile = profile_path or paths["operator_profile"]
    artifacts = {
        "operator_profile": str(resolved_profile),
        "knowledge_layer": str(paths["knowledge_layer"]),
        "retrieval_feedback": str(paths["retrieval_feedback"]),
        "edit_memory": str(paths["edit_memory"]),
        "edits_log": str(paths["state_dir"] / "edits.jsonl"),
        "learn_suggestions": str(paths["state_dir"] / "learn_suggestions.json"),
        "exemplars_dir": str(paths["exemplars_dir"]),
    }
    if category:
        artifacts["category_exemplars"] = str(paths["exemplars_dir"] / f"{category}.jsonl")
    return {
        "surface": "edit-memory",
        "mode": "settings",
        "state_dir": str(paths["state_dir"]),
        "artifacts": artifacts,
        "settings": {
            "learning_guidance": features.learning_guidance,
            "operator_profile_guidance": features.operator_profile_guidance,
            "knowledge_layer_guidance": features.knowledge_layer_guidance,
            "exemplar_guidance": features.exemplar_guidance,
            "retrieval_feedback": features.retrieval_feedback,
            "playbook_risk": features.playbook_risk,
        },
        "commands": {
            "capture_edit": (
                f"pipeline learn --draft <draft.md> --edited <edited.md> --state-dir {paths['state_dir']}"
            ),
            "export_dpo": f"pipeline export-dpo --edits-log {paths['state_dir'] / 'edits.jsonl'} --output {paths['state_dir'] / 'dpo_pairs.jsonl'}",
        },
        "notes": [
            "Use `pipeline edit-memory query` to retrieve operator preference suggestions.",
            "AI critique remains advisory unless captured as a human operator edit.",
        ],
    }


def _print_settings(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return
    print(f"{payload['surface']} settings ({payload['mode']})")
    for key, value in payload.get("settings", {}).items():
        print(f"- {key}: {value}")
    artifacts = payload.get("artifacts", {})
    if artifacts:
        print("Artifacts:")
        for key, value in artifacts.items():
            print(f"- {key}: {value}")
    notes = payload.get("notes", [])
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    raise SystemExit(main())
