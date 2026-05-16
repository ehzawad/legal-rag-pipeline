# Evaluation

The evaluator is intentionally artifact-based. It reads persisted pipeline
outputs (`processed_documents.json`, `retrieved_evidence.json`, `draft.json`)
and reports deterministic metrics before any optional judge-style scoring is
considered.

## Commands

Static single-draft checks:

```bash
uv run pipeline evaluate --run outputs/all_categories
```

Expanded suite:

```bash
uv run pipeline eval-suite \
  --run outputs/all_categories \
  --gold-dir eval \
  --output outputs/all_categories/evaluation.md \
  --json-output outputs/all_categories/evaluation.json
```

Harness cases:

```bash
uv run pipeline harness run \
  --manifest eval/harness_cases.json \
  --output outputs/harness \
  --json
```

Public-data smoke:

```bash
uv run pipeline public-data download \
  --manifest eval/public_eval_manifest.json \
  --output eval/public-documents

uv run pipeline public-data verify \
  --manifest eval/public_eval_manifest.json \
  --dataset eval/public-documents

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

A/B edit-loop check:

```bash
uv run pipeline ab-eval \
  --case-dir outputs/all_categories \
  --profile state/operator_profile.json \
  --state-dir state \
  --edited-reference eval/operator-edits/messy_case_edited_memo.md \
  --draft-output-dir outputs/all_categories/ab_drafts
```

Playbook risk screen:

```bash
uv run pipeline risk \
  --run outputs/all_categories \
  --playbook playbooks/default_legal_playbook.json \
  --output outputs/all_categories/risk_report.json \
  --markdown-output outputs/all_categories/risk_report.md
```

## Metric Surface

- `citation_id_validity`: supported sections whose `evidence_ids` resolve to
  retrieved chunks.
- `sentence_grounding_score`: quote-substring validity. It checks that each
  `citation_quotes[evidence_id]` value appears in the cited chunk. It is not
  semantic entailment.
- `claim_citation_rate`: stricter body scan over factual-looking sentences with
  inline citation tokens.
- `claim_grounding_pass_rate` / `claim_citation_coverage`: claim-graph metrics
  for structured case fact summaries. Deterministic grounding is quote
  substring validation unless the optional entailment judge is enabled.
- Eval-points criteria:
  - `document_understanding`: file/category coverage, page caps, and extraction
    term checks.
  - `grounded_drafting`: citation validity, claim coverage, grounding, raw-page
    citation share, and negative anti-hallucination checks.
  - `edit_improvement`: simulated operator edits routed through `learn_from_files`
    and scored for memory recall, guidance adoption, retrieval rank movement,
    knowledge-rule rendering, and grounding non-regression.
- `cited_field_chunk_share` / `cited_raw_page_chunk_share`: whether citations
  point at derived structured-field chunks or raw page chunks.
- `page_term_recall`, `field_term_recall`, and
  `combined_term_recall`: distinguish page-text hits from structured-field
  extraction hits.
- retrieval recall/precision/MRR: scores the persisted production
  `retrieved_evidence.json` (the same hybrid BM25 + `text-embedding-3-large`
  result the drafter actually saw) against gold filenames or evidence ids.
- A/B non-regression: per-side citation validity, quote-substring validity,
  claim citation rate, raw-page citation share, and optional edited-reference
  body/section similarity.
- Harness pass/fail: deterministic gates around citation validity,
  quote-substring grounding, and unsupported-section counts.
- Playbook risk: deterministic findings for missing preferred terms,
  escalation terms, unsupported draft sections, and evidence ids supporting
  the finding.

## Curated 20-Document Run

Artifacts: `outputs/all_categories/`.

Key results from `outputs/all_categories/evaluation.md`:

| Metric | Result |
|---|---:|
| Documents / pages | 20 / 70 |
| Rendered image pages / text-layer pages | 25 / 45 |
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

Interpretation:

- Processing covers all five messy categories plus a `truepdf` control set,
  and exercises both text-layer and image-based extraction paths.
- Structured-field extraction is strong on the gold term set; page-text
  recall sits below field recall, so the report does not conflate visual
  transcription with model-derived fields.
- The draft cites raw page chunks 81.25% of the time and synthetic
  `:fields` chunks 18.75%. Two retrieval diversity caps drive this:
  `max_field_chunks=1` (one structured-field slot per query) and
  `max_chunks_per_document=3` (no single source can fill the top-k).
- Retrieval recall/precision/MRR are scored against the persisted
  production `retrieved_evidence.json` — the same hybrid (BM25 +
  `text-embedding-3-large`) result the drafter actually saw. The 33.33%
  recall@k reflects the diversity-capped production retrieval over the
  20-doc corpus and the single multi-document gold query in
  `eval/gold_retrieval.json`. The numbers above are honest snapshots of
  one specific run; they drift between runs because GPT-5.5 generation is
  stochastic. The default evaluation run does not enable the optional
  Cohere reranker unless `PIPELINE_RERANK_PROVIDER=cohere` is set. The
  contract metrics (citation id validity, quote-substring validity, claim
  citation rate) are pinned at 100% by validators; the share/recall
  metrics are not.

## Public-Data Run

Artifacts: `eval/public-documents/` and `outputs/public_eval/`.

The public bundle downloads 13 external files, ~2-3 per messy category, with
byte, SHA-256, expected page count, and `max_pages=5` checks in
`eval/public_eval_manifest.json`. The bundle is strictly distinct from
`datasets/`: zero filename and zero SHA-256 overlap.

Key results from `outputs/public_eval/evaluation.md`:

| Metric | Result |
|---|---:|
| Documents / pages | 15 / 40 |
| Manifest matched files | 15 / 15 |
| Required categories present | true |
| Max pages per file | 5 |
| Eval-points required pass rate | 100.00% |
| Document-understanding required pass rate | 100.00% |
| Grounded-drafting required pass rate | 100.00% |
| Edit-improvement required pass rate | 100.00% |
| Citation id validity | 100.00% |
| Claim citation coverage | 100.00% |
| Claim grounding pass rate | 88.89% |
| Raw-page citation share | 100.00% |
| Simulated edit-improvement pass rate | 100.00% |

This is not meant to replace the curated corpus. It proves the pipeline can
materialize, verify, run, and evaluate external public documents without relying
on only hand-curated local examples.

## A/B Edit Loop

Artifacts: `outputs/all_categories/operator_profile.json`,
`outputs/all_categories/edits.jsonl`,
`outputs/all_categories/learn_suggestions.json`,
`outputs/all_categories/ab_eval.md`, and
`outputs/all_categories/ab_drafts/`.

Current A/B result.

Three preference edits across distinct document categories (terse/cautious
restructure, per-document/page-cited, full-prose compliance-review) plus
one `knowledge_rule` and one `evidence_dispute` are captured into
`state/`. The with-profile draft sees the union: operator profile,
knowledge layer, and the per-category exemplar matching the dominant
retrieved-document category. The dispute populates
`state/retrieval_feedback.json`; `retrieve()` now consumes its evidence
boosts/demotes as score adjustments before candidate selection.

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
| Edited-reference section F1 vs messy_case, without / with | 0.00% / 0.00% |

Interpretation:

- The profile improves two proxy signals on this run — section labels
  (+1) and caution tone (+3 signed) — but does not improve
  edited-reference body similarity or section F1, both of which are
  flat or slightly worse with profile vs. without
  (`ab_eval.md` "## Notes" calls this out explicitly).
- "Future results" means later pipeline artifacts and metrics: generated
  drafts, selected evidence, unsupported flags, claim-citation rate, and
  edited-reference similarity. It does not mean legal outcomes or autonomous
  legal decisions.
- This is human preference feedback, not a RAG self-improvement signal. The
  grounding contract is evaluated separately through citation id validity,
  quote-substring validity, claim citation rate, and raw-page citation share.
- `edits.jsonl` is the source-of-truth edit log. `operator_profile.json` is
  derived preference state. `learn_suggestions.json` is not scored until a
  human promotes it into `eval/gold_documents.json` or
  `eval/gold_retrieval.json`.
- `outputs/all_categories/ab_drafts/with_profile_draft.md` is the concrete
  future-draft artifact produced with learned guidance.
- The honest signal of the multi-edit run on this snapshot is
  **section-label adoption (+1)** and **caution-tone shift (+3
  signed)** while citation id validity, quote-substring validity, and
  claim citation rate all stay 100% on both sides (the A/B harness
  reports claim-grounding pass rate as `not available` here because
  the claim graph for the case-fact-summary draft type is not
  populated in this run).
  Verbatim preferred-phrase adoption is 0% on this run; a prior run on
  the same inputs reported 20%. The earlier "80% preferred-phrase
  adoption" number was citation-token pollution in `_preferred_phrases`
  — `[d61bba9a2aabfec3:p4:c0]`-style bracket strings were being
  captured as phrases. The fix strips citation tokens before phrase
  extraction and adds a defense-in-depth guard in
  `_is_reusable_phrase`. Verbatim phrase reuse stays the noisiest of
  the three preference signals; section labels and caution tone are the
  more reliable measurements.
- The per-document retrieval-diversity cap (`max_chunks_per_document=3`)
  may raise the ceiling for verbatim operator-style prose by surfacing
  more candidate source spans, but it does not guarantee a non-zero
  hit per run (this run: 0%; prior run on same inputs: 20%). Pushing
  verbatim adoption to a consistent non-zero floor would likely
  require LoRA SFT on captured pairs.
- Section F1 against the messy_case_edited_memo reference is 0% / 0%
  on this run; a prior run landed at 0% / 13.33%. The profile pulls
  structure toward the operator's edited reference some of the time;
  absolute F1 stays low because the multi-edit profile is intentionally
  not overfit to any single reference.
- An earlier ab_eval reported a claim-citation-rate regression (80% →
  62.5%). That was a sentence-splitter mismatch between drafting and
  evaluation (drafting attached trailing citations before splitting;
  evaluation did not), and an abbreviation handling bug (`9 a.m.`, `v.`,
  `U.S.C.`, etc. produced spurious sentence boundaries that inflated the
  uncited-factual count). The two splitters now share a single
  implementation in `drafting.sentence_units` and protect abbreviations
  before splitting.
- The guidance renderer appends a citation-preservation clause so a
  single terse edit cannot teach "collapse + drop citations" — every
  shortened section must keep grounding for any retained factual claim.
- A production-grade improvement claim would still aggregate held-out
  A/B runs over many edited cases and run statistical tests rather than
  reporting deltas on a single curated case.

## Research Decision

Recent RAG-evaluation work reinforces the design choice to keep deterministic
artifact checks as the core. CUE-R (arXiv:2604.05467, April 2026) argues that
final-answer and citation-faithfulness scores miss per-evidence utility; it
evaluates evidence by perturbing retrieved items and measuring behavior changes.
That maps directly to a future extension here: remove or replace one evidence
chunk, regenerate, and measure whether required claims, grounding, and confidence
change.

GroundedKG-RAG (arXiv:2604.04359, April 2026) reaches a similar conclusion from
the long-document side: interpretable units grounded in original source
sentences make auditing and error analysis easier. That is why this repository
keeps citation ids, verbatim citation quotes, draft warnings, `edits.jsonl`, and
profile/no-profile A/B draft artifacts rather than relying only on final-answer
scores or judge summaries.

Libraries and platforms such as Ragas, DeepEval, Langfuse, Phoenix, Argilla,
Label Studio, LangChain, LlamaIndex, Haystack, and DSPy provide useful pieces:
LLM-judge metrics, trace UIs, annotation queues, RAG abstractions, and prompt
optimization. They are not the primary implementation here because this
take-home also needs image-based extraction routing, public-data integrity, structured-field
extraction, strict quote validation, and human operator-edit reuse in one
inspectable workflow. Those are easier to defend with persisted artifacts and
small deterministic gold checks. Observability or LLM-judge tooling can be
added later as an optional second opinion, not the source of truth.
