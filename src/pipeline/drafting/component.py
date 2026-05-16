from __future__ import annotations

from dataclasses import replace
from typing import Callable

from pipeline.orchestration.artifacts import PipelinePaths, load_draft
from pipeline.config import PipelineFeatures, ProviderConfig
from pipeline.drafting.entailment import judge_claim_entailment
from pipeline.drafting.grounding import apply_claim_grounding
from pipeline.drafting.memo import sections_from_case_summary
from pipeline.drafting.validation import validate_draft_contract
from pipeline.io import write_json
from pipeline.schemas import Draft, EvidenceChunk, EvidencePack, ProcessedDocument, to_jsonable


class CaseFactSummaryDraftingComponent:
    name = "generate_draft"

    def __init__(
        self,
        *,
        generate_case_fact_summary: Callable[..., Draft],
        render_draft_markdown: Callable[[Draft], str],
    ) -> None:
        self._generate_case_fact_summary = generate_case_fact_summary
        self._render_draft_markdown = render_draft_markdown

    def run(
        self,
        *,
        processed: list[ProcessedDocument],
        evidence: list[EvidenceChunk],
        evidence_pack: EvidencePack | None = None,
        task: str,
        guidance: str,
        paths: PipelinePaths,
        config: ProviderConfig,
        features: PipelineFeatures,
        case_id: str = "",
    ) -> Draft:
        last_error: ValueError | None = None
        attempt_guidance = guidance
        for attempt in range(2):
            draft = self._generate_case_fact_summary(
                processed_documents=processed,
                evidence=evidence,
                task=task,
                learned_guidance=attempt_guidance,
                provider=config.generation_provider,
                config=config,
                claim_support_check=features.claim_support_check,
                draft_type=config.draft_type,
                evidence_pack=evidence_pack,
                case_id=case_id,
            )
            if draft.case_summary is None:
                raise ValueError("Draft adapter did not return the required CaseFactSummary")
            summary, grounding_report = apply_claim_grounding(draft.case_summary, draft_type=draft.draft_type)
            if features.claim_entailment_judge:
                entailment = judge_claim_entailment(summary, provider=config.generation_provider, config=config)
                summary, grounding_report = apply_claim_grounding(
                    summary,
                    draft_type=draft.draft_type,
                    entailment_results=entailment,
                )
            draft = replace(
                draft,
                case_summary=summary,
                sections=sections_from_case_summary(summary),
            )
            try:
                validate_draft_contract(
                    draft,
                    grounding_report,
                    require_entailment=features.claim_entailment_judge,
                )
                break
            except ValueError as exc:
                last_error = exc
                if attempt:
                    raise
                attempt_guidance = _with_validation_repair_guidance(guidance, exc)
        else:
            raise last_error or ValueError("Draft contract failed")
        write_json(paths.case_fact_summary, to_jsonable(summary))
        write_json(paths.grounding_report, grounding_report)
        write_json(paths.draft_json, to_jsonable(draft))
        paths.draft_markdown.write_text(self._render_draft_markdown(draft), encoding="utf-8")
        return draft

    def load(self, paths: PipelinePaths) -> Draft:
        return load_draft(paths.draft_json)


def _with_validation_repair_guidance(guidance: str, error: ValueError) -> str:
    repair = (
        "Mandatory validation repair: the previous draft was discarded before artifact write because "
        f"{error}. Regenerate every factual claim with citation quotes copied exactly from the supplied "
        "evidence JSON. If an exact quote is not available, move that item to open_questions instead of "
        "writing it as a factual claim."
    )
    return f"{guidance.rstrip()}\n\n{repair}" if guidance.strip() else repair
