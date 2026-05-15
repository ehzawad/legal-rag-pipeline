from __future__ import annotations

from dataclasses import replace
from typing import Callable

from pipeline.orchestration.artifacts import PipelinePaths, load_draft
from pipeline.config import PipelineFeatures, ProviderConfig
from pipeline.drafting.entailment import judge_claim_entailment
from pipeline.drafting.grounding import apply_claim_grounding
from pipeline.drafting.memo import sections_from_case_summary
from pipeline.io import write_json
from pipeline.schemas import Draft, EvidenceChunk, EvidencePack, ProcessedDocument, to_jsonable


class MemoDraftingComponent:
    name = "generate_draft"

    def __init__(
        self,
        *,
        generate_internal_memo: Callable[..., Draft],
        render_draft_markdown: Callable[[Draft], str],
    ) -> None:
        self._generate_internal_memo = generate_internal_memo
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
        draft = self._generate_internal_memo(
            processed_documents=processed,
            evidence=evidence,
            task=task,
            learned_guidance=guidance,
            provider=config.generation_provider,
            config=config,
            claim_support_check=features.claim_support_check,
            claim_first_drafting=features.claim_first_drafting,
            evidence_pack=evidence_pack,
            case_id=case_id,
        )
        grounding_report = None
        if draft.case_summary is not None:
            summary, grounding_report = apply_claim_grounding(draft.case_summary)
            if features.claim_entailment_judge:
                entailment = judge_claim_entailment(summary, provider=config.generation_provider, config=config)
                summary, grounding_report = apply_claim_grounding(summary, entailment_results=entailment)
            draft = replace(
                draft,
                case_summary=summary,
                sections=sections_from_case_summary(summary),
            )
            write_json(paths.case_fact_summary, to_jsonable(summary))
            write_json(paths.grounding_report, grounding_report)
        write_json(paths.draft_json, to_jsonable(draft))
        paths.draft_markdown.write_text(self._render_draft_markdown(draft), encoding="utf-8")
        return draft

    def load(self, paths: PipelinePaths) -> Draft:
        return load_draft(paths.draft_json)
