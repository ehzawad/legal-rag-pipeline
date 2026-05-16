from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.io import write_json
from pipeline.orchestration.artifacts import PipelinePaths
from pipeline.orchestration.run import _load_valid_draft_checkpoint, _try_load_valid_draft_checkpoint
from pipeline.schemas import CaseFactSummary, ClaimCitation, Draft, EvidenceChunk, FactClaim, GroundingVerdict, to_jsonable


def _invalid_draft() -> Draft:
    return Draft(
        draft_type="case_fact_summary",
        title="Case Fact Summary",
        generated_at="2026-05-17T00:00:00+00:00",
        sections=[],
        evidence=[],
        case_summary=None,
    )


def _valid_draft() -> Draft:
    chunk = EvidenceChunk(
        evidence_id="E1",
        document_id="doc1",
        filename="doc1.pdf",
        page_number=1,
        text="The agreement requires Acme to deliver final copies by May 1.",
    )
    claim = FactClaim(
        claim_id="claim1",
        section_id="material_facts",
        section="Material Facts",
        text="Acme must deliver final copies by May 1.",
        claim_type="fact",
        confidence="high",
        citations=[ClaimCitation(evidence_id="E1", quote="requires Acme to deliver final copies by May 1")],
        grounding=GroundingVerdict(
            status="grounded",
            reason="All citation quotes are grounded in cited evidence.",
            cited_evidence_ids=["E1"],
            substring_grounded_count=1,
            action="keep",
        ),
    )
    summary = CaseFactSummary(
        case_id="case1",
        generated_at="2026-05-17T00:00:00+00:00",
        title="Case Fact Summary",
        section_order=["material_facts"],
        claims=[claim],
        evidence=[chunk],
    )
    return Draft(
        draft_type="case_fact_summary",
        title="Case Fact Summary",
        generated_at="2026-05-17T00:00:00+00:00",
        sections=[],
        evidence=[chunk],
        case_summary=summary,
    )


def test_resume_cache_does_not_reuse_invalid_draft_checkpoint(tmp_path):
    paths = PipelinePaths.for_output_dir(tmp_path)
    write_json(paths.grounding_report, {"unsupported_claim_count": 0})
    components = SimpleNamespace(drafting=SimpleNamespace(load=lambda _paths: _invalid_draft()))

    result = _try_load_valid_draft_checkpoint(
        components=components,
        paths=paths,
        require_entailment=False,
    )

    assert result is None


def test_disabled_drafting_fails_loudly_on_invalid_checkpoint(tmp_path):
    paths = PipelinePaths.for_output_dir(tmp_path)
    write_json(paths.grounding_report, {"unsupported_claim_count": 0})
    write_json(paths.case_fact_summary, {"case_id": "case1", "claims": []})
    components = SimpleNamespace(drafting=SimpleNamespace(load=lambda _paths: _invalid_draft()))

    with pytest.raises(ValueError, match="missing CaseFactSummary"):
        _load_valid_draft_checkpoint(
            components=components,
            paths=paths,
            require_entailment=False,
        )


def test_resume_cache_requires_case_fact_summary_artifact(tmp_path):
    paths = PipelinePaths.for_output_dir(tmp_path)
    draft = _valid_draft()
    write_json(paths.grounding_report, {"unsupported_claim_count": 0, "citation_coverage": 1.0, "grounding_pass_rate": 1.0})
    components = SimpleNamespace(drafting=SimpleNamespace(load=lambda _paths: draft))

    with pytest.raises(ValueError, match="missing case_fact_summary.json"):
        _load_valid_draft_checkpoint(
            components=components,
            paths=paths,
            require_entailment=False,
        )


def test_resume_cache_rejects_stale_case_fact_summary_artifact(tmp_path):
    paths = PipelinePaths.for_output_dir(tmp_path)
    draft = _valid_draft()
    write_json(paths.grounding_report, {"unsupported_claim_count": 0, "citation_coverage": 1.0, "grounding_pass_rate": 1.0})
    stale_summary = to_jsonable(draft.case_summary)
    stale_summary["title"] = "Stale Case Fact Summary"
    write_json(paths.case_fact_summary, stale_summary)
    components = SimpleNamespace(drafting=SimpleNamespace(load=lambda _paths: draft))

    with pytest.raises(ValueError, match="does not match draft.json"):
        _load_valid_draft_checkpoint(
            components=components,
            paths=paths,
            require_entailment=False,
        )
