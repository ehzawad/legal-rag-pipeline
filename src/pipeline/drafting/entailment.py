from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import Any

from pipeline.config import ProviderConfig, resolve_provider_name
from pipeline.providers import ProviderUnavailable, parse_json_response, responses_create
from pipeline.schemas import CaseFactSummary, EvidenceChunk, FactClaim


def judge_claim_entailment(
    summary: CaseFactSummary,
    *,
    provider: str | None = None,
    model: str | None = None,
    config: ProviderConfig | None = None,
    generator: Callable[[str], str] | None = None,
    max_calls: int = 8,
) -> dict[str, dict[str, Any]]:
    """Return optional semantic entailment judgments keyed by claim id."""

    claims = [
        claim
        for claim in summary.claims
        if claim.claim_type != "scaffolding" and claim.citations and _all_quotes_grounded(claim)
    ]
    if not claims or max_calls <= 0:
        return {}
    config = config or ProviderConfig.from_env()
    resolved_provider = resolve_provider_name(provider, fallback=config.generation_provider, reject_test_names=False)
    evidence_by_id = {chunk.evidence_id: chunk for chunk in summary.evidence}
    results: dict[str, dict[str, Any]] = {}
    for batch in _batches(claims, size=max(1, len(claims) // min(max_calls, len(claims)) or 1)):
        prompt = _entailment_prompt(batch, evidence_by_id)
        response_text = _call_judge(
            prompt,
            provider=resolved_provider,
            model=model or config.openai_model,
            reasoning_effort=config.openai_reasoning_effort,
            generator=generator,
        )
        payload = parse_json_response(response_text)
        for item in _result_items(payload):
            claim_id = str(item.get("claim_id") or "").strip()
            if claim_id:
                results[claim_id] = dict(item)
    return results


def _call_judge(
    prompt: str,
    *,
    provider: str,
    model: str,
    reasoning_effort: str,
    generator: Callable[[str], str] | None,
) -> str:
    if generator is not None:
        return generator(prompt)
    if provider != "openai":
        raise ProviderUnavailable(f"Unsupported entailment judge provider: {provider!r}")
    return responses_create(prompt, model, reasoning_effort=reasoning_effort)


def _entailment_prompt(claims: Sequence[FactClaim], evidence_by_id: dict[str, EvidenceChunk]) -> str:
    payload = []
    for claim in claims:
        payload.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "evidence": [
                    {
                        "evidence_id": citation.evidence_id,
                        "quote": citation.quote,
                        "chunk_text": evidence_by_id.get(citation.evidence_id).text
                        if evidence_by_id.get(citation.evidence_id)
                        else "",
                    }
                    for citation in claim.citations
                ],
            }
        )
    return (
        "You are a grounding judge. For each claim, decide whether the cited evidence entails, "
        "contradicts, does not support, or leaves unclear the claim. Use only the supplied evidence. "
        "Return strict JSON only with shape {\"results\":[{\"claim_id\":\"...\","
        "\"verdict\":\"entailed|contradicted|not_supported|unclear\",\"reason\":\"short reason\"}]}.\n\n"
        f"Claims JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def _result_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _all_quotes_grounded(claim: FactClaim) -> bool:
    return bool(claim.citations) and all(citation.substring_grounded for citation in claim.citations)


def _batches(items: Sequence[FactClaim], *, size: int) -> list[list[FactClaim]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]
