from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_DRAFT_TYPE = "case_fact_summary"
CASE_FACT_SUMMARY_ADAPTER = "case_fact_summary_claims_v1"


class DraftSpecError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DraftResponseAdapter:
    id: str
    version: str
    response_shape: str


@dataclass(frozen=True, slots=True)
class DraftSpec:
    id: str
    version: str
    adapter_id: str
    prompt_template: str

    @property
    def content_digest(self) -> str:
        adapter = _ADAPTERS.get(self.adapter_id)
        payload = {
            "id": self.id,
            "version": self.version,
            "adapter": {
                "id": adapter.id,
                "version": adapter.version,
                "response_shape": adapter.response_shape,
            }
            if adapter is not None
            else {"id": self.adapter_id, "missing": True},
            "prompt_template": self.prompt_template,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint_payload(self) -> dict[str, str]:
        return {
            "id": self.id,
            "version": self.version,
            "content_digest": self.content_digest,
        }

    def build_prompt(
        self,
        *,
        task_payload: Mapping[str, Any],
        evidence_payload: Sequence[Mapping[str, Any]],
        learned_guidance: str,
        computed_unsupported: Sequence[str],
        evidence_pack_payload: Mapping[str, Any],
    ) -> str:
        return self.prompt_template.format(
            task_json=json.dumps(dict(task_payload), ensure_ascii=True),
            guidance=learned_guidance,
            unsupported_json=json.dumps(list(computed_unsupported), ensure_ascii=True),
            evidence_pack_json=json.dumps(dict(evidence_pack_payload), ensure_ascii=True),
            evidence_json=json.dumps(list(evidence_payload), ensure_ascii=True),
        )


_CASE_FACT_SUMMARY_PROMPT = (
    "You are drafting a grounded case fact summary for operator review. "
    "Use only the supplied evidence. The canonical output is a claim graph, not Markdown. "
    "Do not invent claim_ids; the system assigns stable claim ids after parsing. "
    "Every claim in claims must cite at least one evidence_id from the evidence JSON "
    "and include a short verbatim quote for each citation. Each quote must appear inside the cited "
    "chunk text after whitespace/case normalization. If evidence is unclear, set confidence to low. "
    "Write each quote in the citations array as {{\"evidence_id\":\"...\",\"quote\":\"...\"}}. "
    "If a useful fact is unsupported, place it in open_questions rather than as a factual claim. "
    "Do not infer legal conclusions such as breach, fault, default, liability, or legal sufficiency "
    "unless a cited source says that directly. Learned guidance may affect tone or section preferences, "
    "but it must not remove citations. Choose sections dynamically from the supplied document type, "
    "evidence, and task. Do not force litigation-only sections such as procedural posture or relief "
    "sought when the document is a deed, probate record, notice, contract, filing, form, checklist, "
    "or other non-litigation record. Warnings should be brief operator review flags about this run's "
    "extraction, citation, or source-quality issues; do not restate broad learned guidance or "
    "knowledge-layer rules unless the supplied evidence directly triggers them.\n\n"
    "Return strict JSON only with this shape:\n"
    "{{"
    '"title":"Case Fact Summary",'
    '"claims":[{{"section":"short_snake_case_section_id_chosen_for_this_document_type",'
    '"text":"one factual claim","claim_type":"fact|summary",'
    '"confidence":"high|medium|low",'
    '"citations":[{{"evidence_id":"E1","quote":"verbatim substring from E1"}}]}}],'
    '"open_questions":["..."],'
    '"warnings":["..."]'
    "}}\n\n"
    "Drafting task JSON:\n{task_json}\n\n"
    "Learned guidance:\n{guidance}\n\n"
    "Unsupported or unclear facts that must be preserved as open questions:\n"
    "{unsupported_json}\n\n"
    "Evidence pack JSON:\n{evidence_pack_json}\n\n"
    "Evidence JSON:\n{evidence_json}"
)


_ADAPTERS = {
    CASE_FACT_SUMMARY_ADAPTER: DraftResponseAdapter(
        id=CASE_FACT_SUMMARY_ADAPTER,
        version="1",
        response_shape="case_fact_summary_claims",
    )
}

_SPECS = {
    DEFAULT_DRAFT_TYPE: DraftSpec(
        id=DEFAULT_DRAFT_TYPE,
        version="1",
        adapter_id=CASE_FACT_SUMMARY_ADAPTER,
        prompt_template=_CASE_FACT_SUMMARY_PROMPT,
    )
}


def normalize_draft_type(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9_:-]+", "_", str(value or "").strip().casefold()).strip("_")
    return normalized or DEFAULT_DRAFT_TYPE


def resolve_draft_spec(value: str | None) -> DraftSpec:
    draft_type = normalize_draft_type(value)
    try:
        return _SPECS[draft_type]
    except KeyError as exc:
        raise DraftSpecError(
            f"Unknown draft_type {value!r}. Registered draft types: {', '.join(sorted(_SPECS))}"
        ) from exc


def resolve_draft_adapter(spec: DraftSpec) -> DraftResponseAdapter:
    try:
        return _ADAPTERS[spec.adapter_id]
    except KeyError as exc:
        raise DraftSpecError(
            f"Draft spec {spec.id!r} references unknown adapter {spec.adapter_id!r}"
        ) from exc


def draft_spec_fingerprint(value: str | None) -> dict[str, str]:
    return resolve_draft_spec(value).fingerprint_payload()


def registered_draft_types() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))
