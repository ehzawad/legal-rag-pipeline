from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.learning.edit_memory import (
    load_edit_memory,
    render_edit_memory_suggestions,
    retrieve_edit_memory,
)
from pipeline.orchestration.artifacts import PipelinePaths
from pipeline.config import PipelineFeatures
from pipeline.io import write_json
from pipeline.schemas import EvidenceChunk, ProcessedDocument


class LearningGuidanceComponent:
    name = "learning_guidance"

    def build(
        self,
        *,
        processed: list[ProcessedDocument],
        evidence: list[EvidenceChunk],
        paths: PipelinePaths,
        profile_path: Path,
        state_paths: dict[str, Path],
        learning_api: dict[str, Any],
        features: PipelineFeatures,
    ) -> tuple[str, dict[str, str]]:
        if not features.learning_guidance:
            paths.learned_guidance.unlink(missing_ok=True)
            return "", {}

        guidance_parts: list[str] = []
        guidance_sources: dict[str, str] = {}

        if features.operator_profile_guidance and profile_path.exists():
            profile = learning_api["load_profile"](profile_path)
            profile_text = learning_api["profile_guidance"](profile)
            if profile_text:
                guidance_parts.append(profile_text)
                guidance_sources["operator_profile"] = str(profile_path)

        knowledge_path = state_paths["knowledge_layer"]
        if features.knowledge_layer_guidance and knowledge_path.exists():
            knowledge_layer = learning_api["load_knowledge_layer"](knowledge_path)
            knowledge_text = learning_api["render_knowledge_layer_guidance"](knowledge_layer)
            if knowledge_text:
                guidance_parts.append(knowledge_text)
                guidance_sources["knowledge_layer"] = str(knowledge_path)

        if features.exemplar_guidance:
            category_for_exemplars = dominant_category(processed, evidence)
            if category_for_exemplars:
                exemplars = learning_api["load_exemplars_for_category"](
                    state_paths["exemplars_dir"],
                    category_for_exemplars,
                    limit=1,
                )
                exemplars_text = learning_api["render_exemplars_guidance"](exemplars)
                if exemplars_text:
                    guidance_parts.append(exemplars_text)
                    guidance_sources["exemplars_category"] = category_for_exemplars

        edit_memory_path = state_paths.get("edit_memory")
        if edit_memory_path and edit_memory_path.exists():
            category_for_memory = dominant_category(processed, evidence)
            evidence_context = "\n".join(
                " ".join(chunk.text.split())[:600] for chunk in evidence[:6]
            )
            memory = load_edit_memory(edit_memory_path)
            memory_matches = retrieve_edit_memory(
                memory,
                draft_text=evidence_context,
                category=category_for_memory,
                limit=5,
            )
            memory_text = render_edit_memory_suggestions(memory_matches)
            if memory_text:
                guidance_parts.append(memory_text)
                guidance_sources["edit_memory"] = str(edit_memory_path)

        guidance = "\n\n".join(guidance_parts)
        if guidance:
            write_json(
                paths.learned_guidance,
                {
                    "profile_path": str(profile_path),
                    "guidance": guidance,
                    "sources": guidance_sources,
                },
            )
        else:
            paths.learned_guidance.unlink(missing_ok=True)
        return guidance, guidance_sources


def dominant_category(
    processed: list[ProcessedDocument],
    evidence: list[EvidenceChunk],
) -> str:
    by_doc_id: dict[str, str] = {}
    for document in processed:
        try:
            source_path = Path(document.source.path)
        except Exception:
            continue
        parts = source_path.parts
        category = ""
        for index, part in enumerate(parts):
            if part == "datasets" and index + 1 < len(parts):
                category = parts[index + 1]
                if category in {"public-eval", "public"} and index + 2 < len(parts):
                    category = parts[index + 2]
                break
        if not category and len(parts) >= 2:
            category = parts[-2]
        if category:
            by_doc_id[document.source.document_id] = category

    if not by_doc_id:
        return ""

    counts: dict[str, int] = {}
    for chunk in evidence:
        category = by_doc_id.get(chunk.document_id)
        if category:
            counts[category] = counts.get(category, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
