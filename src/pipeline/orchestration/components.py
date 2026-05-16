from __future__ import annotations

from dataclasses import dataclass

from pipeline.drafting.component import CaseFactSummaryDraftingComponent
from pipeline.ingestion.component import DocumentProcessingComponent
from pipeline.learning.guidance import LearningGuidanceComponent
from pipeline.retrieval.component import EvidenceRetrievalComponent


@dataclass(slots=True)
class PipelineComponents:
    processing: DocumentProcessingComponent
    retrieval: EvidenceRetrievalComponent
    guidance: LearningGuidanceComponent
    drafting: CaseFactSummaryDraftingComponent
