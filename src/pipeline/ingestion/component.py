from __future__ import annotations

from pathlib import Path
from typing import Callable

from pipeline.orchestration.artifacts import PipelinePaths, load_processed_documents
from pipeline.config import ProviderConfig
from pipeline.corpus import write_corpus_store
from pipeline.io import write_json
from pipeline.schemas import ProcessedDocument, to_jsonable


class DocumentProcessingComponent:
    name = "process_documents"

    def __init__(self, process_directory: Callable[..., list[ProcessedDocument]]) -> None:
        self._process_directory = process_directory

    def run(
        self,
        *,
        input_dir: Path,
        paths: PipelinePaths,
        config: ProviderConfig,
    ) -> list[ProcessedDocument]:
        processed = self._process_directory(
            input_dir,
            provider=config.extraction_provider,
            config=config,
        )
        write_json(paths.processed_documents, to_jsonable(processed))
        write_corpus_store(
            input_dir=input_dir,
            processed_documents=processed,
            corpus_dir=paths.corpus_dir,
        )
        return processed

    def load(self, paths: PipelinePaths) -> list[ProcessedDocument]:
        return load_processed_documents(paths.processed_documents)
