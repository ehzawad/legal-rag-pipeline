from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.io import read_json, write_json
from pipeline.orchestration.artifacts import processed_documents_from_json
from pipeline.schemas import ProcessedDocument, now_iso, to_jsonable


CORPUS_MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    document_id: str
    filename: str
    source_path: str
    stored_original_path: str
    sha256: str
    mime_type: str
    page_count: int
    field_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    version: int
    created_at: str
    corpus_dir: str
    parsed_documents_path: str
    entries: list[CorpusEntry]

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def write_corpus_store(
    *,
    input_dir: Path,
    processed_documents: list[ProcessedDocument],
    corpus_dir: Path,
) -> CorpusManifest:
    """Persist originals plus parsed document state under a corpus directory."""

    originals_dir = corpus_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = corpus_dir / "parsed_documents.json"
    write_json(parsed_path, to_jsonable(processed_documents))

    entries: list[CorpusEntry] = []
    for document in processed_documents:
        source = document.source
        source_path = Path(source.path)
        stored_path: Path | None = originals_dir / _stored_original_name(source.filename, source.sha256)
        if source_path.exists() and source_path.is_file():
            try:
                shutil.copy2(source_path, stored_path)
            except OSError:
                stored_path = None
        else:
            stored_path = None
        entries.append(
            CorpusEntry(
                document_id=source.document_id,
                filename=source.filename,
                source_path=_relative_or_absolute(source_path, input_dir),
                stored_original_path=str(stored_path.relative_to(corpus_dir)) if stored_path is not None else "",
                sha256=source.sha256,
                mime_type=source.mime_type,
                page_count=len(document.pages),
                field_count=len(document.fields),
                warnings=list(document.warnings),
            )
        )

    manifest = CorpusManifest(
        version=CORPUS_MANIFEST_VERSION,
        created_at=now_iso(),
        corpus_dir=str(corpus_dir),
        parsed_documents_path=str(parsed_path.relative_to(corpus_dir)),
        entries=entries,
    )
    write_json(corpus_dir / "manifest.json", manifest.to_jsonable())
    return manifest


def load_corpus_manifest(corpus_dir: Path) -> CorpusManifest:
    payload = read_json(corpus_dir / "manifest.json")
    entries = [CorpusEntry(**entry) for entry in payload.get("entries", [])]
    return CorpusManifest(
        version=int(payload["version"]),
        created_at=str(payload["created_at"]),
        corpus_dir=str(payload.get("corpus_dir") or corpus_dir),
        parsed_documents_path=str(payload["parsed_documents_path"]),
        entries=entries,
    )


def load_corpus_documents(corpus_dir: Path) -> list[ProcessedDocument]:
    manifest = load_corpus_manifest(corpus_dir)
    parsed_path = corpus_dir / manifest.parsed_documents_path
    return processed_documents_from_json(read_json(parsed_path))


def corpus_summary(corpus_dir: Path) -> dict[str, Any]:
    manifest = load_corpus_manifest(corpus_dir)
    return {
        "corpus_dir": str(corpus_dir),
        "documents": len(manifest.entries),
        "pages": sum(entry.page_count for entry in manifest.entries),
        "fields": sum(entry.field_count for entry in manifest.entries),
        "originals": sum(1 for entry in manifest.entries if entry.stored_original_path),
    }


def _stored_original_name(filename: str, sha256: str) -> str:
    suffix = Path(filename).suffix
    digest = (sha256 or "unknown")[:16]
    stem = Path(filename).stem or "document"
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)[:80]
    return f"{safe_stem}.{digest}{suffix}"


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
