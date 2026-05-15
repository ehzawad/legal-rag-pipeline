from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from pipeline.io import read_json


SUPPORTED_PUBLIC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}


@dataclass(frozen=True, slots=True)
class PublicEvalItem:
    id: str
    category: str
    url: str
    filename: str
    source: str
    license: str
    sha256: str = ""
    bytes: int | None = None
    pages_expected: int | None = None
    max_pages: int | None = None
    notes: list[str] = field(default_factory=list)
    default: bool = True


@dataclass(frozen=True, slots=True)
class PublicEvalManifest:
    version: int
    name: str
    description: str
    license_notes: str
    task: str
    items: list[PublicEvalItem]
    golden_checks: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicDataItemStatus:
    id: str
    filename: str
    status: str
    path: str
    bytes: int = 0
    sha256: str = ""
    pages: int | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class PublicDataResult:
    manifest: str
    dataset_dir: str
    item_count: int
    downloaded: int
    reused: int
    failed: int
    items: list[PublicDataItemStatus]
    notes: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def load_public_eval_manifest(path: Path) -> PublicEvalManifest:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise ValueError(f"Unsupported public eval manifest version: {version}")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Manifest must include a non-empty items list")

    items: list[PublicEvalItem] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("Every manifest item must be an object")
        item = PublicEvalItem(
            id=_required_str(raw, "id"),
            category=_required_str(raw, "category"),
            url=_required_str(raw, "url"),
            filename=_required_str(raw, "filename"),
            source=_required_str(raw, "source"),
            license=_required_str(raw, "license"),
            sha256=str(raw.get("sha256") or "").strip().lower(),
            bytes=_optional_int(raw.get("bytes")),
            pages_expected=_optional_int(raw.get("pages_expected")),
            max_pages=_optional_int(raw.get("max_pages")),
            notes=[str(note) for note in raw.get("notes", []) or []],
            default=bool(raw.get("default", True)),
        )
        _validate_item(item)
        if item.id in seen_ids:
            raise ValueError(f"Duplicate manifest item id: {item.id}")
        if item.filename in seen_names:
            raise ValueError(f"Duplicate manifest filename: {item.filename}")
        seen_ids.add(item.id)
        seen_names.add(item.filename)
        items.append(item)

    return PublicEvalManifest(
        version=version,
        name=_required_str(payload, "name"),
        description=str(payload.get("description") or ""),
        license_notes=str(payload.get("license_notes") or ""),
        task=str(payload.get("task") or "first-pass internal memo"),
        items=items,
        golden_checks=dict(payload.get("golden_checks") or {}),
    )


def download_public_eval_set(
    manifest_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    include_optional: bool = False,
) -> PublicDataResult:
    manifest = load_public_eval_manifest(manifest_path)
    selected = [item for item in manifest.items if include_optional or item.default]
    statuses: list[PublicDataItemStatus] = []
    downloaded = 0
    reused = 0
    failed = 0

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for item in selected:
        target = _safe_target(output_dir, item.filename)
        if dry_run:
            statuses.append(
                PublicDataItemStatus(
                    id=item.id,
                    filename=item.filename,
                    status="dry-run",
                    path=str(target),
                    message=item.url,
                )
            )
            continue

        if target.exists() and not force:
            status = _verify_file(item, target)
            if status.status == "ok":
                statuses.append(
                    PublicDataItemStatus(
                        id=item.id,
                        filename=item.filename,
                        status="reused",
                        path=str(target),
                        bytes=status.bytes,
                        sha256=status.sha256,
                        pages=status.pages,
                    )
                )
                reused += 1
                continue

        try:
            _download_one(item.url, target)
            status = _verify_file(item, target)
            if status.status != "ok":
                failed += 1
                statuses.append(status)
                continue
            downloaded += 1
            statuses.append(
                PublicDataItemStatus(
                    id=item.id,
                    filename=item.filename,
                    status="downloaded",
                path=str(target),
                bytes=status.bytes,
                sha256=status.sha256,
                pages=status.pages,
            )
            )
        except Exception as exc:  # noqa: BLE001 - CLI should report all item failures.
            failed += 1
            statuses.append(
                PublicDataItemStatus(
                    id=item.id,
                    filename=item.filename,
                    status="failed",
                    path=str(target),
                    message=str(exc),
                )
            )

    notes = []
    optional_count = len(manifest.items) - len(selected)
    if optional_count:
        notes.append(f"{optional_count} optional item(s) skipped; pass include_optional=True to fetch them.")
    return PublicDataResult(
        manifest=str(manifest_path),
        dataset_dir=str(output_dir),
        item_count=len(selected),
        downloaded=downloaded,
        reused=reused,
        failed=failed,
        items=statuses,
        notes=notes,
    )


def verify_public_eval_set(
    manifest_path: Path,
    dataset_dir: Path,
    *,
    include_optional: bool = False,
) -> PublicDataResult:
    manifest = load_public_eval_manifest(manifest_path)
    selected = [item for item in manifest.items if include_optional or item.default]
    statuses: list[PublicDataItemStatus] = []
    ok = 0
    failed = 0
    for item in selected:
        target = _safe_target(dataset_dir, item.filename)
        status = _verify_file(item, target)
        if status.status == "ok":
            ok += 1
        else:
            failed += 1
        statuses.append(status)
    return PublicDataResult(
        manifest=str(manifest_path),
        dataset_dir=str(dataset_dir),
        item_count=len(selected),
        downloaded=0,
        reused=ok,
        failed=failed,
        items=statuses,
    )


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Manifest field is required: {key}")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    integer = int(value)
    if integer < 0:
        raise ValueError("Integer manifest fields must be non-negative")
    return integer


def _validate_item(item: PublicEvalItem) -> None:
    parsed = urlparse(item.url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError(f"Unsupported URL scheme for {item.id}: {parsed.scheme}")
    rel = Path(item.filename)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"Manifest filename must be a safe relative path: {item.filename}")
    if rel.suffix.lower() not in SUPPORTED_PUBLIC_EXTENSIONS:
        raise ValueError(
            f"Unsupported manifest file extension for {item.filename}; "
            f"allowed: {', '.join(sorted(SUPPORTED_PUBLIC_EXTENSIONS))}"
        )
    if item.sha256 and not re_fullmatch_sha256(item.sha256):
        raise ValueError(f"sha256 must be 64 lowercase hex characters for {item.id}")


def re_fullmatch_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _safe_target(root: Path, filename: str) -> Path:
    root_resolved = root.resolve()
    target = (root / filename).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside dataset dir: {filename}") from exc
    return target


def _download_one(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    parsed = urlparse(url)
    hasher = hashlib.sha256()
    byte_count = 0
    with tmp_path.open("wb") as handle:
        if parsed.scheme == "file":
            source = Path(unquote(parsed.path))
            with source.open("rb") as src:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    byte_count += len(block)
                    hasher.update(block)
                    handle.write(block)
        else:
            request = Request(url, headers={"User-Agent": "a-rag-pipeline-public-eval/0.1 contact@example.com"})
            with urlopen(request, timeout=120) as response:  # noqa: S310 - manifest controls URLs.
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    byte_count += len(block)
                    hasher.update(block)
                    handle.write(block)
    if byte_count == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded zero bytes from {url}")
    os.replace(tmp_path, target)


def _verify_file(item: PublicEvalItem, target: Path) -> PublicDataItemStatus:
    if not target.exists():
        return PublicDataItemStatus(
            id=item.id,
            filename=item.filename,
            status="missing",
            path=str(target),
            message="file is not present",
        )
    digest = _sha256_file(target)
    size = target.stat().st_size
    if item.bytes is not None and size != item.bytes:
        return PublicDataItemStatus(
            id=item.id,
            filename=item.filename,
            status="failed",
            path=str(target),
            bytes=size,
            sha256=digest,
            pages=_page_count(target),
            message=f"byte count mismatch: expected {item.bytes}, got {size}",
        )
    if item.sha256 and digest != item.sha256:
        return PublicDataItemStatus(
            id=item.id,
            filename=item.filename,
            status="failed",
            path=str(target),
            bytes=size,
            sha256=digest,
            pages=_page_count(target),
            message="sha256 mismatch",
        )
    pages = _page_count(target)
    if item.pages_expected is not None and pages != item.pages_expected:
        return PublicDataItemStatus(
            id=item.id,
            filename=item.filename,
            status="failed",
            path=str(target),
            bytes=size,
            sha256=digest,
            pages=pages,
            message=f"page count mismatch: expected {item.pages_expected}, got {pages}",
        )
    if item.max_pages is not None and pages is not None and pages > item.max_pages:
        return PublicDataItemStatus(
            id=item.id,
            filename=item.filename,
            status="failed",
            path=str(target),
            bytes=size,
            sha256=digest,
            pages=pages,
            message=f"page count {pages} exceeds max_pages {item.max_pages}",
        )
    return PublicDataItemStatus(
        id=item.id,
        filename=item.filename,
        status="ok",
        path=str(target),
        bytes=size,
        sha256=digest,
        pages=pages,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _page_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found]

            doc = fitz.open(path)
            try:
                return int(doc.page_count)
            finally:
                doc.close()
        except Exception:
            return None
    if suffix in {".png", ".jpg", ".jpeg", ".txt"}:
        return 1
    return None
