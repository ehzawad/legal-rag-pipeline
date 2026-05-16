from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pipeline.config import ConfigError
from pipeline.io import write_json
from pipeline.schemas import now_iso


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    wait_seconds: float = 1.0
    backoff: float = 2.0
    max_wait_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.wait_seconds < 0:
            raise ValueError("wait_seconds must be non-negative")
        if self.backoff < 1:
            raise ValueError("backoff must be at least 1")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be non-negative")


@dataclass(slots=True)
class StageRecord:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    attempts: int
    artifacts: list[str] = field(default_factory=list)
    error: str = ""
    retryable: bool = False
    warnings: list[str] = field(default_factory=list)


class WorkflowStageError(RuntimeError):
    def __init__(self, stage_name: str, cause: BaseException):
        super().__init__(f"Workflow stage '{stage_name}' failed: {cause}")
        self.stage_name = stage_name
        self.cause = cause


class WorkflowRecorder:
    """Run named workflow stages and persist an inspectable manifest.

    This is intentionally smaller than a generic node graph. The legal-doc
    workflow is linear today, so the production need is durable stage records,
    retry boundaries, and artifact visibility.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.metadata = dict(metadata or {})
        self.retry_policy = retry_policy or RetryPolicy()
        self.records: list[StageRecord] = []
        self.created_at = now_iso()
        self.status = "running"
        self.finished_at = ""
        self._flush()

    def run_stage(
        self,
        name: str,
        operation: Callable[[], T],
        *,
        artifacts: list[Path] | None = None,
        retry_policy: RetryPolicy | None = None,
        warnings: list[str] | None = None,
    ) -> T:
        policy = retry_policy or self.retry_policy
        artifact_paths = [str(path) for path in artifacts or []]
        stage_warnings = list(warnings or [])
        delay = policy.wait_seconds

        for attempt in range(1, policy.max_attempts + 1):
            started_at = now_iso()
            monotonic_start = time.monotonic()
            try:
                result = operation()
            except BaseException as exc:
                retryable = _is_retryable_exception(exc)
                record = StageRecord(
                    name=name,
                    status="failed",
                    started_at=started_at,
                    finished_at=now_iso(),
                    duration_seconds=round(time.monotonic() - monotonic_start, 3),
                    attempts=attempt,
                    artifacts=artifact_paths,
                    error=f"{exc.__class__.__name__}: {exc}",
                    retryable=retryable,
                    warnings=stage_warnings,
                )
                self.records.append(record)
                if attempt >= policy.max_attempts or not retryable:
                    self.status = "failed"
                    self.finished_at = record.finished_at
                self._flush()
                if attempt >= policy.max_attempts or not retryable:
                    raise WorkflowStageError(name, exc) from exc
                if delay > 0:
                    time.sleep(delay)
                    delay = min(delay * policy.backoff, policy.max_wait_seconds)
                continue

            self.records.append(
                StageRecord(
                    name=name,
                    status="succeeded",
                    started_at=started_at,
                    finished_at=now_iso(),
                    duration_seconds=round(time.monotonic() - monotonic_start, 3),
                    attempts=attempt,
                    artifacts=artifact_paths,
                    warnings=stage_warnings,
                )
            )
            self._flush()
            return result

        raise AssertionError("unreachable workflow retry state")

    def record_skipped(
        self,
        name: str,
        *,
        artifacts: list[Path] | None = None,
        reason: str = "",
        warnings: list[str] | None = None,
    ) -> None:
        self.records.append(
            StageRecord(
                name=name,
                status="skipped",
                started_at=now_iso(),
                finished_at=now_iso(),
                duration_seconds=0.0,
                attempts=0,
                artifacts=[str(path) for path in artifacts or []],
                error=reason,
                warnings=list(warnings or []),
            )
        )
        self._flush()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.finished_at = now_iso()
        self._flush()

    def _flush(self) -> None:
        write_json(
            self.manifest_path,
            {
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "status": self.status,
                "metadata": self.metadata,
                "retry_policy": asdict(self.retry_policy),
                "stages": [asdict(record) for record in self.records],
            },
        )


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (ConfigError, FileNotFoundError, IsADirectoryError, PermissionError, ValueError)):
        return False

    message = f"{exc.__class__.__name__}: {exc}".casefold()
    non_retryable_markers = (
        "missing required environment variable",
        "is required",
        "unsupported",
        "test-only",
        "cannot be used",
        "invalid_api_key",
        "insufficient_quota",
        "does not exist",
        "no source documents",
    )
    if any(marker in message for marker in non_retryable_markers):
        return False

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    retryable_markers = (
        "timeout",
        "timed out",
        "temporarily",
        "rate limit",
        "rate_limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "connection reset",
        "internal server error",
        "service unavailable",
    )
    return any(marker in message for marker in retryable_markers)
