from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipeline.schemas import now_iso, to_jsonable


def append_audit_event(
    path: Path,
    *,
    action: str,
    object_type: str,
    object_id: str,
    actor: str = "system",
    payload: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": now_iso(),
        "actor": actor,
        "action": action,
        "object_type": object_type,
        "object_id": object_id,
        "payload": to_jsonable(dict(payload or {})),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
