from __future__ import annotations

import json

from pipeline.learning import capture_operator_edit, capture_operator_edit_from_files


def test_capture_operator_edit_records_case_fact_summary_draft_type() -> None:
    captured = capture_operator_edit(
        original_draft="Original fact [E1]",
        edited_draft="Edited fact [E1]",
        evidence_ids=["E1"],
    )

    assert captured["draft_type"] == "case_fact_summary"


def test_capture_operator_edit_from_files_logs_draft_type(tmp_path) -> None:
    draft = tmp_path / "draft.md"
    edited = tmp_path / "edited.md"
    profile = tmp_path / "state" / "operator_profile.json"
    edits_log = tmp_path / "state" / "edits.jsonl"
    draft.write_text("Original fact [E1]\n", encoding="utf-8")
    edited.write_text("Edited fact [E1]\n", encoding="utf-8")

    result = capture_operator_edit_from_files(
        draft,
        edited,
        profile,
        event_log_path=edits_log,
    )
    event = json.loads(edits_log.read_text(encoding="utf-8").splitlines()[0])

    assert result.event["draft_type"] == "case_fact_summary"
    assert event["draft_type"] == "case_fact_summary"
