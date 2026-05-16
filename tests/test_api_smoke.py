"""FastAPI smoke: healthz + OpenAPI shape.

No live OpenAI calls. The api module is imported and exercised with
fastapi's TestClient — if anything inside the route registration is
broken (typo in path, bad dependency wiring, missing model), the
import or the smoke call will fail.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from pipeline.api import app


client = TestClient(app)


def test_healthz_returns_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload.get("status") in {"ok", "healthy", "up"}


def test_openapi_lists_core_endpoints():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema.get("openapi", "").startswith("3.")
    paths = schema.get("paths") or {}
    # These four endpoints are the operator-facing surface the README
    # documents and the operator UI consumes. If any of them disappear,
    # something has been unintentionally renamed or removed.
    for required in ("/healthz", "/runs", "/runs/{case_id}", "/runs/{case_id}/edits"):
        assert required in paths, f"OpenAPI is missing required path {required}"


def test_ui_root_redirects_to_ui_or_returns_html():
    # /  should route to the operator UI. Either a 200 HTML response (when
    # the React bundle is built and mounted) or a redirect to /ui is fine.
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {200, 301, 302, 307, 308}
    if response.status_code == 200:
        content_type = response.headers.get("content-type", "")
        assert "html" in content_type.lower() or "json" in content_type.lower()
    else:
        location = response.headers.get("location", "")
        assert location.startswith("/ui")


def test_runs_route_rejects_removed_claim_first_feature(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_API_ALLOWED_ROOTS", str(tmp_path))
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    response = client.post(
        "/runs",
        json={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "features": {"claim_first_drafting": False},
        },
    )

    assert response.status_code == 422


def test_runs_route_passes_top_level_draft_type(tmp_path, monkeypatch):
    import pipeline.orchestration.run as run_module

    monkeypatch.setenv("PIPELINE_API_ALLOWED_ROOTS", str(tmp_path))
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    captured = {}

    def fake_run_case(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(case_id=kwargs["case_id"], run_fingerprint="fp-test")

    monkeypatch.setattr(run_module, "run_case", fake_run_case)

    response = client.post(
        "/runs",
        json={
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "case_id": "api-draft-type",
            "draft_type": "case_fact_summary",
            "features": {"process_documents": False},
        },
    )

    assert response.status_code == 200
    assert response.json()["run_fingerprint"] == "fp-test"
    assert captured["draft_type"] == "case_fact_summary"


def test_runs_list_filters_failed_pre_draft_runs_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_API_ALLOWED_ROOTS", str(tmp_path))
    upload_only = tmp_path / "case-upload-only" / "_inputs"
    upload_only.mkdir(parents=True)
    (upload_only / "source.pdf").write_bytes(b"%PDF-1.4\n")

    failed_run = tmp_path / "case-failed"
    failed_run.mkdir()
    (failed_run / "workflow_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-05-17T00:00:00+00:00",
                "status": "failed",
                "metadata": {
                    "case_id": "case-failed",
                    "task": "Failed before draft.",
                },
                "stages": [
                    {
                        "name": "process_documents",
                        "status": "failed",
                        "error": "ProviderUnavailable: Missing required environment variable: OPENAI_API_KEY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reviewable_run = tmp_path / "case-ready"
    reviewable_run.mkdir()
    draft_payload = {
        "draft_type": "case_fact_summary",
        "title": "Case Fact Summary",
        "sections": [],
        "warnings": [],
        "evidence": [],
    }
    (reviewable_run / "draft.md").write_text("# Case Fact Summary\n", encoding="utf-8")
    (reviewable_run / "draft.json").write_text(json.dumps(draft_payload), encoding="utf-8")
    (reviewable_run / "case_run.json").write_text(
        json.dumps(
            {
                "case_id": "case-ready",
                "created_at": "2026-05-17T00:01:00+00:00",
                "task": "Ready for review.",
                "draft": draft_payload,
                "run_fingerprint": "fp-ready",
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/runs", params={"root": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()
    assert [run["case_id"] for run in payload["runs"]] == ["case-ready"]
    assert payload["runs"][0]["reviewable"] is True

    diagnostic_response = client.get(
        "/runs",
        params={"root": str(tmp_path), "include_unreviewable": "true"},
    )
    assert diagnostic_response.status_code == 200
    diagnostic_runs = {run["case_id"]: run for run in diagnostic_response.json()["runs"]}
    assert set(diagnostic_runs) == {"case-failed", "case-ready"}
    assert diagnostic_runs["case-failed"]["reviewable"] is False
    assert diagnostic_runs["case-failed"]["has_draft"] is False
    assert diagnostic_runs["case-failed"]["run_status"] == "failed"
    assert diagnostic_runs["case-failed"]["failure_stage"] == "process_documents"

    failed_summary_response = client.get(
        "/runs/case-failed/summary",
        params={"output_dir": str(failed_run)},
    )
    assert failed_summary_response.status_code == 200
    failed_summary = failed_summary_response.json()
    assert failed_summary["reviewable"] is False
    assert failed_summary["error"].startswith("ProviderUnavailable")
