"""FastAPI smoke: healthz + OpenAPI shape.

No live OpenAI calls. The api module is imported and exercised with
fastapi's TestClient — if anything inside the route registration is
broken (typo in path, bad dependency wiring, missing model), the
import or the smoke call will fail.
"""

from __future__ import annotations

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
