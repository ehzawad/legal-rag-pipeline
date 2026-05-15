"""FastAPI smoke: healthz + OpenAPI shape.

No live OpenAI calls. The api module is imported and exercised with
fastapi's TestClient — if anything inside the route registration is
broken (typo in path, bad dependency wiring, missing model), the
import or the smoke call will fail.
"""

from __future__ import annotations

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
