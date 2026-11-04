"""Storage uploads run as the service account, so they require a valid API key."""

import io
import json
from unittest.mock import AsyncMock, patch

from PIL import Image

_STORAGE = {"provider": "gcs", "bucket": "some-bucket", "path": "out.png", "public": True}


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_anonymous_json_storage_rejected_before_fetch(client):
    """JSON mode: storage without a key -> 401, and nothing is fetched or uploaded."""
    fetch = AsyncMock(return_value=_png_bytes())
    with (
        patch("routers.optimize.fetch_image", new=fetch),
        patch("routers.optimize.gcs_uploader") as mock_gcs,
    ):
        mock_gcs.upload = AsyncMock()
        resp = client.post(
            "/optimize", json={"url": "https://example.com/a.png", "storage": _STORAGE}
        )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"
    fetch.assert_not_called()
    mock_gcs.upload.assert_not_called()


def test_anonymous_multipart_storage_rejected(client):
    """Multipart mode: storage in options without a key -> 401, no upload."""
    with patch("routers.optimize.gcs_uploader") as mock_gcs:
        mock_gcs.upload = AsyncMock()
        resp = client.post(
            "/optimize",
            files={"file": ("a.png", _png_bytes(), "image/png")},
            data={"options": json.dumps({"storage": _STORAGE})},
        )
    assert resp.status_code == 401
    mock_gcs.upload.assert_not_called()


def test_anonymous_optimize_without_storage_still_allowed(client):
    """The public tier (binary response, no storage) is unchanged."""
    resp = client.post("/optimize", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200


def test_cloud_run_without_api_key_rejects_storage(client):
    """K_SERVICE set and API_KEY empty -> bearer tokens fail closed, so storage is refused."""
    with (
        patch("security.auth.settings.api_key", ""),
        patch("security.auth.settings.k_service", "pare"),
        patch("routers.optimize.gcs_uploader") as mock_gcs,
    ):
        mock_gcs.upload = AsyncMock()
        resp = client.post(
            "/optimize",
            files={"file": ("a.png", _png_bytes(), "image/png")},
            data={"options": json.dumps({"storage": _STORAGE})},
            headers={"Authorization": "Bearer guessed"},
        )
    assert resp.status_code == 401
    mock_gcs.upload.assert_not_called()
