"""_fetch_dimensions must not read a whole response when the origin ignores Range."""

import io
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image

from exceptions import SSRFError
from routers.estimate import _fetch_dimensions

_CHUNK = 64 * 1024
_TOTAL_CHUNKS = 1024  # 64 MB, over the 32 MB fetch_image cap


@pytest.mark.asyncio
async def test_fetch_dimensions_stops_after_header_bytes():
    """A 200 with a huge body is read only up to the 8KB header cap."""
    buf = io.BytesIO()
    Image.new("RGB", (320, 240)).save(buf, format="JPEG", quality=50)
    head = buf.getvalue()
    sent = 0

    async def body():
        nonlocal sent
        yield head
        for _ in range(_TOTAL_CHUNKS):
            sent += 1
            yield b"\0" * _CHUNK

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    )
    with (
        patch("utils.url_fetch.validate_url"),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=client)),
        patch("routers.estimate.fetch_image", new=AsyncMock()) as full_fetch,
    ):
        size = await _fetch_dimensions("https://example.com/big.jpg", is_authenticated=True)
    await client.aclose()

    assert size == (320, 240)
    assert sent <= 1
    full_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_dimensions_ssrf_not_retried_via_full_fetch():
    """An SSRF rejection propagates instead of triggering the full-download fallback."""
    with (
        patch("routers.estimate.fetch_partial", new=AsyncMock(side_effect=SSRFError("blocked"))),
        patch("routers.estimate.fetch_image", new=AsyncMock()) as full_fetch,
    ):
        with pytest.raises(SSRFError):
            await _fetch_dimensions("https://example.com/a.jpg", is_authenticated=False)
    full_fetch.assert_not_called()
