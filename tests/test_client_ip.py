"""Rate-limit key must come from the proxy-appended X-Forwarded-For entry."""

from unittest.mock import MagicMock, patch

import pytest

from middleware import _get_client_ip


def _request(forwarded: str | None, peer: str = "203.0.113.9") -> MagicMock:
    req = MagicMock()
    req.headers = {"X-Forwarded-For": forwarded} if forwarded is not None else {}
    req.client.host = peer
    return req


def test_client_supplied_entries_are_ignored():
    """A client rotating fake leftmost entries keeps the same rate-limit key."""
    first = _get_client_ip(_request("198.51.100.1, 192.0.2.50"))
    second = _get_client_ip(_request("198.51.100.2, 192.0.2.50"))
    assert first == second == "192.0.2.50"


@pytest.mark.parametrize(
    ("hops", "forwarded", "expected"),
    [
        (2, "198.51.100.1, 192.0.2.50, 35.191.0.1", "192.0.2.50"),
        (2, "192.0.2.50", "192.0.2.50"),
        (0, "198.51.100.1, 192.0.2.50", "203.0.113.9"),
    ],
)
def test_trusted_proxy_count(hops, forwarded, expected):
    """trusted_proxy_count picks the entry that many places from the right."""
    with patch("middleware.settings.trusted_proxy_count", hops):
        assert _get_client_ip(_request(forwarded)) == expected


def test_no_header_uses_peer():
    """Without X-Forwarded-For the socket peer is used."""
    assert _get_client_ip(_request(None)) == "203.0.113.9"
    assert _get_client_ip(_request(" , ")) == "203.0.113.9"
