"""Tests for security/auth.py — authentication logic."""

from unittest.mock import MagicMock, patch

import pytest

from exceptions import AuthenticationError
from security.auth import authenticate


def _make_request(auth_header=None):
    """Create a mock FastAPI Request with optional Authorization header."""
    req = MagicMock()
    if auth_header:
        req.headers = {"Authorization": auth_header}
    else:
        req.headers = {}
    return req


def test_auth_no_header():
    """No auth header in local dev (no API_KEY, not on Cloud Run) -> returns False."""
    req = _make_request()
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = ""
        mock_settings.k_service = ""
        assert authenticate(req) is False


def test_auth_malformed_header():
    """Non-Bearer auth header -> AuthenticationError."""
    req = _make_request("Basic dXNlcjpwYXNz")
    with pytest.raises(AuthenticationError):
        authenticate(req)


def test_auth_no_api_key_configured():
    """No API_KEY in config (dev mode) -> accept all bearers."""
    req = _make_request("Bearer any-key-works")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = ""
        mock_settings.k_service = ""
        assert authenticate(req) is True


def test_auth_valid_key():
    """Correct API key -> True."""
    req = _make_request("Bearer secret-key-123")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = "secret-key-123"
        assert authenticate(req) is True


def test_auth_invalid_key():
    """Wrong API key -> AuthenticationError."""
    req = _make_request("Bearer wrong-key")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = "correct-key"
        with pytest.raises(AuthenticationError):
            authenticate(req)


def test_auth_no_api_key_configured_on_cloud_run_fails_closed():
    """No API_KEY on Cloud Run (K_SERVICE set) -> bearer tokens are rejected."""
    req = _make_request("Bearer any-key")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = ""
        mock_settings.k_service = "pare"
        with pytest.raises(AuthenticationError):
            authenticate(req)


def test_auth_no_header_on_cloud_run_is_rejected():
    """On Cloud Run a request without a key is refused, even with no API_KEY configured."""
    req = _make_request()
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = ""
        mock_settings.k_service = "pare"
        with pytest.raises(AuthenticationError):
            authenticate(req)


def test_auth_no_header_with_api_key_is_rejected():
    """With API_KEY set, a request without a key is refused, on Cloud Run or not."""
    req = _make_request()
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = "secret-key-123"
        mock_settings.k_service = ""
        with pytest.raises(AuthenticationError):
            authenticate(req)


def test_auth_compares_key_in_constant_time():
    """The key comparison goes through hmac.compare_digest."""
    req = _make_request("Bearer secret-key-123")
    with (
        patch("security.auth.settings") as mock_settings,
        patch("security.auth.hmac.compare_digest", return_value=True) as compare,
    ):
        mock_settings.api_key = "secret-key-123"
        assert authenticate(req) is True
    compare.assert_called_once_with(b"secret-key-123", b"secret-key-123")
