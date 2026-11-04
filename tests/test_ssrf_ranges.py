"""SSRF IP checks: everything that is not globally routable unicast is refused."""

from unittest.mock import patch

import pytest

from exceptions import SSRFError
from security.ssrf import validate_url


def _resolve_to(ip: str):
    family = 10 if ":" in ip else 2
    return patch("security.ssrf.socket.getaddrinfo", return_value=[(family, 1, 6, "", (ip, 443))])


@pytest.mark.parametrize(
    "ip",
    [
        "100.64.0.1",  # shared address space (CGNAT)
        "198.18.0.1",  # benchmarking
        "192.0.0.1",  # IETF protocol assignments
        "240.0.0.1",  # reserved
        "255.255.255.255",
        "224.0.0.1",  # multicast
        "fe80::1",  # IPv6 link-local
        "ff0e::1",  # global-scope multicast
        "::ffff:169.254.169.254",  # IPv4-mapped metadata address
        "::ffff:10.0.0.1",
        "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
        "64:ff9b::a00:1",  # NAT64 of 10.0.0.1
        "2002:a9fe:a9fe::1",  # 6to4 of 169.254.169.254
        "::",
    ],
)
def test_non_global_addresses_blocked(ip):
    """Addresses the old hand-written list missed are refused."""
    with _resolve_to(ip), pytest.raises(SSRFError):
        validate_url("https://images.example.com/a.png")


@pytest.mark.parametrize("ip", ["93.184.216.34", "2606:4700::1", "64:ff9b::808:808"])
def test_global_addresses_allowed(ip):
    """Public unicast addresses (including NAT64 of a public IPv4) pass."""
    with _resolve_to(ip):
        assert validate_url("https://images.example.com/a.png")


def test_blocked_error_does_not_reveal_resolved_ip():
    """The error sent to the caller omits the internal address it resolved to."""
    with _resolve_to("10.1.2.3"), pytest.raises(SSRFError) as exc:
        validate_url("https://images.example.com/a.png")
    assert "resolved_ip" not in exc.value.details
    assert "10.1.2.3" not in exc.value.message
