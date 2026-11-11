import ipaddress
import logging
import socket
import typing
from urllib.parse import urlparse

import anyio
import httpcore

from exceptions import SSRFError

logger = logging.getLogger("pare.security.ssrf")

# NAT64 well-known prefix: is_global is True for it, but a NAT64 gateway maps
# the low 32 bits to an IPv4 address, which must be checked on its own.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")

# Blocked hostnames (case-insensitive)
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.google.internal.",
}


def is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True unless `ip` is a globally routable unicast address.

    Uses the IANA special-purpose registries behind `ipaddress.is_global`
    (private, loopback, link-local, CGNAT 100.64/10, reserved, ...), and
    unwraps IPv4-mapped and NAT64 addresses so the embedded IPv4 is checked.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64_PREFIX:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return not ip.is_global or ip.is_multicast


def validate_url(url: str) -> str:
    """Validate a URL is safe to fetch (not targeting internal resources).

    Args:
        url: User-supplied URL string.

    Returns:
        The validated URL (unchanged if safe).

    Raises:
        SSRFError: If URL targets a private/reserved IP, metadata endpoint,
                   or uses a non-HTTPS scheme.
    """
    parsed = urlparse(url)

    # 1. Scheme check — HTTPS only
    if parsed.scheme != "https":
        raise SSRFError(
            f"Only HTTPS URLs are allowed, got {parsed.scheme}://",
            url=url,
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname", url=url)

    # 2. Hostname blocklist
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise SSRFError(
            "URL targets a blocked metadata endpoint",
            url=url,
        )

    # 3. DNS resolution. An early refusal only: CheckedAddressBackend checks
    # the addresses again when it connects, and connects only to those.
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise SSRFError(f"Could not resolve hostname: {hostname}", url=url)

    # 4. IP validation — check every resolved address
    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if is_blocked_ip(ip):
            # The address stays in our logs; callers only learn it was refused.
            logger.warning("ssrf_blocked host=%s ip=%s", hostname, ip)
            raise SSRFError(
                "URL resolves to a private/reserved IP address",
                url=url,
            )

    return url


class CheckedAddressBackend(httpcore.AnyIOBackend):
    """Network backend that connects only to addresses `is_blocked_ip` allows.

    The host is resolved once here, every address is checked, and the socket
    is opened to a checked address, so a name that resolves differently when
    the client connects (DNS rebinding) can't reach an internal one. TLS
    still verifies, and sends SNI for, the hostname, and the pool stays keyed
    by hostname.
    """

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Resolve host, refuse it if any address is blocked, then connect to the first that answers."""
        try:
            with anyio.fail_after(timeout):
                addr_infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout(f"Resolving {host} timed out") from exc
        except OSError as exc:
            raise httpcore.ConnectError(f"Could not resolve {host}") from exc

        addresses = list(dict.fromkeys(sockaddr[0] for *_, sockaddr in addr_infos))
        for address in addresses:
            if is_blocked_ip(ipaddress.ip_address(address)):
                logger.warning("ssrf_blocked_at_connect host=%s ip=%s", host, address)
                raise SSRFError("URL resolves to a private/reserved IP address", url=host)

        last_error: Exception = httpcore.ConnectError(f"No address for {host}")
        for address in addresses:
            try:
                return await super().connect_tcp(
                    address, port, timeout, local_address, socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        raise last_error
