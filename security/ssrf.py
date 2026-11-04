import ipaddress
import logging
import socket
from urllib.parse import urlparse

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

    # 3. DNS resolution. The HTTP client resolves again when it connects, so
    # this does not stop DNS rebinding; it blocks names that resolve to
    # internal addresses at check time.
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
