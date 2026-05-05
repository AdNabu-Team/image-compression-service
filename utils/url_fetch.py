import asyncio

import httpx

from config import settings
from exceptions import FileTooLargeError, URLFetchError
from security.ssrf import validate_url

# Module-level shared client — created once at first call, reused across requests.
# Avoids per-request TLS handshake + connection setup overhead (~50-200ms on cold hosts).
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    # Per-request timeout is passed to client.stream() instead,
                    # so the client-level timeout is a wide backstop only.
                    timeout=httpx.Timeout(settings.url_fetch_timeout * 4),
                    follow_redirects=False,
                    verify=True,
                )
    return _client


async def close_client() -> None:
    """Close the shared AsyncClient. Called from FastAPI lifespan shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fetch_image(url: str, is_authenticated: bool = False) -> bytes:
    """Fetch image from a URL with SSRF protection and streaming size limits.

    Follows redirects manually (max 5 hops) with SSRF validation at each
    hop to prevent redirect-based SSRF attacks. Streams the response body
    and aborts early if the size limit is exceeded.

    Args:
        url: User-supplied HTTPS URL.
        is_authenticated: Affects timeout (auth=base, public=2x base).

    Returns:
        Raw image bytes.

    Raises:
        SSRFError: URL targets private/reserved IP.
        URLFetchError: Fetch failed (timeout, non-2xx, redirect limit).
        FileTooLargeError: Response body exceeds max file size.
    """
    # 1. SSRF validation on initial URL
    validate_url(url)

    timeout = settings.url_fetch_timeout if is_authenticated else settings.url_fetch_timeout * 2
    max_redirects = settings.url_fetch_max_redirects
    max_size = settings.max_file_size_bytes

    client = await _get_client()
    current_url = url

    try:
        for _hop in range(max_redirects + 1):
            async with client.stream(
                "GET", current_url, timeout=httpx.Timeout(timeout)
            ) as response:
                if response.is_redirect:
                    if response.next_request is None:
                        raise URLFetchError(
                            "Redirect without Location header",
                            url=current_url,
                        )
                    redirect_url = str(response.next_request.url)
                    validate_url(redirect_url)
                    current_url = redirect_url
                    continue

                if not response.is_success:
                    raise URLFetchError(
                        f"URL returned HTTP {response.status_code}",
                        url=url,
                        http_status=response.status_code,
                    )

                # 2. Check Content-Length header for early rejection
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    raise FileTooLargeError(
                        f"URL content too large: {content_length} bytes",
                        file_size=int(content_length),
                        limit=max_size,
                    )

                # 3. Stream body with running size check — abort on the first
                # chunk that pushes past the limit, avoiding full download of
                # oversized payloads before rejection.
                buf = bytearray()
                async for chunk in response.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_size:
                        raise FileTooLargeError(
                            f"URL content exceeds {settings.max_file_size_mb} MB limit",
                            file_size=len(buf),
                            limit=max_size,
                        )

                return bytes(buf)

        raise URLFetchError(
            f"Too many redirects (>{max_redirects})",
            url=url,
        )

    except httpx.TimeoutException:
        raise URLFetchError(
            f"URL fetch timed out after {timeout}s",
            url=url,
        )
    except httpx.RequestError as e:
        raise URLFetchError(
            f"URL fetch failed: {e}",
            url=url,
        )
