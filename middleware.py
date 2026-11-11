import uuid

from fastapi import Request
from PIL import Image as _PIL_Image
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import settings
from exceptions import PareError
from security.auth import authenticate
from security.rate_limiter import safe_check_rate_limit

# Paths served without an API key: uptime checks only. Anything added here is open to the internet.
PUBLIC_PATHS = frozenset({"/health"})


def is_public_request(request: Request) -> bool:
    """Whether the request skips authentication: a public path, or a CORS preflight (which carries no key)."""
    if request.url.path in PUBLIC_PATHS:
        return True
    return request.method == "OPTIONS" and "access-control-request-method" in request.headers


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined middleware for auth, rate limiting, and request ID.

    Order of operations per request:
    1. Inject request ID (UUID)
    2. Authenticate (check Bearer token)
    3. Rate limit check (skip for authenticated if configured)
    4. Process request
    5. Add X-Request-ID to response
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            # 2. Authentication
            is_authenticated = False if is_public_request(request) else authenticate(request)
            request.state.is_authenticated = is_authenticated

            # 3. Rate limiting
            client_ip = _get_client_ip(request)
            await safe_check_rate_limit(client_ip, is_authenticated)

            # 4. Process request
            response = await call_next(request)

        except _PIL_Image.DecompressionBombError:
            limit = settings.max_image_pixels
            if limit >= 1_000_000:
                limit_str = f"{limit // 1_000_000} megapixels"
            else:
                limit_str = f"{limit:,} pixels"
            response = JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": "image_too_large",
                    "message": f"Image exceeds maximum pixel count ({limit_str})",
                },
            )

        except PareError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.error_code,
                    "message": exc.message,
                    **exc.details,
                },
            )

        # 5. Request ID header
        response.headers["X-Request-ID"] = request_id
        return response


def _get_client_ip(request: Request) -> str:
    """Extract the client IP for rate limiting.

    Trusted proxies append to X-Forwarded-For, so the client IP is the entry
    `trusted_proxy_count` places from the right; anything left of it is
    whatever the client sent and must not be used as the rate-limit key.
    """
    peer = request.client.host if request.client else "unknown"
    hops = settings.trusted_proxy_count
    forwarded = request.headers.get("X-Forwarded-For")
    if hops <= 0 or not forwarded:
        return peer
    entries = [e.strip() for e in forwarded.split(",") if e.strip()]
    if not entries:
        return peer
    return entries[-hops] if len(entries) >= hops else entries[0]
