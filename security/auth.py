import hmac

from fastapi import Request

from config import settings
from exceptions import AuthenticationError


def authenticate(request: Request) -> bool:
    """Check if request has a valid API key.

    Args:
        request: FastAPI request object.

    Returns:
        True if authenticated, False for a request without a key in local dev
        (no `API_KEY` and not on Cloud Run).

    Raises:
        AuthenticationError: If the key is missing or invalid. Deployed (Cloud Run)
            or with `API_KEY` set, every request needs a valid key; without
            `API_KEY` on Cloud Run no key is valid (fail closed).
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        if settings.api_key or settings.k_service:
            raise AuthenticationError("A valid API key is required")
        return False  # Local dev without API_KEY

    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Invalid Authorization header format. Expected: Bearer <key>")

    provided_key = auth_header[7:]  # Strip "Bearer "

    if not settings.api_key:
        if settings.k_service:
            # Deployed without API_KEY: no bearer token can be valid.
            raise AuthenticationError("Invalid API key")
        # Local dev / tests: accept any bearer token
        return True

    if not hmac.compare_digest(provided_key.encode(), settings.api_key.encode()):
        raise AuthenticationError("Invalid API key")

    return True


def require_authenticated(request: Request, purpose: str) -> None:
    """Reject the request unless it carried a valid API key.

    Args:
        request: FastAPI request object (after SecurityMiddleware ran).
        purpose: What the key is needed for, used in the error message.

    Raises:
        AuthenticationError: If the request is unauthenticated.
    """
    if not getattr(request.state, "is_authenticated", False):
        raise AuthenticationError(f"A valid API key is required for {purpose}")
