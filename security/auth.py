from fastapi import Request

from config import settings
from exceptions import AuthenticationError


def authenticate(request: Request) -> bool:
    """Check if request has a valid API key.

    Args:
        request: FastAPI request object.

    Returns:
        True if authenticated, False if no auth header present.

    Raises:
        AuthenticationError: If auth header present but key is invalid, or if
            no API key is configured on Cloud Run (fail closed).
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        return False  # No auth — treated as public request

    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Invalid Authorization header format. Expected: Bearer <key>")

    provided_key = auth_header[7:]  # Strip "Bearer "

    if not settings.api_key:
        if settings.k_service:
            # Deployed without API_KEY: no bearer token can be valid.
            raise AuthenticationError("Invalid API key")
        # Local dev / tests: accept any bearer token
        return True

    if provided_key != settings.api_key:
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
