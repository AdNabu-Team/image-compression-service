"""Every route needs an API key once deployed; only PUBLIC_PATHS are open (AdNabu-Team/adnabu-api#7929)."""

import pytest
from starlette.routing import Route

from config import settings
from main import app
from middleware import PUBLIC_PATHS

KEY = "route-auth-test-key"


def app_endpoints() -> list[tuple[str, str]]:
    """(method, path) for every route the app serves: the OpenAPI schema (included routers too) and FastAPI's
    docs routes, which aren't in it."""
    endpoints = {
        (method.upper(), path) for path, ops in app.openapi()["paths"].items() for method in ops
    }
    for route in app.routes:
        if isinstance(route, Route):
            endpoints |= {
                (method, route.path) for method in route.methods or () if method != "HEAD"
            }
    return sorted(endpoints)


def test_the_route_walk_sees_every_router():
    """If this fails, app_endpoints() stopped finding routes and the tests below would pass vacuously."""
    assert {("GET", "/health"), ("POST", "/optimize"), ("POST", "/estimate"), ("GET", "/")} <= set(
        app_endpoints()
    )


@pytest.fixture
def deployed(monkeypatch):
    """Settings as on Cloud Run: an API key and K_SERVICE set."""
    monkeypatch.setattr(settings, "api_key", KEY)
    monkeypatch.setattr(settings, "k_service", "pare")


def test_public_paths_are_only_the_health_check():
    """Opening a path to the internet is a deliberate change: update this test with the reason."""
    assert PUBLIC_PATHS == frozenset({"/health"})


@pytest.mark.parametrize("method,path", app_endpoints())
def test_every_route_refuses_a_request_without_a_key(client, deployed, method, path):
    resp = client.request(method, path)
    if path in PUBLIC_PATHS:
        assert resp.status_code != 401
    else:
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"


@pytest.mark.parametrize("method,path", app_endpoints())
def test_every_route_refuses_a_wrong_key(client, deployed, method, path):
    if path in PUBLIC_PATHS:
        pytest.skip("public path")
    resp = client.request(method, path, headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", app_endpoints())
def test_every_route_accepts_the_key(client, deployed, method, path):
    resp = client.request(method, path, headers={"Authorization": f"Bearer {KEY}"})
    assert resp.status_code != 401


def test_cors_preflight_needs_no_key(client, deployed):
    """Browsers send preflights without credentials; CORSMiddleware answers them."""
    resp = client.options(
        "/optimize",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code != 401


def test_plain_options_still_needs_a_key(client, deployed):
    assert client.options("/optimize").status_code == 401
