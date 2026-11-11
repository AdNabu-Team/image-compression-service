# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Security layers applied via `SecurityMiddleware` (in root `middleware.py`). All modules are invoked per-request before the route handler runs.

## Modules

- **`auth.py`**: Bearer token validation. Deployed (`K_SERVICE` set) or with `API_KEY` set, every request needs a valid key; on Cloud Run an empty `API_KEY` fails closed. Only `middleware.PUBLIC_PATHS` (`/health`) and CORS preflights skip it, and `tests/test_route_auth.py` fails if any other route answers without a key. Locally with no `API_KEY` (dev mode) any token, or none, is accepted; `require_authenticated()` still gates storage uploads there.
- **`rate_limiter.py`**: Redis-backed sliding window (per-minute) + burst limiter (per-10s). Fail-open design: if Redis is unavailable, requests pass through. Lazy Redis init.
- **`ssrf.py`**: Validates URLs before fetch. Blocks every address that is not global unicast (`ipaddress.is_global`, after unwrapping IPv4-mapped and NAT64 addresses), cloud metadata hostnames, non-HTTPS schemes. `validate_url` refuses early; `CheckedAddressBackend` (the shared client's network backend, via `utils.url_fetch.CheckedAddressTransport`) resolves again at connect time, refuses the host if any address is blocked, and opens the socket to a checked address, so DNS rebinding can't reach an internal one. TLS still verifies and sends SNI for the hostname, and the pool stays keyed by hostname. A client built without that transport loses this check (`tests/test_ssrf_rebinding.py` guards the shared one).
- **`svg_sanitizer.py`**: Strips `<script>`, `<foreignObject>`, event handlers (`on*` attributes), `data:text/html` URIs, CSS `@import url()`. Uses `defusedxml` for XXE-safe parsing.
- **`file_validation.py`**: Size limit check + format detection (delegates to `utils/format_detect.py`).

## Design Decisions

- Rate limiter uses **fail-open** — Redis outage should not take down the API
- There is no anonymous tier in production: the Cloud Run services set no `REDIS_URL`, so the rate limiter is off there, and the API key is what keeps strangers from spending the service's CPU
- The rate-limit key is the `X-Forwarded-For` entry `TRUSTED_PROXY_COUNT` from the right (`middleware._get_client_ip`); the leftmost entries are client-controlled, so keying on them lets a caller dodge the limit
- SSRF protection resolves DNS and checks **every** resolved IP address; the blocked IP is logged, never returned to the caller
- SVG sanitization happens during optimization (in `optimizers/svg.py`), not in the security middleware
