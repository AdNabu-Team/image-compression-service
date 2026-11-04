# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Security layers applied via `SecurityMiddleware` (in root `middleware.py`). All modules are invoked per-request before the route handler runs.

## Modules

- **`auth.py`**: Bearer token validation. Empty `API_KEY` = dev mode (accepts any token), except on Cloud Run (`K_SERVICE` set), where it fails closed. No token header = public request (not an error), but `require_authenticated()` gates features that act with the service account (storage uploads).
- **`rate_limiter.py`**: Redis-backed sliding window (per-minute) + burst limiter (per-10s). Fail-open design: if Redis is unavailable, requests pass through. Lazy Redis init.
- **`ssrf.py`**: Validates URLs before fetch. Blocks every address that is not global unicast (`ipaddress.is_global`, after unwrapping IPv4-mapped and NAT64 addresses), cloud metadata hostnames, non-HTTPS schemes. httpx resolves the name again when it connects, so a DNS-rebinding window remains; pinning the connection to the checked IP is not done because the shared client pools connections by host, and an IP-keyed pool would reuse one TLS/SNI session across hostnames.
- **`svg_sanitizer.py`**: Strips `<script>`, `<foreignObject>`, event handlers (`on*` attributes), `data:text/html` URIs, CSS `@import url()`. Uses `defusedxml` for XXE-safe parsing.
- **`file_validation.py`**: Size limit check + format detection (delegates to `utils/format_detect.py`).

## Design Decisions

- Rate limiter uses **fail-open** — Redis outage should not take down the API
- The rate-limit key is the `X-Forwarded-For` entry `TRUSTED_PROXY_COUNT` from the right (`middleware._get_client_ip`); the leftmost entries are client-controlled, so keying on them lets a caller dodge the limit
- SSRF protection resolves DNS and checks **every** resolved IP address; the blocked IP is logged, never returned to the caller
- SVG sanitization happens during optimization (in `optimizers/svg.py`), not in the security middleware
