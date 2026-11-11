"""DNS rebinding: the fetch connects only to the addresses it checked, whatever the name resolves to later."""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
import pytest

from exceptions import SSRFError
from security.ssrf import CheckedAddressBackend
from utils import url_fetch


def _addr_infos(*ips: str, port: int = 443) -> list:
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
        for ip in ips
    ]


@pytest.mark.asyncio
async def test_rebinding_to_a_private_address_is_refused_at_connect():
    """The URL check sees a public address; the connect-time resolution returns loopback and is refused."""
    first_check = patch(
        "security.ssrf.socket.getaddrinfo", return_value=_addr_infos("93.184.216.34")
    )
    rebound = patch(
        "security.ssrf.anyio.getaddrinfo", new=AsyncMock(return_value=_addr_infos("127.0.0.1"))
    )
    connect = patch.object(httpcore.AnyIOBackend, "connect_tcp", new=AsyncMock())
    with first_check, rebound, connect as mock_connect:
        client = httpx.AsyncClient(transport=url_fetch.CheckedAddressTransport())
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=client)):
            with pytest.raises(SSRFError):
                await url_fetch.fetch_image(
                    "https://rebind.example/image.png", is_authenticated=True
                )
        await client.aclose()
    mock_connect.assert_not_called()


@pytest.mark.asyncio
async def test_a_name_with_any_blocked_address_is_refused():
    backend = CheckedAddressBackend()
    resolved = AsyncMock(return_value=_addr_infos("93.184.216.34", "10.0.0.1"))
    with (
        patch("security.ssrf.anyio.getaddrinfo", new=resolved),
        patch.object(httpcore.AnyIOBackend, "connect_tcp", new=AsyncMock()) as mock_connect,
    ):
        with pytest.raises(SSRFError):
            await backend.connect_tcp("mixed.example", 443)
    mock_connect.assert_not_called()


@pytest.mark.asyncio
async def test_connects_to_the_checked_address_and_falls_back_to_the_next():
    backend = CheckedAddressBackend()
    stream = object()
    resolved = AsyncMock(return_value=_addr_infos("93.184.216.34", "93.184.216.35"))
    connect = AsyncMock(side_effect=[httpcore.ConnectError("refused"), stream])
    with (
        patch("security.ssrf.anyio.getaddrinfo", new=resolved),
        patch.object(httpcore.AnyIOBackend, "connect_tcp", new=connect),
    ):
        assert await backend.connect_tcp("cdn.example", 443, timeout=5) is stream
    assert [call.args[0] for call in connect.call_args_list] == ["93.184.216.34", "93.184.216.35"]


@pytest.mark.asyncio
async def test_resolution_failure_is_a_connect_error():
    backend = CheckedAddressBackend()
    with patch(
        "security.ssrf.anyio.getaddrinfo",
        new=AsyncMock(side_effect=socket.gaierror("no such host")),
    ):
        with pytest.raises(httpcore.ConnectError):
            await backend.connect_tcp("missing.example", 443)


@pytest.mark.asyncio
async def test_request_keeps_the_hostname_while_connecting_to_the_checked_ip():
    """End to end over a real socket: the server sees the original Host header, reached at the pinned address."""
    seen = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        seen["host"] = next(
            line.split(b":", 1)[1].strip()
            for line in head.split(b"\r\n")
            if line.lower().startswith(b"host:")
        )
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    resolved = AsyncMock(return_value=_addr_infos("127.0.0.1", port=port))
    async with server:
        with (
            patch("security.ssrf.anyio.getaddrinfo", new=resolved),
            patch("security.ssrf.is_blocked_ip", return_value=False),
        ):
            async with httpx.AsyncClient(transport=url_fetch.CheckedAddressTransport()) as client:
                response = await client.get(f"http://images.example:{port}/a.png")
    assert response.text == "ok"
    assert seen["host"] == f"images.example:{port}".encode()
    assert resolved.call_args.args[0] == "images.example"


@pytest.mark.asyncio
async def test_shared_client_uses_the_checked_transport():
    """Guards an httpx upgrade or refactor that would drop the connect-time check."""
    await url_fetch.close_client()
    client = await url_fetch._get_client()
    try:
        assert isinstance(client._transport, url_fetch.CheckedAddressTransport)
        assert isinstance(client._transport._pool._network_backend, CheckedAddressBackend)
    finally:
        await url_fetch.close_client()
