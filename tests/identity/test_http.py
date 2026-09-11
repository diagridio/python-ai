import asyncio
from unittest.mock import MagicMock, patch

import httpx2
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from diagrid.identity import OAuthConfig
from diagrid.identity.asgi import OAuthMiddleware

from diagrid.identity.http import (
    AsyncClient,
    Client,
    attach_identity_headers,
    attach_identity_headers_async,
)
from diagrid.identity.outbound import (
    USER_TOKEN_HEADER,
    clear_current_token,
    set_current_token,
)

BASE_URL = "http://mcp.invalid"
BARRIER_TIMEOUT = 10


@pytest.fixture(autouse=True)
def clear_token():
    clear_current_token()
    yield
    clear_current_token()


def _recording_transport(seen):
    """MockTransport recording the identity header of every request by path."""

    def handler(request):
        seen[request.url.path] = request.headers.get(USER_TOKEN_HEADER)
        return httpx2.Response(200)

    return httpx2.MockTransport(handler)


class TestSyncClient:
    def test_attaches_current_token(self):
        seen = {}
        set_current_token("tok-sync")
        with Client(transport=_recording_transport(seen), base_url=BASE_URL) as client:
            client.get("/call")

        assert seen == {"/call": "Bearer tok-sync"}

    def test_omits_header_without_user_context(self):
        seen = {}
        with Client(transport=_recording_transport(seen), base_url=BASE_URL) as client:
            client.get("/call")

        assert seen == {"/call": None}

    def test_client_kwargs_are_forwarded(self):
        with Client(base_url=BASE_URL, timeout=1.5, follow_redirects=True) as client:
            assert str(client.base_url) == BASE_URL
            assert client.timeout.read == 1.5
            assert client.follow_redirects is True


class TestAsyncClient:
    def test_attaches_current_token(self):
        seen = {}

        async def body():
            async with AsyncClient(
                transport=_recording_transport(seen), base_url=BASE_URL
            ) as client:
                await client.get("/call")

        set_current_token("tok-async")
        asyncio.run(body())

        assert seen == {"/call": "Bearer tok-async"}

    def test_token_is_read_at_send_time_not_at_construction(self):
        seen = {}
        client = AsyncClient(transport=_recording_transport(seen), base_url=BASE_URL)

        async def body():
            async with client:
                await client.get("/call")

        set_current_token("set-after-construction")
        asyncio.run(body())

        assert seen == {"/call": "Bearer set-after-construction"}

    def test_omits_header_without_user_context(self):
        seen = {}

        async def body():
            async with AsyncClient(
                transport=_recording_transport(seen), base_url=BASE_URL
            ) as client:
                await client.get("/call")

        asyncio.run(body())

        assert seen == {"/call": None}

    def test_client_kwargs_are_forwarded(self):
        async def body():
            async with AsyncClient(
                base_url=BASE_URL, timeout=1.5, follow_redirects=True
            ) as client:
                assert str(client.base_url) == BASE_URL
                assert client.timeout.read == 1.5
                assert client.follow_redirects is True

        asyncio.run(body())


class TestMcpCallPatterns:
    """The MCP SDK drives a supplied client through send() and stream()."""

    def test_build_request_then_send(self):
        seen = {}
        set_current_token("tok-send")
        with Client(transport=_recording_transport(seen), base_url=BASE_URL) as client:
            client.send(client.build_request("POST", "/call"))

        assert seen == {"/call": "Bearer tok-send"}

    def test_stream(self):
        seen = {}
        set_current_token("tok-stream")
        with Client(transport=_recording_transport(seen), base_url=BASE_URL) as client:
            with client.stream("GET", "/call") as response:
                response.read()

        assert seen == {"/call": "Bearer tok-stream"}

    def test_async_stream(self):
        seen = {}

        async def body():
            async with AsyncClient(
                transport=_recording_transport(seen), base_url=BASE_URL
            ) as client:
                async with client.stream("GET", "/call") as response:
                    await response.aread()

        set_current_token("tok-astream")
        asyncio.run(body())

        assert seen == {"/call": "Bearer tok-astream"}


def test_one_client_carries_each_callers_own_token_concurrently():
    """Two in-flight requests through one client must not cross identities.

    The barrier is what makes this meaningful: without it the requests could
    serialise and the test would pass against a constructor-headers client
    that sends one token to every caller.
    """
    seen = {}
    both_in_flight = asyncio.Barrier(2)

    async def handler(request):
        seen[request.url.path] = request.headers.get(USER_TOKEN_HEADER)
        await both_in_flight.wait()
        return httpx2.Response(200)

    client = AsyncClient(transport=httpx2.MockTransport(handler), base_url=BASE_URL)

    async def call_as(user):
        set_current_token(f"tok-{user}")
        await client.get(f"/{user}")

    async def body():
        async with client:
            await asyncio.gather(call_as("alice"), call_as("bob"))

    # Bounded so a regression stops the run rather than hanging on the barrier.
    asyncio.run(asyncio.wait_for(body(), timeout=BARRIER_TIMEOUT))

    assert seen == {"/alice": "Bearer tok-alice", "/bob": "Bearer tok-bob"}


def test_stale_header_is_stripped_when_context_has_no_token():
    seen = {}

    async def body():
        async with AsyncClient(
            transport=_recording_transport(seen),
            base_url=BASE_URL,
            headers={USER_TOKEN_HEADER: "Bearer leftover"},
        ) as client:
            await client.get("/call")

    asyncio.run(body())

    assert seen == {"/call": None}


def test_identity_header_overrides_a_per_request_header():
    seen = {}

    async def body():
        async with AsyncClient(
            transport=_recording_transport(seen), base_url=BASE_URL
        ) as client:
            await client.get("/call", headers={USER_TOKEN_HEADER: "Bearer spoofed"})

    set_current_token("real")
    asyncio.run(body())

    assert seen == {"/call": "Bearer real"}


class TestRedirects:
    """The hook runs on every hop, so a redirect must not carry the token off."""

    @staticmethod
    def _redirecting_transport(seen, location):
        def handler(request):
            seen.append((request.url.origin, request.headers.get(USER_TOKEN_HEADER)))
            if len(seen) == 1:
                return httpx2.Response(307, headers={"Location": location})
            return httpx2.Response(200)

        return httpx2.MockTransport(handler)

    @pytest.mark.parametrize(
        ("called", "location", "forwarded"),
        [
            ("https://mcp.invalid/call", "https://mcp.invalid/call/", True),
            ("http://mcp.invalid/call", "https://mcp.invalid/call", True),
            ("https://mcp.invalid/call", "https://evil.example/steal", False),
            ("https://mcp.invalid/call", "https://mcp.invalid:9999/steal", False),
            ("http://mcp.invalid:8080/call", "https://mcp.invalid:9999/steal", False),
            ("https://mcp.invalid/call", "http://mcp.invalid/call", False),
            ("https://mcp.invalid/call", "https://sub.mcp.invalid/call", False),
        ],
        ids=[
            "same-origin",
            "https-upgrade",
            "different-host",
            "different-port",
            "upgrade-to-different-port",
            "downgrade-to-http",
            "subdomain",
        ],
    )
    def test_token_travels_only_to_the_origin_called(self, called, location, forwarded):
        seen = []
        set_current_token("secret-obo")
        with Client(
            transport=self._redirecting_transport(seen, location),
            follow_redirects=True,
        ) as client:
            client.get(called)

        assert [origin for origin, _ in seen] == [
            httpx2.URL(called).origin,
            httpx2.URL(location).origin,
        ]
        assert [header for _, header in seen] == [
            "Bearer secret-obo",
            "Bearer secret-obo" if forwarded else None,
        ]


class TestEventHookMerging:
    def test_caller_request_hooks_are_preserved_and_not_mutated(self):
        seen = {}
        called = []

        async def caller_hook(request):
            called.append(request.url.path)

        hooks = {"request": [caller_hook]}

        async def body():
            async with AsyncClient(
                transport=_recording_transport(seen),
                base_url=BASE_URL,
                event_hooks=hooks,
            ) as client:
                await client.get("/call")

        set_current_token("tok")
        asyncio.run(body())

        assert called == ["/call"]
        assert seen == {"/call": "Bearer tok"}
        assert hooks == {"request": [caller_hook]}

    def test_a_shared_mapping_does_not_accumulate_a_hook_per_client(self):
        hooks = {"request": []}

        first = Client(event_hooks=hooks, base_url=BASE_URL)
        second = Client(event_hooks=hooks, base_url=BASE_URL)

        assert len(first.event_hooks["request"]) == 1
        assert len(second.event_hooks["request"]) == 1
        assert hooks == {"request": []}

    def test_caller_response_hooks_are_preserved(self):
        statuses = []

        async def on_response(response):
            statuses.append(response.status_code)

        async def body():
            async with AsyncClient(
                transport=_recording_transport({}),
                base_url=BASE_URL,
                event_hooks={"response": [on_response]},
            ) as client:
                await client.get("/call")

        asyncio.run(body())

        assert statuses == [200]


class TestHooksOnCallerOwnedClients:
    def test_sync_hook(self):
        seen = {}
        set_current_token("byo")
        with httpx2.Client(
            transport=_recording_transport(seen),
            base_url=BASE_URL,
            event_hooks={"request": [attach_identity_headers]},
        ) as client:
            client.get("/call")

        assert seen == {"/call": "Bearer byo"}

    def test_async_hook(self):
        seen = {}

        async def body():
            async with httpx2.AsyncClient(
                transport=_recording_transport(seen),
                base_url=BASE_URL,
                event_hooks={"request": [attach_identity_headers_async]},
            ) as client:
                await client.get("/call")

        set_current_token("byo-async")
        asyncio.run(body())

        assert seen == {"/call": "Bearer byo-async"}


def test_middleware_verified_token_reaches_concurrent_outbound_calls():
    """End-to-end: inbound middleware -> contextvar -> outbound client.

    Pins the contextvar propagation through Starlette's BaseHTTPMiddleware
    that the whole design rests on, which the unit tests fake by setting the
    contextvar themselves.
    """
    seen = {}
    both_in_flight = asyncio.Barrier(2)

    async def outbound_handler(request):
        seen[request.url.path] = request.headers.get(USER_TOKEN_HEADER)
        await both_in_flight.wait()
        return httpx2.Response(200)

    outbound = AsyncClient(
        transport=httpx2.MockTransport(outbound_handler), base_url=BASE_URL
    )

    async def endpoint(request):
        user = request.state.user
        await outbound.get(f"/{user.subject}")
        return JSONResponse({"subject": user.subject})

    app = Starlette(routes=[Route("/invoke", endpoint)])
    app.add_middleware(OAuthMiddleware, config=OAuthConfig())

    verifier = MagicMock()
    verifier.verify.side_effect = lambda token: {"sub": token.split(".")[0]}

    async def body():
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://agent.invalid"
        ) as inbound:
            await asyncio.wait_for(
                asyncio.gather(
                    *(
                        inbound.get(
                            "/invoke",
                            headers={USER_TOKEN_HEADER: f"Bearer {user}.raw.token"},
                        )
                        for user in ("alice", "bob")
                    )
                ),
                timeout=BARRIER_TIMEOUT,
            )

    with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
        asyncio.run(body())

    assert seen == {
        "/alice": "Bearer alice.raw.token",
        "/bob": "Bearer bob.raw.token",
    }
