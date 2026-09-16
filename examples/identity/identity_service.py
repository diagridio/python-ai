#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: Verified Identity with Starlette

This example demonstrates the three things `diagrid.identity` does for an app:
it verifies the caller's Catalyst user token before any handler runs, it hands
the handler a typed `VerifiedUser`, and it puts that same caller's identity
back on the outbound call the handler makes.

Prerequisites:
    1. Required packages: uv sync --all-packages --extra identity
    2. An ASGI server: uv pip install uvicorn
    3. A Catalyst sidecar, which sets X-Diagrid-User-Token on inbound
       requests and publishes the issuer and JWKS coordinates the
       middleware discovers.  An OSS Dapr sidecar publishes no identity
       block, so `diagrid dev run` is what supplies them.

Run:
    diagrid dev run -- python3 identity_service.py

Run it bare (python3 identity_service.py) and a tokenless request is still
401 oauth.missing_token, while a token-carrying one is 503
oauth.not_configured: there is nothing to verify it against.
"""

import contextlib
import os
from typing import AsyncIterator

import httpx2
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from diagrid.identity import OAuthConfig
from diagrid.identity.asgi import OAuthMiddleware, verified_user
from diagrid.identity.http import AsyncClient

DEFAULT_DOWNSTREAM_URL = "http://localhost:8081/whoami"
DOWNSTREAM_UNREACHABLE = "downstream_unreachable"
HTTP_BAD_GATEWAY = 502
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8080


async def whoami(request: Request) -> JSONResponse:
    """Return the verified caller the middleware put on this request.

    It cannot be None here: `require_auth` is left at its fail-closed
    default, so a request carrying no token is rejected with 401 before this
    handler runs.
    """
    user = verified_user(request)
    return JSONResponse(
        {
            "subject": user.subject,
            "tenant": user.tenant,
            "scopes": list(user.scopes),
            "hasRead": user.has_scope("read"),
        }
    )


async def downstream(request: Request) -> JSONResponse:
    """Call one downstream service as the caller who reached this route.

    One client built at startup is shared by every request: `AsyncClient`
    reads the caller's token at *send* time rather than at construction.
    """
    client: httpx2.AsyncClient = request.app.state.downstream_client
    url = os.environ.get("DOWNSTREAM_URL", DEFAULT_DOWNSTREAM_URL)
    try:
        response = await client.get(url)
    except httpx2.HTTPError:
        return JSONResponse(
            {"error": DOWNSTREAM_UNREACHABLE}, status_code=HTTP_BAD_GATEWAY
        )
    return JSONResponse({"downstream": response.text})


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    """Build the outbound client once, and close it on shutdown."""
    async with AsyncClient() as client:
        app.state.downstream_client = client
        yield


app = Starlette(
    routes=[
        Route("/whoami", whoami),
        Route("/downstream", downstream),
    ],
    lifespan=lifespan,
)

# Issuer, audience and JWKS URI are left unset so the sidecar's metadata
# endpoint supplies them.  No scopes are required of every caller, which keeps
# the example runnable — /whoami shows the per-caller check with has_scope.
config = OAuthConfig()
app.add_middleware(OAuthMiddleware, config=config)


if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
