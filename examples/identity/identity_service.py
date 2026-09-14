#!/usr/bin/env python3

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""
Example: Verified Identity with Starlette

This example demonstrates the three things `diagrid.identity` does for an app:
it verifies the caller's Catalyst user token before any handler runs, it hands
the handler a typed `VerifiedUser`, and it puts that same caller's identity
back on the outbound call the handler makes.

It is deliberately minimal — no agent, no model, no workflow, no state store.
Two routes and the identity surface, nothing else.

Prerequisites:
    1. Required packages: uv sync --all-packages --extra identity
    2. An ASGI server: uv pip install uvicorn
    3. A Catalyst sidecar, which sets X-Diagrid-User-Token on inbound
       requests and publishes the issuer and JWKS coordinates the
       middleware discovers.

Run:
    python3 identity_service.py
"""

import os

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

    `verified_user` is the typed accessor — a `VerifiedUser`, not a cast out
    of a string-keyed bag. It cannot be None here: `require_auth` is left at
    its fail-closed default, so a request carrying no token is rejected with
    401 before this handler runs.
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

    `AsyncClient` is an `httpx2.AsyncClient` that attaches the caller's
    identity headers at send time, so the callee verifies the same user.
    """
    url = os.environ.get("DOWNSTREAM_URL", DEFAULT_DOWNSTREAM_URL)
    try:
        async with AsyncClient() as client:
            response = await client.get(url)
    except httpx2.HTTPError:
        return JSONResponse(
            {"error": DOWNSTREAM_UNREACHABLE}, status_code=HTTP_BAD_GATEWAY
        )
    return JSONResponse({"downstream": response.text})


app = Starlette(
    routes=[
        Route("/whoami", whoami),
        Route("/downstream", downstream),
    ]
)

# The whole install. Issuer, audience and JWKS URI are left unset so the
# sidecar's metadata endpoint supplies them, and no scopes are required of
# every caller — /whoami demonstrates the scope check with has_scope instead,
# which keeps the example runnable without any scope setup.
config = OAuthConfig()
app.add_middleware(OAuthMiddleware, config=config)


if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
