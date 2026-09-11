# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Identity-aware HTTP clients for outbound on-behalf-of calls.

The sidecar mints an OBO token for whoever it identifies on the *inbound*
request.  An outbound call to the MCP proxy or a sub-agent is a separate,
stateless request, so the caller's token has to ride on it explicitly::

    from mcp.client.streamable_http import streamable_http_client

    from diagrid.identity.http import AsyncClient

    client = AsyncClient()

    async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
        ...

The header is read from the contextvar at *send* time rather than baked in at
construction.  That is what makes one long-lived client safe: concurrent
requests each carry their own caller's token, where a constructor header would
send whichever user was current when the client was built.

The token only ever goes to the origin the caller addressed; a redirect away
from it drops the header.  Past that the client is as wide as you make it, so
call third-party APIs with a plain ``httpx2`` client instead.

Two limitations.  mcp 1.x annotates ``http_client`` as ``httpx.AsyncClient``
where 2.x uses ``httpx2``, so a type checker rejects the call on 1.x even
though the object satisfies it.  And ``X-Diagrid-User-Token`` is not a header
name log scrubbers and tracing SDKs redact by default.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional

import httpx2

from diagrid.identity.outbound import USER_TOKEN_HEADER, outbound_identity_headers

logger = logging.getLogger(__name__)

_REQUEST_EVENT_HOOK = "request"

# httpx2 hands the same ``extensions`` dict to every redirect hop, so an
# origin stamped on the first hop survives to be compared against later ones.
_ORIGIN_EXTENSION = "diagrid_identity_origin"

_HTTP_DEFAULT_PORT = 80
_HTTPS_DEFAULT_PORT = 443

_EventHook = Callable[..., Any]

__all__ = [
    "AsyncClient",
    "Client",
    "attach_identity_headers",
    "attach_identity_headers_async",
]


def attach_identity_headers(request: httpx2.Request) -> None:
    """Set the caller's identity headers on *request*, in place.

    An httpx2 ``request`` event hook for a synchronous client — use it when
    you already own a client you cannot replace::

        httpx2.Client(event_hooks={"request": [attach_identity_headers]})

    The header is cleared first, so a request never carries an identity the
    current context does not hold, whatever set it.
    """
    request.headers.pop(USER_TOKEN_HEADER, None)
    if not _is_original_origin(request):
        logger.debug("identity withheld: %s is not the origin called", request.url)
        return
    headers = outbound_identity_headers()
    if not headers:
        logger.debug("no inbound user context; calling %s unauthenticated", request.url)
        return
    for name, value in headers.items():
        request.headers[name] = value


async def attach_identity_headers_async(request: httpx2.Request) -> None:
    """Async counterpart of :func:`attach_identity_headers`.

    httpx2 requires an awaitable event hook on an ``AsyncClient``.
    """
    attach_identity_headers(request)


def AsyncClient(**kwargs: Any) -> httpx2.AsyncClient:
    """An ``httpx2.AsyncClient`` that carries the calling user's identity.

    Takes every ``httpx2.AsyncClient`` keyword argument.  Caller-supplied
    ``event_hooks`` are kept; the identity hook runs last, so it wins over a
    request hook setting the same header.

    A function returning a plain ``httpx2.AsyncClient``, not a subclass.
    """
    return httpx2.AsyncClient(
        event_hooks=_with_identity_hook(
            kwargs.pop("event_hooks", None), attach_identity_headers_async
        ),
        **kwargs,
    )


def Client(**kwargs: Any) -> httpx2.Client:
    """A synchronous ``httpx2.Client`` that carries the calling user's identity.

    See :func:`AsyncClient`.
    """
    return httpx2.Client(
        event_hooks=_with_identity_hook(
            kwargs.pop("event_hooks", None), attach_identity_headers
        ),
        **kwargs,
    )


def _is_original_origin(request: httpx2.Request) -> bool:
    """Whether *request* still addresses the origin the caller asked for.

    The hook reruns on every redirect hop, and httpx2 strips only
    ``Authorization`` when a hop leaves the origin — so without this a
    redirect from the MCP proxy would hand the OBO token to whatever host it
    names.  Mirrors httpx2's own rule, allowing only a same-host upgrade from
    HTTP on port 80 to HTTPS on port 443.
    """
    origin = request.url.origin
    original = request.extensions.get(_ORIGIN_EXTENSION)
    if original is None:
        request.extensions[_ORIGIN_EXTENSION] = origin
        return True
    return origin == original or (
        origin.host == original.host
        and original.scheme == "http"
        and original.port == _HTTP_DEFAULT_PORT
        and origin.scheme == "https"
        and origin.port == _HTTPS_DEFAULT_PORT
    )


def _with_identity_hook(
    event_hooks: Optional[Mapping[str, List[_EventHook]]], hook: _EventHook
) -> Dict[str, List[_EventHook]]:
    """Return a copy of *event_hooks* with *hook* appended to the request hooks.

    Copied so a mapping the caller reuses across clients is neither mutated
    nor left accumulating one hook per client.
    """
    merged = {name: list(hooks) for name, hooks in (event_hooks or {}).items()}
    merged[_REQUEST_EVENT_HOOK] = [*merged.get(_REQUEST_EVENT_HOOK, []), hook]
    return merged
