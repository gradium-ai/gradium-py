"""Transport layer shared by all Gradium-style clients.

This module provides :class:`BaseClient`, which handles HTTP requests and
WebSocket connections against a configurable ``base_url``. It has no knowledge
of the speech/voices/usages API surface — those live in the concrete clients
(:class:`~gradium.client.GradiumClient`, :class:`~gradium.baseten.BasetenClient`)
built on top of it.
"""

import asyncio
import json
import os
import types
import urllib.parse
from collections.abc import AsyncGenerator, Callable
from typing import Any

import aiohttp

from . import version

SOURCE = f"python-client/{version.__version__}"


async def send(
    ws: aiohttp.ClientWebSocketResponse,
    setup: dict,
    messages: list | AsyncGenerator[dict, None],
    map_fn: Callable | None = None,
) -> None:
    """Send setup and list of messages to WebSocket.

    Args:
        ws: WebSocket connection to send data through.
        setup: Initial setup configuration dictionary.
        messages: List or async generator of message dictionaries to send.
        map_fn: Optional function to transform each message before sending.
    """

    async def send_one(msg):
        msg = map_fn(msg) if map_fn is not None else msg
        if msg is not None:
            await ws.send_str(json.dumps(msg))

    await ws.send_str(json.dumps(setup))
    if isinstance(messages, types.AsyncGeneratorType):
        async for msg in messages:
            await send_one(msg)
    else:
        for msg in messages:
            await send_one(msg)
    await ws.send_str(json.dumps({"type": "end_of_stream"}))


async def receive(
    ws: aiohttp.ClientWebSocketResponse, map_fn: Callable | None = None
) -> AsyncGenerator[Any, None]:
    """Receive messages from WebSocket and yield them.

    Args:
        ws: WebSocket connection to receive data from.
        map_fn: Optional function to transform each received message.

    Yields:
        Transformed message dictionaries.

    Raises:
        Exception: If a WebSocket error occurs or connection is closed abnormally.
    """
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            chunk = json.loads(msg.data)
            chunk = map_fn(chunk) if map_fn is not None else chunk
            if chunk is None:
                continue

            if chunk.get("type") == "error":
                code = chunk.get("code")
                error = chunk.get("message")
                raise Exception(f"Websocket connection error: {error} ({code})")

            yield chunk
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise Exception("WebSocket error")
        elif (
            msg.type == aiohttp.WSMsgType.CLOSE
            or msg.type == aiohttp.WSMsgType.CLOSED
        ):
            close_code = msg.data  # the close code
            close_reason = msg.extra  # the close reason
            if close_code is not None and close_code != 1000:
                raise Exception(
                    f"WebSocket closed (code {close_code}): "
                    f"{close_reason or 'No reason provided'}"
                )


class BaseClient:
    """Transport layer for communicating with a Gradium-style API.

    Handles HTTP requests and WebSocket connections against a ``base_url``.
    Concrete clients build the speech/voices/usages API surface on top of this
    class.

    Attributes:
        _base_url: Base URL for API endpoints.
        _api_key: API key for authentication.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.gradium.ai/api/",
        api_key: str | None = None,
    ):
        """Initialize the client.

        Args:
            base_url: Base URL for the API. Defaults to the Gradium API server.
                Automatically adds protocol (http/https) if missing.
            api_key: API key for authentication. If not provided, reads from the
                GRADIUM_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        if not base_url.endswith("/"):
            base_url = base_url + "/"
        if not base_url.startswith("http"):
            islocal = base_url.startswith(("127", "local"))
            protocol = "http" if islocal else "https"
            base_url = f"{protocol}://{base_url}"
        self._base_url = base_url

        api_key = (
            api_key
            if api_key is not None
            else os.environ.get("GRADIUM_API_KEY")
        )
        if api_key is None:
            raise ValueError(
                "Missing api-key as cli or as env (GRADIUM_API_KEY)"
            )
        self._api_key = api_key

    @property
    def headers(self) -> dict:
        """Get HTTP headers with authentication."""
        return {"x-api-key": self._api_key, "x-api-source": SOURCE}

    def ws(self, session, route: str) -> aiohttp.ClientWebSocketResponse:
        """Create a WebSocket connection to the specified route.

        Args:
            session: aiohttp ClientSession.
            route: API endpoint route.

        Returns:
            WebSocket connection context manager.
        """
        url = urllib.parse.urljoin(self._base_url, route).replace("http", "ws")
        return session.ws_connect(url)

    async def stream(
        self,
        route: str,
        setup: Any,
        input_stream: list[dict[str, str]] | AsyncGenerator[dict, None],
        map_input_fn: Callable | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream data to a WebSocket endpoint and receive responses.

        This is the core method for bidirectional streaming communication with
        the API. It handles both sending and receiving concurrently.

        Args:
            route: API endpoint route.
            setup: Initial setup configuration dictionary.
            input_stream: List or async generator of input messages.
            map_input_fn: Optional function to transform input messages.

        Yields:
            Response messages from the server.

        Raises:
            Exception: If WebSocket communication fails.
        """
        setup |= {"type": "setup"}
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with self.ws(session, route) as ws:
                receive_queue = asyncio.Queue()

                async def receive_worker():
                    """Collect received chunks into queue concurrently."""
                    try:
                        async for chunk in receive(ws):
                            await receive_queue.put(chunk)
                    except Exception as e:
                        await receive_queue.put(e)
                    finally:
                        await receive_queue.put(None)  # End marker

                send_task = asyncio.create_task(
                    send(ws, setup, input_stream, map_fn=map_input_fn)
                )
                receive_task = asyncio.create_task(receive_worker())

                try:
                    # Yield chunks as they arrive from the queue
                    while True:
                        chunk = await receive_queue.get()
                        if chunk is None:  # End marker
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        yield chunk
                finally:
                    # Ensure both tasks complete
                    await asyncio.gather(
                        send_task, receive_task, return_exceptions=True
                    )

    async def _fetch(
        self, method: str, route, parse_response: bool = True, **kwargs
    ):
        """Generic HTTP request handler.

        Args:
            method: HTTP method name (get, post, put, delete, etc.).
            route: API endpoint route.
            parse_response: Whether to parse response as JSON.
            **kwargs: Additional arguments to pass to the HTTP method.

        Returns:
            Parsed JSON response or raw response object.

        Raises:
            ValueError: If the HTTP method is not supported.
            aiohttp.ClientError: If the HTTP request fails.
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = urllib.parse.urljoin(self._base_url, route)
            fn = getattr(session, method)
            if fn is None:
                raise ValueError(f"No such HTTP method {method}")
            response = await fn(url, **kwargs)

            if not response.ok:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    msg = await response.json()
                    if (reason := msg.get("detail")) is not None:
                        response.reason = reason
                else:
                    response.reason = await response.text()
            response.raise_for_status()
            return await response.json() if parse_response else response

    async def post(self, route: str, parse: bool = True, **kwargs):
        """Make a POST request to the API.

        Args:
            route: API endpoint route.
            parse: Whether to parse response as JSON.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response or raw response object.
        """
        return await self._fetch("post", route, parse_response=parse, **kwargs)

    async def put(self, route: str, **kwargs):
        """Make a PUT request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("put", route, **kwargs)

    async def get(self, route: str, **kwargs):
        """Make a GET request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("get", route, **kwargs)

    async def delete(self, route: str, **kwargs):
        """Make a DELETE request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("delete", route, **kwargs)
