"""Tests for the abnormal-close handling in Tts/S2s/Stt.recv().

When the server drops the connection without a clean close handshake (e.g.
enforcing a session limit by resetting the TCP connection instead of sending
a "type": "error" text frame), aiohttp's `receive()` keeps returning the same
WSMsgType.ERROR/CLOSED message forever rather than raising. Before this fix,
`recv()`'s `while True` loop treated any non-TEXT message as "skip and retry",
so it spun forever instead of surfacing a diagnosable error to the caller.
"""

import json
from unittest.mock import AsyncMock

import aiohttp
import pytest

from gradium import stream


class FakeWs:
    """Minimal stand-in for aiohttp.ClientWebSocketResponse.

    `receive()` returns each of `messages` in order; `exception()` and
    `close_code` mirror what aiohttp exposes after an abnormal close.
    """

    def __init__(self, messages, exception=None, close_code=None):
        self._messages = list(messages)
        self.receive = AsyncMock(side_effect=self._next)
        self._exception = exception
        self.close_code = close_code

    async def _next(self):
        return self._messages.pop(0)

    def exception(self):
        return self._exception


def make_text_message(payload: dict) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(payload), None)


@pytest.mark.parametrize(
    "stream_cls",
    [stream.Tts, stream.S2s, stream.Stt],
)
async def test_recv_returns_none_on_clean_close(stream_cls):
    obj = stream_cls(client=None, send_setup_on_start=False)
    obj._ws = FakeWs([aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, None, None)])

    assert await obj.recv() is None


@pytest.mark.parametrize(
    "stream_cls",
    [stream.Tts, stream.S2s, stream.Stt],
)
async def test_recv_raises_on_websocket_error(stream_cls):
    """A raw connection reset (no close handshake) must raise, not loop."""
    obj = stream_cls(client=None, send_setup_on_start=False)
    exc = ConnectionResetError("Connection reset without closing handshake")
    obj._ws = FakeWs(
        [aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, exc, None)],
        exception=exc,
        close_code=None,
    )

    with pytest.raises(RuntimeError, match="closed unexpectedly"):
        await obj.recv()


@pytest.mark.parametrize(
    "stream_cls",
    [stream.Tts, stream.S2s, stream.Stt],
)
async def test_recv_raises_on_abnormal_closed_without_prior_close(stream_cls):
    """WSMsgType.CLOSED reached without ever seeing CLOSE means the
    connection died abnormally; the old code looped on this forever."""
    obj = stream_cls(client=None, send_setup_on_start=False)
    obj._ws = FakeWs(
        [aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None)],
        close_code=1006,
    )

    with pytest.raises(RuntimeError, match="close_code=1006"):
        await obj.recv()


async def test_recv_still_raises_on_explicit_error_frame():
    """A well-behaved "type": "error" text frame keeps working exactly as
    before — this fix only touches connection-level (non-TEXT) failures."""
    obj = stream.Tts(client=None, send_setup_on_start=False)
    obj._ws = FakeWs(
        [make_text_message({"type": "error", "message": "too long"})]
    )

    with pytest.raises(RuntimeError, match="Error from server"):
        await obj.recv()


async def test_recv_returns_audio_message_unaffected():
    """Sanity check that normal traffic through the loop is untouched."""
    obj = stream.Tts(client=None, send_setup_on_start=False)
    obj._ws = FakeWs(
        [
            make_text_message(
                {
                    "type": "audio",
                    "audio": "AAAA",
                    "start_s": 0.0,
                    "stop_s": 1.0,
                }
            )
        ]
    )

    msg = await obj.recv()
    assert msg["type"] == "audio"
    assert msg["audio"] == b"\x00\x00\x00"
