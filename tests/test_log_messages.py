"""Server `log` messages go to the `gradium` logger, not to the stream."""

import json
import logging
import types

import aiohttp
import pytest

from gradium import client as gradium_client
from gradium import stream

LOG = {
    "type": "log",
    "level": "warning",
    "message": "unused json_config keys: tmep",
}
READY = {"type": "ready", "request_id": "r"}
STEP = {"type": "step", "step_idx": 0}


def _text(payload: dict):
    return types.SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload)
    )


class _FakeWs:
    """A websocket that plays back `payloads`, then closes."""

    def __init__(self, payloads: list[dict]):
        self._msgs = [_text(p) for p in payloads]

    async def receive(self):
        if self._msgs:
            return self._msgs.pop(0)
        return types.SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._msgs:
            raise StopAsyncIteration
        return self._msgs.pop(0)


def _logged(caplog) -> list[tuple[int, str]]:
    return [(r.levelno, r.getMessage()) for r in caplog.records]


async def test_receive_logs_and_skips_log_messages(caplog):
    caplog.set_level(logging.DEBUG, logger="gradium")
    ws = _FakeWs([READY, LOG, STEP])

    chunks = [chunk async for chunk in gradium_client.receive(ws)]

    assert chunks == [READY, STEP]
    assert _logged(caplog) == [
        (logging.WARNING, "server: unused json_config keys: tmep")
    ]


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("error", logging.ERROR),
        ("something-new", logging.WARNING),
    ],
)
def test_handle_log_uses_the_message_level(caplog, level, expected):
    caplog.set_level(logging.DEBUG, logger="gradium")

    assert gradium_client.handle_log(
        {"type": "log", "level": level, "message": "m"}
    )
    assert _logged(caplog) == [(expected, "server: m")]


def test_handle_log_leaves_other_messages_alone(caplog):
    assert not gradium_client.handle_log(READY)
    assert _logged(caplog) == []


@pytest.mark.parametrize(
    "cls",
    [stream.Tts, stream.Stt, stream.S2s],
)
async def test_streams_skip_log_messages(caplog, cls):
    """A log before or after `ready` reaches neither wait_for_ready nor recv."""
    caplog.set_level(logging.DEBUG, logger="gradium")
    s = cls(client=None)
    s._ws = _FakeWs([LOG, READY, LOG, STEP])

    assert await s.wait_for_ready() == READY
    assert await s.recv() == STEP
    assert await s.recv() is None
    assert len(_logged(caplog)) == 2
