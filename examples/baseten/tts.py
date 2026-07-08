#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradium"]
#
# [tool.uv.sources]
# gradium = { path = "../..", editable = true }
# ///
"""Debug text-to-speech call against a Baseten-hosted Gradium model.

A single-model Baseten deployment serves the TTS websocket at the base URL
itself and authenticates with an ``Authorization: Api-Key`` header rather than
Gradium's ``x-api-key``. :class:`gradium.BasetenClient` wires both up and
exposes only the text-to-speech endpoint.

This script opens the connection first, then sends the setup message (with the
voice) and the text, and prints debug information (the setup sent, the server
``ready`` message, latency, time-to-first-audio, throughput and per-segment
timestamps). The synthesized audio is written to a file.

Run it with uv (no manual environment setup needed)::

    uv run examples/baseten/tts.py \
        --base-url model-XX.api.baseten.co/environments/production/websocket \
        --api-key "$BASETEN_API_KEY" \
        --text "Hello from Baseten" \
        --voice Ve1zknlflaRwcAQw

Use ``--voice`` to pick a predefined voice name. The API key also falls back to
the GRADIUM_API_KEY environment variable.
"""

import argparse
import asyncio
import os
import pathlib
import platform
import sys
import time

from gradium import BasetenClient, speech, version

DEFAULT_TEXT = (
    "This is a test of the Gradium text to speech model served on Baseten."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a debug text-to-speech call against a Baseten model.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help=(
            "Baseten model websocket URL, e.g. "
            "model-XXXX.api.baseten.co/environments/production/websocket"
        ),
    )
    parser.add_argument(
        "--api-key",
        help="Baseten API key. Defaults to the GRADIUM_API_KEY env var.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: a short sample).",
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="Predefined voice name to use (default: %(default)s).",
    )
    parser.add_argument(
        "--output-format",
        default="wav",
        choices=["wav", "pcm"],
        help="Audio output format (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output file (default: baseten-tts-<session>.<format>).",
    )
    return parser


def _estimate_duration_s(
    raw_data: bytes, sample_rate: int | None, output_format: str
) -> float | None:
    """Best-effort duration of mono 16-bit audio, in seconds."""
    if not sample_rate:
        return None
    n_bytes = len(raw_data)
    if output_format == "wav":
        # Subtract a standard 44-byte PCM WAV header.
        n_bytes = max(0, n_bytes - 44)
    return (n_bytes / 2) / sample_rate


async def _run(args: argparse.Namespace) -> int:
    # BasetenClient serves TTS at the base URL itself and sends the
    # `Authorization: Api-Key <key>` header for you.
    client = BasetenClient(base_url=args.base_url, api_key=args.api_key)

    # Keep the setup minimal: the Baseten model server rejects a Setup message
    # carrying extra fields (e.g. model_name / client_req_id) that the hosted
    # Gradium API tolerates, and then reports "expected initial Setup message".
    setup: speech.TTSSetup = {
        "voice": args.voice,
        "output_format": args.output_format,
    }

    print("=== Baseten TTS debug call ===")
    print("-" * 30)
    print("SETUP")
    print("-" * 30)
    print(f"Client version : {version.__version__}")
    print(f"Python         : {platform.python_version()} ({sys.executable})")
    print(f"Platform       : {platform.platform()}")
    print(f"Base URL       : {client._base_url}")
    print("Endpoint       : (base URL websocket)")

    # Timestamps for each step of the connection (None until reached).
    t_ws_open: float | None = None
    t_eos_sent: float | None = None
    t_ready: float | None = None
    t_first_audible: float | None = None  # first frame with start_s < stop_s
    t_last_audio: float | None = None

    chunks: list[bytes] = []
    segments: list[tuple[str, float, float]] = []
    sample_rate: int | None = None
    session_id: str | None = None

    # Drive the websocket manually (send_setup_on_start=False) so we can open
    # the connection, then send the setup (with the voice) and the text, and
    # time each step separately.
    t0 = time.perf_counter()
    async with client.tts_realtime(send_setup_on_start=False) as tts:
        t_ws_open = time.perf_counter()
        await tts.send_setup(setup)
        await tts.send_text(args.text)
        await tts.send_eos()
        t_eos_sent = time.perf_counter()

        async for msg in tts:
            now = time.perf_counter()
            msg_type = msg.get("type")
            if msg_type == "ready":
                t_ready = now
                sample_rate = msg.get("sample_rate")
                # The wire field is "request_id" but it carries the server's
                # session id.
                session_id = msg.get("request_id")
            elif msg_type == "audio":
                audio = msg["audio"]  # already base64-decoded by the client
                if (
                    msg.get("start_s", 0.0) < msg.get("stop_s", 0.0)
                    and t_first_audible is None
                ):
                    t_first_audible = now
                t_last_audio = now
                chunks.append(audio)
            elif msg_type == "text":
                start_s = msg.get("start_s", 0.0)
                segments.append(
                    (msg.get("text", ""), start_s, msg.get("stop_s", start_s))
                )
            elif msg_type == "end_of_stream":
                break
    t_done = time.perf_counter()

    raw_data = b"".join(chunks)
    duration_s = _estimate_duration_s(raw_data, sample_rate, args.output_format)

    def at(t: float | None) -> str:
        """Cumulative time from connect start."""
        return "-" if t is None else f"{(t - t0) * 1000:.1f} ms"

    print("-" * 30)
    print("REQUEST")
    print("-" * 30)
    print(f"Text           : {args.text!r}")
    print(f"Session id     : {session_id}")
    print(f"Voice          : {setup.get('voice')}")
    print(f"Output format  : {setup['output_format']}")
    print("-" * 30)
    print("AUDIO RESPONSE")
    print("-" * 30)
    print(f"Sample rate    : {sample_rate} Hz")
    print(f"Messages       : {len(chunks)}")
    print(f"Bytes          : {len(raw_data)}")
    if t_ws_open is not None and t_first_audible is not None:
        print(f"TTFA           : {(t_first_audible - t_ws_open) * 1000:.1f} ms")
    if segments:
        print(
            f"Silence        : {segments[0][1]:.2f} s  "
            "(start_s of first segment, lead-in before first word)"
        )

    if duration_s is not None:
        print(f"Duration       : {duration_s:.2f} s")
        # Real-time factor: audio duration / session duration, where the
        # session spans connection-open -> stream-end. >1 = faster than realtime.
        session_s = (t_done - t_ws_open) if t_ws_open is not None else None
        if session_s and session_s > 0:
            print(
                f"RTF            : {duration_s / session_s:.2f}x  "
                "(audio/s per wall/s; >1 = faster than realtime)"
            )

    if raw_data:
        out_path = args.output or pathlib.Path(
            f"baseten-tts-{session_id or 'unknown'}.{args.output_format}"
        )
        out_path.write_bytes(raw_data)
        print(f"Saved          : {out_path.resolve()}")

    print("-" * 30)
    print("TIMELINE")
    print("-" * 30)
    print(f"Connection open    : {at(t_ws_open)}  (TCP + TLS handshake)")
    print(f"Request sent       : {at(t_eos_sent)}  (setup + text + eos)")
    print(f"Server accepted    : {at(t_ready)}  ('ready' message)")
    print(f"First audio packet : {at(t_first_audible)}  <- what a listener hears")
    print(f"Last audio packet  : {at(t_last_audio)}")
    print(f"Stream finished    : {at(t_done)}")

    if not raw_data:
        print("WARNING: no audio data received.")
        return 1

    return 0


def main() -> None:
    args = _build_parser().parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key: pass --api-key or set GRADIUM_API_KEY."
        )
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
