#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradium"]
#
# [tool.uv.sources]
# gradium = { path = "../..", editable = true }
# ///
"""Batched time-to-first-audio (TTFA) benchmark for a Baseten TTS model.

Measures how a single-model Baseten deployment handles a burst of concurrent
requests. It runs in two phases so the TTFA numbers reflect server latency
only, not connection setup:

1. Open all N websockets concurrently and wait until every connection is
   established (the TCP + TLS handshake and the WebSocket upgrade are done).
2. Fire the setup + text on all of them at once, and for each request measure
   TTFA = time from sending the request to the first audible audio frame.

Then it prints the TTFA distribution (min / mean / median / p90 / p99 / max).

Run it with uv (no manual environment setup needed)::

    uv run examples/baseten/ttfa_batch.py \
        --base-url XX/websocket \
        --api-key "$BASETEN_API_KEY" \
        --num 8 \
        --text "Hello from Basetenm, this is generated online" \
        --voice default

The API key also falls back to the GRADIUM_API_KEY environment variable.
"""

import argparse
import asyncio
import contextlib
import math
import os
import time

from gradium import BasetenClient, speech

DEFAULT_TEXT = (
    "This is a test of the Gradium text to speech model served on Baseten."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batched TTFA benchmark against a Baseten TTS model.",
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
        default=os.environ.get("GRADIUM_API_KEY"),
        help="Baseten API key. Defaults to the GRADIUM_API_KEY env var.",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=8,
        help="Number of concurrent requests in the batch (default: %(default)s).",
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
    return parser


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 1]) of a non-empty list."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


async def _drive(tts, setup: speech.TTSSetup, text: str) -> float | None:
    """Send the request on an already-open connection; return TTFA in ms.

    TTFA is measured from just before the setup is sent to the first audible
    audio frame (a frame whose ``start_s < stop_s``; the server may emit an
    initial priming frame with ``start_s == stop_s == 0``). Returns ``None`` if
    the stream ends without any audible audio.
    """
    t_send = time.perf_counter()
    await tts.send_setup(setup)
    await tts.send_text(text)
    await tts.send_eos()

    async for msg in tts:
        if msg.get("type") == "audio":
            if msg.get("start_s", 0.0) < msg.get("stop_s", 0.0):
                return (time.perf_counter() - t_send) * 1000.0
    return None


async def _run(args: argparse.Namespace) -> int:
    client = BasetenClient(base_url=args.base_url, api_key=args.api_key)
    setup: speech.TTSSetup = {
        "voice": args.voice,
        "output_format": args.output_format,
    }
    n = args.num

    print("=== Baseten batched TTFA benchmark ===")
    print(f"Base URL    : {client._base_url}")
    print(f"Batch size  : {n}")
    print(f"Text        : {args.text!r}")
    print(f"Voice       : {args.voice}")
    print("-" * 40)

    async with contextlib.AsyncExitStack() as stack:
        # Phase 1: open all websockets concurrently and wait until every
        # connection is established (send_setup_on_start=False, so no setup is
        # sent yet).
        t_open0 = time.perf_counter()
        conns = await asyncio.gather(
            *(
                stack.enter_async_context(
                    client.tts_realtime(send_setup_on_start=False)
                )
                for _ in range(n)
            )
        )
        t_open1 = time.perf_counter()
        print(
            f"Opened {n} connections in {(t_open1 - t_open0) * 1000:.1f} ms "
            "(all established)"
        )

        # Phase 2: fire setup + text on all connections at once and measure
        # per-request TTFA.
        t_fire0 = time.perf_counter()
        results = await asyncio.gather(
            *(_drive(tts, setup, args.text) for tts in conns),
            return_exceptions=True,
        )
        t_fire1 = time.perf_counter()

    # Partition results into successes (TTFA) and failures.
    ttfas: list[float] = []
    failures: list[str] = []
    print("-" * 40)
    print("PER-REQUEST TTFA")
    print("-" * 40)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            failures.append(f"#{i}: {type(r).__name__}: {r}")
            print(f"  #{i:>3}  ERROR  {type(r).__name__}: {r}")
        elif r is None:
            failures.append(f"#{i}: no audible audio")
            print(f"  #{i:>3}  ERROR  no audible audio")
        else:
            ttfas.append(r)
            print(f"  #{i:>3}  {r:8.1f} ms")

    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Requests    : {n}")
    print(f"Succeeded   : {len(ttfas)}")
    print(f"Failed      : {len(failures)}")
    print(f"Batch wall  : {(t_fire1 - t_fire0) * 1000:.1f} ms")
    if ttfas:
        print(f"TTFA min    : {min(ttfas):8.1f} ms")
        print(f"TTFA mean   : {sum(ttfas) / len(ttfas):8.1f} ms")
        print(f"TTFA median : {_percentile(ttfas, 0.50):8.1f} ms")
        print(f"TTFA p90    : {_percentile(ttfas, 0.90):8.1f} ms")
        print(f"TTFA p99    : {_percentile(ttfas, 0.99):8.1f} ms")
        print(f"TTFA max    : {max(ttfas):8.1f} ms")

    return 0 if ttfas and not failures else 1


def main() -> None:
    args = _build_parser().parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing API key: pass --api-key or set GRADIUM_API_KEY."
        )
    if args.num < 1:
        raise SystemExit("--num must be >= 1.")
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
