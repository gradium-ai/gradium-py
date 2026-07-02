#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradium"]
#
# [tool.uv.sources]
# gradium = { path = "../..", editable = true }
# ///
"""Example: text-to-speech through a Baseten-hosted Gradium model.

A single-model Baseten deployment serves the TTS websocket at the base URL
itself and authenticates with an ``Authorization: Api-Key`` header rather than
Gradium's ``x-api-key``. :class:`gradium.BasetenClient` wires both up and
exposes only the text-to-speech endpoint.

Run it with uv (no manual environment setup needed)::

    uv run examples/baseten/tts.py \
        --base-url model-XXX.api.baseten.co/environments/production/websocket \
        --api-key  \
        --text "Hello this TTS is running on prem." \
        --voice Ve1zknlflaRwcAQw	

Use ``--voice`` to pick a predefined voice name. The API key also falls back to
the GRADIUM_API_KEY environment variable.
"""

import argparse
import asyncio
import os

from gradium import BasetenClient

DEFAULT_TEXT = (
    "This is a test of the Gradium text to speech model served on Baseten."
)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize speech through a Baseten-hosted Gradium model"
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
        help="Baseten API key (falls back to the GRADIUM_API_KEY env var)",
    )
    parser.add_argument(
        "--text", default=DEFAULT_TEXT, help="Text to synthesize"
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="Predefined voice name to use (e.g. 'default')",
    )
    parser.add_argument("--output", default="out.wav", help="Output WAV file")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GRADIUM_API_KEY")
    if not api_key:
        parser.error(
            "Baseten API key must be provided via --api-key or the "
            "GRADIUM_API_KEY environment variable"
        )

    # BasetenClient serves TTS at the base URL itself and sends the
    # `Authorization: Api-Key <key>` header for you.
    client = BasetenClient(base_url=args.base_url, api_key=api_key)

    setup = {"voice": args.voice, "output_format": "wav"}
    result = await client.tts(setup, args.text)

    with open(args.output, "wb") as f:
        f.write(result.raw_data)
    print(f"Wrote {len(result.raw_data)} bytes to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
