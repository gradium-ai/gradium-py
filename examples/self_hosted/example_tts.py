#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradium"]
#
# [tool.uv.sources]
# gradium = { path = "../..", editable = true }
# ///
"""Text-to-speech against a self-hosted / proxied Gradium model.
    uv run examples/self_hosted/example_tts.py \\
        --base-url wss-host.example.com/websocket \\
        --api-key "$GRADIUM_API_KEY" \\
        --text "Hello world" \\
        --voice default

"""

import argparse
import asyncio
import os

from gradium import GradiumClient
from gradium.client import SOURCE


class SelfHostedTTSClient(GradiumClient):
    """GradiumClient for a self-hosted deployment.
    """

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Api-Key {self._api_key}",
            "x-api-source": SOURCE,
        }


async def _run(args: argparse.Namespace) -> None:
    # tts_route="" -> the TTS websocket is served at the base URL directly.
    client = SelfHostedTTSClient(
        base_url=args.base_url, api_key=args.api_key, tts_route=""
    )
    setup = {"voice_id": args.voice, "output_format": args.output_format}

    stream = await client.tts_stream(setup, args.text)
    audio = b"".join([chunk async for chunk in stream.iter_bytes()])

    with open(args.output, "wb") as f:
        f.write(audio)
    print(f"Wrote {len(audio)} bytes to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-to-speech against a self-hosted Gradium model.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Model websocket base URL, e.g. wss-host.example.com/websocket",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GRADIUM_API_KEY"),
        help="API key. Defaults to the GRADIUM_API_KEY env var.",
    )
    parser.add_argument(
        "--text",
        default="This is a test of the Gradium text to speech model.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--voice", default="default", help="Predefined voice name to use."
    )
    parser.add_argument(
        "--output-format",
        default="wav",
        choices=["wav", "pcm"],
        help="Audio output format (default: %(default)s).",
    )
    parser.add_argument(
        "--output", default="out.wav", help="Output audio file."
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing API key: pass --api-key or set GRADIUM_API_KEY."
        )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
