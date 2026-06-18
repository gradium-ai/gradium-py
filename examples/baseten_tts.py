#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gradium"]
#
# [tool.uv.sources]
# gradium = { path = "../", editable = true }
# ///
"""Minimal TTS snippet hitting a Gradium model served through Baseten.

The Baseten deployment is reached via ``additional_headers`` carrying the
Baseten ``Authorization`` credential instead of a Gradium ``x-api-key``.

Run with uv (uses the local gradium checkout):

    uv run examples/baseten_tts.py \\
        --base-url model-XX.api.baseten.co/environments/production/websocket \\
        --api-key XXXX \\
        --voice default
"""

import argparse
import asyncio

from gradium import GradiumClient, RoutesBasetenTTS


async def synthesize(
    base_url: str,
    api_key: str,
    voice: str,
    text: str,
    output_format: str,
    out_path: str,
) -> None:
    # On a single-model Baseten deployment the websocket is served at the base
    # URL itself, so RoutesBasetenTTS sets the TTS route to "" (no "speech/tts"
    # suffix). stt/voices/usages are unsupported there and raise
    # NotImplementedError.
    client = GradiumClient(
        base_url=base_url,
        additional_headers={"Authorization": f"Api-Key {api_key}"},
        routes=RoutesBasetenTTS(),
    )

    audio = await client.tts(
        setup={"voice": voice, "output_format": output_format},
        text=text,
    )

    with open(out_path, "wb") as f:
        f.write(audio.raw_data)
    print(f"Wrote {len(audio.raw_data)} bytes to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradium TTS via a Baseten deployment"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Baseten websocket base URL "
        "(e.g. model-XX.api.baseten.co/environments/production/websocket)",
    )
    parser.add_argument("--api-key", required=True, help="Baseten API key")
    parser.add_argument("--voice", default="default", help="Voice id to use")
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the Gradium text-to-speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output-format", default="wav", help="Audio output format"
    )
    parser.add_argument(
        "--out", default="out.wav", help="Output audio file path"
    )
    args = parser.parse_args()

    asyncio.run(
        synthesize(
            base_url=args.base_url,
            api_key=args.api_key,
            voice=args.voice,
            text=args.text,
            output_format=args.output_format,
            out_path=args.out,
        )
    )


if __name__ == "__main__":
    main()
