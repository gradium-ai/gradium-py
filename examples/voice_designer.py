#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gradium",
# ]
# ///
"""
Example script demonstrating the voice designer flow.

This script:
1. Generates a few draft voices from a text prompt
2. Waits for the drafts to be computed
3. Renders a short preview of each draft to a wav file
4. Optionally promotes one draft to a permanent voice
"""

import argparse
import asyncio
import pathlib

import gradium


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="A deep, warm male voice with a slight British accent",
        help="Description of the wanted voice",
    )
    parser.add_argument(
        "--language", default="en", help="Voice language: en, fr, es, pt, de"
    )
    parser.add_argument(
        "--n-samples", type=int, default=2, help="Number of drafts (1-5)"
    )
    parser.add_argument(
        "--text",
        default="Hello! This is what my new voice sounds like.",
        help="Preview text (keep it short)",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory where the preview wav files are written",
    )
    parser.add_argument(
        "--promote",
        type=int,
        default=None,
        metavar="INDEX",
        help="Index of the draft to promote to a permanent voice",
    )
    parser.add_argument(
        "--name", default="Designed voice", help="Name of the promoted voice"
    )
    parser.add_argument(
        "--api-key", default=None, help="API key (default: GRADIUM_API_KEY)"
    )
    parser.add_argument(
        "--base-url", default="https://api.gradium.ai/api", help="API base URL"
    )
    args = parser.parse_args()

    client = gradium.GradiumClient(base_url=args.base_url, api_key=args.api_key)

    print(f"Generating {args.n_samples} draft(s) for: {args.prompt!r}")
    drafts = await client.voice_generate(
        args.prompt,
        language=args.language,
        n_samples=args.n_samples,
        wait=True,
    )
    embeddings = drafts["embeddings"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, draft in enumerate(embeddings):
        embedding_id = draft["embedding_id"]
        preview = await client.voice_tts(embedding_id, args.text)
        out = args.out_dir / f"draft_{index}_{embedding_id}.wav"
        out.write_bytes(preview.raw_data)
        expires = draft["expires_at"]
        print(f"[{index}] {embedding_id} -> {out} (expires {expires})")

    if args.promote is not None:
        embedding_id = embeddings[args.promote]["embedding_id"]
        voice = await client.voice_from_embedding(embedding_id, name=args.name)
        print(f"Promoted draft {args.promote} to voice {voice['uid']}")


if __name__ == "__main__":
    asyncio.run(main())
