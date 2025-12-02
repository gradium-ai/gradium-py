"""Example script for TTS WebSocket API."""

import argparse
import asyncio
import base64
import sphn
import time

from gradium import client as gradium_client


DEFAULT_TEXT = """
This is a test of the text to speech streaming capabilities of the Gradium API.
"""


async def main():
    parser = argparse.ArgumentParser(description="Test TTS WebSocket API")
    parser.add_argument("--url", default="https://eu.api.gradium.ai/api")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--text", help="Text to synthesize", default=DEFAULT_TEXT
    )
    parser.add_argument("--voice", help="The predefined voice to be used")
    parser.add_argument("--voice-id", help="The predefined voice to be used")
    parser.add_argument(
        "--model", help="The model to be used", default="default"
    )
    parser.add_argument(
        "--num-runs", type=int, default=1, help="Number of runs to perform"
    )
    parser.add_argument("--max-padding", type=int)
    parser.add_argument("--padding-between", type=int)
    parser.add_argument("--max-padding-per-token", type=int)
    parser.add_argument("--padding-bonus", type=float)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--rewrite-rules", type=str)
    args = parser.parse_args()

    tasks = [run_one(args, id=i) for i in range(args.num_runs)]
    await asyncio.gather(*tasks)


async def run_one(args, id: int):
    if args.api_key is None:
        args.api_key = "dummy"
    grc = gradium_client.GradiumClient(base_url=args.url, api_key=args.api_key)

    setup = {
        "model_name": args.model,
        "output_format": "wav",
        "retry_for_s": 30.0,
        "json_config": {},
    }
    if args.voice is None and args.voice_id is None:
        setup["voice_id"] = "m86j6D7UZpGzHsNu"
    elif args.voice is None:
        setup["voice_id"] = args.voice_id
    elif args.voice_id is None:
        setup["voice"] = args.voice
    else:
        raise ValueError("cannot set both --voice and --voice-id")

    if args.max_padding is not None:
        setup["json_config"]["max_padding"] = args.max_padding
    if args.max_padding_per_token is not None:
        setup["json_config"]["max_padding_per_token"] = (
            args.max_padding_per_token
        )
    if args.padding_bonus is not None:
        setup["json_config"]["padding_bonus"] = args.padding_bonus
    if args.padding_between is not None:
        setup["json_config"]["padding_between"] = args.padding_between
    if args.rewrite_rules is not None:
        setup["json_config"]["rewrite_rules"] = args.rewrite_rules

    async def text_gen():
        if args.no_stream:
            yield args.text
            return
        for text in args.text.split(" "):
            yield text

    start_time = time.time()
    stream = await grc.tts_stream(setup, text_gen())
    all_bytes = []
    all_text = []
    first_chunk = True
    async for msg in stream._stream:
        msg_type = msg.get("type")
        if msg_type == "audio":
            if msg["stop_s"] > 0.0 and first_chunk:
                dt = time.time() - start_time
                print(f"Time to first audio: {1000 * dt:.2f}ms")
                first_chunk = False
            bytes_ = base64.b64decode(msg["audio"])
            all_bytes.append(bytes_)
        elif msg_type == "text":
            all_text.append(msg["text"])
    total_time = time.time() - start_time
    all_bytes = b"".join(all_bytes)
    filename = f"out-{id}.wav"
    with open(filename, "wb") as f:
        f.write(all_bytes)
    pcm, sample_rate = sphn.read(filename)
    audio_duration = len(pcm[0]) / sample_rate
    print(
        f"Total time: {total_time:.2f}s, rtf: {audio_duration / total_time:.2f}x"
    )
    print(f"Generated text: {' '.join(all_text)}")


if __name__ == "__main__":
    asyncio.run(main())
