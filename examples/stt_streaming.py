"""Example script for the Speech-to-Text API."""

import argparse
import asyncio
import numpy as np
import sphn

from gradium import client as gradium_client


async def main():
    parser = argparse.ArgumentParser(description="Test STT WebSocket API")
    parser.add_argument(
        "--url",
        default="https://eu.api.gradium.ai/api",
    )
    parser.add_argument(
        "--api-key", required=True, help="API key for authentication"
    )
    parser.add_argument(
        "--audio", type=str, help="Audio to transcribe", default=None
    )
    parser.add_argument(
        "--model-name", type=str, help="ASR model name", default="default"
    )
    parser.add_argument("--language", type=str)
    parser.add_argument("--temp", type=str)
    args = parser.parse_args()

    pcm, _ = sphn.read(args.audio, sample_rate=24000)
    # Convert to single channel, int16 PCM, we expect 24khz
    pcm = (pcm[0] * 32768).astype(np.int16)
    print(f"loaded audio, {len(pcm)} samples, {pcm.shape}")

    grc = gradium_client.GradiumClient(base_url=args.url, api_key=args.api_key)
    setup = {
        "model_name": args.model_name,
        "input_format": "pcm",
    }
    json_config = {}
    if args.language is not None:
        json_config["language"] = args.language
    if args.temp is not None:
        json_config["temp"] = float(args.temp)
    if len(json_config) > 0:
        setup["json_config"] = json_config

    async def audio_gen(audio, chunk_size: int):
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    stream = await grc.stt_stream(setup, audio_gen(pcm, 1920))

    # Use iter_text to only retrieve the text with timestamps.
    # For VAD information, we use the _stream directly.
    vad_step = 0
    async for msg in stream._stream:
        type_ = msg.get("type")
        if type_ == "text":
            print(msg)
        elif type_ == "step":  # VAD information
            vad_step += 1
            # VAD steps occur every 80ms. We recommend using
            # msg["vad"][2]["inactivity_prob"]
            # As the probability for the turn to be finished.
            if vad_step < 10:
                print("VAD", msg)


if __name__ == "__main__":
    asyncio.run(main())
