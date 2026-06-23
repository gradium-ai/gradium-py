"""Example script for the Speech-to-Speech (S2S) WebSocket API.

Streams input audio to the server and writes the synthesized output audio
back to a file while printing the transcribed (optionally translated) text.
"""

import argparse
import asyncio
import os
import time

import numpy as np
import sphn

from gradium import client as gradium_client

# PCM input is 24kHz mono, output is 48kHz mono (16-bit signed).
INPUT_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 48000


async def main():
    parser = argparse.ArgumentParser(description="Test S2S WebSocket API")
    parser.add_argument("--url", default="https://api.gradium.ai/api")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GRADIUM_API_KEY"),
        help="API key for authentication (defaults to $GRADIUM_API_KEY)",
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Input audio file"
    )
    parser.add_argument(
        "--out", type=str, default="s2s-out.wav", help="Output audio file"
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="S2S model name"
    )
    parser.add_argument(
        "--stt-model-name",
        type=str,
        default="default",
        help="STT model name (default: 'default')",
    )
    parser.add_argument(
        "--tts-model-name",
        type=str,
        default="default",
        help="TTS model name (default: 'default')",
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default="cLONiZ4hQ8VpQ4Sz",
        help="Output voice id",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        help="Language to translate to (omit to keep the original language)",
    )
    args = parser.parse_args()

    # Read and resample the input to 24kHz mono int16 PCM with sphn.
    pcm, _ = sphn.read(args.audio, sample_rate=INPUT_SAMPLE_RATE)
    pcm = (pcm[0] * 32768).astype(np.int16)
    print(f"loaded audio, {len(pcm)} samples, {pcm.shape}")

    grc = gradium_client.GradiumClient(base_url=args.url, api_key=args.api_key)
    setup = {
        "model_name": args.model_name,
        "input_format": "pcm_24000",
        "output_format": "pcm_48000",
        "voice_id": args.voice_id,
        "stt_model_name": args.stt_model_name,
        "tts_model_name": args.tts_model_name,
    }
    if args.target_language is not None:
        setup["json_config"] = {"target_language": args.target_language}

    all_bytes = []
    all_text = []
    start_time = time.time()

    # Open a real-time duplex S2S connection (see gradium/stream.py). Audio is
    # streamed in while output audio and text are streamed back concurrently.
    async with grc.s2s_realtime(wait_for_ready_on_start=True, **setup) as s2s:
        print("Ready:", s2s.ready)

        async def send_loop():
            # 1920 samples == 80ms at 24kHz.
            for i in range(0, len(pcm), 1920):
                await s2s.send_audio(pcm[i : i + 1920])
            await s2s.send_eos()

        async def recv_loop():
            async for msg in s2s:
                if msg["type"] == "audio":
                    all_bytes.append(msg["audio"])  # already decoded bytes
                elif msg["type"] == "text":
                    all_text.append(msg["text"])
                    # Print transcribed text as it arrives.
                    print(msg["text"], end=" ", flush=True)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(send_loop())
            tg.create_task(recv_loop())

    total_time = time.time() - start_time

    # Output is raw 48kHz int16 PCM; convert to float [-1, 1] and write a wav.
    out_pcm = np.frombuffer(b"".join(all_bytes), dtype=np.int16)
    out_float = out_pcm.astype(np.float32) / 32768.0
    sample_rate = (s2s.ready or {}).get("sample_rate", OUTPUT_SAMPLE_RATE)
    sphn.write_wav(args.out, out_float, sample_rate)
    print(
        f"Wrote {len(out_pcm)} samples ({sample_rate}Hz) to {args.out} "
        f"in {total_time:.2f}s"
    )

    print(f"\nTranscript: {' '.join(all_text)}")


if __name__ == "__main__":
    asyncio.run(main())
