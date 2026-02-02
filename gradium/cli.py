"""Command-line interface for Gradium API.

Provides TTS (text-to-speech) and STT (speech-to-text) commands.

Usage:
    gradium tts "Hello world" -o output.wav
    gradium stt audio.wav
    gradium stt audio.wav --json
"""

import argparse
import asyncio
import json
import sys

from .client import GradiumClient


def infer_format_from_filename(filename: str) -> str | None:
    """Infer audio format from filename extension."""
    if filename.endswith(".wav"):
        return "wav"
    elif filename.endswith(".pcm"):
        return "pcm"
    elif filename.endswith(".ogg"):
        return "opus"
    return None


async def run_tts(args: argparse.Namespace) -> int:
    """Run text-to-speech conversion."""
    if args.output is None:
        raise ValueError("Output file must be specified for TTS command.")

    client = GradiumClient(
        base_url=args.gradium_base_url,
        api_key=args.api_key,
    )

    # Determine output format
    output_format = args.format
    if output_format is None and args.output:
        output_format = infer_format_from_filename(args.output)
    if output_format is None:
        output_format = "wav"

    setup = {
        "output_format": output_format,
    }
    if args.voice_id:
        setup["voice_id"] = args.voice_id

    result = await client.tts(setup, args.text)

    if args.output:
        with open(args.output, "wb") as f:
            f.write(result.raw_data)
    else:
        sys.stdout.buffer.write(result.raw_data)

    return 0


async def run_stt(args: argparse.Namespace) -> int:
    """Run speech-to-text transcription."""
    client = GradiumClient(
        base_url=args.gradium_base_url,
        api_key=args.api_key,
    )

    # For .wav files, read directly; for other formats, use sphn
    if args.audio_file.endswith(".wav"):
        with open(args.audio_file, "rb") as f:
            audio_data = f.read()

        async def audio_stream():
            chunk_size = 4096
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i : i + chunk_size]

        setup = {"input_format": "wav"}
        stream = await client.stt_stream(setup, audio_stream())
    else:
        try:
            import sphn
        except ImportError:
            print(
                "sphn is required for non-.wav files. "
                "Install with: pip install gradium[cli]",
                file=sys.stderr,
            )
            return 1

        import numpy as np

        pcm, _ = sphn.read(args.audio_file, sample_rate=24000)
        # Convert to single channel int16 PCM
        pcm = (pcm[0] * 32768).astype(np.int16)

        async def audio_stream():
            chunk_size = 1920
            for i in range(0, len(pcm), chunk_size):
                yield pcm[i : i + chunk_size]

        setup = {"input_format": "pcm"}
        stream = await client.stt_stream(setup, audio_stream())

    if args.json:
        segments = []
        async for seg in stream.iter_text():
            segment = {
                "text": seg.text,
                "start_s": seg.start_s,
                "stop_s": seg.stop_s,
            }
            segments.append(segment)
            print(json.dumps(segment), flush=True)
        # Print summary at end
        print(json.dumps({"request_id": stream.request_id}), flush=True)
    else:
        async for seg in stream.iter_text():
            print(seg.text, end=" ", flush=True)
        print()

    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="gradium",
        description="Gradium Voice AI CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TTS subcommand
    tts_parser = subparsers.add_parser(
        "tts",
        help="Convert text to speech",
        description="Convert text to speech audio",
    )
    tts_parser.add_argument("text", help="Text to convert to speech")
    tts_parser.add_argument(
        "--voice-id",
        help="Voice ID to use for synthesis",
    )
    tts_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
    )
    tts_parser.add_argument(
        "-f",
        "--format",
        help="Output audio format (inferred from filename if not specified)",
    )
    tts_parser.add_argument(
        "--gradium-base-url",
        default="https://eu.api.gradium.ai/api",
        help="Gradium API base URL",
    )
    tts_parser.add_argument(
        "--api-key",
        help="API key (default: GRADIUM_API_KEY env var)",
    )

    # STT subcommand
    stt_parser = subparsers.add_parser(
        "stt",
        help="Transcribe speech to text",
        description="Transcribe audio file to text",
    )
    stt_parser.add_argument("audio_file", help="Audio file to transcribe")
    stt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON with timestamps",
    )
    stt_parser.add_argument(
        "--gradium-base-url",
        default="https://eu.api.gradium.ai/api",
        help="Gradium API base URL",
    )
    stt_parser.add_argument(
        "--api-key",
        help="API key (default: GRADIUM_API_KEY env var)",
    )

    args = parser.parse_args()

    if args.command == "tts":
        return asyncio.run(run_tts(args))
    elif args.command == "stt":
        return asyncio.run(run_stt(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())
