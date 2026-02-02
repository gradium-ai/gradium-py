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
        return "ogg"
    return None


async def run_tts(args: argparse.Namespace) -> int:
    """Run text-to-speech conversion."""
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

    # Read audio file
    with open(args.audio_file, "rb") as f:
        audio_data = f.read()

    # Infer input format from filename
    input_format = infer_format_from_filename(args.audio_file)
    if input_format is None:
        input_format = "wav"

    setup = {
        "input_format": input_format,
    }

    result = await client.stt(setup, audio_data)

    if args.json:
        output = {
            "text": result.text,
            "request_id": result.request_id,
            "segments": [
                {
                    "text": seg.text,
                    "start_s": seg.start_s,
                    "stop_s": seg.stop_s,
                }
                for seg in result.text_with_timestamps
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.text)

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
