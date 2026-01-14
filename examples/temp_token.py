#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "websockets",
# ]
# ///
"""
Example script demonstrating temporary token authentication for TTS.

This script:
1. Fetches a temporary token using an API key
2. Connects to the TTS WebSocket endpoint using the token
3. Sends text for synthesis
4. Receives and saves audio chunks to out.wav
"""

import argparse
import asyncio
import base64
import json
import os
from urllib.parse import urlencode

import requests
import websockets


async def synthesize_speech(uri: str, api_key: str, text: str):
    """
    Synthesize speech using temporary token authentication.

    Args:
        uri: Base API URI (e.g., https://eu.api.gradium.ai/api)
        api_key: API key for authentication
        text: Text to synthesize
    """
    # Step 1: Get temporary token
    print(f"Fetching temporary token from {uri}/api-keys/token...")
    response = requests.get(
        f"{uri}/api-keys/token", headers={"x-api-key": api_key}
    )
    response.raise_for_status()

    token_data = response.json()
    token = token_data["token"]
    print(f"Received token: {token[:20]}...")

    # Step 2: Construct WebSocket URL
    # Replace http(s) with ws(s) for WebSocket
    ws_uri = uri.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_uri}/speech/tts?token={token}"
    print(f"Connecting to WebSocket: {ws_url}")

    # Step 3: Connect to WebSocket and send messages
    audio_chunks = []

    async with websockets.connect(ws_url) as websocket:
        print("WebSocket connected")

        # Send setup message
        setup_msg = {
            "type": "setup",
            "voice_id": "YTpq7expH9539ERJ",
            "output_format": "wav",
        }
        await websocket.send(json.dumps(setup_msg))
        print(f"Sent setup message: {setup_msg}")

        # Send text message
        text_msg = {"type": "text", "text": text}
        await websocket.send(json.dumps(text_msg))
        print(f"Sent text message: {text}")

        # Send end of stream
        end_msg = {"type": "end_of_stream"}
        await websocket.send(json.dumps(end_msg))
        print("Sent end_of_stream message")

        # Step 4: Receive audio messages
        print("Receiving audio chunks...")
        async for message in websocket:
            data = json.loads(message)

            if data.get("type") == "audio":
                # Decode base64 audio and append
                audio_data = base64.b64decode(data["audio"])
                audio_chunks.append(audio_data)
                print(f"Received audio chunk: {len(audio_data)} bytes")
            else:
                print(f"Received message: {data}")

    # Step 5: Save audio to file
    if audio_chunks:
        output_file = "out.wav"
        with open(output_file, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        print(
            f"Audio saved to {output_file} ({sum(len(c) for c in audio_chunks)} bytes total)"
        )
    else:
        print("No audio chunks received")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using Gradium TTS API with temporary token"
    )
    parser.add_argument(
        "--uri",
        default="https://eu.api.gradium.ai/api",
        help="Base API URI (default: https://eu.api.gradium.ai/api)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (default: from GRADIUM_API_KEY env var)",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the temporary token authentication.",
        help="Text to synthesize",
    )

    args = parser.parse_args()

    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get("GRADIUM_API_KEY")
    if not api_key:
        parser.error(
            "API key must be provided via --api-key or GRADIUM_API_KEY environment variable"
        )

    # Run the async function
    asyncio.run(synthesize_speech(args.uri, api_key, args.text))


if __name__ == "__main__":
    main()
