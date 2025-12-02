"""Gradium Python Client.

The Gradium Python client provides a simple and async-first interface to interact
with the Gradium API for speech synthesis, speech recognition, voice management,
and usage tracking.

Main Components:
    - GradiumClient: Main client class for API communication
    - speech: Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) modules
    - voices: Voice creation and management functionality
    - usages: Credit and usage tracking

Example:
    Basic usage of the Gradium client:

    >>> from gradium import GradiumClient, TTSSetup
    >>> import asyncio
    >>>
    >>> async def main():
    ...     client = GradiumClient(api_key="your-api-key")
    ...     result = await client.tts(
    ...         setup=TTSSetup(voice="default"),
    ...         text="Hello, world!"
    ...     )
    ...     # result contains audio data
    >>>
    >>> asyncio.run(main())
"""

from . import speech, usages, voices
from .client import GradiumClient
from .speech import TTSSetup, STTSetup

__all__ = ["GradiumClient", "TTSSetup", "STTSetup"]
