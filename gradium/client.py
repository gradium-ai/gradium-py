"""A generic client for Gradium API.

This module provides the main GradiumClient class for communicating with the
Gradium API via both HTTP and WebSocket connections.
"""

import asyncio
import json
import os
import types
import urllib.parse
from collections.abc import AsyncGenerator, Callable
from typing import Any

import aiohttp
import numpy as np

from . import speech, usages, version, voices

SOURCE = f"python-client/{version.__version__}"


async def send(
    ws: aiohttp.ClientWebSocketResponse,
    setup: dict,
    messages: list | AsyncGenerator[dict, None],
    map_fn: Callable | None = None,
) -> None:
    """Send setup and list of messages to WebSocket.

    Args:
        ws: WebSocket connection to send data through.
        setup: Initial setup configuration dictionary.
        messages: List or async generator of message dictionaries to send.
        map_fn: Optional function to transform each message before sending.
    """

    async def send_one(msg):
        msg = map_fn(msg) if map_fn is not None else msg
        if msg is not None:
            await ws.send_str(json.dumps(msg))

    await ws.send_str(json.dumps(setup))
    if isinstance(messages, types.AsyncGeneratorType):
        async for msg in messages:
            await send_one(msg)
    else:
        for msg in messages:
            await send_one(msg)
    await ws.send_str(json.dumps({"type": "end_of_stream"}))


async def receive(
    ws: aiohttp.ClientWebSocketResponse, map_fn: Callable | None = None
) -> AsyncGenerator[Any, None]:
    """Receive messages from WebSocket and yield them.

    Args:
        ws: WebSocket connection to receive data from.
        map_fn: Optional function to transform each received message.

    Yields:
        Transformed message dictionaries.

    Raises:
        Exception: If a WebSocket error occurs or connection is closed abnormally.
    """
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            chunk = json.loads(msg.data)
            chunk = map_fn(chunk) if map_fn is not None else chunk
            if chunk is None:
                continue

            if chunk.get("type") == "error":
                code = chunk.get("code")
                error = chunk.get("message")
                raise Exception(f"Websocket connection error: {error} ({code})")

            yield chunk
        elif msg.type == aiohttp.WSMsgType.ERROR:
            raise Exception("WebSocket error")
        elif (
            msg.type == aiohttp.WSMsgType.CLOSE
            or msg.type == aiohttp.WSMsgType.CLOSED
        ):
            close_code = msg.data  # the close code
            close_reason = msg.extra  # the close reason
            if close_code is not None and close_code != 1000:
                raise Exception(
                    f"WebSocket closed (code {close_code}): "
                    f"{close_reason or 'No reason provided'}"
                )


class GradiumClient:
    """Client for communicating with the Gradium API.

    This client handles all HTTP requests and WebSocket connections to the Gradium
    API, including text-to-speech, speech recognition, voice management, and usage
    tracking.

    Attributes:
        _base_url: Base URL for API endpoints.
        _api_key: API key for authentication.

    Example:
        >>> client = GradiumClient(api_key="your-api-key")
        >>> # Use client methods for API operations
    """

    def __init__(
        self,
        *,
        base_url: str = "https://eu.api.gradium.ai/api/",
        api_key: str | None = None,
    ):
        """Initialize the Gradium client.

        Args:
            base_url: Base URL for the API. Defaults to the eu-api server.
                Automatically adds protocol (http/https) if missing.
            api_key: API key for authentication. If not provided, reads from the
                GRADIUM_API_KEY environment variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        if not base_url.endswith("/"):
            base_url = base_url + "/"
        if not base_url.startswith("http"):
            islocal = base_url.startswith(("127", "local"))
            protocol = "http" if islocal else "https"
            base_url = f"{protocol}://{base_url}"
        self._base_url = base_url

        api_key = (
            api_key
            if api_key is not None
            else os.environ.get("GRADIUM_API_KEY")
        )
        if api_key is None:
            raise ValueError(
                "Missing api-key as cli or as env (GRADIUM_API_KEY)"
            )
        self._api_key = api_key

    @property
    def headers(self) -> dict:
        """Get HTTP headers with authentication."""
        return {"x-api-key": self._api_key, "x-api-source": SOURCE}

    def ws(self, session, route: str) -> aiohttp.ClientWebSocketResponse:
        """Create a WebSocket connection to the specified route.

        Args:
            session: aiohttp ClientSession.
            route: API endpoint route.

        Returns:
            WebSocket connection context manager.
        """
        url = urllib.parse.urljoin(self._base_url, route).replace("http", "ws")
        return session.ws_connect(url)

    async def stream(
        self,
        route: str,
        setup: Any,
        input_stream: list[dict[str, str]] | AsyncGenerator[dict, None],
        map_input_fn: Callable | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream data to a WebSocket endpoint and receive responses.

        This is the core method for bidirectional streaming communication with
        the API. It handles both sending and receiving concurrently.

        Args:
            route: API endpoint route.
            setup: Initial setup configuration dictionary.
            input_stream: List or async generator of input messages.
            map_input_fn: Optional function to transform input messages.

        Yields:
            Response messages from the server.

        Raises:
            Exception: If WebSocket communication fails.
        """
        setup |= {"type": "setup"}
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with self.ws(session, route) as ws:
                receive_queue = asyncio.Queue()

                async def receive_worker():
                    """Collect received chunks into queue concurrently."""
                    try:
                        async for chunk in receive(ws):
                            await receive_queue.put(chunk)
                    except Exception as e:
                        await receive_queue.put(e)
                    finally:
                        await receive_queue.put(None)  # End marker

                send_task = asyncio.create_task(
                    send(ws, setup, input_stream, map_fn=map_input_fn)
                )
                receive_task = asyncio.create_task(receive_worker())

                try:
                    # Yield chunks as they arrive from the queue
                    while True:
                        chunk = await receive_queue.get()
                        if chunk is None:  # End marker
                            break
                        if isinstance(chunk, Exception):
                            raise chunk
                        yield chunk
                finally:
                    # Ensure both tasks complete
                    await asyncio.gather(
                        send_task, receive_task, return_exceptions=True
                    )

    async def _fetch(
        self, method: str, route, parse_response: bool = True, **kwargs
    ):
        """Generic HTTP request handler.

        Args:
            method: HTTP method name (get, post, put, delete, etc.).
            route: API endpoint route.
            parse_response: Whether to parse response as JSON.
            **kwargs: Additional arguments to pass to the HTTP method.

        Returns:
            Parsed JSON response or raw response object.

        Raises:
            ValueError: If the HTTP method is not supported.
            aiohttp.ClientError: If the HTTP request fails.
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = urllib.parse.urljoin(self._base_url, route)
            fn = getattr(session, method)
            if fn is None:
                raise ValueError(f"No such HTTP method {method}")
            response = await fn(url, **kwargs)

            if not response.ok:
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    msg = await response.json()
                    if (reason := msg.get("detail")) is not None:
                        response.reason = reason
                else:
                    response.reason = await response.text()
            response.raise_for_status()
            return await response.json() if parse_response else response

    async def post(self, route: str, parse: bool = True, **kwargs):
        """Make a POST request to the API.

        Args:
            route: API endpoint route.
            parse: Whether to parse response as JSON.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response or raw response object.
        """
        return await self._fetch("post", route, parse_response=parse, **kwargs)

    async def put(self, route: str, **kwargs):
        """Make a PUT request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("put", route, **kwargs)

    async def get(self, route: str, **kwargs):
        """Make a GET request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("get", route, **kwargs)

    async def delete(self, route: str, **kwargs):
        """Make a DELETE request to the API.

        Args:
            route: API endpoint route.
            **kwargs: Additional arguments for the request.

        Returns:
            Parsed JSON response.
        """
        return await self._fetch("delete", route, **kwargs)

    async def tts_stream(
        self,
        setup: "speech.TTSSetup",
        text: str | list[str] | AsyncGenerator,
    ) -> "speech.TTSStream":
        """Stream text-to-speech synthesis results.

        Initiates a streaming TTS request and returns a handler for consuming
        audio chunks as they arrive from the server. Use this when you need
        to process audio data incrementally.

        Args:
            setup: TTS configuration including voice, model, and output format.
            text: Text to synthesize. Can be:
                - str: Single text string
                - list[str]: Multiple text strings
                - AsyncGenerator: Stream of text strings

        Returns:
            TTSStream object for iterating over audio chunks.

        Example:
            >>> async def synthesize_stream():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = TTSSetup(voice="default", output_format="wav")
            ...     stream = await client.tts_stream(setup, "Hello world")
            ...     async for audio_chunk in stream.iter_bytes():
            ...         # Process audio chunk
            ...         pass

        Raises:
            RuntimeError: If server doesn't send expected "ready" message.
            aiohttp.ClientError: If the API request fails.
        """
        return await speech.tts_stream(self, setup, text)

    async def tts(
        self,
        setup: "speech.TTSSetup",
        text: str | list[str] | AsyncGenerator,
    ) -> "speech.TTSResult":
        """Synthesize text to speech (buffered).

        Synthesizes text to speech and returns the complete audio data once the
        request completes. This is simpler than tts_stream for when you don't need
        to process audio chunks incrementally.

        Args:
            setup: TTS configuration including voice, model, and output format.
            text: Text to synthesize. Can be:
                - str: Single text string
                - list[str]: Multiple text strings
                - AsyncGenerator: Stream of text strings

        Returns:
            TTSResult containing complete audio data, metadata, and timing info.

        Example:
            >>> async def synthesize():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = TTSSetup(voice="default", output_format="pcm")
            ...     result = await client.tts(setup, "Hello world")
            ...     audio_array = result.pcm()  # Get as numpy array
            ...     print(f"Sample rate: {result.sample_rate}")

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        return await speech.tts(self, setup, text)

    async def stt_stream(
        self,
        setup: "speech.STTSetup",
        audio: AsyncGenerator,
    ) -> "speech.STTStream":
        """Stream speech-to-text recognition results.

        Initiates a streaming STT request and returns a handler for consuming
        transcribed text segments as they arrive from the server.

        Args:
            setup: STT configuration including model and input format.
            audio: Async generator yielding audio chunks. For numpy arrays:
                - dtype must be int16 or float32
                - shape must be 1-dimensional
                - For float32, values should be in range [-1.0, 1.0]

        Returns:
            STTStream object for iterating over transcribed text segments.

        Example:
            >>> async def transcribe_stream():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = STTSetup(model_name="default", input_format="pcm")
            ...
            ...     async def audio_generator():
            ...         # Yield audio chunks
            ...         for chunk in audio_chunks:
            ...             yield chunk
            ...
            ...     stream = await client.stt_stream(setup, audio_generator())
            ...     async for text_segment in stream.iter_text():
            ...         print(f"{text_segment.text} ({text_segment.start_s}s)")

        Raises:
            RuntimeError: If server doesn't send expected "ready" message.
            ValueError: If audio format is invalid.
            aiohttp.ClientError: If the API request fails.
        """
        return await speech.stt_stream(self, setup, audio)

    async def stt(
        self,
        setup: "speech.STTSetup",
        audio: bytes | np.ndarray | AsyncGenerator[bytes],
        sample_rate: int | None = None,
    ) -> "speech.STTResult":
        """Transcribe audio to text (buffered).

        Transcribes audio to text and returns the complete transcription once the
        request completes. This is simpler than stt_stream for when you don't need
        to process results incrementally.

        Args:
            setup: STT configuration including model and input format.
            audio: Audio data. Can be:
                - bytes: Raw audio bytes (sample_rate must be None)
                - np.ndarray: Audio samples (int16 or float32)
                - AsyncGenerator[bytes]: Stream of audio chunks
            sample_rate: Sample rate in Hz. Required for numpy arrays (must be 24000),
                not supported for bytes input.

        Returns:
            STTResult containing complete transcribed text and timing information.

        Example:
            >>> async def transcribe():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = STTSetup(model_name="default", input_format="wav")
            ...
            ...     # From audio file bytes
            ...     with open("audio.wav", "rb") as f:
            ...         audio_data = f.read()
            ...     result = await client.stt(setup, audio_data)
            ...     print(f"Transcription: {result.text}")
            ...
            ...     # From numpy array
            ...     import numpy as np
            ...     setup_pcm = STTSetup(input_format="pcm")
            ...     audio_array = np.zeros(24000, dtype=np.int16)  # 1 sec of silence
            ...     result = await client.stt(setup_pcm, audio_array, sample_rate=24000)

        Raises:
            ValueError: If audio format is invalid or sample_rate mismatch.
            aiohttp.ClientError: If the API request fails.
        """
        return await speech.stt(self, setup, audio, sample_rate)

    async def credits(self) -> dict:
        """Get current credit balance information.

        Retrieves the current credit balance for your account, including how
        many credits remain and how many have been allocated.

        Returns:
            Dictionary containing credit information with keys such as:
                - remaining: Number of credits remaining
                - allocated: Number of credits allocated to the account
                - Other account-specific credit details

        Example:
            >>> async def check_credits():
            ...     client = GradiumClient(api_key="your-key")
            ...     credits = await client.credits()
            ...     print(f"Remaining: {credits['remaining']}")
            ...     print(f"Allocated: {credits['allocated']}")
            ...
            ...     # Check if enough credits before expensive operation
            ...     if credits['remaining'] < 1000:
            ...         print("Warning: Low credits!")

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        return await usages.get(self)

    async def usage_summary(self) -> dict:
        """Get usage summary statistics.

        Retrieves a summary of API usage across all features, including
        text-to-speech, speech recognition, and other services. Useful for
        tracking consumption and understanding usage patterns.

        Returns:
            Dictionary containing usage summary with statistics such as:
                - Total requests made across all services
                - Usage broken down by service/feature
                - Time period information for the statistics
                - Other relevant usage metrics

        Example:
            >>> async def analyze_usage():
            ...     client = GradiumClient(api_key="your-key")
            ...     summary = await client.usage_summary()
            ...
            ...     # Review usage across services
            ...     print("Usage Summary:")
            ...     for service, stats in summary.items():
            ...         print(f"  {service}: {stats}")
            ...
            ...     # Use for billing or quota management
            ...     total_requests = summary.get('total_requests', 0)
            ...     print(f"Total API calls: {total_requests}")

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        return await usages.summary(self)

    async def voice_create(
        self,
        audio_file: "pathlib.Path",
        *,
        name: str | None = None,
        description: str | None = None,
        start_s: float = 0.0,
        input_format: str | None = None,
    ) -> dict:
        """Create a new custom voice from an audio file.

        Uploads an audio file to create a custom voice that can be used for
        text-to-speech synthesis. The voice will be trained on the provided
        audio sample.

        Args:
            audio_file: Path to the audio file to use for voice creation.
            name: Name for the new voice. Defaults to the audio filename.
            description: Optional description of the voice characteristics.
            start_s: Start time in seconds for the audio clip. Use this to skip
                silence or unwanted audio at the beginning. Defaults to 0.
            input_format: Audio format (e.g., "wav", "mp3"). If not provided,
                automatically inferred from the file extension.

        Returns:
            Dictionary containing voice metadata including:
                - uid: Unique identifier for the created voice
                - name: Voice name
                - description: Voice description
                - Other voice-specific metadata

        Example:
            >>> async def create_custom_voice():
            ...     client = GradiumClient(api_key="your-key")
            ...     voice = await client.voice_create(
            ...         audio_file=Path("speaker.wav"),
            ...         name="My Custom Voice",
            ...         description="A warm, friendly voice",
            ...         start_s=0.5  # Skip first 500ms
            ...     )
            ...     print(f"Created voice with UID: {voice['uid']}")
            ...
            ...     # Use the new voice for TTS
            ...     setup = TTSSetup(voice_id=voice['uid'])
            ...     result = await client.tts(setup, "Hello with my custom voice!")

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            aiohttp.ClientError: If the API request fails.
        """
        return await voices.create(
            self,
            audio_file,
            name=name,
            description=description,
            start_s=start_s,
            input_format=input_format,
        )

    async def voice_get(
        self, voice_uid: str | None = None, include_catalog: bool = False
    ) -> dict:
        """Get voice information by UID or list all voices.

        Retrieves detailed information about a specific voice or returns a list
        of all available voices (both default and custom).

        Args:
            voice_uid: UID of the voice to retrieve. If None, returns all available
                voices including default voices and custom voices.
            include_catalog: whether to return the voice of the public Gradium
                catalog or only the custom voice of the org.

        Returns:
            If voice_uid is provided: Dictionary with voice metadata for the
                specific voice including uid, name, description, and other properties.
            If voice_uid is None: Dictionary containing lists of available voices,
                typically organized by category (e.g., default, custom).

        Example:
            >>> async def list_and_get_voices():
            ...     client = GradiumClient(api_key="your-key")
            ...
            ...     # List all available voices
            ...     all_voices = await client.voice_get()
            ...     print(f"Available voices: {all_voices}")
            ...
            ...     # Get specific voice details
            ...     voice_info = await client.voice_get("voice-uid-123")
            ...     print(f"Voice name: {voice_info['name']}")
            ...     print(f"Description: {voice_info['description']}")

        Raises:
            aiohttp.ClientError: If the API request fails or voice_uid not found.
        """
        return await voices.get(
            self, voice_uid, include_catalog=include_catalog
        )

    async def voice_delete(self, voice_uid: str) -> bool:
        """Delete a custom voice by UID.

        Permanently deletes a custom voice and removes it from the available
        voices for text-to-speech synthesis. This operation cannot be undone.
        Default voices cannot be deleted.

        Args:
            voice_uid: UID of the voice to delete.

        Returns:
            True if deletion was successful.

        Example:
            >>> async def remove_voice():
            ...     client = GradiumClient(api_key="your-key")
            ...
            ...     # Delete a custom voice
            ...     success = await client.voice_delete("voice-uid-123")
            ...     if success:
            ...         print("Voice deleted successfully")
            ...
            ...     # Verify it's gone
            ...     voices = await client.voice_list()
            ...     # voice-uid-123 should no longer appear in the list

        Raises:
            aiohttp.ClientError: If the API request fails or voice doesn't exist.
        """
        return await voices.delete(self, voice_uid)

    async def voice_update(
        self,
        voice_uid: str,
        name: str | None = None,
        description: str | None = None,
        start_s: float | None = None,
    ) -> dict | None:
        """Update voice metadata.

        Updates one or more properties of an existing custom voice. Only the
        properties that are provided (not None) will be updated.

        Args:
            voice_uid: UID of the voice to update.
            name: New name for the voice. If None, name is not updated.
            description: New description for the voice. If None, description
                is not updated.
            start_s: New start time in seconds for the audio clip used by this
                voice. If None, start time is not updated.

        Returns:
            Updated voice metadata dictionary if any updates were made,
            or None if no parameters were provided for update.

        Example:
            >>> async def update_voice_info():
            ...     client = GradiumClient(api_key="your-key")
            ...
            ...     # Update just the name
            ...     updated = await client.voice_update(
            ...         voice_uid="voice-uid-123",
            ...         name="Professional Voice"
            ...     )
            ...
            ...     # Update multiple properties
            ...     updated = await client.voice_update(
            ...         voice_uid="voice-uid-123",
            ...         name="Corporate Voice",
            ...         description="Clear, professional tone for business content",
            ...         start_s=1.0
            ...     )
            ...     print(f"Updated voice: {updated['name']}")

        Raises:
            aiohttp.ClientError: If the API request fails or voice doesn't exist.
        """
        return await voices.update(
            self, voice_uid, name=name, description=description, start_s=start_s
        )

    async def voice_list(self) -> dict:
        """List all available voices.

        Retrieves a list of all voices available for text-to-speech synthesis,
        including both default system voices and custom voices created by the user.

        Returns:
            Dictionary containing lists of available voices, typically organized
            by category (e.g., default voices, custom voices). Each voice entry
            includes metadata such as uid, name, description, and other properties.

        Example:
            >>> async def show_available_voices():
            ...     client = GradiumClient(api_key="your-key")
            ...     voices = await client.voice_list()
            ...
            ...     # Iterate through voices
            ...     for category, voice_list in voices.items():
            ...         print(f"\n{category} voices:")
            ...         for voice in voice_list:
            ...             print(f"  - {voice['name']} (UID: {voice['uid']})")

        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        return await voices.get(self)
