"""A streaming client for the Gradium API.

This module provides async streaming interfaces for Text-to-Speech (TTS) and
Speech-to-Text (STT) operations using WebSocket connections.

Example (TTS):
    ```python
    import asyncio
    from gradium import client as gradium_client

    async def main():
        grc = gradium_client.GradiumClient(
            base_url="https://api.gradium.ai",
            api_key="your-api-key"
        )

        setup = {
            "model_name": "default",
            "output_format": "wav",
            "voice": "default",
            "json_config": {},
        }

        async with grc.tts_realtime(**setup) as tts:
            print("Ready:", tts.ready)

            # Stream text word by word
            async def send_loop():
                for word in "Hello world".split():
                    await tts.send_text(word)
                await tts.send_eos()

            # Receive audio chunks
            async def recv_loop():
                async for msg in tts:
                    if msg["type"] == "audio":
                        audio_bytes = msg["audio"]
                        # Process audio...
                    elif msg["type"] == "text":
                        text = msg["text"]
                        # Process text...

            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_loop())
                tg.create_task(recv_loop())

    asyncio.run(main())
    ```

Example (STT):
    ```python
    import asyncio
    import numpy as np
    from gradium import client as gradium_client

    async def main():
        grc = gradium_client.GradiumClient(
            base_url="https://api.gradium.ai",
            api_key="your-api-key"
        )

        setup = {
            "model_name": "default",
            "input_format": "pcm",
            "json_config": {},
        }

        async with grc.stt_realtime(**setup) as stt:
            print("Ready:", stt.ready)

            # Send audio in chunks
            async def send_loop():
                chunk_size = 1920  # 80ms at 24kHz
                for i in range(0, len(pcm_data), chunk_size):
                    await stt.send_audio(pcm_data[i:i + chunk_size])
                await stt.send_eos()

            # Receive transcription
            async def recv_loop():
                async for msg in stt:
                    if msg["type"] == "text":
                        print("Transcription:", msg["text"])
                    elif msg["type"] == "step":
                        # VAD information (every 80ms)
                        vad_prob = msg["vad"][2]["inactivity_prob"]

            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_loop())
                tg.create_task(recv_loop())

    asyncio.run(main())
    ```
"""

import base64
import json
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from . import client, speech


@dataclass
class RawAudioChunk:
    """Raw audio data chunk.

    Attributes:
        data: The raw audio bytes.
    """

    data: bytes


class Tts:
    """Text-to-Speech streaming client.

    This class provides an async context manager interface for streaming TTS operations.
    Text is sent to the server and audio chunks are received in real-time.

    The class implements async iteration, allowing you to use `async for` to receive
    messages from the server.

    Attributes:
        ready: Information returned by the server when the connection is ready.

    Example:
        ```python
        async with grc.tts_realtime(model_name="default", voice="default") as tts:
            await tts.send_text("Hello world")
            await tts.send_eos()

            async for msg in tts:
                if msg["type"] == "audio":
                    process_audio(msg["audio"])
        ```
    """

    def __init__(
        self,
        client: "client.GradiumClient",
        route: str = "speech/tts",
        send_setup_on_start: bool = True,
        **kwargs,
    ):
        """Initialize the TTS streaming client.

        Args:
            client: The GradiumClient instance.
            route: The WebSocket route for TTS (default: "speech/tts").
            **kwargs: Setup parameters to send to the server (model_name, voice, etc.).
        """
        self._client = client
        self._kwargs = kwargs
        self._route = route
        self._ws = None
        self._session = None
        self._ready = None
        self._send_setup_on_start = send_setup_on_start

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self._client.headers)
        self._ws = await self._client.ws(self._session, self._route)
        if self._send_setup_on_start:
            await self.send_setup(self._kwargs)
            ready = await self.wait_for_ready()
            self._ready = ready
        else:
            self._ready = None
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

    async def send_setup(self, setup: Any):
        """Send setup configuration to the server.

        This is called automatically during context manager entry.

        Args:
            setup: Setup parameters (model_name, voice, output_format, json_config, etc.).

        Raises:
            RuntimeError: If the connection is not open.
        """
        if not self._ws:
            raise RuntimeError("Connection not open")
        setup = speech.TTSSetup(**setup) if isinstance(setup, dict) else setup
        setup = {**setup, "type": "setup"}

        if (config := setup.get("json_config")) is not None:
            if not isinstance(config, str):
                setup["json_config"] = json.dumps(config)

        await self._ws.send_json(setup)

    async def send_text(self, text: str, client_req_id: str | None = None):
        """Send text to be synthesized.

        Text can be sent incrementally (e.g., word by word) for streaming synthesis.

        Args:
            text: The text to synthesize.

        Raises:
            RuntimeError: If the connection is not open.

        Example:
            ```python
            for word in "Hello world".split():
                await tts.send_text(word)
            ```
        """
        if not self._ws:
            raise RuntimeError("Connection not open")
        payload = {"type": "text", "text": text}
        if client_req_id is not None:
            payload["client_req_id"] = client_req_id
        await self._ws.send_json(payload)

    async def send_eos(self, client_req_id: str | None = None):
        """Send end-of-stream signal.

        This signals to the server that no more text will be sent, allowing it to
        finalize audio generation and close the stream.

        Args:
            client_req_id: optional identifier of the request used in
                multiplexing.

        Raises:
            RuntimeError: If the connection is not open.

        Example:
            ```python
            await tts.send_text("Hello world")
            await tts.send_eos()
            ```
        """
        if not self._ws:
            raise RuntimeError("Connection not open")
        payload = {"type": "end_of_stream"}
        if client_req_id is not None:
            payload["client_req_id"] = client_req_id
        await self._ws.send_json({"type": "end_of_stream"})

    async def wait_for_ready(self):
        """Wait for the ready message from the server.

        This is called automatically during context manager entry.

        Returns:
            dict: The ready message data from the server.

        Raises:
            RuntimeError: If the connection is not open or unexpected message received.
        """
        if self._ws is None:
            raise RuntimeError("Connection not open")
        msg = await self._ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"Unexpected message type: {msg.type}")
        data = json.loads(msg.data)
        if data.get("type") != "ready":
            raise RuntimeError(f"Expected ready message, got: {data}")
        return data

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Get the next message from the stream.

        Returns:
            dict: The next message from the server.

        Raises:
            StopAsyncIteration: When the stream is closed.
        """
        msg = await self.recv()
        if msg is None:
            raise StopAsyncIteration
        return msg

    async def recv(self) -> dict[str, Any] | None:
        """Receive a message from the server.

        Returns:
            dict or None: The message data, or None if the connection is closed.
                Message types include:
                - "audio": Contains "audio" (bytes), "start_s", "stop_s" fields
                - "text": Contains "text" field with the generated text
                - "error": Error message from the server (raises RuntimeError)

        Raises:
            RuntimeError: If the connection is not open or server returns an error.

        Example:
            ```python
            msg = await tts.recv()
            if msg["type"] == "audio":
                audio_data = msg["audio"]
                duration = msg["stop_s"] - msg.get("start_s", 0.0)
            ```
        """
        if self._ws is None:
            raise RuntimeError("Connection not open")
        while True:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.CLOSE:
                return
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            msg_type = data["type"]
            if msg_type == "error":
                raise RuntimeError(f"Error from server: {data}")
            elif msg_type == "text":
                start_s = data.get("start_s", 0.0)
                data["stop_s"] = data.get("stop_s", start_s)
            elif msg_type == "audio":
                data["audio"] = base64.b64decode(data["audio"])
            elif msg_type == "ready" and self._ready is None:
                self._ready = data
            return data

    @property
    def ready(self):
        """Get the ready message received from the server.

        Returns:
            dict: The ready message data.
        """
        return self._ready


class Stt:
    """Speech-to-Text streaming client.

    This class provides an async context manager interface for streaming STT operations.
    Audio data is sent to the server and transcription results are received in real-time.

    The class implements async iteration, allowing you to use `async for` to receive
    messages from the server.

    Attributes:
        ready: Information returned by the server when the connection is ready.

    Example:
        ```python
        async with grc.stt_realtime(model_name="default", input_format="pcm") as stt:
            # Send audio in chunks (e.g., 80ms at 24kHz)
            for i in range(0, len(pcm_data), 1920):
                await stt.send_audio(pcm_data[i:i + 1920])
            await stt.send_eos()

            # Receive transcription
            async for msg in stt:
                if msg["type"] == "text":
                    print(msg["text"])
                elif msg["type"] == "step":
                    # VAD information
                    vad_prob = msg["vad"][2]["inactivity_prob"]
        ```
    """

    def __init__(
        self,
        client: "client.GradiumClient",
        route: str = "speech/asr",
        **kwargs,
    ):
        """Initialize the STT streaming client.

        Args:
            client: The GradiumClient instance.
            route: The WebSocket route for STT (default: "speech/asr").
            **kwargs: Setup parameters to send to the server (model_name, input_format, etc.).
        """
        self._client = client
        self._kwargs = kwargs
        self._route = route
        self._ws = None
        self._session = None
        self._setup = None
        self._ready = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=self._client.headers)
        self._ws = await self._client.ws(self._session, self._route)
        await self.send_setup(self._kwargs)
        ready = await self.wait_for_ready()
        self._ready = ready
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

    async def send_setup(self, setup: Any):
        """Send setup configuration to the server.

        This is called automatically during context manager entry.

        Args:
            setup: Setup parameters (model_name, input_format, json_config, etc.).

        Raises:
            RuntimeError: If the connection is not open.
        """
        if not self._ws:
            raise RuntimeError("Connection not open")
        setup = speech.STTSetup(**setup) if isinstance(setup, dict) else setup
        setup = {**setup, "type": "setup"}

        if (config := setup.get("json_config")) is not None:
            if not isinstance(config, str):
                setup["json_config"] = json.dumps(config)

        self._setup = setup
        await self._ws.send_json(setup)

    async def send_audio(self, audio: bytes | np.ndarray):
        """Send audio data to be transcribed.

        Audio can be sent in chunks for streaming transcription. For PCM format,
        audio is expected to be 24kHz, mono, int16 or float32.

        Args:
            audio: Audio data as bytes or numpy array. If numpy array:
                - Must be 1-dimensional
                - For int16: values in range [-32768, 32767]
                - For float32: values in range [-1.0, 1.0]

        Raises:
            RuntimeError: If the connection is not open.
            ValueError: If audio format is invalid.

        Example:
            ```python
            # Send audio in 80ms chunks (1920 samples at 24kHz)
            chunk_size = 1920
            for i in range(0, len(pcm_data), chunk_size):
                await stt.send_audio(pcm_data[i:i + chunk_size])
            ```
        """
        if not self._ws or self._setup is None:
            raise RuntimeError("Connection not open")
        if isinstance(audio, np.ndarray):
            if not self._setup["input_format"].startswith("pcm"):
                raise ValueError(
                    "audio np.ndarray can only be sent when input_format is 'pcm'"
                )
            if audio.dtype == np.int16:
                pass
            elif audio.dtype == np.float32:
                audio = (audio * 32768).astype(np.int16)
            else:
                raise ValueError("audio np.ndarray must be int16 or float32")
            if audio.ndim != 1:
                raise ValueError("audio np.ndarray must be 1-dimensional")
            audio = audio.tobytes()

        await self._ws.send_json(
            {"type": "audio", "audio": base64.b64encode(audio).decode("utf-8")}
        )

    async def send_eos(self):
        """Send end-of-stream signal.

        This signals to the server that no more audio will be sent, allowing it to
        finalize transcription and close the stream.

        Raises:
            RuntimeError: If the connection is not open.

        Example:
            ```python
            await stt.send_audio(audio_data)
            await stt.send_eos()
            ```
        """
        if not self._ws:
            raise RuntimeError("Connection not open")
        await self._ws.send_json({"type": "end_of_stream"})

    async def send_flush(self, flush_id: int = 0):
        if not self._ws:
            raise RuntimeError("Connection not open")
        await self._ws.send_json({"type": "flush", "flush_id": flush_id})

    async def wait_for_ready(self):
        """Wait for the ready message from the server.

        This is called automatically during context manager entry.

        Returns:
            dict: The ready message data from the server.

        Raises:
            RuntimeError: If the connection is not open or unexpected message received.
        """
        if self._ws is None:
            raise RuntimeError("Connection not open")
        msg = await self._ws.receive()
        if msg.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError(f"Unexpected message type: {msg.type}")
        data = json.loads(msg.data)
        if data.get("type") != "ready":
            raise RuntimeError(f"Expected ready message, got: {data}")
        return data

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Get the next message from the stream.

        Returns:
            dict: The next message from the server.

        Raises:
            StopAsyncIteration: When the stream is closed.
        """
        msg = await self.recv()
        if msg is None:
            raise StopAsyncIteration
        return msg

    async def recv(self) -> dict[str, Any] | None:
        """Receive a message from the server.

        Returns:
            dict or None: The message data, or None if the connection is closed.
                Message types include:
                - "text": Contains "text" field with transcribed text
                - "step": Contains "vad" field with voice activity detection info.
                    VAD steps occur every 80ms. Use msg["vad"][2]["inactivity_prob"]
                    to determine the probability that the turn is finished.
                - "error": Error message from the server (raises RuntimeError)
                - "flushed": Indicates that previous audio has been flushed (contains "flush_id")

        Raises:
            RuntimeError: If the connection is not open or server returns an error.

        Example:
            ```python
            msg = await stt.recv()
            if msg["type"] == "text":
                print("Transcription:", msg["text"])
            elif msg["type"] == "step":
                vad_prob = msg["vad"][2]["inactivity_prob"]
            ```
        """
        if self._ws is None:
            raise RuntimeError("Connection not open")
        while True:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.CLOSE:
                return None
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            msg_type = data["type"]
            if msg_type == "error":
                raise RuntimeError(f"Error from server: {data}")
            return data

    @property
    def ready(self):
        """Get the ready message received from the server.

        Returns:
            dict: The ready message data.
        """
        return self._ready
