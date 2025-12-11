"""Text-to-Speech and Speech-to-Text functionality for Gradium API.

This module provides high-level interfaces for:
- Text-to-Speech (TTS) conversion with streaming and buffered modes
- Speech-to-Text (STT) transcription with streaming and buffered modes
- Voice configuration and management

Classes:
    TTSSetup: Configuration dictionary for TTS requests.
    STTSetup: Configuration dictionary for STT requests.
    TextWithTimestamps: Text segment with timestamp information.
    TTSStream: Streaming TTS result handler.
    TTSResult: Buffered TTS result with audio data.
    STTStream: Streaming STT result handler.
    STTResult: Buffered STT result with transcription.

Functions:
    tts_stream: Stream TTS results.
    tts: Get buffered TTS result.
    stt_stream: Stream STT results.
    stt: Get buffered STT result.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from . import client as gradium_client


class TTSSetup(TypedDict, total=False):
    """Configuration for Text-to-Speech requests.

    Attributes:
        model_name: TTS model to use. Defaults to "default".
        voice: Voice profile to use. Defaults to "default". Can be a name or None.
        voice_id: Specific voice UID to use instead of voice name.
        output_format: Audio output format (e.g., "wav", "pcm"). Defaults to "wav".
        json_config: Additional JSON configuration for the TTS model.
    """

    model_name: str = "default"
    voice: str | None = None
    voice_id: str | None = None
    output_format: str = "wav"
    json_config: Any | None = None


class STTSetup(TypedDict, total=False):
    """Configuration for Automatic Speech Recognition requests.

    Attributes:
        model_name: STT model to use. Defaults to "default".
        input_format: Audio input format (e.g., "wav", "pcm"). Defaults to "wav".
        json_config: Additional JSON configuration for the STT model.
    """

    model_name: str = "default"
    input_format: str = "wav"
    json_config: Any | None = None


@dataclass
class TextWithTimestamps:
    """Text segment with timestamp information.

    Attributes:
        text: The text content.
        start_s: Start time in seconds.
        stop_s: Stop time in seconds.
    """

    text: str
    start_s: float
    stop_s: float


class TTSStream:
    """Stream handler for Text-to-Speech results.

    Provides async iteration over audio chunks from a TTS request while also
    collecting text timing information.

    Attributes:
        _stream: Underlying async message stream.
        _setup: TTS configuration used for this request.
        _text_with_timestamps: Collected text segments with timing.
    """

    def __init__(
        self,
        stream: AsyncGenerator,
        setup: TTSSetup,
        ready: Any,
    ):
        """Initialize TTSStream.

        Args:
            stream: Async generator yielding TTS messages.
            sample_rate: Sample rate of output audio or None.
            request_id: Unique request identifier.
            setup: TTS configuration dictionary.
        """
        self._stream = stream
        self._setup = setup
        self._text_with_timestamps = []
        self._ready = ready

    @property
    def sample_rate(self) -> int | None:
        """Get the sample rate of the output audio."""
        return self._ready.get("sample_rate")

    @property
    def request_id(self) -> str | None:
        """Get the unique request ID."""
        return self._ready.get("request_id")

    async def iter_bytes(self) -> AsyncGenerator[bytes]:
        """Stream audio chunks as bytes.

        Iterates over audio chunks from the server, yielding raw audio bytes
        and collecting text timing information. Use this method when you need
        to process audio data incrementally, such as streaming to a player
        or saving to a file while synthesis is in progress.

        Yields:
            Raw audio data chunks (base64 decoded) in the format specified
            by the setup configuration (e.g., WAV, PCM).

        Example:
            >>> async def stream_to_file():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = TTSSetup(voice_id="YTpq7expH9539ERJ", output_format="wav")
            ...     stream = await client.tts_stream(setup, "Hello world")
            ...
            ...     with open("output.wav", "wb") as f:
            ...         async for chunk in stream.iter_bytes():
            ...             f.write(chunk)
            ...
            ...     # Access timing information after streaming
            ...     for twt in stream._text_with_timestamps:
            ...         print(f"{twt.text}: {twt.start_s}s - {twt.stop_s}s")
        """
        async for msg in self._stream:
            msg_type = msg.get("type")
            if msg_type == "text":
                start_s = msg.get("start_s", 0.0)
                twt = TextWithTimestamps(
                    text=msg.get("text", ""),
                    start_s=start_s,
                    stop_s=msg.get("stop_s", start_s),
                )
                self._text_with_timestamps.append(twt)
            elif msg_type == "audio":
                yield base64.b64decode(msg["audio"])


@dataclass
class TTSResult:
    """Buffered Text-to-Speech result.

    Contains the complete audio data from a TTS request along with metadata
    and utility methods for audio format conversion.

    Attributes:
        raw_data: Raw audio bytes in the specified format.
        sample_rate: Sample rate of the audio or None.
        output_format: Audio format (e.g., "wav", "pcm").
        request_id: Unique ID for this TTS request.
        text_with_timestamps: List of text segments with timing information.
    """

    raw_data: bytes
    sample_rate: int | None
    output_format: str | None
    request_id: str | None
    text_with_timestamps: list[TextWithTimestamps]

    def pcm16(self) -> np.array:
        """Get PCM16 numpy array from raw audio data.

        Returns:
            Numpy array with int16 audio samples.

        Raises:
            ValueError: If output_format is not "pcm".
        """
        _format = self.output_format
        if _format is None or not _format.startswith("pcm"):
            raise ValueError("output_format is not 'pcm'")
        return np.frombuffer(self.raw_data, dtype=np.int16)

    def pcm(self) -> np.array:
        """Get PCM float numpy array from raw audio data.

        Converts PCM16 audio to float32 with values in range [-1.0, 1.0].

        Returns:
            Numpy array with float32 audio samples.

        Raises:
            ValueError: If output_format is not "pcm".
        """
        pcm16 = self.pcm16()
        return pcm16.astype(np.float32) / 32768.0


async def tts_stream(
    client: "gradium_client.GradiumClient",
    setup: TTSSetup,
    text: str | list[str] | AsyncGenerator,
    tts_endpoint: str = "speech/tts",
) -> TTSStream:
    """Stream Text-to-Speech synthesis results.

    Initiates a streaming TTS request and returns a handler for consuming
    audio chunks as they arrive from the server.

    Args:
        client: GradiumClient instance.
        setup: TTS configuration (TTSSetup TypedDict).
        text: Text to synthesize. Can be a string, list of strings, or
            async generator of strings.

    Returns:
        TTSStream object for iterating over audio chunks.

    Raises:
        RuntimeError: If server doesn't send expected "ready" message first.
    """
    if isinstance(text, str):
        text = [text]

    def format_text(text: str):
        return {"type": "text", "text": text}

    if (config := setup.get("json_config")) is not None:
        if not isinstance(config, str):
            # Make a copy to avoid modifying the original setup
            setup = dict(setup)
            setup["json_config"] = json.dumps(config)

    stream = client.stream(tts_endpoint, setup, text, map_input_fn=format_text)
    ready = await anext(stream)
    if (msg_type := ready.get("type")) != "ready":
        raise RuntimeError(f"unexpected first message type `{msg_type}`")

    return TTSStream(stream, setup=setup, ready=ready)


async def tts(
    client: "gradium_client.GradiumClient",
    setup: TTSSetup,
    text: str | list[str] | AsyncGenerator,
) -> TTSResult:
    """Buffered Text-to-Speech synthesis.

    Synthesizes text to speech and returns the complete audio data once the
    request completes. This is simpler than tts_stream for when you don't need
    to process audio chunks as they arrive.

    Args:
        client: GradiumClient instance.
        setup: TTS configuration (TTSSetup TypedDict).
        text: Text to synthesize. Can be a string, list of strings, or
            async generator of strings.

    Returns:
        TTSResult containing complete audio data and metadata.
    """
    chunks = []
    stream = await tts_stream(client, setup, text)
    async for chunk in stream.iter_bytes():
        chunks.append(chunk)
    raw_data = b"".join(chunks)
    return TTSResult(
        raw_data=raw_data,
        sample_rate=stream.sample_rate,
        request_id=stream.request_id,
        text_with_timestamps=stream._text_with_timestamps,
        output_format=setup.get("output_format"),
    )


class STTStream:
    """Stream handler for Speech-to-Text results.

    Provides async iteration over transcribed text segments from an STT request.

    Attributes:
        _stream: Underlying async message stream.
        _setup: STT configuration used for this request.
    """

    def __init__(
        self,
        stream: AsyncGenerator,
        setup: STTSetup,
        ready: Any,
    ):
        """Initialize STTStream.

        Args:
            stream: Async generator yielding STT messages.
            setup: STT configuration dictionary.
        """
        self._stream = stream
        self._setup = setup
        self._ready = ready

    async def iter_text(self) -> AsyncGenerator[TextWithTimestamps]:
        """Stream transcribed text segments.

        Iterates over text segments as they are transcribed from the audio stream.
        Use this method when you need real-time transcription results, such as
        live captioning or interactive voice applications.

        Yields:
            TextWithTimestamps objects containing transcribed text and timing
            information (start_s and stop_s in seconds).

        Example:
            >>> import numpy as np
            >>> async def transcribe_realtime():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = STTSetup(model_name="default", input_format="pcm")
            ...
            ...     async def audio_stream():
            ...         # Stream audio chunks from microphone or file
            ...         for i in range(10):
            ...             chunk = np.random.randint(-1000, 1000, 1920, dtype=np.int16)
            ...             yield chunk
            ...
            ...     stream = await client.stt_stream(setup, audio_stream())
            ...     async for text_segment in stream.iter_text():
            ...         timestamp = f"[{text_segment.start_s:.2f}s]"
            ...         print(f"{timestamp} {text_segment.text}")
        """
        async for msg in self._stream:
            type_ = msg.get("type")
            if type_ == "text":
                start_s = msg.get("start_s", 0.0)
                yield TextWithTimestamps(
                    text=msg.get("text", ""),
                    start_s=start_s,
                    stop_s=msg.get("stop_s", start_s),
                )

    @property
    def request_id(self) -> str | None:
        """Get the unique request ID."""
        return self._ready.get("request_id")


@dataclass
class STTResult:
    """Buffered Speech-to-Text result.

    Contains the transcribed text from a speech recognition request along with
    timing information and metadata.

    Attributes:
        text: Complete transcribed text with segments joined by spaces.
        text_with_timestamps: List of individual text segments with timing.
        request_id: Unique ID for this STT request.
    """

    text: str
    text_with_timestamps: list[TextWithTimestamps]
    request_id: str | None


async def stt_stream(
    client: "gradium_client.GradiumClient",
    setup: STTSetup,
    audio: AsyncGenerator,
    stt_endpoint: str = "speech/asr",
) -> STTStream:
    """Stream Speech-to-Text transcription results.

    Initiates a streaming STT request and returns a handler for consuming
    transcribed text segments as they arrive from the server.

    Args:
        client: GradiumClient instance.
        setup: STT configuration (STTSetup TypedDict).
        audio: Async generator yielding audio chunks. For numpy arrays:
            - dtype must be int16 or float32
            - shape must be 1-dimensional
            - For float32, values should be in range [-1.0, 1.0]

    Returns:
        STTStream object for iterating over transcribed text segments.

    Raises:
        RuntimeError: If server doesn't send expected "ready" message first.
        ValueError: If audio format is invalid.
    """

    def format_audio(audio):
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.int16:
                pass
            elif audio.dtype == np.float32:
                audio = (audio * 32768).astype(np.int16)
            else:
                raise ValueError("audio np.ndarray must be int16 or float32")
            if audio.ndim != 1:
                raise ValueError("audio np.ndarray must be 1-dimensional")
            audio = audio.tobytes()

        return {
            "type": "audio",
            "audio": base64.b64encode(audio).decode("utf8"),
        }

    if (config := setup.get("json_config")) is not None:
        if not isinstance(config, str):
            # Make a copy to avoid modifying the original setup
            setup = dict(setup)
            setup["json_config"] = json.dumps(config)

    stream = client.stream(
        stt_endpoint, setup, audio, map_input_fn=format_audio
    )
    ready = await anext(stream)
    if ready.get("type") != "ready":
        raise RuntimeError(f"unexpected first message type {ready.get('type')}")
    return STTStream(stream, setup=setup, ready=ready)


async def stt(
    client: "gradium_client.GradiumClient",
    setup: STTSetup,
    audio: bytes | np.ndarray | AsyncGenerator[bytes],
    sample_rate: int | None = None,
) -> STTResult:
    """Buffered Speech-to-Text transcription.

    Transcribes audio to text and returns the complete transcription once the
    request completes. This is simpler than stt_stream for when you don't need
    to process results as they arrive.

    Args:
        client: GradiumClient instance.
        setup: STT configuration (STTSetup TypedDict).
        audio: Audio data. Can be:
            - bytes: Raw audio bytes (sample_rate must be None)
            - np.ndarray: Audio samples (int16 or float32)
            - AsyncGenerator[bytes]: Stream of audio chunks
        sample_rate: Sample rate in Hz. Required for numpy arrays (must be 24000),
            not supported for bytes input.

    Returns:
        STTResult containing transcribed text and metadata.

    Raises:
        ValueError: If audio format is invalid or sample_rate mismatch.
    """

    async def bytes_stream_gen(audio, chunk_size: int) -> AsyncGenerator[bytes]:
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    if isinstance(audio, bytes):
        bytes_stream = bytes_stream_gen(audio, 4096)
        if sample_rate is not None:
            raise ValueError(
                "sample_rate is not supported for bytes audio input"
            )
    elif isinstance(audio, np.ndarray):
        if setup.get("input_format") != "pcm":
            raise ValueError(
                "input_format must be 'pcm' for np.ndarray audio input"
            )
        if sample_rate != 24000:
            raise ValueError(
                "sample_rate must be 24000 for np.ndarray audio input"
            )
        if audio.dtype == np.int16:
            pass
        elif audio.dtype == np.float32:
            audio = (audio * 32768).astype(np.int16)
        else:
            raise ValueError("audio np.ndarray must be int16 or float32")
        if audio.ndim != 1:
            raise ValueError("audio np.ndarray must be 1-dimensional")

        bytes_stream = bytes_stream_gen(audio, 1920)
    else:
        if sample_rate is not None:
            raise ValueError(
                "sample_rate is not supported for bytes audio input"
            )
        bytes_stream = audio

    stream = await stt_stream(client, setup, bytes_stream)
    all_texts = []
    async for text in stream.iter_text():
        all_texts.append(text)
    return STTResult(
        text=" ".join(t.text for t in all_texts),
        text_with_timestamps=all_texts,
        request_id=stream.request_id,
    )
