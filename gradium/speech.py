"""Text-to-Speech, Speech-to-Text and Speech-to-Speech functionality.

This module provides high-level interfaces for:
- Text-to-Speech (TTS) conversion with streaming and buffered modes
- Speech-to-Text (STT) transcription with streaming and buffered modes
- Speech-to-Speech (S2S) translation/conversion with streaming and buffered modes
- Voice configuration and management

Classes:
    TTSSetup: Configuration dictionary for TTS requests.
    STTSetup: Configuration dictionary for STT requests.
    S2SSetup: Configuration dictionary for S2S requests.
    TextWithTimestamps: Text segment with timestamp information.
    TTSStream: Streaming TTS result handler.
    TTSResult: Buffered TTS result with audio data.
    STTStream: Streaming STT result handler.
    STTResult: Buffered STT result with transcription.
    S2SStream: Streaming S2S result handler.
    S2SResult: Buffered S2S result with output audio and transcription.

Functions:
    tts_stream: Stream TTS results.
    tts: Get buffered TTS result.
    stt_stream: Stream STT results.
    stt: Get buffered STT result.
    s2s_stream: Stream S2S results.
    s2s: Get buffered S2S result.
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
    pronunciation_id: str | None = None
    output_format: str = "wav"
    json_config: Any | None = None
    client_req_id: str | None = None


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
    client_req_id: str | None = None


class S2SSetup(TypedDict, total=False):
    """Configuration for Speech-to-Speech requests.

    Speech-to-Speech pipes input audio through speech recognition, optional
    translation and speech synthesis, returning both the synthesized output
    audio and the (optionally translated) transcribed text.

    Attributes:
        model_name: S2S model alias to use. Defaults to "default".
        stt_model_name: Speech-to-text model used for transcription.
        tts_model_name: Text-to-speech model used for synthesis.
        input_format: Audio input format. One of "pcm", "wav", "opus",
            "ulaw_8000", "alaw_8000", or an explicit rate like "pcm_24000".
            For "pcm" the input is 24 kHz, 16-bit signed mono. Defaults to "pcm".
        output_format: Audio output format (e.g. "pcm", "wav", "opus",
            "ulaw_8000"). For "pcm" the output is 48 kHz, 16-bit signed mono.
            Defaults to "pcm".
        voice_id: Voice UID used for the synthesized output.
        json_config: Additional JSON configuration for the S2S pipeline. Set
            "target_language" (e.g. "en") to translate, omit to keep the
            original language.
        client_req_id: Client supplied identifier used for multiplexing.
        close_ws_on_eos: Whether the server closes the socket after sending its
            end_of_stream. Defaults to True.
    """

    model_name: str = "default"
    stt_model_name: str | None = None
    tts_model_name: str | None = None
    input_format: str = "pcm"
    output_format: str = "pcm"
    voice_id: str | None = None
    json_config: Any | None = None
    client_req_id: str | None = None
    close_ws_on_eos: bool | None = None


@dataclass
class TextWithTimestamps:
    """Text segment with timestamp information.

    Attributes:
        text: The text content.
        start_s: Start time in seconds.
        stop_s: Stop time in seconds.
        client_req_id: Client supplied identifier used for multiplexing.
        stream_id: Stream identifier when present (used by S2S).
    """

    text: str
    start_s: float
    stop_s: float
    client_req_id: str | None = None
    stream_id: int | None = None


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
                    client_req_id=msg.get("client_req_id"),
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
    tts_endpoint: str = "speech/tts",
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
        tts_endpoint: WebSocket route for the TTS endpoint.

    Returns:
        TTSResult containing complete audio data and metadata.
    """
    chunks = []
    stream = await tts_stream(client, setup, text, tts_endpoint=tts_endpoint)
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

    @property
    def delay_in_frames(self) -> int | None:
        """Get the delay in frames configured for this STT request."""
        return self._ready.get("delay_in_frames")


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
    stt_endpoint: str = "speech/asr",
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
        stt_endpoint: WebSocket route for the STT endpoint.

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

    stream = await stt_stream(
        client, setup, bytes_stream, stt_endpoint=stt_endpoint
    )
    all_texts = []
    async for text in stream.iter_text():
        all_texts.append(text)
    return STTResult(
        text=" ".join(t.text for t in all_texts),
        text_with_timestamps=all_texts,
        request_id=stream.request_id,
    )


def _encode_audio_chunk(audio: bytes | np.ndarray) -> dict:
    """Encode an audio chunk into a server `audio` message.

    Args:
        audio: Audio chunk as raw bytes or a 1-D numpy array (int16 or
            float32). float32 samples are expected in the range [-1.0, 1.0].

    Returns:
        A dict with the base64 encoded audio ready to be sent to the server.

    Raises:
        ValueError: If the numpy array dtype or shape is invalid.
    """
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


class S2SStream:
    """Stream handler for Speech-to-Speech results.

    Speech-to-Speech is duplex: input audio is streamed to the server while
    output audio chunks and transcribed (optionally translated) text segments
    are streamed back. Use :meth:`iter_audio` to consume the output audio while
    text segments are collected into ``_text_with_timestamps``, or
    :meth:`iter_events` to handle every server message yourself.

    Attributes:
        _stream: Underlying async message stream.
        _setup: S2S configuration used for this request.
        _text_with_timestamps: Collected text segments with timing.
    """

    def __init__(
        self,
        stream: AsyncGenerator,
        setup: S2SSetup,
        ready: Any,
    ):
        """Initialize S2SStream.

        Args:
            stream: Async generator yielding S2S messages.
            setup: S2S configuration dictionary.
            ready: The server's `ready` message.
        """
        self._stream = stream
        self._setup = setup
        self._ready = ready
        self._text_with_timestamps = []

    @property
    def sample_rate(self) -> int | None:
        """Get the sample rate of the output audio."""
        return self._ready.get("sample_rate")

    @property
    def frame_size(self) -> int | None:
        """Get the output frame size in samples."""
        return self._ready.get("frame_size")

    @property
    def request_id(self) -> str | None:
        """Get the unique request ID."""
        return self._ready.get("request_id")

    async def iter_audio(self) -> AsyncGenerator[bytes]:
        """Stream output audio chunks as bytes.

        Iterates over the output audio chunks from the server, yielding raw
        audio bytes and collecting transcribed text segments into
        ``_text_with_timestamps`` along the way.

        Yields:
            Raw output audio data chunks (base64 decoded) in the format
            specified by the setup configuration (e.g. PCM, WAV).

        Example:
            >>> async def translate_to_file():
            ...     client = GradiumClient(api_key="your-key")
            ...     setup = S2SSetup(
            ...         input_format="pcm",
            ...         output_format="wav",
            ...         json_config={"target_language": "en"},
            ...     )
            ...     stream = await client.s2s_stream(setup, audio_generator())
            ...     with open("output.wav", "wb") as f:
            ...         async for chunk in stream.iter_audio():
            ...             f.write(chunk)
            ...     for twt in stream._text_with_timestamps:
            ...         print(f"{twt.text}: {twt.start_s}s - {twt.stop_s}s")
        """
        async for msg in self._stream:
            msg_type = msg.get("type")
            if msg_type == "text":
                start_s = msg.get("start_s", 0.0)
                self._text_with_timestamps.append(
                    TextWithTimestamps(
                        text=msg.get("text", ""),
                        start_s=start_s,
                        stop_s=msg.get("stop_s", start_s),
                        client_req_id=msg.get("client_req_id"),
                        stream_id=msg.get("stream_id"),
                    )
                )
            elif msg_type == "audio":
                yield base64.b64decode(msg["audio"])

    async def iter_events(self) -> AsyncGenerator[dict]:
        """Stream every server message, with audio fields decoded.

        Unlike :meth:`iter_audio` this yields both ``text`` and ``audio``
        messages as dictionaries. ``audio`` messages have their ``audio`` field
        replaced by the base64 decoded raw bytes. ``text`` messages are also
        collected into ``_text_with_timestamps``.

        Yields:
            Server message dictionaries.
        """
        async for msg in self._stream:
            msg_type = msg.get("type")
            if msg_type == "text":
                start_s = msg.get("start_s", 0.0)
                self._text_with_timestamps.append(
                    TextWithTimestamps(
                        text=msg.get("text", ""),
                        start_s=start_s,
                        stop_s=msg.get("stop_s", start_s),
                        client_req_id=msg.get("client_req_id"),
                        stream_id=msg.get("stream_id"),
                    )
                )
            elif msg_type == "audio":
                msg = {**msg, "audio": base64.b64decode(msg["audio"])}
            yield msg


@dataclass
class S2SResult:
    """Buffered Speech-to-Speech result.

    Contains the complete output audio together with the transcribed
    (optionally translated) text and metadata.

    Attributes:
        raw_data: Raw output audio bytes in the specified format.
        sample_rate: Sample rate of the output audio or None.
        output_format: Audio output format (e.g. "pcm", "wav").
        request_id: Unique ID for this S2S request.
        text: Complete transcribed text with segments joined by spaces.
        text_with_timestamps: List of text segments with timing information.
    """

    raw_data: bytes
    sample_rate: int | None
    output_format: str | None
    request_id: str | None
    text: str
    text_with_timestamps: list[TextWithTimestamps]

    def pcm16(self) -> np.array:
        """Get PCM16 numpy array from raw output audio.

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
        """Get PCM float numpy array from raw output audio.

        Converts PCM16 audio to float32 with values in range [-1.0, 1.0].

        Returns:
            Numpy array with float32 audio samples.

        Raises:
            ValueError: If output_format is not "pcm".
        """
        return self.pcm16().astype(np.float32) / 32768.0


async def s2s_stream(
    client: "gradium_client.GradiumClient",
    setup: S2SSetup,
    audio: AsyncGenerator,
    s2s_endpoint: str = "speech/s2s",
) -> S2SStream:
    """Stream Speech-to-Speech results.

    Initiates a streaming S2S request and returns a handler for consuming the
    output audio chunks and transcribed text as they arrive from the server.

    Args:
        client: GradiumClient instance.
        setup: S2S configuration (S2SSetup TypedDict).
        audio: Async generator yielding audio chunks. For numpy arrays:
            - dtype must be int16 or float32
            - shape must be 1-dimensional
            - For float32, values should be in range [-1.0, 1.0]
            For "pcm" input the expected rate is 24 kHz, mono.
        s2s_endpoint: WebSocket route for S2S (default: "speech/s2s").

    Returns:
        S2SStream object for iterating over output audio and text.

    Raises:
        RuntimeError: If server doesn't send expected "ready" message first.
        ValueError: If audio format is invalid.
    """
    if (config := setup.get("json_config")) is not None:
        if not isinstance(config, str):
            # Make a copy to avoid modifying the original setup
            setup = dict(setup)
            setup["json_config"] = json.dumps(config)

    stream = client.stream(
        s2s_endpoint, setup, audio, map_input_fn=_encode_audio_chunk
    )
    ready = await anext(stream)
    if (msg_type := ready.get("type")) != "ready":
        raise RuntimeError(f"unexpected first message type `{msg_type}`")

    return S2SStream(stream, setup=setup, ready=ready)


async def s2s(
    client: "gradium_client.GradiumClient",
    setup: S2SSetup,
    audio: bytes | np.ndarray | AsyncGenerator[bytes],
    sample_rate: int | None = None,
    s2s_endpoint: str = "speech/s2s",
) -> S2SResult:
    """Buffered Speech-to-Speech conversion.

    Pipes audio through the speech-to-speech pipeline and returns the complete
    output audio together with the transcribed (optionally translated) text
    once the request completes. This is simpler than s2s_stream for when you
    don't need to process results as they arrive.

    Args:
        client: GradiumClient instance.
        setup: S2S configuration (S2SSetup TypedDict).
        audio: Audio data. Can be:
            - bytes: Raw audio bytes (sample_rate must be None)
            - np.ndarray: Audio samples (int16 or float32, "pcm" input only)
            - AsyncGenerator[bytes]: Stream of audio chunks
        sample_rate: Sample rate in Hz. Required for numpy arrays (must be
            24000), not supported for bytes input.
        s2s_endpoint: WebSocket route for the S2S endpoint.

    Returns:
        S2SResult containing the output audio, transcription and metadata.

    Raises:
        ValueError: If audio format is invalid or sample_rate mismatch.
    """

    async def bytes_stream_gen(audio, chunk_size: int) -> AsyncGenerator[bytes]:
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    if isinstance(audio, bytes):
        if sample_rate is not None:
            raise ValueError(
                "sample_rate is not supported for bytes audio input"
            )
        bytes_stream = bytes_stream_gen(audio, 4096)
    elif isinstance(audio, np.ndarray):
        if not setup.get("input_format", "pcm").startswith("pcm"):
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

    stream = await s2s_stream(
        client, setup, bytes_stream, s2s_endpoint=s2s_endpoint
    )
    chunks = []
    async for chunk in stream.iter_audio():
        chunks.append(chunk)
    return S2SResult(
        raw_data=b"".join(chunks),
        sample_rate=stream.sample_rate,
        output_format=setup.get("output_format"),
        request_id=stream.request_id,
        text=" ".join(t.text for t in stream._text_with_timestamps),
        text_with_timestamps=stream._text_with_timestamps,
    )
