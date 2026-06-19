"""Client for a Gradium TTS model served through Baseten.

A single-model Baseten deployment serves the TTS WebSocket at the base URL
itself and authenticates with a Baseten ``Authorization: Api-Key`` header rather
than a Gradium ``x-api-key``. :class:`BasetenClient` wires both up and exposes
only the text-to-speech endpoint; speech-to-text, voices and usage are not
implemented on this deployment.
"""

from collections.abc import AsyncGenerator

from . import speech, stream
from .base import BaseClient, SOURCE

__all__ = ["BasetenClient"]


class BasetenClient(BaseClient):
    """Text-to-speech client for a single-model Baseten deployment.

    Inherits the transport from :class:`~gradium.base.BaseClient` and implements
    only text-to-speech (:meth:`tts`, :meth:`tts_stream`, :meth:`tts_realtime`),
    served at the base URL (empty route) and authenticated with
    ``Authorization: Api-Key <key>``. Speech-to-text, voices and usage endpoints
    are not provided.

    Example:
        >>> client = BasetenClient(
        ...     base_url="model-XX.api.baseten.co/environments/production/websocket",
        ...     api_key="your-baseten-key",
        ... )
        >>> audio = await client.tts(
        ...     setup={"voice": "default", "output_format": "wav"},
        ...     text="Hello world",
        ... )
    """

    def __init__(self, *, base_url: str, api_key: str | None = None):
        """Initialize the Baseten TTS client.

        Args:
            base_url: Baseten websocket base URL for the model, e.g.
                ``model-XX.api.baseten.co/environments/production/websocket``.
                The TTS websocket is served at this URL directly.
            api_key: Baseten API key, sent as ``Authorization: Api-Key <key>``.
                If not provided, reads from the GRADIUM_API_KEY environment
                variable.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        super().__init__(base_url=base_url, api_key=api_key)

    @property
    def headers(self) -> dict:
        """HTTP headers with Baseten ``Api-Key`` authentication."""
        return {
            "Authorization": f"Api-Key {self._api_key}",
            "x-api-source": SOURCE,
        }

    async def tts_stream(
        self,
        setup: "speech.TTSSetup",
        text: str | list[str] | AsyncGenerator,
    ) -> "speech.TTSStream":
        """Stream text-to-speech synthesis results from the Baseten model."""
        return await speech.tts_stream(self, setup, text, tts_endpoint="")

    async def tts(
        self,
        setup: "speech.TTSSetup",
        text: str | list[str] | AsyncGenerator,
    ) -> "speech.TTSResult":
        """Synthesize text to speech (buffered) from the Baseten model."""
        return await speech.tts(self, setup, text, tts_endpoint="")

    def tts_realtime(self, **kwargs) -> "stream.Tts":
        """Create a real-time TTS WebSocket connection to the Baseten model."""
        kwargs.setdefault("route", "")
        return stream.Tts(self, **kwargs)
