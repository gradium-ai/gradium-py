"""Voice management functionality for Gradium API.

This module provides functionality for creating, retrieving, updating, and
deleting custom voices for use with text-to-speech synthesis.

Functions:
    create: Create a new voice from an audio file.
    get: Retrieve voice information by UID.
    update: Update voice metadata.
    delete: Delete a voice by UID.

Voice designer (text-prompted voices):
    generate: Generate draft voice embeddings from a text prompt.
    embedding_get: Poll the status of a draft embedding.
    tts: Preview a draft embedding with a short one-shot TTS request.
    from_embedding: Promote a draft embedding to a permanent voice.
"""

import asyncio
import json
import pathlib
import time

import aiohttp

from . import client as gradium_client
from . import speech

ROUTE = "voices/"
GENERATOR_ROUTE = "voice-generator/"
POST_TTS_ROUTE = "post/speech/tts"

# Prefix of draft embedding ids returned by `generate`. Passing such an id as
# `voice_id` to the one-shot TTS endpoint tells the API to use the draft.
EMBEDDING_ID_PREFIX = "vox_emb_"


async def create(
    client: "gradium_client.GradiumClient",
    audio_file: pathlib.Path,
    *,
    language: str,
    name: str | None = None,
    description: str | None = None,
    start_s: float = 0.0,
    input_format: str | None = None,
) -> dict:
    """Create a new voice from an audio file.

    Uploads an audio file to create a custom voice that can be used for
    text-to-speech synthesis.

    Args:
        client: GradiumClient instance.
        audio_file: Path to the audio file to use for the voice.
        language: ISO language code of the voice (required).
        name: Name for the new voice. Defaults to the audio filename.
        description: Optional description of the voice.
        start_s: Start time in seconds for the audio clip to use. Defaults to 0.
        input_format: Audio format (e.g., "wav", "mp3"). If not provided,
            inferred from the file extension.

    Returns:
        Dictionary containing voice metadata including the new voice UID.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        aiohttp.ClientError: If the API request fails.
    """
    audio_file = pathlib.Path(audio_file)
    input_format = (
        input_format if input_format else audio_file.suffix.strip(".")
    )

    form_data = aiohttp.FormData()
    content_type = f"audio/{input_format}"
    with open(audio_file, "rb") as file:
        form_data.add_field(
            "audio_file",
            file,
            filename=audio_file.name,
            content_type=content_type,
        )

        fields = {
            "name": name if name is not None else audio_file.name,
            "language": language,
            "start_s": start_s,
            "description": description,
            "input_format": input_format,
        }
        for key, value in fields.items():
            if value is not None:
                form_data.add_field(key, str(value))

        result = await client.post(ROUTE, data=form_data)
    return result


async def get(
    client: "gradium_client.GradiumClient",
    voice_uid: str | None = None,
    include_catalog: bool = False,
) -> dict:
    """Get voice information.

    Args:
        client: GradiumClient instance.
        voice_uid: UID of the voice to retrieve. If None, returns all voices.

    Returns:
        Dictionary containing voice metadata. If voice_uid is None, returns
        a list of all available voices.

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    voice_uid = "" if voice_uid is None else voice_uid
    return await client.get(
        f"{ROUTE}{voice_uid}",
        params={"limit": 0, "include_catalog": int(include_catalog)},
    )


async def update(
    client: "gradium_client.GradiumClient",
    voice_uid: str,
    name: str | None = None,
    description: str | None = None,
    start_s: float | None = None,
    language: str | None = None,
) -> dict | None:
    """Update voice metadata.

    Updates one or more properties of an existing voice. Only non-None
    parameters will be updated.

    Args:
        client: GradiumClient instance.
        voice_uid: UID of the voice to update.
        name: New name for the voice. If None, not updated.
        description: New description. If None, not updated.
        start_s: New start time in seconds. If None, not updated.
        language: New ISO language code. If None, not updated.

    Returns:
        Updated voice metadata dictionary, or None if no updates were made.

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    data = {
        "name": name,
        "description": description,
        "start_s": start_s,
        "language": language,
    }
    data = {k: v for k, v in data.items() if v is not None}
    if data:
        return await client.put(
            f"{ROUTE}{voice_uid}",
            json=data,
        )


async def delete(
    client: "gradium_client.GradiumClient", voice_uid: str
) -> bool:
    """Delete a voice by UID.

    Permanently deletes a custom voice and removes it from available voices
    for text-to-speech synthesis.

    Args:
        client: GradiumClient instance.
        voice_uid: UID of the voice to delete.

    Returns:
        True if deletion was successful.

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    return await client.delete(f"{ROUTE}{voice_uid}")


async def generate(
    client: "gradium_client.GradiumClient",
    prompt: str,
    *,
    language: str,
    n_samples: int = 1,
    wait: bool = False,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> dict:
    """Generate draft voice embeddings from a text description.

    Each generated sample is a draft embedding identified by a ``vox_emb_``
    id. Drafts are computed asynchronously on the server and expire after a
    while unless promoted with `from_embedding`. Preview a draft with `tts`
    and poll its status with `embedding_get`.

    Generation is billed per sample. The request fails with a 402 status if
    the account does not have enough credits.

    Args:
        client: GradiumClient instance.
        prompt: Free-text description of the wanted voice, e.g.
            "A deep, warm male voice with a slight British accent".
        language: Language of the voice: "en", "fr", "es", "pt" or "de".
        n_samples: Number of candidate voices to generate (1 to 5).
        wait: If True, poll the drafts until they are all ready and return
            their latest status instead of the initial queued entries.
        timeout: Maximum time in seconds to wait for the drafts when `wait`
            is True.
        poll_interval: Delay in seconds between two status polls when `wait`
            is True.

    Returns:
        Dictionary with an ``embeddings`` list. Each entry has an
        ``embedding_id``, a ``ready`` flag and an ``expires_at`` timestamp.

    Raises:
        TimeoutError: If `wait` is True and the drafts are not ready within
            `timeout` seconds.
        aiohttp.ClientError: If the API request fails.
    """
    result = await client.post(
        f"{GENERATOR_ROUTE}generate",
        json={"prompt": prompt, "language": language, "n_samples": n_samples},
    )
    if wait:
        deadline = time.monotonic() + timeout
        embeddings = []
        for entry in result["embeddings"]:
            embedding_id = entry["embedding_id"]
            while not entry.get("ready"):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"draft embedding {embedding_id} not ready after "
                        f"{timeout} seconds"
                    )
                await asyncio.sleep(poll_interval)
                entry = await embedding_get(client, embedding_id)
            embeddings.append(entry)
        result = {**result, "embeddings": embeddings}
    return result


async def embedding_get(
    client: "gradium_client.GradiumClient", embedding_id: str
) -> dict:
    """Get the status of a draft voice embedding.

    Args:
        client: GradiumClient instance.
        embedding_id: Draft embedding id (``vox_emb_...``) as returned by
            `generate`.

    Returns:
        Dictionary describing the embedding: ``embedding_id``, ``ready``,
        ``expires_at``, ``prompt``, ``language`` and ``created_at``.

    Raises:
        LookupError: If the API returns no entry for this embedding id.
        aiohttp.ClientError: If the API request fails, e.g. 404 when the
            embedding does not exist, has expired or belongs to another
            organization.
    """
    result = await client.get(
        f"{GENERATOR_ROUTE}embeddings", params={"embedding_id": embedding_id}
    )
    # The endpoint answers with the same `{"embeddings": [...]}` envelope as
    # `generate`; unwrap the single entry we asked for.
    entries = result.get("embeddings") or []
    if not entries:
        raise LookupError(f"no draft embedding with id {embedding_id}")
    return entries[0]


async def tts(
    client: "gradium_client.GradiumClient",
    embedding_id: str,
    text: str,
    *,
    output_format: str = "wav",
    model_name: str | None = None,
    json_config: dict | str | None = None,
) -> "speech.TTSResult":
    """Preview a draft voice embedding with a one-shot TTS request.

    Renders a short text with the draft embedding through the HTTP TTS
    endpoint, so a candidate voice can be judged before promoting it with
    `from_embedding`. The request is billed as regular TTS.

    Args:
        client: GradiumClient instance.
        embedding_id: Draft embedding id (``vox_emb_...``) as returned by
            `generate`. The draft must be ready, otherwise the API returns
            a 404.
        text: Text to synthesize. Previews are meant to be short: the API
            rejects text above its length limit with a 400.
        output_format: Audio format: "wav", "opus" (ogg wrapped) or "pcm".
        model_name: Optional TTS model name, as for any TTS request.
        json_config: Optional extra TTS configuration, as a dict or a JSON
            string.

    Returns:
        TTSResult with the audio bytes in `raw_data`. The sample rate and
        request id are not reported by this endpoint and are left to None.

    Raises:
        aiohttp.ClientError: If the API request fails, e.g. 404 when the
            embedding is missing, expired or not ready yet.
    """
    body = {
        "voice_id": embedding_id,
        "text": text,
        "output_format": output_format,
        "only_audio": True,
    }
    if model_name is not None:
        body["model_name"] = model_name
    if json_config is not None:
        if not isinstance(json_config, str):
            json_config = json.dumps(json_config)
        body["json_config"] = json_config

    response = await client.post(POST_TTS_ROUTE, parse=False, json=body)
    raw_data = await response.read()
    return speech.TTSResult(
        raw_data=raw_data,
        sample_rate=None,
        output_format=output_format,
        request_id=None,
        text_with_timestamps=[],
    )


async def from_embedding(
    client: "gradium_client.GradiumClient",
    embedding_id: str,
    *,
    name: str,
    description: str | None = None,
) -> dict:
    """Create a permanent voice from a draft voice embedding.

    Promotes a draft produced by `generate` to a regular voice. The new
    voice behaves like any other custom voice and never expires. Creating
    the voice is free, generation was already billed.

    Args:
        client: GradiumClient instance.
        embedding_id: Draft embedding id (``vox_emb_...``) to promote.
        name: Name of the new voice.
        description: Optional description of the voice.

    Returns:
        Dictionary containing the voice metadata, including its UID.

    Raises:
        aiohttp.ClientError: If the API request fails. The API returns a 409
            when the custom voices quota is reached, when the embedding is
            missing, not ready or belongs to another organization, or when
            a voice was already created from this embedding.
    """
    data = {"voxium_embedding_id": embedding_id, "name": name}
    if description is not None:
        data["description"] = description
    return await client.post(f"{ROUTE}from-embedding", json=data)
