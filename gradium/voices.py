"""Voice management functionality for Gradium API.

This module provides functionality for creating, retrieving, updating, and
deleting custom voices for use with text-to-speech synthesis.

Functions:
    create: Create a new voice from an audio file.
    get: Retrieve voice information by UID.
    update: Update voice metadata.
    delete: Delete a voice by UID.
"""

import pathlib

import aiohttp

from . import client as gradium_client

ROUTE = "voices/"


async def create(
    client: "gradium_client.GradiumClient",
    audio_file: pathlib.Path,
    *,
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
    file = open(audio_file, "rb")
    form_data.add_field(
        "audio_file", file, filename=audio_file.name, content_type=content_type
    )

    fields = {
        "name": name if name is not None else audio_file.name,
        "start_s": start_s,
        "description": description,
        "input_format": input_format,
    }
    for key, value in fields.items():
        if value is not None:
            form_data.add_field(key, str(value))

    result = await client.post(ROUTE, data=form_data)
    file.close()
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

    Returns:
        Updated voice metadata dictionary, or None if no updates were made.

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    data = {"name": name, "description": description, "start_s": start_s}
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
