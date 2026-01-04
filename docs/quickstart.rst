Quick Start Guide
=================

This guide will help you get started with the Gradium Python client.

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install gradium

Getting Your API Key
--------------------

To use the Gradium API, you'll need an API key. You can obtain one by signing up at `gradium.ai <https://gradium.ai>`_.

Basic Usage
-----------

Text-to-Speech
~~~~~~~~~~~~~~

Convert text to speech with the TTS API:

.. code-block:: python

   from gradium import GradiumClient, TTSSetup
   import asyncio

   async def text_to_speech():
       client = GradiumClient(api_key="gsk_...")

       result = await client.tts(
           setup=TTSSetup(voice="default"),
           text="Hello, this is a test of the Gradium TTS API."
       )

       # Save the audio to a file
       with open("output.wav", "wb") as f:
           f.write(result)

   asyncio.run(text_to_speech())

TTS Streaming
~~~~~~~~~~~~~

For real-time streaming of audio:

.. code-block:: python

   from gradium import GradiumClient, TTSSetup
   import asyncio

   async def stream_tts():
       client = GradiumClient(api_key="gsk_...")

       async with client.tts_stream(
           setup=TTSSetup(voice="default"),
           text="This is streaming text-to-speech."
       ) as stream:
           async for chunk in stream:
               # Process audio chunk
               print(f"Received {len(chunk)} bytes")

   asyncio.run(stream_tts())

Speech-to-Text
~~~~~~~~~~~~~~

Transcribe audio to text:

.. code-block:: python

   from gradium import GradiumClient, STTSetup
   import asyncio

   async def speech_to_text():
       client = GradiumClient(api_key="gsk_...")

       with open("audio.wav", "rb") as f:
           audio_data = f.read()

       result = await client.stt(
           setup=STTSetup(),
           audio=audio_data
       )

       print(f"Transcription: {result}")

   asyncio.run(speech_to_text())

STT Streaming
~~~~~~~~~~~~~

Stream audio for real-time transcription:

.. code-block:: python

   from gradium import GradiumClient, STTSetup
   import asyncio

   async def stream_stt():
       client = GradiumClient(api_key="gsk_...")

       async with client.stt_stream(setup=STTSetup()) as stream:
           # Send audio chunks
           with open("audio.wav", "rb") as f:
               while chunk := f.read(4096):
                   await stream.write(chunk)

           # Get transcription results
           async for result in stream:
               print(f"Transcription: {result}")

   asyncio.run(stream_stt())

Managing Voices
~~~~~~~~~~~~~~~

List available voices:

.. code-block:: python

   from gradium import GradiumClient
   import asyncio

   async def list_voices():
       client = GradiumClient(api_key="gsk_...")
       voices = await client.voices.list()

       for voice in voices:
           print(f"Voice: {voice.name}")

   asyncio.run(list_voices())

Checking Usage
~~~~~~~~~~~~~~

Monitor your API usage:

.. code-block:: python

   from gradium import GradiumClient
   import asyncio

   async def check_usage():
       client = GradiumClient(api_key="gsk_...")
       usage = await client.usages.get()

       print(f"Credits used: {usage.credits_used}")
       print(f"Credits remaining: {usage.credits_remaining}")

   asyncio.run(check_usage())

Next Steps
----------

* Check out the :doc:`api` for detailed API reference
* Visit the `GitHub repository <https://github.com/gradium/gradium-py>`_ for examples
* Try the `Google Colab notebooks <https://github.com/gradium-ai/gradium-py/tree/main/notebooks>`_
