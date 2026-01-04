Gradium Python Client Documentation
====================================

Welcome to the Gradium Python Client documentation. This library provides a simple and async-first interface to interact with the Gradium API for speech synthesis, speech recognition, voice management, and usage tracking.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api

Features
--------

* **Text-to-Speech (TTS)**: Convert text to natural-sounding speech
* **Speech-to-Text (STT)**: Transcribe audio to text
* **Voice Management**: Create and manage custom voices
* **Usage Tracking**: Monitor your API usage and credits
* **Async-First**: Built with asyncio for efficient asynchronous operations
* **Streaming Support**: Stream audio data in real-time

Installation
------------

Install the Gradium Python client using pip:

.. code-block:: bash

   pip install gradium

Quick Example
-------------

Here's a simple example of using the Gradium client for text-to-speech:

.. code-block:: python

   from gradium import GradiumClient, TTSSetup
   import asyncio

   async def main():
       client = GradiumClient(api_key="your-api-key")
       result = await client.tts(
           setup=TTSSetup(voice="default"),
           text="Hello, world!"
       )
       # result contains audio data

   asyncio.run(main())

Links
-----

* `GitHub Repository <https://github.com/gradium/gradium-py>`_
* `Gradium Website <https://gradium.ai>`_
* `API Documentation <https://docs.gradium.ai>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
