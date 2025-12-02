# Gradium Python Client

Python client library for the [Gradium Voice AI API](https://gradium.ai).

## Text-to-Speech Streaming Example

You can find some examples in the `examples/` directory. Try them out with
[uv](https://docs.astral.sh/uv/getting-started/installation/). For
the text-to-speech streaming example, run:

```bash
uv run --with sphn examples/tts_streaming.py \
    --text "Hello, this is a test of the Gradium TTS streaming API." \
    --api-key gsk_...
```

## Speech-to-Text Streaming Example

You can also try the speech-to-text streaming example:

```bash
uv run --with sphn examples/tts_streaming.py \
    --audio test_file.mp3 \
    --api-key gsk_...
```
