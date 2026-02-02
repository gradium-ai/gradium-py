# Gradium Python Client

Python client library for the [Gradium Voice AI API](https://gradium.ai).

## Examples

See the [examples/](examples/) directory for usage examples:

- Text-to-Speech Streaming: `examples/tts_streaming.py`
- Speech-to-Text Streaming: `examples/stt_streaming.py`

You can also try it out in Google Colab.

- EU Server <a target="_blank" href="https://colab.research.google.com/github/gradium-ai/gradium-py/blob/main/notebooks/gradium_example_eu.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- US server <a target="_blank" href="https://colab.research.google.com/github/gradium-ai/gradium-py/blob/main/notebooks/gradium_example_us.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Command Line Interface

The package includes a CLI that can be run via `uvx gradium`:

```bash
# Text-to-speech
uvx gradium tts "Hello world" -o output.wav
uvx gradium tts "Hello world" --voice-id abc123 -o output.ogg

# Speech-to-text
uvx gradium stt audio.wav
uvx gradium stt audio.wav --json  # outputs JSON with timestamps

# To handle non-wav file, additional dependencies are required
uvx --from 'gradium[cli]' gradium stt audio.mp3
```

Run `uvx gradium --help` for all available options.

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
uv run --with sphn examples/stt_streaming.py \
    --audio test_file.mp3 \
    --api-key gsk_...
```
