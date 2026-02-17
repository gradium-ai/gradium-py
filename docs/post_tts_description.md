# POST Endpoint for Text-to-Speech

Use this HTTP POST endpoint for simple, text-to-speech conversion. The audio
data is sent back in a streaming way.

**Endpoint URL:**

For Europe
```
https://eu.api.gradium.ai/api/post/speech/tts
```

For the USA
```
https://us.api.gradium.ai/api/post/speech/tts
```

**Authentication:**
Include your API key in the request header:
- Header: `x-api-key: your_api_key`

---

## Quick Example

```bash
curl -L -X POST https://eu.api.gradium.ai/api/post/speech/tts \
  -H "x-api-key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the text to speech system.", "voice_id": "YTpq7expH9539ERJ", "output_format": "wav", "only_audio": true}' \
  > output.wav
```

---

## Request Format

**Method:** POST
**Content-Type:** application/json

**Request Body:**
```json
{
  "text": "Hello, this is a test of the text to speech system.",
  "voice_id": "YTpq7expH9539ERJ",
  "output_format": "wav",
  "json_config": "{}",
  "only_audio": true
}
```

**Fields:**
- `text` (string, required): The text to be converted to speech
- `voice_id` (string, required): Voice ID from the library (e.g.,
  "YTpq7expH9539ERJ") or a custom voice ID
- `output_format` (string, required): Audio format - "wav", "pcm", or "opus"
  (ogg wrapped opus data).
- `json_config` (string, optional): Additional configuration in JSON string format (e.g., `{"padding_bonus": -1.2}`)
- `model_name` (string, optional): The TTS model to use (default: "default")
- `only_audio` (boolean, optional): When `true`, returns only the raw audio
  bytes. When `false` or omitted, returns a stream of JSON messages containing
  the audio and metadata. The format is the same as with the websocket endpoint.

---

## Response Format

### When `only_audio` is `true`

The response body contains the raw audio bytes in the requested format. Save directly to a file:

```bash
curl ... > output.wav
```

**Content-Type:** Depends on the output format:
- `audio/wav` for WAV format
- `audio/ogg` for Ogg wrapped Opus format
- `audio/pcm` for PCM format

### When `only_audio` is `false` or omitted

The response is a stream of JSON messages using the same format as the WebSocket
endpoint.

## Error Handling

When errors occur, the server returns an error response:

**Error Response Format:**
```json
{
  "error": "Error description explaining what went wrong",
  "code": 400
}
```

**Common HTTP Status Codes:**
- `400`: Bad Request (e.g., missing required fields, invalid voice ID)
- `401`: Unauthorized (invalid or missing API key)
- `500`: Internal Server Error

---

## When to Use POST vs WebSocket

The POST endpoint is ideal for simple, text-to-speech generations.
The main difference with the WebSocket endpoint is that the input is not
handled in a streaming way; the entire text is sent in one request. The audio is
still streamed back to the client, allowing for efficient handling of large
audio outputs and lower latency.

So if your use case involves sending complete text blocks and receiving audio
responses, the POST endpoint is a straightforward choice. For more interactive
or real-time applications where text input is streamed, the WebSocket endpoint
is more suitable.
