# WebSocket Endpoint for Text-to-Speech Streaming

Connect to this endpoint via WebSocket for real-time text-to-speech conversion with low latency audio streaming.

**Connection URL:**

For Europe
```
wss://eu.api.gradium.ai/api/speech/tts
```

For the USA
```
wss://us.api.gradium.ai/api/speech/tts
```


**Authentication:**
Include your API key in the WebSocket connection header:
- Header: `x-api-key: your_api_key`

---

## Quick Reference

| Direction | Message Type | Example |
|-----------|-------------|---------|
| 🔵⬆️ Client→Server | Setup (first) | `{"type": "setup", "voice_id": "YTpq7expH9539ERJ", "model_name": "default", "output_format": "wav"}` |
| 🟢⬇️ Server→Client | Ready | `{"type": "ready", "request_id": "uuid"}` |
| 🔵⬆️ Client→Server | Text | `{"type": "text", "text": "Hello, world!"}` |
| 🟢⬇️ Server→Client | Audio (stream) | `{"type": "audio", "audio": "base64..."}` |
| 🔵⬆️ Client→Server | EndOfStream | `{"type": "end_of_stream"}` |
| 🟢⬇️ Server→Client | AEndOfStream | `{"type": "end_of_stream"}` |
| 🔴⬇️ Server→Client | Error | `{"type": "error", "message": "Error description", "code": 1008}` |

---

## Message Types

### 1. Setup Message (First Message)

**Direction:** Client → Server
**Format:** JSON Object

```json
{
  "type": "setup",
  "model_name": "default",
  "voice_id": "YTpq7expH9539ERJ",
  "output_format": "wav"
}
```

**Fields:**
- `type` (string, required): Must be "setup"
- `model_name` (string, required): The TTS model to use (default: "default")
- `voice_id` (string, required): Voice ID from the library (e.g., "YTpq7expH9539ERJ" for Emma's voice) or custom voice ID
- `output_format` (string, required): Audio format - either "wav", "pcm", or "opus".

**Important:** This must be the very first message sent after connection. The server will close the connection if any other message is sent first.

---

### 2. Ready Message

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "ready",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Fields:**
- `type` (string): Will be "ready"
- `request_id` (string): Unique identifier for the session

This message is sent by the server after receiving the setup message, indicating that the connection is ready to receive text messages.

---

### 3. Text Message (Subsequent Messages)

**Direction:** Client → Server
**Format:** JSON Object

```json
{
  "type": "text",
  "text": "Hello, world!"
}
```

**Fields:**
- `type` (string, required): Must be "text"
- `text` (string, required): The text to be converted to speech

Send text messages to be converted to speech. You can send multiple text messages in sequence. The server will stream audio back as it's generated.

---

### 4. Audio Response

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "audio",
  "audio": "base64_encoded_audio_data..."
}
```

**Fields:**
- `type` (string): Will be "audio"
- `audio` (string): Base64-encoded audio data in the requested format

**Important:** Multiple audio messages will be streamed for each text message. Continue receiving until you detect the end of speech or receive a new message type.

---

### 5. End Of Stream

**Direction:** Client → Server and Server → Client
**Format:** JSON Object

```json
{
  "type": "end_of_stream",
}
```

This message is sent by the client when it has submitted all the text that it
wants to be considered. The server will then send back all the remaining audio
until all the text has been processed, then an `EndOfStream` message, and then
closes the websocket connection.

---

## Error Handling

When errors occur, the server sends an error message as JSON before closing the connection:

**Error Message Format:**
```json
{
  "type": "error",
  "message": "Error description explaining what went wrong",
  "code": 1008
}
```

**Common Error Codes:**
- `1008`: Policy Violation (e.g., invalid API key, missing setup message)
- `1011`: Internal Server Error (unexpected server-side error)

---

## Best Practices

1. **Always send setup first**: The server expects a setup message immediately after connection
2. **Handle audio streaming**: Audio responses are streamed in chunks - buffer and process appropriately
3. **Implement reconnection logic**: Network issues happen - build in automatic reconnection with exponential backoff
4. **Monitor connection health**: Implement ping/pong or periodic checks to detect stale connections
5. **Graceful error handling**: Parse error messages and handle different error codes appropriately
6. **Reuse connections**: For multiple utterances, keep the connection alive and send multiple text messages
7. **Close cleanly**: Always close WebSocket connections properly when done

---
