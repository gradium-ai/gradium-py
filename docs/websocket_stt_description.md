# WebSocket Endpoint for Speech-to-Text Streaming

Connect to this endpoint via WebSocket for real-time speech-to-text conversion with streaming audio input.

**Connection URL:**

For Europe
```
wss://eu.api.gradium.ai/api/speech/asr
```

For the USA
```
wss://us.api.gradium.ai/api/speech/asr
```

**Authentication:**
Include your API key in the WebSocket connection header:
- Header: `x-api-key: your_api_key`

---

## Quick Reference

| Direction | Message Type | Example |
|-----------|-------------|---------|
| 🔵⬆️ Client→Server | Setup (first) | `{"type": "setup", "model_name": "default", "input_format": "pcm"}` |
| 🟢⬇️ Server→Client | Ready | `{"type": "ready", "request_id": "uuid", "model_name": "default", "sample_rate": 24000}` |
| 🔵⬆️ Client→Server | Audio | `{"type": "audio", "audio": "base64..."}` |
| 🟢⬇️ Server→Client | Text (result) | `{"type": "text", "text": "Hello world", "start_s": 0.5}` |
| 🟢⬇️ Server→Client | VAD (activity) | `{"type": "step", "vad": [...], "step_idx": 5, "step_duration_s": 0.08}` |
| 🟢⬇️ Server→Client | End Text | `{"type": "end_text", "stop_s": 2.5}` |
| 🔵⬆️ Client→Server | EndOfStream | `{"type": "end_of_stream"}` |
| 🟢⬇️ Server→Client | EndOfStream | `{"type": "end_of_stream"}` |
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
  "input_format": "pcm"
}
```

**Fields:**
- `type` (string, required): Must be "setup"
- `model_name` (string, required): The Speech-To-Text model to use (default: "default")
- `input_format` (string, required): Audio format - "pcm", "wav", or "opus"

**Important:** This must be the very first message sent after connection. The server will close the connection if any other message is sent first.

---

### 2. Ready Message

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "ready",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_name": "default",
  "sample_rate": 24000,
  "frame_size": 1920,
  "delay_in_frames": 0,
  "text_stream_names": []
}
```

**Fields:**
- `type` (string): Will be "ready"
- `request_id` (string): Unique identifier for the session
- `model_name` (string): The Speech To Text model being used
- `sample_rate` (integer): Expected sample rate in Hz (typically 24000)
- `frame_size` (int): Number of samples by which the model processes data (typically 1920 which is equivalent to 80ms at 24kHz)
- `delay_in_frames` (integer): Delay in audio frames for the model
- `text_stream_names` (array): List of text stream names

This message is sent by the server after receiving the setup message, indicating that the connection is ready to receive audio.

---

### 3. Audio Message

**Direction:** Client → Server
**Format:** JSON Object (with binary audio data)

```json
{
  "type": "audio",
  "audio": "base64_encoded_audio_data..."
}
```

**Fields:**
- `type` (string, required): Must be "audio"
- `audio` (string, required): Base64-encoded audio data

**Audio Format Requirements (for PCM input):**
- **Sample Rate**: 24000 Hz (24kHz)
- **Format**: PCM (Pulse Code Modulation)
- **Bit Depth**: 16-bit signed integer (little-endian)
- **Channels**: Single channel (mono)
- **Chunk Size**: Recommended 1920 samples per frame (80ms at 24kHz)

Send audio messages to be transcribed. You can send multiple audio messages in sequence. The server will stream text and VAD responses as it processes the audio.

---

### 4. Text Response

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "text",
  "text": "Hello world",
  "start_s": 0.5,
  "stream_id": null
}
```

**Fields:**
- `type` (string): Will be "text"
- `text` (string): The transcribed text
- `start_s` (float): Start time of the transcription in seconds
- `stream_id` (integer or null): Stream identifier for tracking multiple concurrent streams

Text messages contain the transcribed speech. Multiple text messages will be streamed as the audio is processed.

---

### 5. VAD Response (Voice Activity Detection)

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "step",
  "vad": [
    {
      "horizon_s": 0.5,
      "inactivity_prob": 0.05
    },
    {
      "horizon_s": 1.0,
      "inactivity_prob": 0.08
    },
    {
      "horizon_s": 2.0,
      "inactivity_prob": 0.12
    }
  ],
  "step_idx": 5,
  "step_duration_s": 0.08,
  "total_duration_s": 0.4
}
```

**Fields:**
- `type` (string): Will be "step"
- `vad` (array): List of VAD predictions with future horizons
  - `horizon_s` (float): Lookahead duration in seconds
  - `inactivity_prob` (float): Probability that voice activity has ended by this horizon in seconds.
- `step_idx` (integer): The step index (increments every 80ms)
- `step_duration_s` (float): Duration of this step in seconds (typically 0.08)
- `total_duration_s` (float): Total duration of audio processed so far

**VAD Interpretation:**
- VAD messages are emitted every 80ms (one per audio frame)
- Use the `inactivity_prob` value from the longest horizon to determine if the speaker has likely finished
- Higher `inactivity_prob` values indicate higher confidence that speaking has ended
- Recommended threshold: Use `vad[2]["inactivity_prob"]` (third prediction) as the turn-taking indicator

---

### 6. End Text Response

**Direction:** Server → Client
**Format:** JSON Object

```json
{
  "type": "end_text",
  "stop_s": 2.5,
  "stream_id": null
}
```

**Fields:**
- `type` (string): Will be "end_text"
- `stop_s` (float): Stop time of last `text` message in seconds
- `stream_id` (integer or null): Stream identifier

Sent when the previous text segment has a finished and its end timestamp is
available.

---

### 7. End Of Stream

**Direction:** Client → Server and Server → Client
**Format:** JSON Object

```json
{
  "type": "end_of_stream"
}
```

This message is sent by the client when it has finished sending audio. The server will then process any remaining audio and send back all outstanding text results, VAD information, and then an `end_of_stream` message before closing the connection.

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
- `1008`: Policy Violation (e.g., invalid API key, missing setup message, invalid audio format)
- `1011`: Internal Server Error (unexpected server-side error)

---

## Best Practices for STT

1. **Always send setup first**: The server expects a setup message immediately after connection
2. **Use correct audio format**: When using PCM, ensure audio is 24kHz PCM 16-bit mono
3. **Send appropriately sized chunks**: 1920 samples (80ms) per message is recommended
4. **Graceful shutdown**: Send `end_of_stream` when done to properly close the session
5. **VAD Threshold**: Our VAD provides estimated probabilities that the speaker would be silent for a fixed number of seconds in the future. The thresholds to trigger the end-of-the-turn decisions might be application-dependent; as a starting point we recommend looking at the horizon of 2s and trigger when the inactivity_prob is above 0.5: `turn_ended = msg["vad"][2]["inactivity_prob"] > 0.5`.
5. **Acting on VAD**: Whenever you decide that the VAD probabilities warrant a decision to consider the turn ended, there is still up to `delay_in_frames` audio frames processed by the model. Instead of feeding silence from the speaker, the system can be made more reactive by flushing the remainder of the turn's transcript. For that, you can feed in `delay_in_frames` chunks of silence (vectors of zeros). If those are fed in faster than realtime, the API also has a possibility to process them faster, allowing a considerably more reactive turn-around.