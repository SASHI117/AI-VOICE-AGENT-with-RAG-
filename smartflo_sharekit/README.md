# SmartFlo (Tata Tele) — Pipecat Integration Kit

Everything you need to connect your Pipecat voice bot to SmartFlo telephony.

This kit is customized to run your existing agents:

- `apsagent` (default old agent behavior)
- `rnf_agnet` (RNF profile, default in this SmartFlo bridge)

---

## What's in this kit

```
smartflo/
  smartflo_serializer.py   ← The serializer (only SmartFlo-specific file)
example_bot.py             ← Minimal working Pipecat bot using the serializer
example_server.py          ← FastAPI server with /ws/smartflo endpoint
requirements.txt
README.md (this file)
```

---

## How SmartFlo works with Pipecat

SmartFlo streams audio over WebSocket using the same JSON protocol as Exotel and Twilio Media Streams:

```
SmartFlo  ──(WebSocket)──►  Your Server  ──►  Pipecat Pipeline
          ◄──(WebSocket)──                ◄──
```

The only difference from Exotel is the **audio codec**:

| Platform  | Audio codec      | Encoding   |
|-----------|-----------------|------------|
| Exotel    | Raw PCM         | linear16   |
| SmartFlo  | µ-law (PCMU / G.711) | mulaw  |
| Twilio    | µ-law (PCMU / G.711) | mulaw  |

**Critical:** If you send raw PCM to SmartFlo, the caller hears loud static. The `SmartFloFrameSerializer` handles the PCM ↔ µ-law conversion automatically.

---

## Integration — 3 steps

### Step 1: Copy the serializer into your project

```
your_project/
  smartflo/
    __init__.py               ← empty file
    smartflo_serializer.py    ← copy from this kit
```

### Step 2: Replace your serializer

In your `bot.py`, import and swap:

```python
# Before (Exotel):
from pipecat.serializers.exotel import ExotelFrameSerializer
serializer = ExotelFrameSerializer(stream_sid=stream_sid)

# After (SmartFlo):
from smartflo.smartflo_serializer import SmartFloFrameSerializer
serializer = SmartFloFrameSerializer(stream_sid=stream_sid, call_sid=call_sid)
```

Everything else — transport, pipeline, LLM, STT, TTS — stays identical.

### Step 3: Add a `/ws/smartflo` endpoint to your server

```python
@app.websocket("/ws/smartflo")
async def smartflo_endpoint(websocket: WebSocket):
    await websocket.accept()
    await run_bot_smartflo(websocket)
```

See `example_server.py` for the full version with error handling.

---

## SmartFlo JSON protocol

SmartFlo sends these events over the WebSocket (same as Exotel/Twilio):

### Incoming events (SmartFlo → your server)

| Event       | Description                                | Action required         |
|-------------|-------------------------------------------|-------------------------|
| `connected` | WebSocket established                      | Ignore                  |
| `start`     | Call started — contains `streamSid`, `callSid` | Extract IDs, create serializer |
| `media`     | Audio chunk (base64 µ-law)                 | Decode → PCM → STT      |
| `mark`      | Synchronisation marker                     | **Must echo back**      |
| `dtmf`      | Keypad digit pressed                       | Optional handling        |
| `stop`      | Call ended                                 | Ignore / cleanup        |
| `clear`     | Tells bot to stop speaking                 | Flush buffer, stop TTS  |

### Outgoing events (your server → SmartFlo)

| Event   | When to send              | Format                                           |
|---------|--------------------------|--------------------------------------------------|
| `media` | Send audio to caller      | `{"event":"media","streamSid":"...","media":{"payload":"<base64-mulaw>"}}` |
| `mark`  | Acknowledge received mark | `{"event":"mark","streamSid":"...","mark":{"name":"<name>"}}` |
| `clear` | Interrupt caller audio    | `{"event":"clear","streamSid":"..."}` |

### Critical: 160-byte alignment

SmartFlo **requires** all outgoing `media` payloads to be a multiple of 160 bytes (raw µ-law).
Sending misaligned chunks causes choppy or silent audio.
The serializer handles this automatically by buffering and padding with µ-law silence bytes (`0xFF`).

### Parsing the `start` event

```python
import json

async for message in websocket.iter_text():
    data = json.loads(message)
    if data["event"] == "start":
        start = data["start"]
        stream_sid = start["streamSid"]   # use for all outgoing messages
        call_sid   = start["callSid"]     # optional, for logging
        break
```

---

## Local setup & testing

### Prerequisites

- Python 3.11+
- [ngrok](https://ngrok.com) (free tier works) — SmartFlo can't call `localhost`

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

Use your existing server root `.env` (already used by `apsagent.py` and `rnf_gemini/rnf_agnet.py`).

Minimum required:

```env
GOOGLE_API_KEY=...
```

Optional profile selector for SmartFlo bridge:

```env
SMARTFLO_AGENT_PROFILE=rnf_agnet
```

Use `SMARTFLO_AGENT_PROFILE=apsagent` to switch profile.

### 3. Run the server

```bash
uvicorn example_server:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected JSON includes active profile and endpoint.

### 4. Expose with ngrok

In a second terminal:

```bash
ngrok http 8000
```

ngrok gives you a URL like `https://abc123.ngrok.io`.
Your SmartFlo WebSocket URL is:

```
wss://abc123.ngrok.io/ws/smartflo
```

### 5. Configure SmartFlo

1. Log in to SmartFlo dashboard
2. Go to **Voice Bot** → your bot → **Stream URL**
3. Paste: `wss://abc123.ngrok.io/ws/smartflo`
4. Make a test call

---

## Production deployment

For production, host the server on any HTTPS endpoint (the `wss://` URL SmartFlo requires is just HTTPS WebSocket):

- **Docker**: standard Python container, expose port 8000
- **Cloud Run / Azure Container Apps / AWS ECS**: any works
- Must be HTTPS — SmartFlo rejects plain `ws://`

Production URL format:
```
wss://your-domain.com/ws/smartflo
```

---

## Common mistakes

### 1. Sending raw PCM (static noise on call)

**Wrong:**
```python
# ExotelFrameSerializer sends raw PCM — DO NOT use for SmartFlo
from pipecat.serializers.exotel import ExotelFrameSerializer
```

**Right:**
```python
from smartflo.smartflo_serializer import SmartFloFrameSerializer
```

### 2. Not acknowledging `mark` events

SmartFlo uses `mark` events to synchronise playback. If you don't echo them back, the call can behave unexpectedly. The serializer handles this automatically — just make sure you're passing `OutputTransportMessageFrame` back through your pipeline output.

### 3. Forgetting the 160-byte chunk alignment

Raw µ-law chunks must be multiples of 160 bytes. The serializer pads with `0xFF` (µ-law silence). If you write your own serializer, this is easy to miss — it causes intermittent choppy audio that's hard to debug.

### 4. Using `ws://` instead of `wss://`

SmartFlo requires TLS. Local testing with ngrok is fine — ngrok provides the TLS termination. In production, make sure your reverse proxy / load balancer terminates TLS.

---

## Pipecat version compatibility

Tested with `pipecat-ai >= 0.0.40`.

The serializer imports from:
- `pipecat.audio.utils` — `pcm_to_ulaw`, `ulaw_to_pcm`, `create_stream_resampler`
- `pipecat.serializers.base_serializer` — `FrameSerializer`, `FrameSerializerType`
- `pipecat.frames.frames` — standard frame types

If you're on an older version of Pipecat, check that `pcm_to_ulaw` / `ulaw_to_pcm` exist in `pipecat.audio.utils`. They were added in 0.0.35.

---

## File reference

| File | Purpose |
|------|---------|
| `smartflo/smartflo_serializer.py` | The serializer — copy this into your project |
| `smartflo/__init__.py` | Empty init file (required for Python package) |
| `example_bot.py` | Minimal bot — shows how to wire serializer into Pipecat |
| `example_server.py` | FastAPI server — shows the `/ws/smartflo` endpoint pattern |
| `requirements.txt` | Dependencies |
