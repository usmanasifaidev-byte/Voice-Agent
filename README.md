# Agentic System

A small voice + text booking assistant. It walks a caller through a fixed script — collect name, age, preferred time, and party size, confirm the details, register the booking — over WebSocket-based real-time voice or plain text chat.

> The codebase also contains RAG (document upload/search) and generic API-endpoint-calling code (`app/rag.py`, `app/api/rag.py`, `app/api/setup.py`, `public/setup.html`, `developer_api/`). None of it is wired into the running app (`src/server.py` doesn't mount those routers) — it's left in place but dormant, not part of the current booking flow.

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install External Tools

#### FFmpeg
Download and install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or use a pre-built distribution.

Set the path in your `.env` file:
```env
FFMPEG_BIN=path/to/ffmpeg.exe  # Windows
# or
FFMPEG_BIN=ffmpeg  # Linux/Mac (if in PATH)
```

#### Faster-Whisper (ASR)
The system uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech recognition, which provides GPU acceleration and built-in VAD filtering.

**Option 1: Use model name (auto-downloads from HuggingFace)**
```env
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large-v3, etc.
```

**Option 2: Use local converted model**
If you have a converted CTranslate2 model, specify the path:
```env
WHISPER_MODEL=path/to/whisper-base-ct2
```

**GPU Acceleration:**
- GPU is auto-detected if CUDA is available
- Set `USE_CUDA=true` in `.env` to force GPU, or `USE_CUDA=false` to force CPU
- GPU uses `float16` compute type, CPU uses `int8` for optimal performance

#### Piper TTS (Optional)
Download a Piper voice model and set the path:
```env
PIPER_VOICE=path/to/models/en_US-lessac-medium.onnx
```

### 3. Environment Variables

Create a `.env` file in the root directory (see `.env.example`):

```env
TOGETHER_API_KEY=your_together_ai_api_key_here
TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo

# Voice processing (see above for setup instructions)
FFMPEG_BIN=ffmpeg
WHISPER_MODEL=base  # Model name (tiny, base, small, medium, large-v3) or local path
PIPER_VOICE=models/en_US-lessac-medium.onnx
# Optional: Force GPU/CPU (auto-detected if not set)
# USE_CUDA=true  # Force GPU
# USE_CUDA=false  # Force CPU
```

Get your API key from [Together AI](https://together.ai/)

### 4. Run the Server

```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```

Or using Python directly:

```bash
python -m uvicorn src.server:app --reload
```

The server will start on `http://localhost:8000`

## Usage

Visit `http://localhost:8000` — it serves the chat UI directly (text + voice call). The conversation script is defined in `simpleScript.txt` at the repo root; edit that file to change what the assistant asks for or how it behaves.

## Deploying to Render

The repo includes a `Dockerfile`, `.dockerignore`, `start.sh`, and `render.yaml` for a Docker-based Render Web Service.

1. Push the repo to GitHub/GitLab and create a new Render **Blueprint** (or Web Service) pointing at it — `render.yaml` configures the service automatically.
2. In the Render dashboard, set the `TOGETHER_API_KEY` secret (left blank in `render.yaml` on purpose — never commit real keys). `TOGETHER_MODEL`, `WHISPER_MODEL`, and `USE_CUDA` are pre-filled but can be overridden per-environment.
3. Deploy. The Docker build installs CPU-only PyTorch, ffmpeg, and downloads a Piper voice model (`en_US-lessac-medium`, chosen for faster CPU synthesis over the larger `-high` variant) — no manual file uploads needed.

**Things to know before you deploy:**
- **Single instance only.** Voice-call state (`SESS`/`TURN` in `voice/service/shared_session.py`) lives in in-process memory, not a database. `render.yaml` pins `numInstances: 1` and `start.sh` runs uvicorn with `--workers 1` — don't change either without moving session state out of memory first.
- **No GPU on Render.** `USE_CUDA=false` is set explicitly; faster-whisper and Piper both run on CPU (`int8` compute type). The `base` Whisper model is a reasonable default — `small`/`medium` will be noticeably slower on a shared CPU instance.
- **Nothing is persisted across deploys.** Uploaded/recorded audio in `storage/voice/` is ephemeral container disk — fine for this app since no booking data is currently written to disk (see `simpleScript.txt` / `app/api/state.py` — the assistant confirms bookings conversationally only). Add a Render Disk or external DB if you need durability later.
- **Cold starts.** First request after a deploy/restart triggers a faster-whisper model download (cached after that) plus PyTorch/model init — expect the first call to be slow.
- **Plan size.** `render.yaml` defaults to the `starter` plan. Torch + faster-whisper + Piper together are memory-hungry; if responses are slow or the instance OOMs, bump to `standard` in `render.yaml`.

## Project Structure

- `src/server.py` - FastAPI application entry point
- `app/api/state.py` - the live `/api/scripted_chat` endpoint (loads `simpleScript.txt`, calls the LLM)
- `app/api/voice.py`, `app/api/webrtc.py` - voice session + WebSocket/WebRTC handling
- `voice/` - ASR (faster-whisper), VAD (Silero), TTS (Piper) and the turn/session managers
- `app/services/` - external service integrations (Together AI client, shared HTTP client)
- `public/` - frontend (`chat.html` is the only page actually served)
- `simpleScript.txt` - the conversation script driving the assistant's behavior
- `configs/`, `storage/` - present but unused by the current booking flow (see note at the top)

## Features

- **Script-driven conversation**: `simpleScript.txt` defines a small booking flow — collect name, age, time, and party size, confirm, then end the call
- **Real-time voice interaction**: WebSocket-based voice communication with:
  - Voice Activity Detection (VAD) using Silero VAD
  - Automatic Speech Recognition (ASR) using faster-whisper
  - Text-to-Speech (TTS) using Piper
  - Buffer-based silence detection for efficient processing
  - Automatic call end once a booking is confirmed
- **Text chat**: same script/backend, no voice required

## Voice Features

The system supports real-time voice conversations:

1. **Voice Activity Detection**: Detects when the user is speaking vs. silent
2. **Silence-based processing**: Only processes audio when silence is detected (reduces CPU usage)
3. **In-memory processing**: Efficient audio processing with minimal disk I/O
4. **Segment archival**: All voice segments are saved with transcripts for debugging
5. **WebSocket streaming**: Low-latency audio streaming for real-time interaction
