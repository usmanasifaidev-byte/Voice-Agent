FROM python:3.11-slim

# ffmpeg: required for webm->wav conversion (voice pipeline)
# curl: used below to fetch the Piper TTS voice model at build time
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Piper TTS voice model (medium quality — good speed/quality balance on CPU-only hosting).
# Not committed to git (binary, too large for a normal push) so it's fetched here instead.
RUN mkdir -p models \
    && curl -fsSL -o models/en_US-lessac-medium.onnx \
       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
    && curl -fsSL -o models/en_US-lessac-medium.onnx.json \
       https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

COPY . .
RUN chmod +x start.sh

ENV FFMPEG_BIN=ffmpeg \
    WHISPER_MODEL=base \
    PIPER_VOICE=models/en_US-lessac-medium.onnx \
    USE_CUDA=false

EXPOSE 8000

CMD ["./start.sh"]
