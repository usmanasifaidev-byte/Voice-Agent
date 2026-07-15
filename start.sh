#!/bin/sh
set -e

# Render assigns $PORT dynamically. webrtc.py calls its own /api/scripted_chat
# over HTTP via API_BASE_URL, so that loopback URL must track the real port —
# otherwise it silently breaks (defaults to localhost:8000, which may not be
# where uvicorn is actually listening).
PORT="${PORT:-8000}"
export API_BASE_URL="http://127.0.0.1:${PORT}"

# Single worker: voice-call state (SESS/TURN) lives in in-process memory, so
# more than one worker/instance would break requests landing on a different
# process mid-call.
exec uvicorn src.server:app --host 0.0.0.0 --port "${PORT}" --workers 1
