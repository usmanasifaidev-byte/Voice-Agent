from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import os
import re
import logging
from app.services.together_client import call_llm

router = APIRouter()
logger = logging.getLogger(__name__)

SCRIPT_PATH = os.path.join(os.getcwd(), "simpleScript.txt")

def _load_script() -> str:
    if os.path.exists(SCRIPT_PATH):
        with open(SCRIPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""

END_CALL_TAG = "[END_CALL]"

def _sanitize(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.MULTILINE).strip()

def _extract_end_call(text: str) -> tuple[str, bool]:
    """Strip the [END_CALL] tag the script appends once a booking is confirmed."""
    if END_CALL_TAG in text:
        return text.replace(END_CALL_TAG, "").strip(), True
    return text, False

@router.post("/scripted_chat")
async def scripted_chat(req: Request):
    body = await req.json()
    content = body.get("content", "")
    history = body.get("history", [])

    script = _load_script()
    system = (
        "You are a concise, friendly voice assistant. "
        "Keep replies under two short sentences — natural speech, no lists, no markdown.\n"
        + (f"\n---\n{script}" if script else "")
    )

    messages = [{"role": "system", "content": system}]
    for h in history[-20:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append(h)
    messages.append({"role": "user", "content": content})

    raw = call_llm(messages)
    reply, end_call = _extract_end_call(_sanitize(raw))
    logger.info("reply: %s (end_call=%s)", reply[:200], end_call)
    return JSONResponse({"reply": reply, "end_call": end_call})

@router.get("/ping")
async def ping():
    return JSONResponse({"status": "ok"})
