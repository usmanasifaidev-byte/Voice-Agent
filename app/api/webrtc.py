"""
WebRTC signaling and data channel handling for real-time voice communication.
Uses WebSocket for signaling and WebRTC DataChannel for audio streaming.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import logging
import asyncio
from voice.service.shared_session import SESS, TURN
import os
from uvicorn.protocols.utils import ClientDisconnected
from app.utils.text_processing import clean_text_for_tts

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebRTC connections
active_connections: dict[str, WebSocket] = {}

async def _handle_scripted_chat_response(websocket: WebSocket, session_id: str, transcript: str):
    """Handle scripted_chat API call asynchronously and send response"""
    try:
        import httpx
        from voice.service.shared_session import TURN
        
        # Get session to access conversation history
        s = TURN.get(session_id)
        if not s:
            logger.error("_handle_scripted_chat_response: session %s not found", session_id)
            return
        
        # Ensure processing_active is set (should already be set, but double-check)
        s.processing_active = True
        logger.info("_handle_scripted_chat_response: set processing_active=True for session_id=%s", session_id)
        
        # Add user message to conversation history
        s.conversation_history.append({"role": "user", "content": transcript})
        history = s.conversation_history[-20:]
        
        # Call scripted_chat API asynchronously
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        scripted_chat_url = f"{api_base_url}/api/scripted_chat"
        payload = {
            "content": transcript,
            "turn": s.turn_number,
            "history": history
        }
        
        logger.info("_handle_scripted_chat_response: calling scripted_chat for session_id=%s", session_id)
        
        # Use shared HTTP client for connection pooling
        from app.services.external_api_client import get_http_client
        client = get_http_client()
        response = await client.post(scripted_chat_url, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data.get("reply", "")
        end_call = bool(data.get("end_call", False))

        # Add assistant reply to conversation history
        if reply:
            s.conversation_history.append({"role": "assistant", "content": reply})
            s.turn_number += 1

        logger.info("_handle_scripted_chat_response: reply received (len=%d, end_call=%s) for session_id=%s", len(reply), end_call, session_id)

        # Send reply text (don't include transcript - it was already sent in the initial processing_result).
        # State stays "thinking" here — audio hasn't been generated yet, so it isn't "speaking" until
        # _send_audio_ready actually fires.
        if not await safe_send_text(websocket, {
            "type": "processing_result",
            "ok": True,
            "finalized": True,
            "reply": reply,
            "state": "thinking",
            "end_call": end_call
        }):
            return  # Connection closed

        # Generate audio if TTS is available
        if reply and TURN.piper_voice:
            from voice.tts.piper_runner import synthesize_wav_api
            # Clean markdown formatting from reply before TTS (remove asterisks, bold markers, etc.)
            cleaned_reply = clean_text_for_tts(reply)
            logger.debug("_handle_scripted_chat_response: cleaned reply for TTS (original_len=%d, cleaned_len=%d)",
                        len(reply), len(cleaned_reply))

            # Get current segment index (before advance_segment was called)
            segment_index = s.segment_index - 1 if s.segment_index > 0 else 0
            out_wav = os.path.join(s.dir, f"reply_segment_{segment_index}.wav")
            # Run TTS in executor (it's CPU/GPU intensive)
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(
                None,
                synthesize_wav_api,
                TURN.piper_voice,
                cleaned_reply,  # Use cleaned text for TTS
                out_wav,
                TURN.use_cuda
            )
            if ok and os.path.exists(out_wav):
                await _send_audio_ready(websocket, session_id, out_wav, end_call=end_call)
            else:
                # No audio generated - clear processing flag and tell the client what to do next
                # (previously left the frontend stuck showing "thinking" forever)
                TURN.clear_processing_flag(session_id)
                await safe_send_text(websocket, {
                    "type": "processing_result",
                    "ok": True,
                    "finalized": False,
                    "state": "listening",
                    "end_call": end_call
                })
        else:
            # No audio to play - clear processing flag and tell the client what to do next
            TURN.clear_processing_flag(session_id)
            await safe_send_text(websocket, {
                "type": "processing_result",
                "ok": True,
                "finalized": False,
                "state": "listening",
                "end_call": end_call
            })

    except Exception as e:
        logger.error("_handle_scripted_chat_response error: %s", str(e), exc_info=True)
        # Clear processing flag on error
        TURN.clear_processing_flag(session_id)
        if not await safe_send_text(websocket, {
            "type": "error",
            "error": f"Failed to generate response: {str(e)}"
        }):
            pass  # Connection closed

async def _send_audio_ready(websocket: WebSocket, session_id: str, audio_path: str, end_call: bool = False):
    """Send audio_ready message to client"""
    s = SESS.get(session_id)
    if not s:
        from voice.service.voice_session import VoiceSession
        s = VoiceSession(session_id, SESS.base)
        SESS.sessions[session_id] = s
    if s:
        s.reply_audio_path = audio_path
        logger.info("_send_audio_ready: set reply_audio_path=%s for session_id=%s", audio_path, session_id)

    import time
    audio_url = f"/api/voice/audio/{session_id}?t={int(time.time() * 1000)}"
    await safe_send_text(websocket, {
        "type": "audio_ready",
        "audio_path": audio_url,
        "audio_file": audio_path,
        "end_call": end_call
    })

async def safe_send_text(websocket: WebSocket, message: dict) -> bool:
    """
    Safely send a text message over WebSocket, handling connection errors gracefully.
    Returns True if sent successfully, False if connection is closed.
    """
    try:
        await websocket.send_text(json.dumps(message))
        return True
    except (WebSocketDisconnect, ClientDisconnected, RuntimeError) as e:
        # Connection is closed or closing, this is expected when client disconnects
        logger.debug("WebSocket send failed (connection closed): %s", str(e))
        return False
    except Exception as e:
        logger.error("Unexpected error sending WebSocket message: %s", str(e), exc_info=True)
        return False

@router.websocket("/voice/webrtc/{session_id}")
async def webrtc_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for WebRTC signaling and audio data streaming.
    Handles:
    1. WebRTC offer/answer exchange (signaling)
    2. Audio chunk streaming via DataChannel messages
    3. Real-time processing with buffer-based silence detection
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info("WebRTC WebSocket connected: session_id=%s", session_id)
    
    # Initialize session in both TURN and SESS (ensure they're in sync)
    TURN.start(session_id)
    # Ensure session exists in SESS as well
    if not SESS.get(session_id):
        from voice.service.voice_session import VoiceSession
        s = VoiceSession(session_id, SESS.base)
        SESS.sessions[session_id] = s
        logger.info("webrtc_websocket: created session in SESS for session_id=%s", session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                msg_type = message.get("type")
                
                if msg_type == "offer":
                    # WebRTC offer received - for now, we'll use DataChannel directly
                    # In a full WebRTC implementation, you'd handle SDP exchange here
                    if not await safe_send_text(websocket, {
                        "type": "answer",
                        "session_id": session_id,
                        "status": "ready"
                    }):
                        break  # Connection closed, exit loop
                    logger.info("WebRTC offer received for session_id=%s", session_id)
                
                elif msg_type == "audio_chunk":
                    # Audio data received via DataChannel (base64 encoded)
                    import base64
                    chunk_data = base64.b64decode(message.get("data", ""))
                    respond = message.get("respond", False)
                    
                    # Check if processing is active BEFORE processing the chunk
                    s = TURN.get(session_id)
                    if s and s.processing_active:
                        logger.debug("webrtc_websocket: discarding audio chunk (processing_active=True for session_id=%s)", session_id)
                        # Still send a response to keep the connection alive, but don't process
                        if not await safe_send_text(websocket, {
                            "type": "processing_result",
                            "ok": True,
                            "finalized": False,
                            "state": "speaking"
                        }):
                            break
                        continue
                    
                    # Process chunk with buffer-based silence detection (run in executor to avoid blocking)
                    # Use lower VAD threshold (0.3) and shorter min_speech_ms (100) for better sensitivity
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(
                        None, 
                        lambda: TURN.push_chunk(session_id, chunk_data, vad_threshold=0.5, min_speech_ms=300, min_silence_ms=1200, respond=False)
                    )
                    
                    # Send response back
                    response_msg = {
                        "type": "processing_result",
                        **res
                    }
                    logger.info("webrtc_websocket: sending processing_result: finalized=%s, transcript='%s'", 
                               res.get("finalized"), res.get("transcript", ""))
                    if not await safe_send_text(websocket, response_msg):
                        break  # Connection closed, exit loop
                    
                    # If finalized and response requested, handle scripted_chat call asynchronously
                    if res.get("finalized") and respond and res.get("transcript"):
                        # Run scripted_chat call asynchronously
                        asyncio.create_task(_handle_scripted_chat_response(websocket, session_id, res.get("transcript", "")))
                    elif res.get("finalized") and res.get("audio_path") and os.path.exists(res["audio_path"]):
                        # Audio already generated (from previous call), send it
                        await _send_audio_ready(websocket, session_id, res["audio_path"])
                
                elif msg_type == "ping":
                    # Keep-alive
                    if not await safe_send_text(websocket, {"type": "pong"}):
                        break  # Connection closed, exit loop
                
                elif msg_type == "playback_complete":
                    # Client notifies that playback has finished - clear processing flag
                    TURN.clear_processing_flag(session_id)
                    logger.info("webrtc_websocket: playback_complete received for session_id=%s", session_id)
                
                else:
                    logger.warning("Unknown message type: %s", msg_type)
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received: %s", data[:100])
            except Exception as e:
                logger.error("Error processing message: %s", str(e), exc_info=True)
                # Try to send error, but don't break if connection is closed
                if not await safe_send_text(websocket, {
                    "type": "error",
                    "error": str(e)
                }):
                    break  # Connection closed, exit loop
                
    except WebSocketDisconnect:
        logger.info("WebRTC WebSocket disconnected: session_id=%s", session_id)
    except Exception as e:
        logger.error("WebRTC WebSocket error: %s", str(e), exc_info=True)
    finally:
        active_connections.pop(session_id, None)
        # Sessions were never otherwise removed from TURN/SESS (in-memory dicts) — an unbounded
        # leak over the process lifetime. Any in-flight background task (push_chunk's executor
        # thread, _handle_scripted_chat_response) already holds its own reference to the session
        # object and checks TURN.get()/safe_send_text for None/closed before continuing, so
        # dropping it from the dict here is safe even if work is still finishing up.
        TURN.remove(session_id)
        SESS.remove(session_id)
        logger.info("WebRTC connection closed: session_id=%s", session_id)

@router.post("/voice/webrtc/start")
async def webrtc_start():
    """Start a new WebRTC session"""
    sid = SESS.start()
    logger.info("webrtc_start sid=%s", sid)
    TURN.start(sid)
    return JSONResponse({"ok": True, "session_id": sid})

