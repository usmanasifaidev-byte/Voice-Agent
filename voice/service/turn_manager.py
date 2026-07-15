import os
import subprocess
import threading
import shutil
import time
import io
import json
from typing import Dict, Optional, List
from voice.vad.silero_runner import get_speech_segments_from_audio
from voice.asr.whisper_runner import transcribe_wav_bytes
from voice.tts.piper_runner import synthesize_wav_api

class TurnSession:
    def __init__(self, sid: str, base_dir: str, ffmpeg_bin: str):
        self.sid = sid
        self.dir = os.path.join(base_dir, sid)
        os.makedirs(self.dir, exist_ok=True)
        self.segment_index = 0
        self.webm_path = os.path.join(self.dir, f"segment_{self.segment_index}.webm")
        self.wav_path = os.path.join(self.dir, f"segment_{self.segment_index}.wav")
        self.finalized: bool = False
        self.transcript: Optional[str] = None
        self.ffmpeg_bin = ffmpeg_bin
        self.chunk_count = 0
        self.last_duration = 0.0
        self.last_conversion_time = 0.0
        self.segment_start_time = time.time()  # Track when segment started
        self.conversion_lock = threading.Lock()  # Prevent concurrent conversions
        self.processing_active: bool = False  # Flag to prevent processing new chunks during processing/playback
        
        # Conversation history for scripted_chat
        self.conversation_history: List[Dict[str, str]] = []
        self.turn_number: int = 0
        
        # In-memory buffers for processing (files still saved for archival)
        self.webm_buffer = bytearray()  # Accumulate webm chunks in memory
        self.webm_header: Optional[bytes] = None  # Store the first chunk (contains EBML header) for reuse
        self.wav_bytes: Optional[bytes] = None  # Converted wav in memory
        self.wav_audio: Optional[any] = None  # Audio array for VAD (numpy array)

    def append_chunk(self, data: bytes):
        # Store to both file (for archival) and memory buffer (for processing)
        # File storage for archival/debugging
        try:
            with open(self.webm_path, "ab") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("append_chunk: failed to write file: %s", str(e))
        
        # Store the first chunk as the webm header (contains EBML header needed for ffmpeg)
        # This is needed because after advance_segment(), new chunks won't have the header
        if self.webm_header is None and len(data) >= 4:
            # Check if this looks like an EBML header (0x1A45DFA3)
            if data[0] == 0x1A and data[1] == 0x45 and data[2] == 0xDF and data[3] == 0xA3:
                # Store first ~8KB as header (enough for EBML header and initial segment info)
                self.webm_header = bytes(data[:min(8192, len(data))])
                import logging
                logging.getLogger(__name__).info("append_chunk: stored webm header (%d bytes) for segment=%d", 
                                                 len(self.webm_header), self.segment_index)
            else:
                # Log if we don't see the expected header
                import logging
                logging.getLogger(__name__).warning("append_chunk: first chunk doesn't have EBML header (first 4 bytes: %02x %02x %02x %02x)", 
                                                    data[0] if len(data) > 0 else 0,
                                                    data[1] if len(data) > 1 else 0,
                                                    data[2] if len(data) > 2 else 0,
                                                    data[3] if len(data) > 3 else 0)
        
        # In-memory buffer for processing
        self.webm_buffer.extend(data)
        self.chunk_count += 1

    def convert_to_wav_memory(self) -> bool:
        """Convert webm to wav in memory. Returns True if conversion succeeded. Also saves to disk for archival."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Prevent concurrent conversions
        if not self.conversion_lock.acquire(blocking=False):
            logger.debug("convert_to_wav_memory: conversion already in progress, skipping")
            return False
        
        try:
            # Throttle: don't convert more than once per 0.3 seconds (reduced from 0.5 for faster response)
            now = time.time()
            if self.last_conversion_time > 0 and (now - self.last_conversion_time) < 0.3:
                logger.debug("convert_to_wav_memory: throttled (%.2fs since last conversion)", now - self.last_conversion_time)
                return False
            
            # Check if we have enough data in memory
            # WebM files need at least ~5KB for a valid header and initial data
            # After advance_segment(), we reset the buffer, so we need to accumulate enough chunks first
            if len(self.webm_buffer) < 500:  # Reduced from 1000 to 500 bytes for faster processing
                logger.debug("convert_to_wav_memory: webm buffer too small: %d bytes (need 500)", len(self.webm_buffer))
                return False
            
            # Prepare webm data for conversion
            # If we have a stored header and the buffer doesn't start with it, prepend the header
            webm_data = bytes(self.webm_buffer)
            if self.webm_header is not None:
                # Check if buffer already starts with the header
                if len(webm_data) < len(self.webm_header) or webm_data[:len(self.webm_header)] != self.webm_header:
                    # Prepend header to make a valid webm file
                    webm_data = self.webm_header + webm_data
                    logger.debug("convert_to_wav_memory: prepended header (total size: %d bytes)", len(webm_data))
            
            # Convert webm to wav in memory using ffmpeg with stdin/stdout
            # Use -err_detect ignore_err to handle incomplete webm files
            cmd = [
                self.ffmpeg_bin, "-y",
                "-err_detect", "ignore_err",
                "-f", "webm",  # Input format
                "-i", "pipe:0",  # Read from stdin
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                "-f", "wav",  # Output format
                "pipe:1"  # Write to stdout
            ]
            
            try:
                res = subprocess.run(
                    cmd,
                    input=webm_data,  # Send webm bytes (with header if needed) via stdin
                    capture_output=True,
                    # 15s, not 5s: on constrained/shared CPU hosting (e.g. Render starter), ffmpeg can
                    # get starved by concurrent whisper/VAD work from an overlapping call and blow past 5s
                    timeout=15
                )
                
                if res.returncode != 0:
                    stderr_full = res.stderr.decode('utf-8', errors='ignore') if res.stderr else "no stderr"
                    logger.warning("convert_to_wav_memory: ffmpeg failed rc=%d (0x%X) stderr_len=%d, input_size=%d", 
                                 res.returncode, res.returncode & 0xFFFFFFFF, len(stderr_full), len(webm_data))
                    if stderr_full:
                        logger.warning("convert_to_wav_memory: stderr (last 500): %s", stderr_full[-500:])
                    return False
                
                if not res.stdout or len(res.stdout) == 0:
                    logger.warning("convert_to_wav_memory: no output from ffmpeg")
                    return False
                
                # Store wav bytes in memory
                self.wav_bytes = res.stdout
                
                # Also save to disk for archival
                try:
                    with open(self.wav_path, "wb") as f:
                        f.write(self.wav_bytes)
                        f.flush()
                        os.fsync(f.fileno())
                except Exception as e:
                    logger.warning("convert_to_wav_memory: failed to save wav to disk: %s", str(e))
                    # Continue anyway - we have it in memory
                
                # Load audio array for VAD (in memory)
                try:
                    import soundfile as sf
                    import numpy as np
                    # Read from memory buffer
                    audio, sr = sf.read(io.BytesIO(self.wav_bytes))
                    # Convert to mono if stereo
                    if len(audio.shape) > 1 and audio.shape[1] > 1:
                        audio = np.mean(audio, axis=1)
                    # Ensure it's float32 and normalized to [-1, 1]
                    if audio.dtype != np.float32:
                        if audio.dtype == np.int16:
                            audio = audio.astype(np.float32) / 32768.0
                        elif audio.dtype == np.int32:
                            audio = audio.astype(np.float32) / 2147483648.0
                        else:
                            audio = audio.astype(np.float32)
                    self.wav_audio = audio
                    logger.info("convert_to_wav_memory: success - wav size=%d bytes, audio samples=%d, duration=%.2fs, sample_rate=%d", 
                               len(self.wav_bytes), len(audio), len(audio) / 16000.0, sr)
                except Exception as e:
                    logger.warning("convert_to_wav_memory: failed to load audio array: %s", str(e))
                    # Continue anyway - we can still use wav_bytes for whisper
                
                self.last_conversion_time = now
                return True
                
            except subprocess.TimeoutExpired:
                logger.warning("convert_to_wav_memory: ffmpeg timed out")
                return False
            except Exception as e:
                logger.warning("convert_to_wav_memory: exception: %s", str(e), exc_info=True)
                return False
        finally:
            self.conversion_lock.release()

    def advance_segment(self):
        import logging
        logger = logging.getLogger(__name__)
        # Wait for any ongoing conversion to finish
        with self.conversion_lock:
            old_index = self.segment_index
            self.segment_index += 1
            
            # Save copies of old segment files before advancing
            old_webm = os.path.join(self.dir, f"segment_{old_index}.webm")
            old_wav = os.path.join(self.dir, f"segment_{old_index}.wav")
            old_webm_copy = os.path.join(self.dir, f"segment_{old_index}_final.webm")
            old_wav_copy = os.path.join(self.dir, f"segment_{old_index}_final.wav")
            
            try:
                # Copy old files to _final versions to preserve them
                if os.path.exists(old_webm):
                    shutil.copy2(old_webm, old_webm_copy)
                    logger.debug("advance_segment: saved copy of webm: %s -> %s", old_webm, old_webm_copy)
                if os.path.exists(old_wav):
                    shutil.copy2(old_wav, old_wav_copy)
                    logger.debug("advance_segment: saved copy of wav: %s -> %s", old_wav, old_wav_copy)
                
                # Also save transcript if available
                if self.transcript:
                    transcript_file = os.path.join(self.dir, f"segment_{old_index}_transcript.txt")
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(self.transcript)
                    logger.debug("advance_segment: saved transcript: %s", transcript_file)
            except Exception as e:
                logger.warning("advance_segment: failed to save copies of old files: %s", str(e))
            
            # Set new paths for next segment
            self.webm_path = os.path.join(self.dir, f"segment_{self.segment_index}.webm")
            self.wav_path = os.path.join(self.dir, f"segment_{self.segment_index}.wav")
            self.finalized = False
            self.transcript = None
            self.chunk_count = 0
            self.last_duration = 0.0
            self.last_conversion_time = 0.0
            self.segment_start_time = time.time()  # Reset segment start time
            
            # Reset in-memory buffers (but keep the header for future segments)
            self.webm_buffer = bytearray()
            # Don't reset webm_header - we need it for future segments
            self.wav_bytes = None
            self.wav_audio = None
            
            logger.info("advance_segment: sid=%s segment %d -> %d (saved copies, new paths: webm=%s, wav=%s)", 
                       self.sid, old_index, self.segment_index, self.webm_path, self.wav_path)

class TurnManager:
    def __init__(self, base_dir: str, ffmpeg_bin: str, whisper_model: str, piper_voice: str = "", 
                 use_cuda: bool = False, whisper_bin: str = None):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)
        self.ffmpeg_bin = ffmpeg_bin
        self.whisper_model = whisper_model
        self.piper_voice = piper_voice
        self.use_cuda = use_cuda  # Enable GPU acceleration for Piper TTS and Whisper
        # Deprecated: whisper_bin kept for backward compatibility but not used
        if whisper_bin:
            import logging
            logging.getLogger(__name__).warning("whisper_bin parameter is deprecated (using faster-whisper)")
        self.whisper_bin = whisper_bin  # Keep for backward compat
        # Determine device and compute type for Whisper
        self.whisper_device = "cuda" if use_cuda else "cpu"
        self.whisper_compute_type = "float16" if use_cuda else "int8"
        self.sessions: Dict[str, TurnSession] = {}

    def start(self, sid: str) -> TurnSession:
        s = TurnSession(sid, self.base, self.ffmpeg_bin)
        self.sessions[sid] = s
        return s

    def get(self, sid: str) -> Optional[TurnSession]:
        return self.sessions.get(sid)

    def remove(self, sid: str) -> None:
        """Drop a session once its call ends — sessions are never otherwise cleaned up,
        so leaving this out is an unbounded in-memory leak over the life of the process."""
        self.sessions.pop(sid, None)


    def clear_processing_flag(self, sid: str) -> bool:
        """Clear the processing_active flag for a session (called after playback completes)"""
        s = self.get(sid)
        if s:
            s.processing_active = False
            import logging
            logging.getLogger(__name__).info("clear_processing_flag: sid=%s - processing_active cleared, ready for new input", sid)
            return True
        return False

    def push_chunk(self, sid: str, data: bytes, vad_threshold: float = 0.35, min_speech_ms: int = 150, min_silence_ms: int = 1000, respond: bool = False) -> Dict:
        """
        Buffer-based processing: Only process when silence is detected.
        Accumulates chunks and only converts/processes when silence threshold is met.
        """
        import logging
        logger = logging.getLogger(__name__)
        s = self.get(sid)
        if not s:
            return {"ok": False, "error": "invalid session"}
        
        # Skip accumulating/processing if we're currently processing/playing back (prevent new audio from being processed)
        if s.processing_active:
            logger.debug("push_chunk: sid=%s segment=%d - discarding chunk (processing/playback active, chunk_count=%d)", 
                        sid, s.segment_index, s.chunk_count)
            # Don't accumulate chunks during processing/playback - discard them
            # This prevents audio spoken during playback from being processed
            return {"ok": True, "finalized": False, "state": "speaking"}
        
        # Just accumulate chunks - don't process yet
        s.append_chunk(data)
        
        # Need at least 2 chunks before checking (to avoid checking on chunk 0 after advancing)
        if s.chunk_count < 2:
            logger.debug("push_chunk: sid=%s segment=%d chunk=%d - accumulating (need at least 2)", 
                        sid, s.segment_index, s.chunk_count)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        # Only check for silence periodically (every 0.5s or every 4 chunks, but not on chunk 0)
        now = time.time()
        # Use segment_start_time for new segments, last_conversion_time for ongoing segments
        if s.last_conversion_time > 0:
            time_since_last = now - s.last_conversion_time
        else:
            # New segment - use segment start time
            time_since_last = now - s.segment_start_time
        
        should_check = (
            (s.chunk_count % 4 == 0 and s.chunk_count > 0) or  # Every 4 chunks (but not 0)
            time_since_last >= 0.5  # Or every 0.5s
        )
        
        if not should_check:
            logger.debug("push_chunk: sid=%s segment=%d chunk=%d - skipping check (time_since_last=%.2f)", 
                        sid, s.segment_index, s.chunk_count, time_since_last)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        logger.info("push_chunk: sid=%s segment=%d chunk=%d - checking silence (time_since_last=%.2f, buffer_size=%d)", 
                    sid, s.segment_index, s.chunk_count, time_since_last, len(s.webm_buffer))
        
        # Convert webm to wav in memory
        converted = s.convert_to_wav_memory()
        if not converted:
            logger.debug("push_chunk: sid=%s segment=%d chunk=%d - conversion skipped/failed (will retry)", 
                       sid, s.segment_index, s.chunk_count)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        # Ensure we have audio array for VAD
        if s.wav_audio is None:
            logger.warning("push_chunk: sid=%s segment=%d chunk=%d - audio array not ready after conversion", 
                       sid, s.segment_index, s.chunk_count)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        # Get audio duration from in-memory audio array
        duration = len(s.wav_audio) / 16000.0  # 16kHz sampling rate
        
        # Only run VAD if we have enough new audio (0.5s minimum)
        if (duration - s.last_duration) < 0.5:
            logger.debug("push_chunk: sid=%s segment=%d duration=%.2f last=%.2f - not enough new audio", 
                        sid, s.segment_index, duration, s.last_duration)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        s.last_duration = duration
        
        # Run VAD on in-memory audio array
        logger.info("push_chunk: sid=%s segment=%d running VAD duration=%.2fs, audio_samples=%d, threshold=%.2f", 
                   sid, s.segment_index, duration, len(s.wav_audio), vad_threshold)
        try:
            segs = get_speech_segments_from_audio(s.wav_audio, sampling_rate=16000, threshold=vad_threshold, min_speech_ms=min_speech_ms, min_silence_ms=min_silence_ms)
            logger.info("push_chunk: sid=%s segment=%d VAD returned %d segments", sid, s.segment_index, len(segs) if segs else 0)
        except Exception as e:
            logger.error("push_chunk: sid=%s segment=%d VAD call failed: %s", sid, s.segment_index, str(e), exc_info=True)
            segs = []
        
        if not segs:
            # No speech detected yet - keep listening
            logger.info("push_chunk: sid=%s segment=%d - no speech segments detected (duration=%.2fs, threshold=%.2f, min_speech_ms=%d)", 
                       sid, s.segment_index, duration, vad_threshold, min_speech_ms)
            return {"ok": True, "finalized": False, "state": "listening"}
        
        # Calculate silence after last speech segment
        last_end = max([seg['end'] for seg in segs])
        silence = max(0.0, duration - last_end)
        
        logger.debug("push_chunk: sid=%s segment=%d segs=%d last_end=%.2fs duration=%.2fs silence=%.2fs (need %.2fs)", 
                    sid, s.segment_index, len(segs), last_end, duration, silence, min_silence_ms / 1000.0)
        
        # Only process when silence threshold is met (buffer-based approach)
        if silence * 1000 >= min_silence_ms:
            # Silence detected - process the buffered audio
            # Set processing flag IMMEDIATELY to prevent new chunks from being processed
            # This must be set before transcription to prevent race conditions
            s.processing_active = True
            logger.info("push_chunk: sid=%s SILENCE DETECTED - processing segment=%d (processing_active=True, will prevent new chunks)", sid, s.segment_index)
            
            # Transcribe using in-memory wav bytes (piped to whisper via stdin)
            if s.wav_bytes is None or len(s.wav_bytes) == 0:
                logger.warning("push_chunk: sid=%s segment=%d - wav_bytes is None or empty, cannot transcribe", sid, s.segment_index)
                txt = ""
            else:
                logger.info("push_chunk: sid=%s segment=%d - transcribing wav_bytes=%d bytes", sid, s.segment_index, len(s.wav_bytes))
                txt = transcribe_wav_bytes(
                    self.whisper_model, 
                    s.wav_bytes,
                    device=self.whisper_device,
                    compute_type=self.whisper_compute_type,
                    vad_filter=True,
                    whisper_bin=self.whisper_bin  # For backward compat
                ) or ""
                logger.info("push_chunk: sid=%s segment=%d - transcription result: '%s' (len=%d)", sid, s.segment_index, txt, len(txt))
            s.transcript = txt
            s.finalized = True
            # "thinking" once a reply will actually be generated; otherwise back to "listening" right away
            # (this used to always say "speaking", which showed while the LLM/TTS hadn't even started)
            will_respond = bool(respond and txt)
            res: Dict = {"ok": True, "finalized": True, "transcript": txt, "state": "thinking" if will_respond else "listening"}
            logger.info("push_chunk: sid=%s segment=%d - returning finalized result: transcript='%s'", sid, s.segment_index, txt)

            # Note: scripted_chat API call is now handled asynchronously in webrtc.py
            # This prevents blocking the event loop. We just set processing_active and return.
            # The actual API call and audio generation happens in _handle_scripted_chat_response()
            if will_respond:
                # Keep processing_active=True - it will be cleared after audio playback completes
                # or in _handle_scripted_chat_response if no audio is generated
                logger.info("push_chunk: sid=%s - transcript finalized, response will be handled asynchronously (processing_active=True)", sid)
            else:
                # No response requested or no transcript - clear processing flag immediately
                s.processing_active = False
                logger.info("push_chunk: sid=%s - no response requested, clearing processing_active", sid)
            
            # IMPORTANT: Advance segment AFTER setting processing_active
            # This ensures new chunks go to the new segment, but processing_active prevents them from being processed
            s.advance_segment()
            if s.processing_active:
                logger.info("push_chunk: sid=%s advanced to segment=%d, processing_active=True (new chunks will be discarded until cleared)", sid, s.segment_index)
            else:
                logger.info("push_chunk: sid=%s advanced to segment=%d, processing_active=False (ready for new input)", sid, s.segment_index)
            return res
        
        # Still in speech or short silence - keep buffering
        state = "recording" if (duration - last_end) < (min_silence_ms / 1000.0) else "listening"
        return {"ok": True, "finalized": False, "state": state}