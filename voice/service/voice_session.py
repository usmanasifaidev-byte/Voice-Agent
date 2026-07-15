import os
import uuid
import wave
from typing import Dict, Optional

class VoiceSession:
    def __init__(self, sid: str, base_dir: str):
        self.sid = sid
        self.dir = os.path.join(base_dir, sid)
        os.makedirs(self.dir, exist_ok=True)
        self.input_path: Optional[str] = None
        self.transcript: Optional[str] = None
        self.reply_text: Optional[str] = None
        self.reply_audio_path: Optional[str] = None

class VoiceSessionManager:
    def __init__(self, base_dir: str):
        self.base = base_dir
        os.makedirs(self.base, exist_ok=True)
        self.sessions: Dict[str, VoiceSession] = {}

    def start(self) -> str:
        sid = str(uuid.uuid4())
        self.sessions[sid] = VoiceSession(sid, self.base)
        return sid

    def get(self, sid: str) -> Optional[VoiceSession]:
        return self.sessions.get(sid)

    def remove(self, sid: str) -> None:
        self.sessions.pop(sid, None)

    def set_input(self, sid: str, path: str):
        s = self.get(sid)
        if s:
            s.input_path = path

    def set_transcript(self, sid: str, text: str):
        s = self.get(sid)
        if s:
            s.transcript = text

    def set_reply_text(self, sid: str, text: str):
        s = self.get(sid)
        if s:
            s.reply_text = text

    def set_reply_audio(self, sid: str, path: str):
        s = self.get(sid)
        if s:
            s.reply_audio_path = path

def synth_silence_wav(out_path: str, seconds: float = 1.0, rate: int = 16000):
    frames = int(seconds * rate)
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * frames)