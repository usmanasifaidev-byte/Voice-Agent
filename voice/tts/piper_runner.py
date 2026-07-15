import os
import logging
import wave

logger = logging.getLogger(__name__)

def synthesize_wav_api(voice_model: str, text: str, out_wav: str, use_cuda: bool = False) -> bool:
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(voice_model, use_cuda=use_cuda)
        with wave.open(out_wav, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 0
    except Exception as e:
        logger.error("piper TTS failed: %s", e, exc_info=True)
        return False