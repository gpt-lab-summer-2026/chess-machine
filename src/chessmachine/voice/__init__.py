"""Speech I/O: microphone capture, distil-whisper STT, Kokoro TTS.

All heavy/optional dependencies (faster-whisper, kokoro-onnx, sounddevice,
webrtcvad, numpy) are imported lazily inside methods, so this package imports
cleanly on a machine that only has the core deps (e.g. for `--dev` mode).
"""
from .stt import STT, create_stt
from .tts import TTS, create_tts

__all__ = ["STT", "create_stt", "TTS", "create_tts"]
