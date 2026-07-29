"""Speech-to-text backends.

`DistilWhisperSTT` runs distil-whisper through faster-whisper (CTranslate2,
int8 on CPU — fast on the Pi 5). `StdinSTT` reads typed lines for `--dev` mode.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..config import AudioConfig, SttConfig
from .console import safe_print

log = logging.getLogger(__name__)


class STT(ABC):
    @abstractmethod
    def listen(self) -> str:
        """Capture one utterance and return its transcript (may be empty)."""

    def close(self) -> None:  # pragma: no cover
        pass


class StdinSTT(STT):
    """Typed input stand-in for development."""

    def listen(self) -> str:
        try:
            return input("you> ").strip()
        except EOFError:
            return "quit"


class DistilWhisperSTT(STT):
    def __init__(self, cfg: SttConfig, audio_cfg: AudioConfig, capture=None):
        from faster_whisper import WhisperModel  # lazy: heavy dependency

        log.info("Loading distil-whisper model %s (%s/%s)...",
                 cfg.model, cfg.device, cfg.compute_type)
        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        # `capture` is any object with record_utterance() -> 16 kHz float32. Default
        # is the local sounddevice mic; the esp32 backend injects a serial capture.
        if capture is None:
            from .audio import AudioCapture
            capture = AudioCapture(audio_cfg)
        self.capture = capture
        # Optional hook fired the moment the mic CLOSES, before transcription. The
        # pipeline uses it to switch the status LED from "listening" to "thinking":
        # whisper takes a few seconds on the Pi, and without this the light would
        # still say "speak now" while the mic was already shut.
        self.on_capture_done = None

    def transcribe(self, samples) -> str:
        if samples is None or len(samples) == 0:
            return ""
        segments, _info = self.model.transcribe(
            samples, language=self.cfg.language, beam_size=self.cfg.beam_size,
            vad_filter=True,   # drop non-speech regions -> fewer silence hallucinations
            condition_on_previous_text=False,   # curb cross-utterance hallucination drift
            initial_prompt=self.cfg.prompt or None,   # bias toward chess vocabulary
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def listen(self) -> str:
        samples = self.capture.record_utterance()
        if self.on_capture_done is not None:
            try:
                self.on_capture_done()
            except Exception:  # noqa: BLE001 - a UI hook must not break listening
                log.debug("on_capture_done hook failed", exc_info=True)
        text = self.transcribe(samples)
        log.info("STT: %r", text)
        if text:
            safe_print(f"you> {text}")   # mirror the recognised speech in the terminal
        return text


def create_capture(stt_cfg: SttConfig, audio_cfg: AudioConfig, motion=None):
    """Build the mic for a whisper backend: any object with
    `record_utterance() -> 16 kHz float32`. Every backend shares the same Whisper
    model and only differs in where the audio comes from.
    """
    if stt_cfg.backend == "distil_whisper":
        from .audio import AudioCapture  # local mic via sounddevice (USB / built-in)
        return AudioCapture(audio_cfg)
    if stt_cfg.backend == "esp32_whisper":
        # MAX4466 on the ESP32, streamed over the motion controller's serial
        # link (shared port, turn-based: we never record while the crane moves).
        if motion is None:
            raise ValueError("esp32_whisper STT needs the serial motion controller "
                             "(set motion.backend: serial)")
        from .esp32_mic import MotionMicCapture
        return MotionMicCapture(motion)
    if stt_cfg.backend == "network_whisper":
        # A phone streaming over WiFi (e.g. IP Webcam), pulled per-window via PyAV.
        from .network_mic import NetworkMicCapture
        return NetworkMicCapture(stt_cfg, audio_cfg)
    raise ValueError(f"Unknown stt backend: {stt_cfg.backend!r}")


def create_stt(stt_cfg: SttConfig, audio_cfg: AudioConfig, motion=None) -> STT:
    if stt_cfg.backend == "stdin":
        return StdinSTT()
    return DistilWhisperSTT(stt_cfg, audio_cfg,
                            capture=create_capture(stt_cfg, audio_cfg, motion))
