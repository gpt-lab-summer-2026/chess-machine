"""Speech-to-text backends.

`DistilWhisperSTT` runs distil-whisper through faster-whisper (CTranslate2,
int8 on CPU — fast on the Pi 5). `StdinSTT` reads typed lines for `--dev` mode.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..config import AudioConfig, SttConfig

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
    def __init__(self, cfg: SttConfig, audio_cfg: AudioConfig):
        from faster_whisper import WhisperModel  # lazy: heavy dependency

        from .audio import AudioCapture

        log.info("Loading distil-whisper model %s (%s/%s)...",
                 cfg.model, cfg.device, cfg.compute_type)
        self.cfg = cfg
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self.capture = AudioCapture(audio_cfg)

    def transcribe(self, samples) -> str:
        if samples is None or len(samples) == 0:
            return ""
        segments, _info = self.model.transcribe(
            samples, language=self.cfg.language, beam_size=self.cfg.beam_size,
            vad_filter=True,   # drop non-speech regions -> fewer silence hallucinations
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def listen(self) -> str:
        samples = self.capture.record_utterance()
        text = self.transcribe(samples)
        log.info("STT: %r", text)
        if text:
            print(f"you> {text}")   # mirror the recognised speech in the terminal
        return text


def create_stt(stt_cfg: SttConfig, audio_cfg: AudioConfig) -> STT:
    if stt_cfg.backend == "stdin":
        return StdinSTT()
    if stt_cfg.backend == "distil_whisper":
        return DistilWhisperSTT(stt_cfg, audio_cfg)
    raise ValueError(f"Unknown stt backend: {stt_cfg.backend!r}")
