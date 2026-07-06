"""Text-to-speech backends.

`KokoroTTS` synthesizes with Kokoro (ONNX runtime, CPU-friendly) and plays it.
`StdoutTTS` just prints, for `--dev` mode and CI.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..config import TtsConfig
from .console import safe_print

log = logging.getLogger(__name__)


class TTS(ABC):
    @abstractmethod
    def say(self, text: str) -> None:
        """Speak `text`, blocking until playback finishes."""

    def close(self) -> None:  # pragma: no cover
        pass


class StdoutTTS(TTS):
    """Prints instead of speaking (development / headless CI)."""

    def say(self, text: str) -> None:
        safe_print(f"[speaker] {text}")


class KokoroTTS(TTS):
    def __init__(self, cfg: TtsConfig, output_device=None):
        from kokoro_onnx import Kokoro  # lazy: heavy dependency

        log.info("Loading Kokoro TTS (%s)...", cfg.model_path)
        self.cfg = cfg
        self.output_device = output_device
        self.kokoro = Kokoro(cfg.model_path, cfg.voices_path)

    def synth(self, text: str):
        """Return (float32 samples, sample_rate)."""
        samples, sample_rate = self.kokoro.create(
            text, voice=self.cfg.voice, speed=self.cfg.speed, lang="en-us",
        )
        return samples, sample_rate

    def say(self, text: str) -> None:
        if not text:
            return
        # Mirror the reply as text as well as speaking it, so the terminal shows
        # the conversation (same prefix StdoutTTS uses in dev). Print first — the
        # text should appear before the synthesis/playback delay.
        safe_print(f"[speaker] {text}")
        from .audio import play

        samples, sample_rate = self.synth(text)
        play(samples, sample_rate, device=self.output_device)


def create_tts(cfg: TtsConfig, output_device=None) -> TTS:
    if cfg.backend == "stdout":
        return StdoutTTS()
    if cfg.backend == "kokoro":
        return KokoroTTS(cfg, output_device=output_device)
    raise ValueError(f"Unknown tts backend: {cfg.backend!r}")
