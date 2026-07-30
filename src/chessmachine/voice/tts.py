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

        log.info("Loading Kokoro TTS (%s, %d threads)...", cfg.model_path, cfg.intra_op_threads)
        self.cfg = cfg
        self.output_device = output_device
        # Kokoro's own constructor builds an InferenceSession with NO SessionOptions,
        # so onnxruntime takes every core (4 here) AND spins while idle. On a 4-core
        # Pi that pool fights whisper's and llama-server's. from_session() is the only
        # hook kokoro_onnx offers for passing options, so build the session ourselves.
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.intra_op_threads
        opts.inter_op_num_threads = 1
        # Don't busy-wait between inferences; TTS is bursty and idle most of the time.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(cfg.model_path, opts,
                                       providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(session, cfg.voices_path)

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
        play(samples, sample_rate, device=self.output_device,
             lead_in_s=self.cfg.lead_in_s, lead_in_primer=self.cfg.lead_in_primer)


def create_tts(cfg: TtsConfig, output_device=None) -> TTS:
    if cfg.backend == "stdout":
        return StdoutTTS()
    if cfg.backend == "kokoro":
        return KokoroTTS(cfg, output_device=output_device)
    raise ValueError(f"Unknown tts backend: {cfg.backend!r}")
