"""Speech-to-text backends.

`DistilWhisperSTT` runs distil-whisper through faster-whisper (CTranslate2,
int8 on CPU). `StdinSTT` reads typed lines for `--dev` mode.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..config import AudioConfig, SttConfig
from .console import safe_print

log = logging.getLogger(__name__)

# Whisper's fixed input rate: the model is trained on 16 kHz mono, so every capture
# backend resamples to this before transcription (see voice/audio.resample).
WHISPER_SR = 16000


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
    # Optional hook fired the moment the mic CLOSES, before transcription. The
    # pipeline uses it to switch the status LED from "listening" to "thinking":
    # whisper takes a few seconds on the Pi, and without this the light would
    # still say "speak now" while the mic was already shut.
    # Declared on the CLASS so `listen()` works on instances built without
    # __init__ (tests use __new__ to skip the model load).
    on_capture_done = None

    def __init__(self, cfg: SttConfig, audio_cfg: AudioConfig, capture=None):
        from faster_whisper import WhisperModel  # lazy: heavy dependency

        log.info("Loading distil-whisper model %s (%s/%s, %d threads)...",
                 cfg.model, cfg.device, cfg.compute_type, cfg.cpu_threads)
        self.cfg = cfg
        # cfg.model is an ALIAS, so faster-whisper hits the HF Hub for a revision
        # check before loading -- seconds of httpx chatter on every boot. Try the
        # cache first and only reach for the network if the model isn't there yet
        # (first run), which keeps a fresh install working.
        kwargs = dict(device=cfg.device, compute_type=cfg.compute_type,
                      cpu_threads=cfg.cpu_threads)
        try:
            self.model = WhisperModel(cfg.model, local_files_only=True, **kwargs)
        except Exception:  # noqa: BLE001 - not cached yet; fall through to download
            log.info("%s not in the local cache -- downloading", cfg.model)
            self.model = WhisperModel(cfg.model, **kwargs)
        # The rate Whisper is fed at. Every capture backend downsamples to this
        # before returning: the local mic records at audio.capture_rate (48 kHz,
        # its native rate) and resamples in voice/audio.resample; esp32 and network
        # do the same from their own rates. Whisper is trained at 16 kHz, so a
        # different value here would silently change the perceived speaking rate.
        self.sample_rate = audio_cfg.sample_rate
        if self.sample_rate != WHISPER_SR:
            log.warning("audio.sample_rate is %d Hz, but Whisper expects %d Hz -- "
                        "transcription will be inaccurate.", self.sample_rate, WHISPER_SR)
        # `capture` is any object with record_utterance() -> 16 kHz float32. Default
        # is the local sounddevice mic; the esp32 backend injects a serial capture.
        if capture is None:
            from .audio import AudioCapture
            capture = AudioCapture(audio_cfg)
        self.capture = capture

    def _prep_audio(self, samples):
        """Level-normalize and pad a captured utterance before Whisper.

        This is the single biggest lever on REAL-vs-synthetic accuracy. faster-
        whisper does NO amplitude normalization, and Whisper hallucinates confident
        text on quiet input -- which is why a loud TTS round-trip transcribes fine
        while a real (quieter) mic mangles the same words. So: RMS-normalize toward
        a consistent loudness (gain capped so it can never clip), then frame the
        clip with a little silence so a short, tightly VAD-gated command gets clean
        onset/offset instead of being clipped mid-plosive. Runs at the ONE choke
        point every mic backend funnels through, so local/esp32/network all benefit.
        Measured on distil-small.en (2026-07-30): on degraded chess audio this,
        together with beam search and vad_filter off, turned "to do four" back into
        "d2 to d4" and "to-f3" into "knight to f3", at no extra latency.
        """
        import numpy as np

        x = np.asarray(samples, dtype="float32").reshape(-1)
        if x.size == 0:
            return x
        if self.cfg.normalize:
            rms = float(np.sqrt(np.mean(x * x)))
            peak = float(np.max(np.abs(x)))
            if rms > 1e-5 and peak > 1e-5:
                gain = min(self.cfg.target_rms / rms, 0.97 / peak)   # cap: never clip
                x = (x * gain).astype("float32")
        # Capture has already downsampled to Whisper's rate by the time we get here.
        pad = int(getattr(self, "sample_rate", WHISPER_SR) * self.cfg.pad_ms / 1000)
        if pad > 0:
            z = np.zeros(pad, dtype="float32")
            x = np.concatenate([z, x, z])
        return x

    def transcribe(self, samples) -> str:
        if samples is None or len(samples) == 0:
            return ""
        samples = self._prep_audio(samples)   # normalize level + pad (see _prep_audio)
        segments, _info = self.model.transcribe(
            samples, language=self.cfg.language, beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,   # capture already VAD-gates; off avoids double-VAD
            condition_on_previous_text=False,   # curb cross-utterance hallucination drift
            initial_prompt=self.cfg.prompt or None,   # bias toward chess vocabulary
            max_new_tokens=self.cfg.max_new_tokens,   # bounds worst-case decode time
            temperature=self.cfg.temperature,         # single greedy pass, no retry ladder
            repetition_penalty=self.cfg.repetition_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if self.cfg.max_chars and len(text) > self.cfg.max_chars:
            log.warning("STT transcript ran to %d chars (over the %d-char cap) -- "
                       "likely a hallucination; truncating: %r",
                       len(text), self.cfg.max_chars, text)
            text = text[:self.cfg.max_chars]
        return text

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
