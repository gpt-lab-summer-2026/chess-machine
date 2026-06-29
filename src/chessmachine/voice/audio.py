"""Microphone capture (with VAD) and speaker playback.

Captures a single spoken utterance: it waits for speech to start, records until
`silence_ms` of trailing silence, and returns 16 kHz mono float32 samples ready
for whisper. Heavy deps are imported lazily so the module loads without them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..config import AudioConfig

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class AudioCapture:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def record_utterance(self) -> np.ndarray:
        """Record one utterance; return a float32 numpy array at cfg.sample_rate."""
        if self.cfg.vad.enabled:
            return self._record_vad()
        return self._record_fixed(self.cfg.vad.max_utterance_s)

    # -- VAD-gated capture --------------------------------------------------- #
    def _record_vad(self) -> np.ndarray:
        import numpy as np
        import sounddevice as sd
        import webrtcvad

        sr = self.cfg.sample_rate
        frame_ms = self.cfg.vad.frame_ms
        frame_len = int(sr * frame_ms / 1000)          # samples per frame
        silence_frames = int(self.cfg.vad.silence_ms / frame_ms)
        max_frames = int(self.cfg.vad.max_utterance_s * 1000 / frame_ms)

        vad = webrtcvad.Vad(self.cfg.vad.aggressiveness)
        collected: list[bytes] = []
        triggered = False
        num_silent = 0

        with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                               blocksize=frame_len, device=self.cfg.input_device) as stream:
            log.debug("Listening (VAD)...")
            for _ in range(max_frames):
                buf, _overflowed = stream.read(frame_len)
                frame = bytes(buf)
                if len(frame) < frame_len * 2:
                    continue
                is_speech = vad.is_speech(frame, sr)
                if not triggered:
                    if is_speech:
                        triggered = True
                        collected.append(frame)
                else:
                    collected.append(frame)
                    num_silent = num_silent + 1 if not is_speech else 0
                    if num_silent >= silence_frames:
                        break

        if not collected:
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(b"".join(collected), dtype=np.int16)
        return (pcm.astype(np.float32) / 32768.0)

    # -- fixed-duration capture (VAD disabled) ------------------------------- #
    def _record_fixed(self, seconds: float) -> np.ndarray:
        import sounddevice as sd

        sr = self.cfg.sample_rate
        log.debug("Recording %.1fs...", seconds)
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                       dtype="float32", device=self.cfg.input_device)
        sd.wait()
        return audio.reshape(-1)


def play(samples: np.ndarray, sample_rate: int, device: int | str | None = None) -> None:
    """Play float32 samples on the speaker, blocking until done."""
    import sounddevice as sd

    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()
