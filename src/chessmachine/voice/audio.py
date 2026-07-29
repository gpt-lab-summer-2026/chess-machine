"""Microphone capture (with VAD) and speaker playback.

Captures a single spoken utterance: it waits for speech to start, records until
`silence_ms` of trailing silence, and returns 16 kHz mono float32 samples ready
for whisper. Heavy deps are imported lazily so the module loads without them.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..config import AudioConfig, VadConfig

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# A Bluetooth sink wakes from idle with ~0.3-0.5 s of latency, which clips the
# start of a fresh utterance ("Chess machine ready" -> "ess machine ready"). We
# prepend this much silence to each pw-play so only silence is lost, never speech.
_BT_LEAD_IN_S = 0.5


def collect_utterance(frames: Iterable[Any], vad: Any, vad_cfg: VadConfig,
                      sr: int = 16000) -> np.ndarray:
    """VAD-gate a stream of fixed-size int16 mono frames into one utterance.

    `frames` yields `frame_len`-sample int16 numpy arrays (frame_len = sr *
    frame_ms / 1000); `vad.is_speech(bytes, sr)` classifies each. Waits for
    speech to start, then ends after `silence_ms` of trailing silence. Returns
    float32 in [-1, 1] — empty if too little actual speech was heard.

    This is the ONE end-of-utterance state machine, shared by every mic backend
    (local sounddevice, network stream): pure and I/O-free, so it is unit-tested
    with a fake VAD and each backend only has to supply frames. Any per-window
    cap (max_utterance_s) belongs to the frame producer.
    """
    import numpy as np

    frame_ms = vad_cfg.frame_ms
    silence_frames = max(1, int(vad_cfg.silence_ms / frame_ms))
    collected: list[np.ndarray] = []
    triggered = False
    num_silent = 0
    speech_frames = 0

    for chunk in frames:
        is_speech = vad.is_speech(chunk.tobytes(), sr)
        if not triggered:
            if is_speech:
                triggered = True
                collected.append(chunk)
                speech_frames += 1
        else:
            collected.append(chunk)
            if is_speech:
                speech_frames += 1
                num_silent = 0
            else:
                num_silent += 1
            if num_silent >= silence_frames:
                break

    # Reject blips: a click can trip the VAD for a frame or two. Require a
    # minimum amount of actual speech before we bother transcribing.
    if not collected or speech_frames * frame_ms < vad_cfg.min_speech_ms:
        return np.zeros(0, dtype="float32")
    return np.concatenate(collected).astype("float32") / 32768.0


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
        """Open the mic and feed its frames to the shared `collect_utterance`
        state machine (see there for the end-of-utterance rules)."""
        import sounddevice as sd
        import webrtcvad

        sr = self.cfg.sample_rate
        frame_len = int(sr * self.cfg.vad.frame_ms / 1000)   # samples per frame
        vad = webrtcvad.Vad(self.cfg.vad.aggressiveness)

        with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                               blocksize=frame_len, device=self.cfg.input_device) as stream:
            log.debug("Listening (VAD)...")
            frames = self._iter_frames(stream, frame_len)
            return collect_utterance(frames, vad, self.cfg.vad, sr)

    def _iter_frames(self, stream: Any, frame_len: int):
        """Yield int16 mono frames from an open input stream, capped at
        `max_utterance_s` so a caller who never stops talking still returns."""
        import numpy as np

        max_frames = int(self.cfg.vad.max_utterance_s * 1000 / self.cfg.vad.frame_ms)
        for _ in range(max_frames):
            buf, _overflowed = stream.read(frame_len)
            frame = bytes(buf)
            if len(frame) < frame_len * 2:       # short read: skip the partial frame
                continue
            yield np.frombuffer(frame, dtype="<i2")

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
    """Play float32 samples on the speaker, blocking until done.

    On the Pi, output is routed through PipeWire via `pw-play`: PortAudio/ALSA
    only sees the raw HDMI device and can't reach the Bluetooth sink, whereas
    PipeWire owns the BT speaker and transparently resamples/reformats to it.
    Falls back to sounddevice where `pw-play` isn't installed (e.g. dev boxes).
    """
    pw_play = shutil.which("pw-play")
    if pw_play is not None:
        _play_via_pipewire(pw_play, samples, sample_rate)
        return
    import sounddevice as sd

    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()


def _play_via_pipewire(pw_play: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples to a temp WAV and play it through PipeWire.

    A WAV file (not a raw stdin pipe) is used because pw-play is libsndfile-
    backed and needs the header to know the rate/format; PipeWire then routes
    to the current default sink (the Bluetooth speaker) and resamples as needed.
    """
    import os
    import subprocess
    import tempfile
    import wave

    import numpy as np

    arr = np.asarray(samples)
    if arr.dtype != np.int16:                     # Kokoro emits float32 in [-1, 1]
        arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    channels = 1 if arr.ndim == 1 else arr.shape[1]

    lead = int(sample_rate * _BT_LEAD_IN_S)       # silence so the BT wake-up clips nothing
    if lead > 0:
        pad = np.zeros((lead,) if arr.ndim == 1 else (lead, channels), dtype="<i2")
        arr = np.concatenate([pad, arr])

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(arr.tobytes())
        subprocess.run([pw_play, path], check=True)
    finally:
        os.unlink(path)
