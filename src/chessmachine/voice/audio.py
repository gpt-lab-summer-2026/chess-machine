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

# Default silence prepended to each pw-play, so the speaker's amplifier is awake
# before the first syllable. Two separate causes clip that syllable:
#   1. PipeWire SUSPENDING the idle BT node -> A2DP renegotiation drops the start.
#      Fixed out-of-band by session.suspend-timeout-seconds = 0 for bluez nodes
#      (~/.config/wireplumber/wireplumber.conf.d/50-bt-no-suspend.conf).
#   2. The SPEAKER'S OWN amp powering down between utterances and swallowing the
#      first ~350 ms when it wakes. Suspend-disable does nothing for this -- only a
#      long-enough lead-in does. Kokoro emits just ~40 ms of its own leading
#      silence, so this pad carries the rest.
# Overridable per call (and via tts.lead_in_s); tune by ear with
# scripts/tts_leadin_test.py.
_BT_LEAD_IN_S = 0.5


def _lead_in(n: int, channels: int, primer: bool):
    """Build `n` frames of lead-in for the speaker's amp to wake into.

    Silence by default. With `primer=True`, inaudible low-level noise (~-62 dBFS)
    instead, for amps that only wake on actual SIGNAL rather than on a silent but
    active link -- pure silence never rouses those, so the first syllable still
    clips no matter how long the pad. Pure and I/O-free so it can be unit-tested.
    """
    import numpy as np

    shape = (n,) if channels == 1 else (n, channels)
    if n <= 0 or not primer:
        return np.zeros(shape, dtype="<i2")
    rng = np.random.default_rng(0)
    return rng.integers(-24, 25, size=shape).astype("<i2")


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
    import collections

    import numpy as np

    frame_ms = vad_cfg.frame_ms
    silence_frames = max(1, int(vad_cfg.silence_ms / frame_ms))
    collected: list[np.ndarray] = []
    triggered = False
    num_silent = 0
    speech_frames = 0
    saw_signal = False
    # Pre-roll: webrtcvad needs a frame or two to latch onto speech onset, so the
    # frames it spends deciding used to be thrown away -- taking the leading
    # plosive with them ("pawn to e4" transcribed as "on to e4"). Keep a short
    # ring buffer of pre-trigger frames and prepend it once speech starts.
    pre_roll_frames = max(0, int(vad_cfg.pre_roll_ms / frame_ms))
    pending: collections.deque | None = (
        collections.deque(maxlen=pre_roll_frames) if pre_roll_frames else None)

    for chunk in frames:
        saw_signal = saw_signal or bool(chunk.any())
        is_speech = vad.is_speech(chunk.tobytes(), sr)
        if not triggered:
            if is_speech:
                triggered = True
                if pending:
                    collected.extend(pending)   # recover the speech onset
                    pending.clear()
                collected.append(chunk)
                speech_frames += 1
            elif pending is not None:
                pending.append(chunk)
        else:
            collected.append(chunk)
            if is_speech:
                speech_frames += 1
                num_silent = 0
            else:
                num_silent += 1
            if num_silent >= silence_frames:
                break

    # A capture path that yields DIGITAL SILENCE (every sample exactly zero) is a
    # dead device, not a quiet room -- e.g. input_device "default" resolving to a
    # PipeWire graph with no source because the USB mic dropped off the bus. It
    # must be called out: whisper happily hallucinates text from near-silence
    # ("oot Saber-s-h-h-huh"), the SLM then turns that into a confident chess
    # move, and the failure looks like bad accuracy instead of a missing mic.
    if not saw_signal:
        log.error("Microphone produced ZERO audio for the whole window -- the capture "
                  "device is not delivering samples. Check that the mic is connected "
                  "(`arecord -l`, `wpctl status`); refusing to transcribe silence.")
        return np.zeros(0, dtype="float32")

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


def make_beep(sample_rate: int = 24000, freq: float = 880.0, ms: int = 160,
              volume: float = 0.22):
    """A short sine 'earcon' with 8 ms fades, as float32 in [-1, 1].

    Played the instant the mic opens so the user gets an unmistakable "speak now"
    cue (like a voice assistant's chime). Kept brief and mid-volume so it's clear
    but not startling. Pure/I-O-free for unit testing.
    """
    import numpy as np

    n = int(sample_rate * ms / 1000)
    t = np.arange(n, dtype="float32") / sample_rate
    tone = np.sin(2 * np.pi * freq * t).astype("float32") * volume
    fade = max(1, int(sample_rate * 0.008))            # 8 ms in/out, no click
    ramp = np.linspace(0.0, 1.0, fade, dtype="float32")
    tone[:fade] *= ramp
    tone[-fade:] *= ramp[::-1]
    return tone


def play(samples: np.ndarray, sample_rate: int, device: int | str | None = None,
         *, lead_in_s: float = _BT_LEAD_IN_S, lead_in_primer: bool = False) -> None:
    """Play float32 samples on the speaker, blocking until done.

    On the Pi, output is routed through PipeWire via `pw-play`: PortAudio/ALSA
    only sees the raw HDMI device and can't reach the Bluetooth sink, whereas
    PipeWire owns the BT speaker and transparently resamples/reformats to it.
    Falls back to sounddevice where `pw-play` isn't installed (e.g. dev boxes).

    `lead_in_s` / `lead_in_primer` prepend an amp-wake lead-in (see `_lead_in`).
    """
    pw_play = shutil.which("pw-play")
    if pw_play is not None:
        _play_via_pipewire(pw_play, samples, sample_rate, lead_in_s, lead_in_primer)
        return
    import sounddevice as sd

    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()


def _play_via_pipewire(pw_play: str, samples: np.ndarray, sample_rate: int,
                       lead_in_s: float = _BT_LEAD_IN_S,
                       lead_in_primer: bool = False) -> None:
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

    lead = int(sample_rate * lead_in_s)           # amp-wake lead-in so nothing clips
    if lead > 0:
        arr = np.concatenate([_lead_in(lead, channels, lead_in_primer), arr])

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
