"""Microphone capture (with VAD) and speaker playback.

Captures a single spoken utterance: it waits for speech to start, records until
`silence_ms` of trailing silence, and returns 16 kHz mono float32 samples ready
for whisper. Heavy deps are imported lazily so the module loads without them.
"""
from __future__ import annotations

import contextlib
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

    End-of-utterance is a TOLERANT window, not a consecutive run. webrtcvad
    mislabels a real percentage of room noise as speech (MEASURED on this mic:
    38/200 frames of an empty room at aggressiveness 2, 9/200 at 3), and the old
    rule -- `silence_ms` of *consecutive* silence, counter reset to zero by any
    single speech frame -- could then never fire: one blip every couple of seconds
    is enough to hold it open until the max_utterance_s cap, so Whisper got the
    whole 7 s window every time instead of the ~2 s that was spoken. Now the
    utterance ends when the trailing `silence_ms` window is *mostly* quiet
    (`end_tolerance`), which a scattered false positive can't prevent.

    This is the ONE end-of-utterance state machine, shared by every mic backend
    (local sounddevice, network stream): pure and I/O-free, so it is unit-tested
    with a fake VAD and each backend only has to supply frames. Any per-window
    cap (max_utterance_s) belongs to the frame producer.
    """
    import collections

    import numpy as np

    frame_ms = vad_cfg.frame_ms
    silence_frames = max(1, int(vad_cfg.silence_ms / frame_ms))
    # How many stray "speech" frames the trailing window may contain and still
    # count as silence. 0.0 restores the old strict consecutive-run behaviour.
    allowed_speech = int(silence_frames * max(0.0, vad_cfg.end_tolerance))
    recent: collections.deque = collections.deque(maxlen=silence_frames)
    collected: list[np.ndarray] = []
    triggered = False
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
            recent.append(is_speech)
            # Full trailing window, and it's mostly quiet -> the utterance is over.
            if len(recent) == silence_frames and sum(recent) <= allowed_speech:
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


def _lowpass(x: np.ndarray, cutoff_hz: float, rate: int, taps: int = 161) -> np.ndarray:
    """Windowed-sinc FIR low-pass, applied before any downsample.

    Without this, decimating 48 kHz -> 16 kHz folds everything above 8 kHz back
    into the speech band as aliasing distortion -- exactly the kind of degradation
    that makes Whisper hallucinate.

    161 taps, not fewer: the transition band scales as ~3.3/taps, and at 101 taps
    the roll-off started biting at 7 kHz (-2.9 dB) and reached -14 dB by 7.5 kHz --
    audible loss of exactly the /t/ /d/ /s/ burst energy that distinguishes
    "d2 to d4" from "d2". At 161 taps the response is flat (-0.0 dB) through 7 kHz
    and still -47 dB at the 8 kHz fold point, i.e. strictly better in BOTH bands.
    Measured on this Pi: 11 ms for a worst-case 7 s clip, against a ~6 s decode.
    """
    import numpy as np

    if x.size < taps:                       # too short to filter meaningfully
        return x
    n = np.arange(taps) - (taps - 1) / 2
    h = (np.sinc(2.0 * (cutoff_hz / rate) * n) * np.hamming(taps)).astype("float32")
    h /= h.sum()                            # unity DC gain
    return np.convolve(x, h, mode="same").astype("float32")


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio, anti-aliasing when downsampling.

    The USB mic runs at its native rate (44.1/48 kHz) and Whisper needs 16 kHz, so
    the conversion happens HERE rather than being left to PipeWire's resampler --
    we keep the full-rate capture and control the filtering ourselves. Band-limit
    first, then sample onto the target grid (an exact decimation when the ratio is
    an integer, e.g. 48k -> 16k).
    """
    import numpy as np

    x = np.asarray(x, dtype="float32").reshape(-1)
    if src_rate == dst_rate or x.size == 0:
        return x
    if dst_rate < src_rate:
        # 0.47 * dst_rate (7520 Hz for 16 kHz out) sits as close to the new Nyquist
        # as the filter's transition band allows: flat through the speech band,
        # -47 dB by 8 kHz where aliasing would fold in. See _lowpass.
        x = _lowpass(x, cutoff_hz=0.47 * dst_rate, rate=src_rate)
    n_out = int(round(x.shape[0] * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    src_idx = np.arange(n_out, dtype="float64") * (src_rate / dst_rate)
    return np.interp(src_idx, np.arange(x.shape[0]), x).astype("float32")


class AudioCapture:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg

    def record_utterance(self) -> np.ndarray:
        """Record one utterance; return float32 at cfg.sample_rate (16 kHz).

        The mic itself is opened at cfg.capture_rate -- its NATIVE rate -- and the
        result is downsampled here (see `resample`).
        """
        if self.cfg.vad.enabled:
            return self._record_vad()
        return self._record_fixed(self.cfg.vad.max_utterance_s)

    def _capture_rate(self) -> int:
        """Rate to open the mic at: cfg.capture_rate, or the Whisper rate if unset.

        webrtcvad only accepts 8/16/32/48 kHz, so a capture rate it can't handle
        (e.g. the card's 44.1 kHz) would raise deep inside the VAD on every frame.
        Fall back to the Whisper rate in that case rather than failing to listen.
        """
        rate = self.cfg.capture_rate or self.cfg.sample_rate
        if self.cfg.vad.enabled and rate not in (8000, 16000, 32000, 48000):
            log.warning("audio.capture_rate %d Hz is not one of webrtcvad's "
                        "8/16/32/48 kHz -- capturing at %d Hz instead.",
                        rate, self.cfg.sample_rate)
            return self.cfg.sample_rate
        return rate

    # -- VAD-gated capture --------------------------------------------------- #
    def _record_vad(self) -> np.ndarray:
        """Open the mic and feed its frames to the shared `collect_utterance`
        state machine (see there for the end-of-utterance rules)."""
        import sounddevice as sd
        import webrtcvad

        sr = self._capture_rate()                            # native mic rate
        frame_len = int(sr * self.cfg.vad.frame_ms / 1000)   # samples per frame
        vad = webrtcvad.Vad(self.cfg.vad.aggressiveness)

        with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                               blocksize=frame_len, device=self.cfg.input_device) as stream:
            log.debug("Listening (VAD) at %d Hz...", sr)
            frames = self._iter_frames(stream, frame_len)
            audio = collect_utterance(frames, vad, self.cfg.vad, sr)
        return resample(audio, sr, self.cfg.sample_rate)     # -> Whisper's 16 kHz

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

        sr = self._capture_rate()
        log.debug("Recording %.1fs at %d Hz...", seconds, sr)
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                       dtype="float32", device=self.cfg.input_device)
        sd.wait()
        return resample(audio.reshape(-1), sr, self.cfg.sample_rate)


class SpeakerKeepAlive:
    """Hold a Bluetooth speaker's amplifier awake with a continuous inaudible stream.

    THE reason a silent lead-in did not fix the clipped first syllable: cheap BT
    speakers power their amplifier down after a few seconds of *silence* and take
    ~350 ms to wake. Padding the clip with silence feeds the amp exactly what puts
    it to sleep, so it sleeps straight through the pad and still wakes on the first
    syllable ("Chess machine ready" -> "ess machine ready", "Your move" -> "r move").
    Disabling PipeWire's node suspend didn't help either -- that fixes the *link*,
    not the speaker's own amp.

    Keeping one very quiet noise stream open means the amp never sleeps at all, so
    speech starts instantly and needs no lead-in latency. PipeWire mixes this stream
    with the speech stream, so nothing else has to change.

    Level is a trade-off only the ear can settle: too quiet and the amp still naps,
    too loud and you hear hiss. Tune `tts.keep_alive_level` with
    scripts/tts_leadin_test.py --keepalive.
    """

    _CHUNK_S = 0.1

    def __init__(self, sample_rate: int = 48000, level: float = 0.002):
        self.sample_rate = sample_rate
        self.level = level
        self._proc: Any = None
        self._thread: Any = None
        self._stop: Any = None

    def start(self) -> bool:
        """Begin streaming. Returns False (and does nothing) if pw-play is absent."""
        import threading

        if shutil.which("pw-play") is None:
            log.debug("pw-play not installed -- speaker keep-alive disabled")
            return False
        if self._thread is not None:
            return True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="spk-keepalive", daemon=True)
        self._thread.start()
        log.info("Speaker keep-alive on (level %.4f) -- holding the BT amp awake so "
                 "the first syllable isn't clipped.", self.level)
        return True

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()
            with contextlib.suppress(Exception):
                proc.terminate()
        self._thread = None

    def _buffer(self) -> bytes:
        """~1 s of very low-level noise, looped. Noise (not a tone) because amp
        detectors are broadband, and it can't beat against anything."""
        import numpy as np

        rng = np.random.default_rng(0)
        amp = max(1, int(self.level * 32767))
        n = int(self.sample_rate)
        return rng.integers(-amp, amp + 1, size=n).astype("<i2").tobytes()

    def _run(self) -> None:
        import subprocess

        chunk_bytes = int(self.sample_rate * self._CHUNK_S) * 2
        payload = self._buffer()
        while self._stop is not None and not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    ["pw-play", "--raw", f"--rate={self.sample_rate}",
                     "--channels=1", "--format=s16", "-"],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                pos = 0
                while not self._stop.is_set():
                    if pos + chunk_bytes > len(payload):
                        pos = 0
                    self._proc.stdin.write(payload[pos:pos + chunk_bytes])
                    self._proc.stdin.flush()
                    pos += chunk_bytes
                    if self._proc.poll() is not None:
                        break               # player died (sink vanished) -> respawn
            except Exception:  # noqa: BLE001 - a comfort stream must never crash the app
                log.debug("speaker keep-alive stream dropped; retrying", exc_info=True)
            finally:
                with contextlib.suppress(Exception):
                    self._proc.stdin.close()
            if self._stop is not None and not self._stop.wait(1.0):
                continue                    # BT dropped out: reconnect after a beat


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
