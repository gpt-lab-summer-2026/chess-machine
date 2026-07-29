"""Capture one spoken utterance from a network audio stream.

Intended for a phone running a mic-streaming app (e.g. IP Webcam:
`http://<phone-ip>:8080/audio.opus`). PyAV/libav opens the HTTP (or RTSP/RTP)
stream, we decode + resample to 16 kHz mono, and run the SAME webrtcvad
end-of-utterance logic as the local sounddevice mic. The stream is opened only
for the listen window and closed immediately after, so the WiFi mic never
competes with the Bluetooth speaker (the machine is turn-based: you speak, THEN
it answers). Drop-in for the sounddevice `AudioCapture` in `DistilWhisperSTT`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from ..config import AudioConfig, SttConfig, VadConfig

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

_DST_RATE = 16000  # Whisper wants 16 kHz mono


def collect_utterance(frames: Iterable[Any], vad: Any, vad_cfg: VadConfig,
                      sr: int = _DST_RATE) -> "np.ndarray":
    """VAD-gate a stream of fixed-size int16 mono frames into one utterance.

    `frames` yields `frame_len`-sample int16 numpy arrays (frame_len = sr *
    frame_ms / 1000). `vad.is_speech(bytes, sr)` classifies each. Returns 16 kHz
    float32 in [-1, 1] — empty if too little speech was heard. This is the pure
    state machine (no I/O) so it can be unit-tested with a fake VAD; the network
    decode path just feeds it. Mirrors `AudioCapture._record_vad`.
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
    pcm = np.concatenate(collected)
    return (pcm.astype("float32") / 32768.0)


class NetworkMicCapture:
    """`record_utterance()` that pulls one window from a network audio stream.

    Bound to an HTTP/RTSP URL (`stt.stream_url`). The stream is (re)opened on
    every window and closed right after, so a dropped phone / wrong URL degrades
    to an empty result (rule-based fallback) rather than crashing the pipeline.
    """

    def __init__(self, stt_cfg: SttConfig, audio_cfg: AudioConfig):
        self.url = stt_cfg.stream_url
        self.timeout_s = stt_cfg.stream_timeout_s
        self.vad_cfg = audio_cfg.vad

    def record_utterance(self) -> "np.ndarray":
        import numpy as np

        if not self.url:
            log.warning("network mic: stt.stream_url is empty — set it to your phone's "
                        "IP Webcam URL (e.g. http://172.20.10.5:8080/audio.opus)")
            return np.zeros(0, dtype="float32")
        try:
            return self._capture()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully like the other captures
            log.warning("network mic: capture from %s failed (%s)", self.url, exc)
            import numpy as np
            return np.zeros(0, dtype="float32")

    def _capture(self) -> "np.ndarray":
        import av
        from av.audio.resampler import AudioResampler

        # libav network I/O timeout is microseconds; applies to open AND reads so
        # a stalled stream can't hang the turn forever.
        to = str(int(self.timeout_s * 1_000_000))
        container = av.open(self.url, options={"timeout": to, "rw_timeout": to})
        try:
            resampler = AudioResampler(format="s16", layout="mono", rate=_DST_RATE)
            frames = self._iter_frames(container, resampler)
            import webrtcvad
            vad = webrtcvad.Vad(self.vad_cfg.aggressiveness)
            return collect_utterance(frames, vad, self.vad_cfg, _DST_RATE)
        finally:
            container.close()

    def _iter_frames(self, container: Any, resampler: Any):
        """Decode the stream, resample to 16 kHz mono s16, and yield exactly
        `frame_len`-sample int16 frames, capped at max_utterance_s."""
        import numpy as np

        frame_len = int(_DST_RATE * self.vad_cfg.frame_ms / 1000)
        max_frames = int(self.vad_cfg.max_utterance_s * 1000 / self.vad_cfg.frame_ms)
        pending = np.zeros(0, dtype="<i2")
        emitted = 0
        for frame in container.decode(audio=0):
            for rs in resampler.resample(frame):
                samples = rs.to_ndarray().reshape(-1).astype("<i2")
                pending = np.concatenate([pending, samples]) if pending.size else samples
                while pending.size >= frame_len:
                    yield pending[:frame_len]
                    pending = pending[frame_len:]
                    emitted += 1
                    if emitted >= max_frames:
                        return
