"""Capture one spoken utterance from a MAX4466 microphone on the ESP32.

The ESP32 samples the MAX4466 on its ADC and streams the audio to the Pi over
the SAME USB-serial link it uses for motor commands. The chess machine is
turn-based (you speak, THEN the crane moves), so the mic and the motor never use
the port at the same time — this capture runs a whole `LISTEN` window
synchronously and hands 16 kHz float32 samples to Whisper, exactly like the
sounddevice `AudioCapture` it stands in for.

Wire protocol (host <-> firmware), all on one serial port:
    ->  LISTEN\\n
    <-  [0xAA 0x55][len_LE:2][len bytes of int16 PCM]   (repeated audio frames)
    <-  AUDIO_END\\n
The firmware does its own on-device silence detection and a hard time cap, so a
window ends by itself; `read_timeout_s` is only a backstop against a dropped link.
Text lines other than AUDIO_END (e.g. '#' debug, stray 'OK') are ignored.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

_FRAME_MAGIC_0 = 0xAA
_FRAME_MAGIC_1 = 0x55
_AUDIO_END = "AUDIO_END"


def resample_to_16k(pcm: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
    """Linear-resample mono float32 audio to `dst_rate` (Whisper wants 16 kHz).

    `np.interp` is plenty for speech STT and needs no scipy. Returns the input
    unchanged when it's already at the target rate or empty.
    """
    import numpy as np

    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype("float32", copy=False)
    n_out = int(round(pcm.shape[0] * dst_rate / src_rate))
    if n_out <= 0:
        return pcm.astype("float32", copy=False)
    x_old = np.linspace(0.0, 1.0, num=pcm.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype("float32")


class Esp32SerialCapture:
    """Records one utterance over the shared ESP32 serial link.

    `conn` is a pyserial-like object (`write`, `flush`, `read(n)`, optional
    `reset_input_buffer`). It is typically the SAME handle the motion transport
    owns — the caller guarantees the two never run concurrently (turn-based).
    """

    def __init__(self, conn: Any, src_rate: int = 8000, dst_rate: int = 16000,
                 listen_cmd: bytes = b"LISTEN\n", read_timeout_s: float = 8.0):
        self.conn = conn
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.listen_cmd = listen_cmd
        self.read_timeout_s = read_timeout_s

    def record_utterance(self) -> np.ndarray:
        """Trigger a LISTEN window and return 16 kHz mono float32 samples
        (empty array if nobody spoke / the link dropped)."""
        import numpy as np

        # Drop anything buffered (a previous move's ack, boot banner) so the
        # first bytes we parse belong to this window.
        with _suppress():
            self.conn.reset_input_buffer()
        self.conn.write(self.listen_cmd)
        self.conn.flush()

        frames: list[np.ndarray] = []
        line = bytearray()
        deadline = time.monotonic() + self.read_timeout_s
        while time.monotonic() < deadline:
            b = self.conn.read(1)
            if not b:
                continue                                  # serial timeout, no data yet
            val = b[0]

            if val == _FRAME_MAGIC_0:
                b2 = self.conn.read(1)
                if b2 and b2[0] == _FRAME_MAGIC_1:
                    length = self._read_exact(2)
                    if length is None:
                        break
                    n = int.from_bytes(length, "little")
                    data = self._read_exact(n)
                    if data is None:
                        break
                    frames.append(np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0)
                else:                                     # a lone 0xAA in a text line
                    line.append(val)
                    if b2:
                        line.append(b2[0])
                continue

            if val == 0x0A:                               # newline: a text line ended
                text = line.decode(errors="replace").strip()
                line.clear()
                if text == _AUDIO_END:
                    break
                if text:
                    log.debug("esp32 mic: %s", text)      # '#' debug, stray acks — ignore
                continue
            if val != 0x0D:                               # ignore CR, buffer the rest
                line.append(val)
        else:
            log.warning("ESP32 mic: no AUDIO_END within %.1fs (link dropped?)", self.read_timeout_s)

        if not frames:
            return np.zeros(0, dtype="float32")
        pcm = np.concatenate(frames)
        return resample_to_16k(pcm, self.src_rate, self.dst_rate)

    def _read_exact(self, n: int) -> bytes | None:
        """Read exactly `n` bytes (serial `read` may return fewer). None if the
        link stalls before `n` arrive, so the caller aborts the window cleanly."""
        buf = bytearray()
        deadline = time.monotonic() + self.read_timeout_s
        while len(buf) < n and time.monotonic() < deadline:
            chunk = self.conn.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None


class MotionMicCapture:
    """`record_utterance()` that pulls mic audio over the MOTION controller's
    serial link (same board, same port). Drop-in for sounddevice `AudioCapture`
    in `DistilWhisperSTT`.

    Bound to a motion controller exposing a `.serial` handle (SerialMotion). The
    handle only exists after `connect()`, so the underlying capture is built
    lazily on first use — by which point the pipeline has connected the crane.
    """

    def __init__(self, motion: Any, src_rate: int = 8000):
        self._motion = motion
        self._src_rate = src_rate
        self._cap: Esp32SerialCapture | None = None

    def record_utterance(self) -> np.ndarray:
        import numpy as np

        conn = getattr(self._motion, "serial", None)
        if conn is None:
            log.warning("ESP32 mic: motion serial not connected yet; no audio")
            return np.zeros(0, dtype="float32")
        if self._cap is None:
            self._cap = Esp32SerialCapture(conn, src_rate=self._src_rate)
        return self._cap.record_utterance()


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib here —
    reset_input_buffer is optional on the connection object."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True
