"""ESP32 serial mic capture: frame parsing, resampling, and robustness — all
against a synthetic serial stream, so no hardware or model is needed."""
import struct

import numpy as np

from chessmachine.voice.esp32_mic import (
    Esp32SerialCapture,
    MotionMicCapture,
    resample_to_16k,
)


def _frame(samples_i16) -> bytes:
    pcm = b"".join(struct.pack("<h", s) for s in samples_i16)
    return bytes([0xAA, 0x55]) + struct.pack("<H", len(pcm)) + pcm


class FakeSerial:
    """Replays a fixed byte stream for read(); records what was written."""

    def __init__(self, stream: bytes):
        self.stream = bytes(stream)
        self.pos = 0
        self.written = bytearray()

    def write(self, data) -> int:
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        pass

    def read(self, n: int = 1) -> bytes:
        chunk = self.stream[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


# -- resample ---------------------------------------------------------------- #
def test_resample_doubles_8k_to_16k():
    out = resample_to_16k(np.array([0.0, 0.5, -0.5, 0.25], dtype="float32"), 8000, 16000)
    assert out.shape[0] == 8 and out.dtype == np.float32


def test_resample_noop_when_rate_matches():
    a = np.array([0.1, 0.2], dtype="float32")
    assert np.array_equal(resample_to_16k(a, 16000, 16000), a)


# -- capture ----------------------------------------------------------------- #
def test_capture_sends_listen_and_returns_16k_audio():
    stream = _frame([100, 200, 300, 400]) + _frame([-100, -200, -300, -400]) + b"AUDIO_END\n"
    conn = FakeSerial(stream)
    cap = Esp32SerialCapture(conn, src_rate=8000, dst_rate=16000)
    out = cap.record_utterance()

    assert bytes(conn.written) == b"LISTEN\n"          # it triggered a window
    assert out.dtype == np.float32
    assert out.shape[0] == 16                          # 8 samples @ 8 kHz -> 16 @ 16 kHz
    assert np.all(np.abs(out) < 1.0)                   # normalized into [-1, 1)


def test_capture_ignores_text_lines_between_frames():
    stream = (_frame([1000, 2000, 3000, 4000])
              + b"# some debug line\n"
              + _frame([500, 600, 700, 800])
              + b"AUDIO_END\n")
    out = Esp32SerialCapture(FakeSerial(stream)).record_utterance()
    assert out.shape[0] == 16                          # both frames parsed, text skipped


def test_capture_empty_window_returns_empty():
    out = Esp32SerialCapture(FakeSerial(b"AUDIO_END\n")).record_utterance()
    assert out.shape[0] == 0 and out.dtype == np.float32


def test_capture_returns_audio_even_without_audio_end():
    # Link drops before AUDIO_END: a short read timeout backstops it and we still
    # return whatever frames arrived rather than hanging.
    stream = _frame([100, 200, 300, 400])              # one frame, no AUDIO_END
    cap = Esp32SerialCapture(FakeSerial(stream), read_timeout_s=0.05)
    out = cap.record_utterance()
    assert out.shape[0] == 8                            # 4 samples -> 8 after resample


# -- MotionMicCapture (pulls audio over the motion controller's shared serial) - #
class FakeMotion:
    def __init__(self, serial):
        self._serial = serial

    @property
    def serial(self):
        return self._serial


def test_motion_mic_capture_uses_shared_serial():
    conn = FakeSerial(_frame([100, 200, 300, 400]) + b"AUDIO_END\n")
    cap = MotionMicCapture(FakeMotion(conn))
    out = cap.record_utterance()
    assert bytes(conn.written) == b"LISTEN\n"
    assert out.shape[0] == 8 and out.dtype == np.float32


def test_motion_mic_capture_returns_empty_when_not_connected():
    out = MotionMicCapture(FakeMotion(None)).record_utterance()
    assert out.shape[0] == 0 and out.dtype == np.float32
