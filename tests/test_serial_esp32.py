"""Tests for the ESP32 serial line protocol, driven by a fake serial port.

A `FakeSerial` records the bytes the host writes and replays scripted reply
lines, so the request/response loop is exercised without any hardware. We set
`sm._ser` directly to bypass the real `serial.Serial` construction in connect().
"""
import pytest

from chessmachine.config import SerialConfig
from chessmachine.motion.serial_esp32 import SerialMotion


class FakeSerial:
    """Minimal pyserial stand-in: scripted reply lines, records writes."""

    def __init__(self, replies=()):
        self._replies = [r if isinstance(r, bytes) else (r + "\n").encode() for r in replies]
        self.written: list[bytes] = []
        self.is_open = True

    def write(self, data):
        self.written.append(data)
        return len(data)

    def flush(self):
        pass

    def readline(self):
        return self._replies.pop(0) if self._replies else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False


def _motion(replies=()):
    sm = SerialMotion(SerialConfig(timeout_s=0.05, home_timeout_s=0.05))
    sm._ser = FakeSerial(replies)
    return sm


def test_feed_formatting():
    assert SerialMotion._feed(None) == ""
    assert SerialMotion._feed(0) == ""            # falsey -> no feed term
    assert SerialMotion._feed(1200) == " F1200"


def test_parse_status():
    out = SerialMotion._parse_status("R10.00 A20.50 H5.00 MAG1 ENDR0 ENDA1")
    assert out == {"r": 10.0, "a": 20.5, "height": 5.0,
                   "magnet": True, "endstop_r": False, "endstop_a": True}


def test_move_xz_writes_protocol_line():
    sm = _motion(["OK"])
    sm.move_xz(70.0, 25.0, 1200)                         # (x,z) -> polar (r mm, a deg)
    assert sm._ser.written == [b"MOVE R74.33 A19.65 F1200\n"]


def test_move_xz_without_feed_omits_feed_term():
    sm = _motion(["OK"])
    sm.move_xz(1.0, 2.0)
    assert sm._ser.written == [b"MOVE R2.24 A63.43\n"]


def test_set_pulley_and_magnet_lines():
    sm = _motion(["OK", "OK"])
    sm.set_pulley(40.0, 800)
    sm.magnet(True)
    assert sm._ser.written == [b"PULLEY H40.00 F800\n", b"MAG ON\n"]


def test_magnet_off_line():
    sm = _motion(["OK"])
    sm.magnet(False)
    assert sm._ser.written == [b"MAG OFF\n"]


def test_command_skips_debug_and_event_lines():
    sm = _motion(["# booting", "EVT bump", "OK"])
    sm.move_xz(0.0, 0.0)                 # consumes the '#' and 'EVT' lines, then OK
    assert sm._ser._replies == []        # all three replies were read


def test_err_reply_raises():
    sm = _motion(["ERR bad arg"])
    with pytest.raises(RuntimeError, match="ESP32 error"):
        sm.move_xz(0.0, 0.0)


def test_expect_mismatch_raises():
    sm = _motion(["OK SOMETHINGELSE"])
    with pytest.raises(RuntimeError, match="Unexpected reply"):
        sm.home()                        # HOME expects "HOMED" in the payload


def test_home_accepts_expected_token():
    sm = _motion(["OK HOMED"])
    sm.home()
    assert sm._ser.written == [b"HOME\n"]


def test_timeout_when_no_reply():
    sm = _motion([])                     # readline only ever returns b""
    with pytest.raises(TimeoutError):
        sm.move_xz(0.0, 0.0)


def test_command_without_connection_raises():
    sm = SerialMotion(SerialConfig())    # _ser stays None (never connected)
    with pytest.raises(RuntimeError, match="not connected"):
        sm.move_xz(0.0, 0.0)


def test_status_parses_reply():
    sm = _motion(["OK R1.00 A2.00 H3.00 MAG0 ENDR1 ENDA0"])
    assert sm.status() == {"r": 1.0, "a": 2.0, "height": 3.0,
                           "magnet": False, "endstop_r": True, "endstop_a": False}
