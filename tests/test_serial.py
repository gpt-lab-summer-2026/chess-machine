"""The serial backend is the one place the planar board frame becomes polar."""
from chessmachine.config import SerialConfig
from chessmachine.motion.serial_esp32 import SerialMotion


def _capture():
    # offset off here: these assert the plain polar conversion; the winch-offset
    # correction has its own tests in test_serial_offset.py.
    m = SerialMotion(SerialConfig(winch_offset_mm=0.0))
    sent: list[str] = []
    m._command = lambda line, **kw: sent.append(line) or ""  # type: ignore[assignment]
    return m, sent


def test_move_xz_converts_to_polar():
    m, sent = _capture()
    m.move_xz(100.0, 0.0)                       # straight out the +x axis
    assert sent[-1] == "MOVE R100.00 A0.00"
    m.move_xz(0.0, 100.0)                       # +90 degrees
    assert sent[-1] == "MOVE R100.00 A90.00"
    m.move_xz(100.0, 100.0)                     # r=hypot, 45 degrees
    assert sent[-1] == "MOVE R141.42 A45.00"
    m.move_xz(80.0, -60.0, feed=4000)           # negative angle keeps the feed
    assert sent[-1] == "MOVE R100.00 A-36.87 F4000"


def test_parse_status_reads_polar_axes():
    out = SerialMotion._parse_status("R150.00 A-12.50 H4.00 MAG1 ENDR0 ENDA1")
    assert out == {
        "r": 150.0, "a": -12.5, "height": 4.0,
        "magnet": True, "endstop_r": False, "endstop_a": True,
    }
