import math

import chess
import pytest

from chessmachine.config import GeometryConfig
from chessmachine.motion.geometry import BoardGeometry, Point


def _r(p: Point) -> float:
    return math.hypot(p.x, p.z)


def _on_board(cfg: GeometryConfig, p: Point) -> bool:
    """True if a planar point lies within the 8x8 board square."""
    near, far = cfg.r_min_mm, cfg.r_min_mm + cfg.board_size_mm
    half = cfg.board_size_mm / 2.0
    return near <= p.x <= far and -half <= p.z <= half


def test_a1_is_origin():
    geo = BoardGeometry(GeometryConfig())
    assert geo.name_to_point("a1") == Point(93.5, -94.5)


def test_square_pitch():
    geo = BoardGeometry(GeometryConfig())  # pitch 27, file->x, rank->z
    assert geo.name_to_point("h1") == Point(93.5 + 7 * 27, -94.5)
    assert geo.name_to_point("a8") == Point(93.5, -94.5 + 7 * 27)
    assert geo.name_to_point("h8") == Point(93.5 + 7 * 27, -94.5 + 7 * 27)


def test_every_square_is_inside_the_reachable_sector():
    cfg = GeometryConfig()
    geo = BoardGeometry(cfg)
    for sq in chess.SQUARES:
        p = geo.square_to_point(sq)
        assert cfg.r_min_mm <= _r(p) <= cfg.r_max_mm        # within the annulus
        assert abs(math.degrees(math.atan2(p.z, p.x))) < 55.0  # within the sweep


def test_invert_file():
    geo = BoardGeometry(GeometryConfig(invert_file=True))
    # a-file now sits at the far end of the x-axis
    assert geo.name_to_point("a1") == Point(93.5 + 7 * 27, -94.5)
    assert geo.name_to_point("h1") == Point(93.5, -94.5)


def test_axis_swap():
    geo = BoardGeometry(GeometryConfig(file_axis="z", rank_axis="x"))
    # files now run along z, ranks along x
    assert geo.name_to_point("a1") == Point(93.5, -94.5)
    assert geo.name_to_point("h1") == Point(93.5, -94.5 + 7 * 27)
    assert geo.name_to_point("a8") == Point(93.5 + 7 * 27, -94.5)


def test_graveyard_dish_scatters_off_board_and_reachable():
    cfg = GeometryConfig()
    geo = BoardGeometry(cfg)
    slots = geo.graveyard_slots()
    assert len(slots) == cfg.graveyard_capacity          # one dish, `capacity` drop points

    ct = math.radians(cfg.graveyard_center_deg)
    cx = cfg.graveyard_center_r_mm * math.cos(ct)
    cz = cfg.graveyard_center_r_mm * math.sin(ct)
    spread = min(cfg.graveyard_jitter_mm, cfg.graveyard_diameter_mm / 2.0)
    # dish sits on the -theta (h1) side, well off the board's h-edge
    assert cz < 0 and cfg.graveyard_center_deg <= -40
    for s in slots:
        assert math.hypot(s.x - cx, s.z - cz) <= spread + 1e-9   # every drop within the dish jitter
        assert cfg.r_min_mm <= _r(s) <= cfg.r_max_mm             # reachable
        assert not _on_board(cfg, s)                              # clears the board


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        BoardGeometry(GeometryConfig(file_axis="x", rank_axis="x"))


def test_diagonal_board_angle_layout():
    # Corner-first (diagonal) real-machine layout: h8 near, a1 far on the bisector,
    # a8 on the right (-z). Mirrors config.example.yaml.
    cfg = GeometryConfig(board_angle_deg=135.0, origin_x_mm=380.0, origin_z_mm=0.0,
                         square_pitch_mm=26.30, r_min_mm=80.0, r_max_mm=455.0)
    geo = BoardGeometry(cfg)
    a1, h8 = geo.name_to_point("a1"), geo.name_to_point("h8")
    a8, h1 = geo.name_to_point("a8"), geo.name_to_point("h1")
    # far/near corners sit on the bisector (z=0); a1 is farthest, h8 nearest
    assert a1.z == pytest.approx(0.0, abs=1e-6)
    assert h8.z == pytest.approx(0.0, abs=1e-6)
    assert _r(a1) == pytest.approx(380.0)
    assert _r(h8) == pytest.approx(120.0, abs=0.5)   # near corner ~120 mm out
    assert _r(h8) < _r(a1)
    # a8 on the right (-z), h1 on the left (+z), mirror images across the bisector
    assert a8.z < 0 < h1.z
    assert a8.z == pytest.approx(-h1.z)
    assert _r(a8) == pytest.approx(_r(h1))
    # every square center falls inside the reachable annulus
    for sq in chess.SQUARES:
        assert cfg.r_min_mm <= _r(geo.square_to_point(sq)) <= cfg.r_max_mm


# -- the graveyard dish must stay inside the FIRMWARE's soft limits ----------- #
def _firmware_angle_limits() -> tuple[float, float]:
    """Read A_MIN_DEG / A_MAX_DEG out of the firmware source.

    Parsed rather than duplicated so this test can't quietly drift away from what
    the ESP32 is actually built with.
    """
    import pathlib
    import re

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "firmware" / "esp32_chess" / "esp32_chess.ino").read_text()
    amin = float(re.search(r"A_MIN_DEG\s*=\s*(-?[\d.]+)f", src).group(1))
    amax = float(re.search(r"A_MAX_DEG\s*=\s*(-?[\d.]+)f", src).group(1))
    return amin, amax


def test_configured_graveyard_stays_inside_the_firmware_sweep():
    """A drop the firmware can't reach is the WORST failure mode here: `clampf` just
    saturates the angle, so pieces pile up short of the dish with no error at all.
    The shipped config's dish -- scatter and winch offset included -- must therefore
    resolve inside the built firmware's sweep, with the real winch_offset applied.
    """
    import pathlib

    from chessmachine.config import load_config
    from chessmachine.motion.serial_esp32 import SerialMotion

    cfg = load_config(pathlib.Path(__file__).resolve().parent.parent
                      / "config" / "config.yaml")
    geo = BoardGeometry(cfg.motion.geometry)
    amin, amax = _firmware_angle_limits()
    offset = cfg.motion.serial.winch_offset_mm

    for slot in geo.graveyard_slots():
        r, a = SerialMotion._magnet_to_cart(slot.x, slot.z, offset)
        assert amin <= a <= amax, (
            f"graveyard slot needs cart angle {a:.2f} deg, outside the firmware sweep "
            f"[{amin}, {amax}] -- it would be silently clamped short of the dish"
        )
        assert cfg.motion.geometry.r_min_mm <= r <= cfg.motion.geometry.r_max_mm, (
            f"graveyard slot needs cart radius {r:.2f} mm, outside the reachable annulus"
        )


def test_board_squares_stay_inside_the_firmware_sweep():
    """Widening A_MIN_DEG for the dish must not be hiding a board-side problem, so
    pin the board's own angular span too."""
    import pathlib

    from chessmachine.config import load_config
    from chessmachine.motion.serial_esp32 import SerialMotion

    cfg = load_config(pathlib.Path(__file__).resolve().parent.parent
                      / "config" / "config.yaml")
    geo = BoardGeometry(cfg.motion.geometry)
    amin, amax = _firmware_angle_limits()
    offset = cfg.motion.serial.winch_offset_mm
    for sq in chess.SQUARES:
        p = geo.square_to_point(sq)
        _r_cart, a = SerialMotion._magnet_to_cart(p.x, p.z, offset)
        assert amin <= a <= amax, f"{chess.square_name(sq)} needs {a:.2f} deg"
