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


def test_graveyard_two_arcs_one_side_reachable_off_board():
    cfg = GeometryConfig()
    geo = BoardGeometry(cfg)
    slots = geo.graveyard_slots()
    assert len(slots) == 16                       # 2 concentric arcs x 8

    inner = math.radians(cfg.graveyard_inner_deg)
    side = cfg.graveyard_side                     # -1 = h1 / -theta (a8/+theta is switch-blocked)
    # first slot: OUTER arc, inner angle, on the configured (reachable) side
    assert slots[0].x == pytest.approx(cfg.graveyard_radius_mm * math.cos(inner))
    assert slots[0].z == pytest.approx(side * cfg.graveyard_radius_mm * math.sin(inner))
    # slot 8 begins the INNER arc: same side + angle, smaller radius
    assert slots[8].x == pytest.approx(cfg.graveyard_radius2_mm * math.cos(inner))
    assert slots[8].z == pytest.approx(side * cfg.graveyard_radius2_mm * math.sin(inner))
    # all storage is on ONE side now (the a8 side is blocked by the base limit switch)
    assert all((s.z < 0) == (side < 0) for s in slots)

    for s in slots:
        assert cfg.r_min_mm <= _r(s) <= cfg.r_max_mm   # reachable
        assert not _on_board(cfg, s)                    # clears the board


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
