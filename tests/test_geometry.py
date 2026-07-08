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


def test_graveyard_arcs_are_reachable_and_off_board():
    cfg = GeometryConfig()
    geo = BoardGeometry(cfg)
    slots = geo.graveyard_slots()
    assert len(slots) == 16                       # 2 sides x 8

    # first slot: positive side, inner angle, at the configured radius
    inner = math.radians(cfg.graveyard_inner_deg)
    assert slots[0].x == pytest.approx(cfg.graveyard_radius_mm * math.cos(inner))
    assert slots[0].z == pytest.approx(cfg.graveyard_radius_mm * math.sin(inner))
    # second side mirrors below the bisector
    assert slots[8].z == pytest.approx(-slots[0].z)

    for s in slots:
        assert cfg.r_min_mm <= _r(s) <= cfg.r_max_mm   # reachable
        assert not _on_board(cfg, s)                    # clears the board


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        BoardGeometry(GeometryConfig(file_axis="x", rank_axis="x"))
