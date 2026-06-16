import chess
import pytest

from chessmachine.config import GeometryConfig
from chessmachine.motion.geometry import BoardGeometry, Point


def test_a1_is_origin():
    geo = BoardGeometry(GeometryConfig())
    assert geo.name_to_point("a1") == Point(20.0, 20.0)


def test_square_pitch():
    geo = BoardGeometry(GeometryConfig())  # pitch 18.75, file->x, rank->z
    assert geo.name_to_point("h1") == Point(20.0 + 7 * 18.75, 20.0)
    assert geo.name_to_point("a8") == Point(20.0, 20.0 + 7 * 18.75)
    assert geo.name_to_point("h8") == Point(20.0 + 7 * 18.75, 20.0 + 7 * 18.75)


def test_invert_file():
    geo = BoardGeometry(GeometryConfig(invert_file=True))
    # a-file now sits at the far end of the x-axis
    assert geo.name_to_point("a1") == Point(20.0 + 7 * 18.75, 20.0)
    assert geo.name_to_point("h1") == Point(20.0, 20.0)


def test_axis_swap():
    geo = BoardGeometry(GeometryConfig(file_axis="z", rank_axis="x"))
    # files now run along z, ranks along x
    assert geo.name_to_point("a1") == Point(20.0, 20.0)
    assert geo.name_to_point("h1") == Point(20.0, 20.0 + 7 * 18.75)
    assert geo.name_to_point("a8") == Point(20.0 + 7 * 18.75, 20.0)


def test_graveyard_slots_count_and_spacing():
    geo = BoardGeometry(GeometryConfig())  # 4 columns x 8 rows
    slots = geo.graveyard_slots()
    assert len(slots) == 32
    assert slots[0] == Point(175.0, 10.0)
    assert slots[1] == Point(175.0, 21.0)        # next row, z + pitch
    assert slots[8] == Point(175.0 + 13.0, 10.0)  # next column, x + pitch


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        BoardGeometry(GeometryConfig(file_axis="x", rank_axis="x"))
