import chess
import pytest

from chessmachine.config import GeometryConfig
from chessmachine.motion.geometry import BoardGeometry, Point


def test_a1_is_origin():
    geo = BoardGeometry(GeometryConfig())
    assert geo.name_to_point("a1") == Point(10.0, 10.0)


def test_square_pitch():
    geo = BoardGeometry(GeometryConfig())  # pitch 15.0, file->x, rank->z
    assert geo.name_to_point("h1") == Point(10.0 + 7 * 15.0, 10.0)
    assert geo.name_to_point("a8") == Point(10.0, 10.0 + 7 * 15.0)
    assert geo.name_to_point("h8") == Point(10.0 + 7 * 15.0, 10.0 + 7 * 15.0)
    # the whole grid stays within the 150 mm linear-motor stroke
    assert geo.name_to_point("h8").x <= 150.0 and geo.name_to_point("h8").z <= 150.0


def test_invert_file():
    geo = BoardGeometry(GeometryConfig(invert_file=True))
    # a-file now sits at the far end of the x-axis
    assert geo.name_to_point("a1") == Point(10.0 + 7 * 15.0, 10.0)
    assert geo.name_to_point("h1") == Point(10.0, 10.0)


def test_axis_swap():
    geo = BoardGeometry(GeometryConfig(file_axis="z", rank_axis="x"))
    # files now run along z, ranks along x
    assert geo.name_to_point("a1") == Point(10.0, 10.0)
    assert geo.name_to_point("h1") == Point(10.0, 10.0 + 7 * 15.0)
    assert geo.name_to_point("a8") == Point(10.0 + 7 * 15.0, 10.0)


def test_graveyard_slots_within_reach():
    geo = BoardGeometry(GeometryConfig())  # 2 columns x 8 rows = 16 ("2 per row")
    slots = geo.graveyard_slots()
    assert len(slots) == 16
    assert slots[0] == Point(127.5, 10.0)
    assert slots[1] == Point(127.5, 25.0)         # next row, z + pitch
    assert slots[8] == Point(127.5 + 15.0, 10.0)  # next column, x + pitch
    # storage must also fit inside the 150 mm stroke
    assert all(s.x <= 150.0 and s.z <= 150.0 for s in slots)


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        BoardGeometry(GeometryConfig(file_axis="x", rank_axis="x"))
