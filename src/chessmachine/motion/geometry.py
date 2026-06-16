"""Map chess squares (and the off-board graveyard) to physical (X,Z) gantry mm.

Origin is the center of square a1. Files (a..h) and ranks (1..8) are assigned to
the two physical axes via config, with optional inversion, so the same code
handles any mounting orientation of the board under the gantry.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from ..config import GeometryConfig


@dataclass(frozen=True)
class Point:
    x: float
    z: float

    def rounded(self, ndigits: int = 2) -> "Point":
        return Point(round(self.x, ndigits), round(self.z, ndigits))


class BoardGeometry:
    def __init__(self, cfg: GeometryConfig):
        if {cfg.file_axis, cfg.rank_axis} != {"x", "z"}:
            raise ValueError(
                "geometry.file_axis and geometry.rank_axis must be 'x' and 'z' (distinct); "
                f"got {cfg.file_axis!r}, {cfg.rank_axis!r}"
            )
        self.cfg = cfg

    def square_to_point(self, square: int) -> Point:
        """python-chess square index (a1=0 .. h8=63) -> gantry Point."""
        file_idx = chess.square_file(square)   # 0..7  (a..h)
        rank_idx = chess.square_rank(square)   # 0..7  (1..8)
        if self.cfg.invert_file:
            file_idx = 7 - file_idx
        if self.cfg.invert_rank:
            rank_idx = 7 - rank_idx

        file_off = file_idx * self.cfg.square_pitch_mm
        rank_off = rank_idx * self.cfg.square_pitch_mm

        if self.cfg.file_axis == "x":
            return Point(self.cfg.origin_x_mm + file_off, self.cfg.origin_z_mm + rank_off)
        # file runs along z, rank along x
        return Point(self.cfg.origin_x_mm + rank_off, self.cfg.origin_z_mm + file_off)

    def name_to_point(self, name: str) -> Point:
        return self.square_to_point(chess.parse_square(name))

    def graveyard_slots(self) -> list[Point]:
        """Off-board storage positions, filled column-major."""
        slots: list[Point] = []
        for col in range(self.cfg.graveyard_columns):
            x = self.cfg.graveyard_x_mm + col * self.cfg.graveyard_x_pitch_mm
            for row in range(self.cfg.graveyard_slots_per_column):
                z = self.cfg.graveyard_z_start_mm + row * self.cfg.graveyard_z_pitch_mm
                slots.append(Point(x, z))
        return slots
