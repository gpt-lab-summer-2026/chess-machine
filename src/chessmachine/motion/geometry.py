"""Map chess squares (and the off-board graveyard) to planar (X,Z) millimetres.

The frame's origin is the crane pivot; the transport converts each (x,z) to
polar (r, theta) for the rotary base + radial railcart. The board itself is a
flat grid, so squares map to a plane exactly as before: origin is the center of
a1, files (a..h) and ranks (1..8) run along the two planar axes (config, with
optional inversion) so the board can sit at any orientation in the sector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import chess

from ..config import GeometryConfig


@dataclass(frozen=True)
class Point:
    x: float
    z: float

    def rounded(self, ndigits: int = 2) -> Point:
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
        """python-chess square index (a1=0 .. h8=63) -> planar Point (mm)."""
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
        """Off-board storage: two symmetric arcs in the leftover sector.

        Slots sit at `graveyard_radius_mm` (beyond the board's far corners, so
        the whole arc clears the board), spread between `graveyard_inner_deg` and
        `graveyard_outer_deg` on each side of the bisector. Filled
        positive-angle side first, inner angle first.
        """
        r = self.cfg.graveyard_radius_mm
        n = self.cfg.graveyard_slots_per_side
        inner = self.cfg.graveyard_inner_deg
        outer = self.cfg.graveyard_outer_deg
        slots: list[Point] = []
        for sign in (1.0, -1.0):
            for k in range(n):
                frac = k / (n - 1) if n > 1 else 0.0
                theta = math.radians(sign * (inner + (outer - inner) * frac))
                slots.append(Point(r * math.cos(theta), r * math.sin(theta)))
        return slots
