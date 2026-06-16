"""Tracks captured pieces in off-board storage.

Each slot holds at most one piece. `store` finds the first free slot; `retrieve`
finds a stored piece of a given type+color (used for promotions and resets).
The piece identities let a reset rebuild the exact starting position.
"""
from __future__ import annotations

from typing import Optional

import chess

from .geometry import Point


class GraveyardFull(RuntimeError):
    pass


class Graveyard:
    def __init__(self, slots: list[Point]):
        self.slots = slots
        self.contents: list[Optional[chess.Piece]] = [None] * len(slots)

    @property
    def capacity(self) -> int:
        return len(self.slots)

    def occupied(self) -> int:
        return sum(1 for c in self.contents if c is not None)

    def free(self) -> int:
        return self.capacity - self.occupied()

    def store(self, piece: chess.Piece) -> Point:
        for i, cur in enumerate(self.contents):
            if cur is None:
                self.contents[i] = piece
                return self.slots[i]
        raise GraveyardFull(
            f"No free storage slot for {piece.symbol()} "
            f"({self.occupied()}/{self.capacity} used)"
        )

    def retrieve(self, piece_type: int, color: bool) -> Optional[Point]:
        """Free and return the slot of a matching piece, or None if absent."""
        for i, cur in enumerate(self.contents):
            if cur is not None and cur.piece_type == piece_type and cur.color == color:
                self.contents[i] = None
                return self.slots[i]
        return None

    def reset(self) -> None:
        self.contents = [None] * len(self.slots)
