"""Turn a chess move into physical pick-and-place operations.

The board is played by lifting a magnetic piece straight up to a safe travel
height, carrying it above all other pieces, and lowering it onto the target
square. That clears arbitrary moves (including knights) without sliding between
pieces. Captures route the taken piece to off-board storage first; castling
moves king then rook; promotions swap in a stored piece when one is available.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import chess

from ..chess_engine.game import classify_move, MoveClassification, MoveKind
from ..config import SpeedsConfig, MagnetConfig
from .base import MotionController
from .geometry import BoardGeometry, Point
from .graveyard import Graveyard

log = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    kind: MoveKind
    description: str = ""
    notes: list[str] = field(default_factory=list)   # manual-intervention prompts
    transfers: int = 0                                # pick+place pairs performed


class Choreographer:
    """Sequences a MotionController to execute chess moves on the physical board."""

    def __init__(
        self,
        controller: MotionController,
        geometry: BoardGeometry,
        graveyard: Graveyard,
        speeds: SpeedsConfig,
        magnet: MagnetConfig,
    ):
        self.ctl = controller
        self.geo = geometry
        self.graveyard = graveyard
        self.speeds = speeds
        self.magnet_cfg = magnet

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        self.ctl.connect()

    def home(self) -> None:
        self.ctl.home()

    def close(self) -> None:
        self.ctl.close()

    def park(self) -> None:
        """Raise the magnet and release it (safe idle state)."""
        self.ctl.set_pulley(self.geo.cfg.travel_height_mm, self.speeds.lift_feed)
        self.ctl.magnet(False)

    # -- low-level primitives ------------------------------------------------ #
    def _sleep(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)

    def _pick(self, point: Point) -> None:
        self.ctl.move_xz(point.x, point.z, self.speeds.travel_feed)
        self._sleep(self.speeds.settle_ms)
        self.ctl.set_pulley(self.geo.cfg.pick_height_mm, self.speeds.lift_feed)
        self.ctl.magnet(True)
        self._sleep(self.magnet_cfg.settle_ms)
        self.ctl.set_pulley(self.geo.cfg.travel_height_mm, self.speeds.lift_feed)
        self._sleep(self.speeds.settle_ms)

    def _place(self, point: Point) -> None:
        self.ctl.move_xz(point.x, point.z, self.speeds.travel_feed)
        self._sleep(self.speeds.settle_ms)
        self.ctl.set_pulley(self.geo.cfg.pick_height_mm, self.speeds.lift_feed)
        self.ctl.magnet(False)
        self._sleep(self.magnet_cfg.settle_ms)
        self.ctl.set_pulley(self.geo.cfg.travel_height_mm, self.speeds.lift_feed)
        self._sleep(self.speeds.settle_ms)

    def _transfer(self, src: Point, dst: Point) -> None:
        self._pick(src)
        self._place(dst)

    def _sq(self, square: int) -> Point:
        return self.geo.square_to_point(square)

    # -- public: execute one move ------------------------------------------- #
    def execute_move(self, board_before: chess.Board, move: chess.Move) -> ExecutionReport:
        """Actuate `move`, given the position *before* it is applied (blocking)."""
        complete, _report = self.begin_move(board_before, move)
        return complete()

    def begin_move(self, board_before: chess.Board, move: chess.Move):
        """Split a move into (complete, report): the captured piece (if any) is
        cleared to storage *now* (blocking), and `complete()` actuates the rest.

        This lets the caller speak the move and its explanation while the gantry
        carries the piece — but only after the capture's discard has finished,
        so the board never looks wrong while we talk. `complete()` is safe to run
        on a worker thread (it touches only the motion controller).
        """
        cls = classify_move(board_before, move)
        report = ExecutionReport(kind=cls.kind)

        # Clear a captured piece to storage first (normal capture or en passant).
        # Castling is never a capture, so this is skipped for it.
        if not cls.is_castle and cls.is_capture and cls.captured_piece is not None:
            slot = self.graveyard.store(cls.captured_piece)
            self._transfer(self._sq(cls.captured_square), slot)
            report.transfers += 1

        def complete() -> ExecutionReport:
            self._actuate_body(cls, report)
            return report

        return complete, report

    def _actuate_body(self, cls: MoveClassification, report: ExecutionReport) -> None:
        """The piece's own travel (after any capture has been discarded)."""
        if cls.is_castle:
            self._transfer(self._sq(cls.from_square), self._sq(cls.to_square))   # king
            self._transfer(self._sq(cls.rook_from), self._sq(cls.rook_to))       # rook
            report.transfers += 2
            report.description = f"castled {cls.castle_side}side"
            return

        if cls.promotion:
            self._do_promotion(cls, report)
            report.description = "promoted"
        else:
            self._transfer(self._sq(cls.from_square), self._sq(cls.to_square))
            report.transfers += 1
            report.description = "moved"

    def _do_promotion(self, cls: MoveClassification, report: ExecutionReport) -> None:
        color = cls.moving_piece.color
        promo_pt = self.graveyard.retrieve(cls.promotion, color)
        promo_name = chess.piece_name(cls.promotion)
        to_name = chess.square_name(cls.to_square)

        if promo_pt is not None:
            # Spare piece available: pawn -> storage, spare -> destination.
            pawn_slot = self.graveyard.store(cls.moving_piece)
            self._transfer(self._sq(cls.from_square), pawn_slot)
            self._transfer(promo_pt, self._sq(cls.to_square))
            report.transfers += 2
        else:
            # No spare: carry the pawn and ask for a manual swap.
            self._transfer(self._sq(cls.from_square), self._sq(cls.to_square))
            report.transfers += 1
            report.notes.append(
                f"I have no spare {promo_name} in storage. "
                f"Please replace the pawn on {to_name} with a {promo_name}."
            )

    # -- public: physically reverse a move (undo) --------------------------- #
    def reverse_move(self, board_before: chess.Board, move: chess.Move) -> ExecutionReport:
        """Undo `move` on the board. `board_before` is the position it was made
        from (i.e. after popping it from the move stack)."""
        cls = classify_move(board_before, move)
        report = ExecutionReport(kind=cls.kind)

        if cls.is_castle:
            self._transfer(self._sq(cls.to_square), self._sq(cls.from_square))   # king back
            self._transfer(self._sq(cls.rook_to), self._sq(cls.rook_from))       # rook back
            report.transfers = 2
            report.description = "undid castling"
            return report

        if cls.promotion:
            # Promotion undo would need to un-swap the piece; ask for manual help.
            self._transfer(self._sq(cls.to_square), self._sq(cls.from_square))
            report.transfers = 1
            report.notes.append(
                f"Undid a promotion — please put a pawn back on "
                f"{chess.square_name(cls.from_square)} by hand."
            )
            return report

        # Move the piece back, then restore any captured piece from storage.
        self._transfer(self._sq(cls.to_square), self._sq(cls.from_square))
        report.transfers += 1
        if cls.is_capture and cls.captured_piece is not None:
            slot = self.graveyard.retrieve(cls.captured_piece.piece_type, cls.captured_piece.color)
            if slot is not None:
                self._transfer(slot, self._sq(cls.captured_square))
                report.transfers += 1
            else:
                report.notes.append(
                    f"Couldn't find the captured "
                    f"{chess.piece_name(cls.captured_piece.piece_type)} in storage."
                )
        report.description = "undid move"
        return report

    # -- public: rebuild the starting position ------------------------------ #
    def setup_starting_position(self, current_board: chess.Board) -> list[str]:
        """Reset the board: clear everything to storage, then place a fresh set.

        Works when total storage capacity covers every physical piece on the
        board (default geometry provides 32 slots = a full set).
        """
        notes: list[str] = []
        on_board = current_board.piece_map()
        if len(on_board) > self.graveyard.free() + self.graveyard.occupied():
            notes.append("Not enough storage to auto-reset; please reset by hand.")
            return notes

        # 1) Clear board into storage.
        for square, piece in list(on_board.items()):
            slot = self.graveyard.store(piece)
            self._transfer(self._sq(square), slot)

        # 2) Place a standard starting set, sourcing pieces from storage.
        start = chess.Board()
        for square, piece in start.piece_map().items():
            pt = self.graveyard.retrieve(piece.piece_type, piece.color)
            if pt is None:
                notes.append(
                    f"Missing a {'white' if piece.color else 'black'} "
                    f"{chess.piece_name(piece.piece_type)} for {chess.square_name(square)}."
                )
                continue
            self._transfer(pt, self._sq(square))
        return notes
