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

import chess

from ..chess_engine.game import MoveClassification, MoveKind, classify_move
from ..config import MagnetConfig, SpeedsConfig
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
    aborted: bool = False                             # nothing actuated; caller must not apply it


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
        """Actuate `move`, given the position *before* it is applied."""
        cls = classify_move(board_before, move)
        report = ExecutionReport(kind=cls.kind)

        # Pre-flight: a capture needs a free storage slot. If none is available,
        # refuse the move without touching the board, so the logical and physical
        # positions stay in sync (the caller will not apply the move).
        if cls.is_capture and cls.captured_piece is not None and self.graveyard.free() == 0:
            report.aborted = True
            report.notes.append(
                "Storage is full — please clear the captured pieces off the board "
                "before I can take again."
            )
            return report

        if cls.is_castle:
            assert cls.rook_from is not None and cls.rook_to is not None
            self._transfer(self._sq(cls.from_square), self._sq(cls.to_square))   # king
            self._transfer(self._sq(cls.rook_from), self._sq(cls.rook_to))       # rook
            report.transfers = 2
            report.description = f"castled {cls.castle_side}side"
            return report

        # 1) Clear a captured piece to storage (normal capture or en passant).
        if cls.is_capture and cls.captured_piece is not None:
            assert cls.captured_square is not None
            slot = self.graveyard.store(cls.captured_piece)
            self._transfer(self._sq(cls.captured_square), slot)
            report.transfers += 1

        # 2) Move the piece (handling promotion).
        if cls.promotion:
            self._do_promotion(cls, report)
            report.description = "promoted"
        else:
            self._transfer(self._sq(cls.from_square), self._sq(cls.to_square))
            report.transfers += 1
            report.description = "moved"

        return report

    def _do_promotion(self, cls: MoveClassification, report: ExecutionReport) -> None:
        assert cls.promotion is not None
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
            assert cls.rook_from is not None and cls.rook_to is not None
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
            assert cls.captured_square is not None
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

        Needs a free storage slot per piece on the board (32 for a full set). The
        reference geometry has only 16 slots, so it falls back to a manual
        re-setup (returns a note and clears its storage bookkeeping); widen the
        graveyard to ~32 slots for automatic resets.
        """
        notes: list[str] = []
        on_board = current_board.piece_map()
        # The reset stages every piece in storage at once, so it needs a free
        # slot per piece on the board. The reference hardware (16 slots) can't
        # hold a full 32-piece set, so it falls back to a manual reset.
        if len(on_board) > self.graveyard.free():
            # The human will physically reset the board (and clear storage), so
            # drop our captured-piece bookkeeping to match.
            self.graveyard.reset()
            notes.append("I don't have enough storage to reset the board myself; "
                         "please set the pieces back up by hand.")
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
