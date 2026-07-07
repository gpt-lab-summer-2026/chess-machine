"""Prototype actuator: one motor pulse per move, direction by whose move it is.

A stand-in for `Choreographer` while the machine is a single relay-driven DC
motor rather than the full crane. It exposes the same surface the pipeline calls
(`connect`/`home`/`close`/`park`, `begin_move`/`execute_move`/`reverse_move`,
`setup_starting_position`), but ignores geometry, captures, castling, promotion
and storage entirely. Every move just runs the motor for a fixed time:

    the user's moves   -> motor forward
    the machine's moves -> motor reverse
    undo                -> the opposite direction of the move being undone

`mover_is_machine` is supplied by the pipeline (it knows the side to move and the
machine's colour); the crane `Choreographer` accepts and ignores the same flag,
so the pipeline drives both the same way. This is a hardware bring-up test, not
real piece movement — see firmware/esp32_dc_prototype.
"""
from __future__ import annotations

import logging

import chess

from ..chess_engine.game import MoveKind
from .choreography import ExecutionReport
from .relay_esp32 import RelayMotion

log = logging.getLogger(__name__)


class RelayChoreographer:
    def __init__(self, controller: RelayMotion, pulse_ms: int):
        self.ctl = controller
        self.pulse_ms = pulse_ms

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        self.ctl.connect()

    def home(self) -> None:
        self.ctl.home()   # no-op on the prototype firmware

    def close(self) -> None:
        self.ctl.close()

    def park(self) -> None:
        self.ctl.stop()

    # -- one pulse ----------------------------------------------------------- #
    def _pulse(self, forward: bool) -> ExecutionReport:
        self.ctl.drive(forward, self.pulse_ms)
        return ExecutionReport(
            kind=MoveKind.NORMAL,
            description=f"pulsed motor {'forward' if forward else 'reverse'}",
            transfers=1,
        )

    # -- pipeline-facing actuation ------------------------------------------ #
    def execute_move(self, board_before: chess.Board, move: chess.Move,
                     mover_is_machine: bool = False) -> ExecutionReport:
        return self._pulse(forward=not mover_is_machine)

    def begin_move(self, board_before: chess.Board, move: chess.Move,
                   mover_is_machine: bool = False):
        """Match Choreographer.begin_move's (complete, report) contract so the
        pipeline's concurrent-actuation path works: the motor pulses on a worker
        thread while the move is spoken."""
        report = ExecutionReport(kind=MoveKind.NORMAL)

        def complete() -> ExecutionReport:
            self.ctl.drive(not mover_is_machine, self.pulse_ms)
            report.transfers = 1
            report.description = f"pulsed motor {'reverse' if mover_is_machine else 'forward'}"
            return report

        return complete, report

    def reverse_move(self, board_before: chess.Board, move: chess.Move,
                     mover_is_machine: bool = False) -> ExecutionReport:
        # Undo runs the motor opposite to the move it undoes.
        return self._pulse(forward=mover_is_machine)

    def setup_starting_position(self, current_board: chess.Board) -> list[str]:
        self.ctl.stop()
        return ["I can't reset the board on the relay prototype — "
                "please set the pieces up by hand."]
