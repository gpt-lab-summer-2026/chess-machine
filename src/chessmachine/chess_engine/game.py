"""Logical game state and move classification.

`GameState` wraps a python-chess board with the bits the application needs
(whose turn, machine color, history, game-over text). `classify_move` turns a
legal move into the physical operations the crane must perform — this is the
bridge between chess rules and the motion choreography.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import chess


class MoveKind(str, Enum):
    NORMAL = "normal"
    CAPTURE = "capture"
    EN_PASSANT = "en_passant"
    CASTLE = "castle"
    PROMOTION = "promotion"


@dataclass
class MoveClassification:
    """Everything the choreographer needs to actuate a single move."""
    move: chess.Move
    from_square: int
    to_square: int
    moving_piece: chess.Piece
    is_capture: bool = False
    is_en_passant: bool = False
    is_castle: bool = False
    castle_side: str | None = None        # "king" | "queen"
    promotion: int | None = None          # chess.PieceType promoted to
    captured_square: int | None = None    # where the captured piece sits
    captured_piece: chess.Piece | None = None
    rook_from: int | None = None          # castling rook travel
    rook_to: int | None = None

    @property
    def kind(self) -> MoveKind:
        if self.is_castle:
            return MoveKind.CASTLE
        if self.promotion:
            return MoveKind.PROMOTION
        if self.is_en_passant:
            return MoveKind.EN_PASSANT
        if self.is_capture:
            return MoveKind.CAPTURE
        return MoveKind.NORMAL


def classify_move(board: chess.Board, move: chess.Move) -> MoveClassification:
    """Inspect `move` against `board` (the position *before* the move)."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"No piece on {chess.square_name(move.from_square)} to move")

    c = MoveClassification(
        move=move,
        from_square=move.from_square,
        to_square=move.to_square,
        moving_piece=moving_piece,
        promotion=move.promotion,
    )

    if board.is_castling(move):
        c.is_castle = True
        rank = chess.square_rank(move.from_square)
        if board.is_kingside_castling(move):
            c.castle_side = "king"
            c.rook_from = chess.square(7, rank)   # h-file
            c.rook_to = chess.square(5, rank)     # f-file
        else:
            c.castle_side = "queen"
            c.rook_from = chess.square(0, rank)   # a-file
            c.rook_to = chess.square(3, rank)     # d-file
        return c

    if board.is_en_passant(move):
        c.is_capture = True
        c.is_en_passant = True
        # Captured pawn sits on the destination file, on the mover's start rank.
        c.captured_square = chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        )
        c.captured_piece = board.piece_at(c.captured_square)
    elif board.is_capture(move):
        c.is_capture = True
        c.captured_square = move.to_square
        c.captured_piece = board.piece_at(move.to_square)

    return c


# --------------------------------------------------------------------------- #
class GameState:
    """Logical board + history. Knows nothing about hardware or speech."""

    def __init__(self, machine_color: bool = chess.BLACK, start_fen: str | None = None):
        self.machine_color = machine_color
        self.board = chess.Board(start_fen) if start_fen else chess.Board()
        self.history: list[tuple[chess.Move, str]] = []
        self._resigned_by: bool | None = None   # color that resigned, if any

    # -- mutation --------------------------------------------------------- #
    def reset(self, start_fen: str | None = None) -> None:
        self.board = chess.Board(start_fen) if start_fen else chess.Board()
        self.history.clear()
        self._resigned_by = None

    def resign(self, color: bool) -> None:
        """Record that `color` resigned; the other side wins."""
        self._resigned_by = color

    def push(self, move: chess.Move) -> str:
        """Apply a (legal) move, returning its SAN."""
        san = self.board.san(move)
        self.board.push(move)
        self.history.append((move, san))
        return san

    def undo(self) -> chess.Move | None:
        self._resigned_by = None   # taking a move back puts the game back in play
        if not self.board.move_stack:
            return None
        move = self.board.pop()
        if self.history:
            self.history.pop()
        return move

    # -- queries ---------------------------------------------------------- #
    def fen(self) -> str:
        return self.board.fen()

    def turn(self) -> bool:
        return self.board.turn

    def is_machine_turn(self) -> bool:
        return self.board.turn == self.machine_color

    def legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def is_legal(self, move: chess.Move) -> bool:
        return move in self.board.legal_moves

    def san(self, move: chess.Move) -> str:
        return self.board.san(move)

    def classify(self, move: chess.Move) -> MoveClassification:
        return classify_move(self.board, move)

    def last_move(self) -> chess.Move | None:
        return self.board.peek() if self.board.move_stack else None

    def is_game_over(self) -> bool:
        return self._resigned_by is not None or self.board.is_game_over(claim_draw=True)

    @staticmethod
    def color_name(color: bool) -> str:
        return "White" if color == chess.WHITE else "Black"

    def result_text(self) -> str:
        if self._resigned_by is not None:
            loser = self.color_name(self._resigned_by)
            winner = self.color_name(not self._resigned_by)
            return f"{loser} resigned. {winner} wins."
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "The game is still in progress."
        if outcome.winner is None:
            reason = outcome.termination.name.replace("_", " ").lower()
            return f"The game is a draw ({reason})."
        winner = self.color_name(outcome.winner)
        if outcome.termination == chess.Termination.CHECKMATE:
            return f"Checkmate. {winner} wins."
        return f"{winner} wins ({outcome.termination.name.replace('_', ' ').lower()})."


# --------------------------------------------------------------------------- #
# Speakable narration of a move (nicer than raw SAN for TTS).
# --------------------------------------------------------------------------- #
_PIECE_WORDS = {
    "K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight",
}


def speak_san(san: str) -> str:
    """Turn SAN like 'Nxf3+' into 'knight takes f3, check' for text-to-speech."""
    if san in ("O-O", "0-0"):
        return "castles kingside"
    if san in ("O-O-O", "0-0-0"):
        return "castles queenside"

    suffix = ""
    core = san
    if core.endswith("#"):
        suffix, core = ", checkmate", core[:-1]
    elif core.endswith("+"):
        suffix, core = ", check", core[:-1]

    promo = ""
    if "=" in core:
        core, promo_pc = core.split("=", 1)
        promo = f", promote to {_PIECE_WORDS.get(promo_pc[0], promo_pc[0].lower())}"

    piece = "pawn"
    if core and core[0] in _PIECE_WORDS:
        piece = _PIECE_WORDS[core[0]]
        core = core[1:]
    
    takes = "takes" in core or "x" in core
    dest = core.replace("x", "").strip()
    # Leftover leading file/rank is a disambiguator (e.g. "Rad1") — read it out.
    dest_sq = dest[-2:] if len(dest) >= 2 else dest
    disambig = dest[:-2]
    disambig_words = " ".join(list(disambig)) + " " if disambig else ""

    verb = "takes" if takes else "to"
    spoken_dest = " ".join(list(dest_sq)) if dest_sq else ""
    return f"{piece} {disambig_words}{verb} {spoken_dest}{promo}{suffix}".replace("  ", " ").strip()
