"""Position analysis -> structured facts -> grounded English.

`describe_position` produces a dict of *facts* (material, evaluation, best line)
computed by python-chess + the engine. `facts_to_summary` renders them to a
deterministic sentence. The SLM may re-phrase the summary, but it is given these
facts as ground truth so it cannot fabricate an evaluation.
"""
from __future__ import annotations

from typing import Optional

import chess

from .engine import ChessEngine, AnalysisResult

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def material_balance(board: chess.Board) -> dict:
    white = sum(PIECE_VALUES[p.piece_type]
                for p in board.piece_map().values() if p.color == chess.WHITE)
    black = sum(PIECE_VALUES[p.piece_type]
                for p in board.piece_map().values() if p.color == chess.BLACK)
    diff = white - black
    leader = "white" if diff > 0 else "black" if diff < 0 else "even"
    return {"white": white, "black": black, "diff": diff, "leader": leader}


def game_phase(board: chess.Board) -> str:
    non_pawn = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.piece_type not in (chess.PAWN, chess.KING)
    )
    if non_pawn <= 6:
        return "endgame"
    if board.fullmove_number <= 8:
        return "opening"
    return "middlegame"


def eval_verdict(score_cp: Optional[int], mate_in: Optional[int]) -> str:
    if mate_in is not None:
        side = "White" if mate_in > 0 else "Black"
        return f"{side} has a forced mate in {abs(mate_in)}"
    if score_cp is None:
        return "the evaluation is unclear"
    pawns = score_cp / 100.0
    a = abs(pawns)
    side = "White" if pawns > 0 else "Black"
    if a < 0.8:
        return "the position is roughly equal"
    if a < 2.0:
        return f"{side} is slightly better"
    if a < 5.0:
        return f"{side} is clearly better"
    return f"{side} is winning"


def _pv_sans(board: chess.Board, pv: list[chess.Move], limit: int = 4) -> list[str]:
    sans, tmp = [], board.copy()
    for mv in pv[:limit]:
        if mv not in tmp.legal_moves:
            break
        sans.append(tmp.san(mv))
        tmp.push(mv)
    return sans


def describe_position(
    board: chess.Board,
    engine: ChessEngine,
    last_move_san: Optional[str] = None,
) -> dict:
    """Build a dict of ground-truth facts about the current position."""
    analysis: AnalysisResult = engine.analyse(board)
    material = material_balance(board)

    best_san = None
    if analysis.best_move and analysis.best_move in board.legal_moves:
        best_san = board.san(analysis.best_move)

    facts = {
        "turn": "white" if board.turn == chess.WHITE else "black",
        "fullmove": board.fullmove_number,
        "phase": game_phase(board),
        "in_check": board.is_check(),
        "legal_moves": board.legal_moves.count(),
        "material": material,
        "score_cp": analysis.score_cp,
        "mate_in": analysis.mate_in,
        "verdict": eval_verdict(analysis.score_cp, analysis.mate_in),
        "best_move_san": best_san,
        "pv_sans": _pv_sans(board, analysis.pv),
        "last_move_san": last_move_san,
        "game_over": board.is_game_over(claim_draw=True),
    }
    return facts


def facts_to_summary(facts: dict) -> str:
    """Deterministic spoken-style summary of the facts (SLM-free fallback)."""
    if facts.get("game_over"):
        return "The game is over."

    parts: list[str] = []
    parts.append(facts["verdict"].capitalize() + ".")

    mat = facts["material"]
    if mat["diff"] != 0:
        leader = "White" if mat["diff"] > 0 else "Black"
        parts.append(f"{leader} is up {abs(mat['diff'])} point"
                     f"{'s' if abs(mat['diff']) != 1 else ''} of material.")
    else:
        parts.append("Material is equal.")

    if facts["in_check"]:
        parts.append(f"{facts['turn'].capitalize()} is in check.")

    if facts.get("best_move_san"):
        line = facts.get("pv_sans") or [facts["best_move_san"]]
        parts.append("Best line: " + " ".join(line) + ".")

    return " ".join(parts)
