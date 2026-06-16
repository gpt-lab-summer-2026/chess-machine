"""Chess truth: game state, engine (Stockfish), and position analysis.

Nothing here depends on speech or hardware, so it is fully unit-testable.
"""
from .game import GameState, MoveKind, classify_move
from .engine import ChessEngine, AnalysisResult, create_engine
from . import analysis

__all__ = [
    "GameState",
    "MoveKind",
    "classify_move",
    "ChessEngine",
    "AnalysisResult",
    "create_engine",
    "analysis",
]
