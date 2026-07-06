"""Chess truth: game state, engine (Stockfish), and position analysis.

Nothing here depends on speech or hardware, so it is fully unit-testable.
"""
from . import analysis
from .engine import AnalysisResult, ChessEngine, create_engine
from .game import GameState, MoveKind, classify_move

__all__ = [
    "GameState",
    "MoveKind",
    "classify_move",
    "ChessEngine",
    "AnalysisResult",
    "create_engine",
    "analysis",
]
