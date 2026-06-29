"""Abstract NLU: understand an utterance, phrase an answer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .intents import Intent

if TYPE_CHECKING:
    from ..chess_engine.analysis import PositionFacts


class NLU(ABC):
    @abstractmethod
    def interpret(self, transcript: str, context: dict) -> Intent:
        """Classify an utterance into an Intent. `context` carries the live board."""

    @abstractmethod
    def phrase_analysis(self, question: str, facts: PositionFacts) -> str:
        """Phrase a spoken answer to a position question, grounded in `facts`."""

    def small_talk(self, transcript: str, context: dict) -> str:
        return ("I can set the difficulty, analyze the board, make my move, "
                "or play your move. What would you like?")

    def comment_on_move(self, info: dict) -> str:
        """One-line reaction to a move just played, grounded in `info`.

        Default: deterministic phrasing (also the SLM's fallback). `info` carries
        the move quality label, computed tactics, difficulty, SAN, and mover.
        """
        from ..chess_engine.analysis import MoveQuality, move_comment_summary

        quality = MoveQuality(
            label=info.get("label", "normal"),
            cp_loss=info.get("cp_loss"),
            is_sacrifice=info.get("is_sacrifice", False),
            only_good_move=info.get("only_good_move", False),
        )
        return move_comment_summary(
            quality, info.get("motifs", []), info.get("difficulty", ""),
            info.get("mover", ""), info.get("san", ""),
        )

    def close(self) -> None:  # pragma: no cover
        pass
