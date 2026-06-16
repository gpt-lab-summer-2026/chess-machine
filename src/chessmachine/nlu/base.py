"""Abstract NLU: understand an utterance, phrase an answer."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .intents import Intent


class NLU(ABC):
    @abstractmethod
    def interpret(self, transcript: str, context: dict) -> Intent:
        """Classify an utterance into an Intent. `context` carries the live board."""

    @abstractmethod
    def phrase_analysis(self, question: str, facts: dict) -> str:
        """Phrase a spoken answer to a position question, grounded in `facts`."""

    def small_talk(self, transcript: str, context: dict) -> str:
        return ("I can set the difficulty, analyze the board, make my move, "
                "or play your move. What would you like?")

    def close(self) -> None:  # pragma: no cover
        pass
