"""Deterministic keyword NLU.

Used as the dev/offline backend and as the fallback whenever the SLM is
unavailable or returns unusable output, so the machine stays operable without
a language model. It leans on `parse_move` to decide whether an utterance is a
chess move.
"""
from __future__ import annotations

import re

from ..chess_engine.analysis import PositionFacts, facts_to_summary
from .base import NLU
from .intents import Intent
from .move_parsing import normalize_spoken, parse_move

_DIFFICULTY_RE = re.compile(r"\b(easy|medium|hard)\b")
_NUMBER_RE = re.compile(r"\b(\d{3,4})\b")
_SQUARE_RE = re.compile(r"[a-h][1-8]")


def _has(text: str, *phrases: str) -> bool:
    return any(p in text for p in phrases)


def _looks_like_move(text: str) -> bool:
    t = normalize_spoken(text)
    return bool(_SQUARE_RE.search(t)) or "castle" in t


class RuleBasedNLU(NLU):
    def interpret(self, transcript: str, context: dict) -> Intent:
        t = transcript.lower().strip()
        board = context.get("board")

        if _has(t, "new game", "reset", "start over", "restart", "new match", "set up the board"):
            return Intent("new_game", text=transcript)
        if _has(t, "undo", "take back", "takeback", "take that back"):
            return Intent("undo", text=transcript)
        if _has(t, "resign", "give up", "concede"):
            return Intent("resign", text=transcript)
        if _has(t, "repeat", "say again", "come again", "pardon", "what did you say"):
            return Intent("repeat", text=transcript)
        if _has(t, "help", "what can you do", "instructions"):
            return Intent("help", text=transcript)

        if _has(t, "difficulty", "level", "elo") or _DIFFICULTY_RE.search(t):
            diff: str | None = None
            m = _DIFFICULTY_RE.search(t)
            if m:
                diff = m.group(1)
            else:
                num = _NUMBER_RE.search(t)
                if num:
                    diff = num.group(1)
            return Intent("set_difficulty", difficulty=diff, text=transcript)

        if _has(t, "your move", "your turn", "you move", "your go", "make your move",
                "make a move", "go ahead", "play your"):
            return Intent("engine_move", text=transcript)

        if _has(t, "whose turn", "whose move", "what turn", "score", "material count"):
            return Intent("status", text=transcript)

        if _has(t, "analy", "best move", "winning", "evaluat", "how am i", "how are we",
                "what should i", "what do you think", "threat", "hint", "advice",
                "advantage", "who is better", "who's better", "is it good"):
            return Intent("analyze", question=transcript, text=transcript)

        # A legal move, or a clear (if unreadable) move attempt, routes to
        # opponent_move so the pipeline can play it or ask the user to repeat.
        if board is not None and parse_move(transcript, board):
            return Intent("opponent_move", move=transcript, text=transcript)
        if _looks_like_move(transcript):
            return Intent("opponent_move", move=transcript, text=transcript)

        return Intent("unknown", text=transcript)

    def phrase_analysis(self, question: str, facts: PositionFacts) -> str:
        q = (question or "").lower()
        summary = facts_to_summary(facts)
        if "best move" in q and facts.get("best_move_san"):
            return f"I'd play {facts['best_move_san']}. {summary}"
        if _has(q, "material", "up", "ahead", "down"):
            mat = facts.get("material", {})
            diff = mat.get("diff", 0)
            if diff == 0:
                return "Material is even. " + summary
            leader = "White" if diff > 0 else "Black"
            return f"{leader} is up {abs(diff)} points of material. " + summary
        return summary
