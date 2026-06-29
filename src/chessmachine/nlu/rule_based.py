"""Deterministic keyword NLU.

Used as the dev/offline backend and as the fallback whenever the SLM is
unavailable or returns unusable output, so the machine stays operable without
a language model. It leans on `parse_move` to decide whether an utterance is a
chess move.
"""
from __future__ import annotations

import re
from typing import Optional

from ..chess_engine.analysis import facts_to_summary
from .base import NLU
from .intents import Intent
from .move_parsing import parse_move, normalize_spoken

_DIFFICULTY_RE = re.compile(r"\b(easy|medium|hard)\b")
_NUMBER_RE = re.compile(r"\b(\d{3,4})\b")
_SQUARE_RE = re.compile(r"[a-h][1-8]")
# Splits a compound utterance on conjunctions; whitespace anchors keep it from
# matching inside words like "command".
_CLAUSE_RE = re.compile(r"\s+and\s+then\s+|\s+and\s+|\s+then\s+|\s*;\s*", re.IGNORECASE)


def _has(text: str, *phrases: str) -> bool:
    return any(p in text for p in phrases)


def _looks_like_move(text: str) -> bool:
    t = normalize_spoken(text)
    return bool(_SQUARE_RE.search(t)) or "castle" in t


_SWAP_PHRASES = ("switch side", "swap side", "switch sides", "swap sides", "switch color",
                 "swap color", "switch colour", "swap colour", "trade side", "other side",
                 "change side", "switch teams")
_MACHINE_SUBJECT = ("you ", "you're", "youre", "your", "you'll", "youll", "you play",
                    "you take", "you be")


def _side_request(t: str) -> Optional[str]:
    """Resolve a side-change request to the MACHINE's target color.

    Returns 'white'/'black' (machine plays it), 'swap' (toggle), or None.
    Perspective: "you play white" -> machine white; "I'll play white" (or just
    "play as white") -> the user wants white, so the machine takes black.
    """
    if _has(t, *_SWAP_PHRASES):
        return "swap"
    has_color = "white" in t or "black" in t
    if has_color and _has(t, "play", "take", "give", "want", "make", "side", "be ",
                          "color", "colour", "switch", "swap"):
        color = "white" if "white" in t else "black"
        if _has(t, *_MACHINE_SUBJECT):
            return color                                  # machine plays this color
        return "black" if color == "white" else "white"   # user wants color -> machine opposite
    return None


class RuleBasedNLU(NLU):
    def interpret(self, transcript: str, context: dict) -> list[Intent]:
        # Deterministic stand-in for the SLM's language work: a single legal move
        # is one intent; otherwise split a compound utterance on conjunctions and
        # interpret each clause.
        board = context.get("board")
        if board is not None and parse_move(transcript, board):
            return [self._interpret_one(transcript, context)]
        clauses = [c.strip() for c in _CLAUSE_RE.split(transcript) if c.strip()]
        if len(clauses) <= 1:
            return [self._interpret_one(transcript, context)]
        return [self._interpret_one(c, context) for c in clauses]

    def _interpret_one(self, transcript: str, context: dict) -> Intent:
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

        side = _side_request(t)
        if side is not None:
            return Intent("set_side", color=(None if side == "swap" else side), text=transcript)

        if _has(t, "difficulty", "level", "elo") or _DIFFICULTY_RE.search(t):
            diff: Optional[str] = None
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

    def phrase_analysis(self, question: str, facts: dict) -> str:
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
