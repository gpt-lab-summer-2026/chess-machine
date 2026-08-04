"""Opening recognition: name the opening being played from its moves.

Stockfish EVALUATES positions but does not NAME openings and ships no opening book,
so this is a small bundled table of common openings keyed by their defining SAN move
sequence. `identify_opening()` returns the LONGEST line the game's moves start with,
so a game that deepens from "the Sicilian Defense" into "the Najdorf Variation" is
recognized as it develops (letting the caller comment once, then again on the deeper
line). It is move-order sensitive (no transposition table) — good enough for mainline
play; unusual/offbeat openings simply aren't named.
"""
from __future__ import annotations


def _norm(san: str) -> str:
    """Strip check/mate/annotation marks so history SAN matches the table."""
    return san.rstrip("+#!?")


# (name, SAN move list). Order-independent; the LONGEST matching prefix wins, so it is
# fine to have both a shallow name ("the Sicilian Defense") and deeper variations.
_OPENINGS: list[tuple[str, list[str]]] = [
    # --- first moves (kept short; depth 1 is not commented on its own) ---
    ("the King's Pawn opening", ["e4"]),
    ("the Queen's Pawn opening", ["d4"]),
    ("the English Opening", ["c4"]),
    ("the Réti Opening", ["Nf3"]),
    ("the Bird's Opening", ["f4"]),
    ("the Nimzo-Larsen Attack", ["b3"]),

    # --- 1.e4 e5 (open games) ---
    ("the Open Game", ["e4", "e5"]),
    ("the Vienna Game", ["e4", "e5", "Nc3"]),
    ("the King's Gambit", ["e4", "e5", "f4"]),
    ("the Petrov Defense", ["e4", "e5", "Nf3", "Nf6"]),
    ("the Philidor Defense", ["e4", "e5", "Nf3", "d6"]),
    ("the Italian Game", ["e4", "e5", "Nf3", "Nc6", "Bc4"]),
    ("the Italian Game, Giuoco Piano", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]),
    ("the Two Knights Defense", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"]),
    ("the Scotch Game", ["e4", "e5", "Nf3", "Nc6", "d4"]),
    ("the Ruy Lopez", ["e4", "e5", "Nf3", "Nc6", "Bb5"]),
    ("the Ruy Lopez, Berlin Defense", ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6"]),
    ("the Ruy Lopez, Morphy Defense", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]),

    # --- 1.e4 c5 (Sicilian) ---
    ("the Sicilian Defense", ["e4", "c5"]),
    ("the Sicilian, Alapin Variation", ["e4", "c5", "c3"]),
    ("the Sicilian, Closed", ["e4", "c5", "Nc3"]),
    ("the Sicilian, Najdorf Variation",
     ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]),
    ("the Sicilian, Dragon Variation",
     ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"]),
    ("the Sicilian, Sveshnikov Variation",
     ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e5"]),

    # --- 1.e4 other ---
    ("the French Defense", ["e4", "e6"]),
    ("the Caro-Kann Defense", ["e4", "c6"]),
    ("the Scandinavian Defense", ["e4", "d5"]),
    ("the Pirc Defense", ["e4", "d6"]),
    ("the Modern Defense", ["e4", "g6"]),
    ("the Alekhine Defense", ["e4", "Nf6"]),

    # --- 1.d4 d5 (closed/Queen's Gambit) ---
    ("the Closed Game", ["d4", "d5"]),
    ("the Queen's Gambit", ["d4", "d5", "c4"]),
    ("the Queen's Gambit Accepted", ["d4", "d5", "c4", "dxc4"]),
    ("the Queen's Gambit Declined", ["d4", "d5", "c4", "e6"]),
    ("the Slav Defense", ["d4", "d5", "c4", "c6"]),
    ("the London System", ["d4", "d5", "Bf4"]),

    # --- 1.d4 Nf6 (Indian defenses) ---
    ("the Indian Defense", ["d4", "Nf6"]),
    ("the Nimzo-Indian Defense", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"]),
    ("the Queen's Indian Defense", ["d4", "Nf6", "c4", "e6", "Nf3", "b6"]),
    ("the King's Indian Defense", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7"]),
    ("the Grünfeld Defense", ["d4", "Nf6", "c4", "g6", "Nc3", "d5"]),
    ("the Benoni Defense", ["d4", "Nf6", "c4", "c5"]),
    ("the Catalan Opening", ["d4", "Nf6", "c4", "e6", "g3"]),

    # --- 1.d4 other ---
    ("the Dutch Defense", ["d4", "f5"]),
]


def identify_opening(san_moves: list[str]) -> tuple[str, int] | None:
    """Longest opening line the game's moves start with, as (name, depth-in-plies),
    or None if nothing matches. Deeper lines win over their shallower parents."""
    played = [_norm(s) for s in san_moves]
    best: tuple[str, int] | None = None
    for name, line in _OPENINGS:
        n = len(line)
        if n <= len(played) and played[:n] == line and (best is None or n > best[1]):
            best = (name, n)
    return best
