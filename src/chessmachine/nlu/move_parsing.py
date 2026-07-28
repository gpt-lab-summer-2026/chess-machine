"""Resolve a spoken/typed move into a concrete *legal* chess move.

Order of attack:
  1. Exact SAN / UCI (what the SLM is told to emit, and what a careful user types).
  2. Fuzzy speech: normalize number-words, glue "e four" -> "e4", detect castling,
     extract squares/piece words, then enumerate legal moves that fit.
Everything is filtered against `board.legal_moves`, so noisy transcripts can only
ever resolve to legal moves (or to an empty/ambiguous list the caller re-asks on).
"""
from __future__ import annotations

import re

import chess

_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
}
# NATO / common spellings STT produces for file letters.
_FILE_WORDS = {
    "alpha": "a", "bravo": "b", "charlie": "c", "delta": "d",
    "echo": "e", "foxtrot": "f", "golf": "g", "hotel": "h",
}
_PIECE_WORDS = {
    "king": chess.KING, "queen": chess.QUEEN, "rook": chess.ROOK,
    "bishop": chess.BISHOP, "knight": chess.KNIGHT, "horse": chess.KNIGHT,
    "pawn": chess.PAWN,
}
_PROMO_DEFAULT = chess.QUEEN
_SQUARE_RE = re.compile(r"[a-h][1-8]")
_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def normalize_spoken(text: str) -> str:
    """Lowercase, strip punctuation, map number/file words, glue 'e 4' -> 'e4'."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    words = t.split()
    words = [_FILE_WORDS.get(w, w) for w in words]
    words = [_NUM_WORDS.get(w, w) for w in words]
    t = " ".join(words)
    # "e 4" / "e to 4" stray spaces between a file letter and a rank digit
    t = re.sub(r"\b([a-h])\s+([1-8])\b", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


# Cues that mark an utterance as a question / analysis request rather than a
# move. Kept STT-robust: matched against the NORMALIZED string (no punctuation),
# because Whisper output usually has no '?'.
_INTERROGATIVES = {
    "what", "whats", "which", "where", "why", "how", "hows", "who",
    "is", "are", "am", "do", "does", "did", "can", "could", "should",
    "would", "will",
}
_ANALYSIS_STEMS = (
    "threat", "attack", "hang", "defend", "protect", "safe", "danger",
    "win", "better", "worse", "good", "best", "weak", "strong",
    "advantage", "worth", "should i", "is it", "how is",
)


def looks_like_analysis(transcript: str) -> bool:
    """True if the utterance reads as a question / analysis request, not a move.

    Used only to DOWNGRADE a move classification to analyze, so a false positive
    costs a re-route (the machine answers instead of moving), never a wrong move.
    Operates on the normalized transcript so it never depends on a '?' that STT
    tends to drop.
    """
    t = normalize_spoken(transcript)
    if not t:
        return False
    if t.split()[0] in _INTERROGATIVES:
        return True
    return any(stem in t for stem in _ANALYSIS_STEMS)


def question_target_square(question: str, board: chess.Board,
                           prefer_color: bool) -> int | None:
    """Resolve a piece/square reference in an analysis question to a board square.

    "...knight on c4..." -> c4; a bare piece word ("my knight") -> that piece if
    the preferred side has exactly one of them, else None (ambiguous -> caller
    falls back to a whole-board answer). `prefer_color` is the asker's side (the
    human); "your"/"you" flips it to the other side. Returns a square index or None.
    """
    t = normalize_spoken(question)
    if re.search(r"\byour\b|\byoure\b|\byou\b", t):
        prefer_color = not prefer_color
    squares = _SQUARE_RE.findall(t)
    if squares:
        return chess.parse_square(squares[0])
    ptype = next((pt for word, pt in _PIECE_WORDS.items()
                  if re.search(rf"\b{word}\b", t)), None)
    if ptype is None:
        return None
    owned = [s for s in chess.SQUARES
             if (p := board.piece_at(s)) is not None
             and p.color == prefer_color and p.piece_type == ptype]
    return owned[0] if len(owned) == 1 else None


def _try_exact(text: str, board: chess.Board) -> chess.Move | None:
    s = text.strip()
    # SAN (handles Nf3, exd5, O-O, e8=Q+, ...). parse_san already checks legality.
    for cand in (s, s.replace("0", "O")):
        try:
            return board.parse_san(cand)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            pass
    # UCI
    tok = s.lower().replace(" ", "")
    if _UCI_RE.match(tok):
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return mv
        except ValueError:
            pass
    return None


def _castling_candidates(t: str, board: chess.Board) -> list[chess.Move]:
    if "castle" not in t and "castling" not in t:
        return []
    side = None
    if "king" in t or "short" in t:
        side = "k"
    elif "queen" in t or "long" in t:
        side = "q"
    out = []
    for mv in board.legal_moves:
        if not board.is_castling(mv):
            continue
        if side is None \
           or (side == "k" and board.is_kingside_castling(mv)) \
           or (side == "q" and board.is_queenside_castling(mv)):
            out.append(mv)
    return out

def parse_move(text: str, board: chess.Board) -> list[chess.Move]:
    """Return the legal moves matching `text`, most-confident first.

    - exactly one element  -> unambiguous, just play it
    - several elements      -> ambiguous, ask the user to clarify
    - empty                 -> not understood / illegal, ask to repeat
    """
    exact = _try_exact(text, board)
    if exact is not None:
        return [exact]

    t = normalize_spoken(text)
    candidates: list[chess.Move] = []

    candidates.extend(_castling_candidates(t, board))

    # Detect a requested promotion piece ("promote to queen", or a trailing piece word).
    promo_type: int | None = None
    m = re.search(r"promot\w*\s+(?:to\s+)?(king|queen|rook|bishop|knight)", t)
    if m:
        promo_type = _PIECE_WORDS[m.group(1)]

    # Moving piece word (knight, queen, ...).
    piece_type: int | None = None
    for word, ptype in _PIECE_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            piece_type = ptype
            break

    squares = _SQUARE_RE.findall(t)

    if len(squares) >= 2:
        frm, to = squares[0], squares[1]
        for suffix in ("", "q", "r", "b", "n"):
            try:
                mv = chess.Move.from_uci(frm + to + suffix)
            except ValueError:
                continue
            if mv in board.legal_moves:
                candidates.append(mv)
    elif len(squares) == 1:
        to_sq = chess.parse_square(squares[0])
        moves_to = [mv for mv in board.legal_moves if mv.to_square == to_sq]
        if piece_type is not None and promo_type is None:
            by_mover = [mv for mv in moves_to
                        if (p := board.piece_at(mv.from_square)) and p.piece_type == piece_type]
            # If naming the piece empties the list but promotions exist, the word
            # was the promotion target (e.g. "knight on a8"), not the mover.
            candidates.extend(by_mover if by_mover else moves_to)
        else:
            candidates.extend(moves_to)

    return _resolve(candidates, board, promo_type or piece_type)


def _resolve(candidates: list[chess.Move], board: chess.Board,
             promo_pref: int | None) -> list[chess.Move]:
    """De-dupe, keep only legal moves, and collapse promotion ambiguity."""
    legal = []
    seen = set()
    for mv in candidates:
        if mv in board.legal_moves and mv.uci() not in seen:
            seen.add(mv.uci())
            legal.append(mv)

    promos = [m for m in legal if m.promotion]
    if promos and len(legal) == len(promos):
        # Every candidate is a promotion of the same pawn: pick the requested
        # piece, defaulting to a queen, so we return a single move.
        want = (
            promo_pref
            if promo_pref in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
            else _PROMO_DEFAULT
        )
        chosen = [m for m in promos if m.promotion == want]
        if chosen:
            return chosen[:1]
        return [m for m in promos if m.promotion == _PROMO_DEFAULT][:1] or promos[:1]
    return legal


def describe_candidates(moves: list[chess.Move], board: chess.Board) -> str:
    """Human-readable list of candidate moves for a clarification prompt."""
    return ", ".join(board.san(m) for m in moves)


def explain_move_failure(text: str, board: chess.Board) -> str:
    """Explain why a spoken move can't be played, suggesting alternatives where
    possible. Used when `parse_move` finds nothing legal, so the machine can say
    *why* instead of a bare "say it again"."""
    t = normalize_spoken(text)
    squares = _SQUARE_RE.findall(t)

    def _dests(sq: int) -> list[str]:
        return sorted(board.san(m) for m in board.legal_moves if m.from_square == sq)

    if len(squares) >= 2:
        frm, to = chess.parse_square(squares[0]), chess.parse_square(squares[1])
        piece = board.piece_at(frm)
        if piece is None:
            return f"There's no piece on {squares[0]} to move."
        if piece.color != board.turn:
            mover = "white" if board.turn == chess.WHITE else "black"
            owner = "white" if piece.color == chess.WHITE else "black"
            return f"It's {mover}'s move, but {squares[0]} holds {owner}'s piece."
        name = chess.piece_name(piece.piece_type)
        promo = (chess.QUEEN if piece.piece_type == chess.PAWN
                 and chess.square_rank(to) in (0, 7) else None)
        candidate = chess.Move(frm, to, promotion=promo)
        if board.is_pseudo_legal(candidate):
            reason = f"moving the {name} to {squares[1]} would leave the king in check"
        else:
            reason = f"the {name} on {squares[0]} can't reach {squares[1]}"
        dests = _dests(frm)
        if dests:
            return f"That move isn't legal: {reason}. It can go to {', '.join(dests)}."
        return f"That move isn't legal: {reason}, and it has nowhere legal to go."

    if len(squares) == 1:
        # If they named a piece, explain from that piece's point of view.
        piece_type = next((pt for word, pt in _PIECE_WORDS.items()
                           if re.search(rf"\b{word}\b", t)), None)
        if piece_type is not None:
            name = chess.piece_name(piece_type)
            owned = [sq for sq in chess.SQUARES
                     if (p := board.piece_at(sq)) is not None
                     and p.piece_type == piece_type and p.color == board.turn]
            if not owned:
                return f"You have no {name} in play to move."
            dests = sorted(board.san(m) for m in board.legal_moves if m.from_square in owned)
            if not dests:
                return f"Your {name} can't move anywhere right now."
            return f"No {name} can reach {squares[0]}. A {name} can go to {', '.join(dests)}."
        to = chess.parse_square(squares[0])
        reachers = sorted(board.san(m) for m in board.legal_moves if m.to_square == to)
        if not reachers:
            return f"Nothing can legally move to {squares[0]} right now. Say it again?"
        return f"Did you mean {', '.join(reachers)}?"

    return ("I couldn't read that as a move. Try the target square, like "
            "'e4', 'knight to f3', or 'e2 to e4'.")
