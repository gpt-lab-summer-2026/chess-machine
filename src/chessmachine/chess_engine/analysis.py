"""Position analysis -> structured facts -> grounded English.

`describe_position` produces a dict of *facts* (material, evaluation, best line)
computed by python-chess + the engine. `facts_to_summary` renders them to a
deterministic sentence. The SLM may re-phrase the summary, but it is given these
facts as ground truth so it cannot fabricate an evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict

import chess

from .engine import AnalysisResult, ChessEngine

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class MaterialBalance(TypedDict):
    white: int           # summed piece values (kings excluded)
    black: int
    diff: int            # white - black (signed)
    leader: str          # "white" | "black" | "even"


class PositionFacts(TypedDict):
    """Ground-truth facts about a position (produced by `describe_position`).

    The contract between analysis and the NLU layer: the SLM and the
    deterministic fallbacks are handed these facts and must not invent others.
    """

    turn: str                    # "white" | "black"
    fullmove: int
    phase: str                   # "opening" | "middlegame" | "endgame"
    in_check: bool
    legal_moves: int
    material: MaterialBalance
    score_cp: int | None         # centipawns, White's POV (None if mate/unclear)
    mate_in: int | None          # +ve = White mates, -ve = Black mates
    verdict: str                 # human-readable evaluation
    best_move_san: str | None
    pv_sans: list[str]           # principal variation in SAN
    last_move_san: str | None
    game_over: bool


def material_balance(board: chess.Board) -> MaterialBalance:
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


def eval_verdict(score_cp: int | None, mate_in: int | None) -> str:
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
    last_move_san: str | None = None,
) -> PositionFacts:
    """Build a dict of ground-truth facts about the current position."""
    analysis: AnalysisResult = engine.analyse(board)
    material = material_balance(board)

    best_san = None
    if analysis.best_move and analysis.best_move in board.legal_moves:
        best_san = board.san(analysis.best_move)

    facts: PositionFacts = {
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


def facts_to_summary(facts: PositionFacts) -> str:
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

    best = facts["best_move_san"]
    if best:
        line = facts["pv_sans"] or [best]
        parts.append("Best line: " + " ".join(line) + ".")

    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Move quality (blunder / mistake / best / great / brilliant) and tactics.
#
# These power the machine's *proactive* commentary: after a move is played we
# judge it (vs the engine's best) and surface any grounded tactical motif, then
# the SLM phrases a one-liner scaled to the difficulty. Only noteworthy moves
# are spoken — normal, book, and forced moves stay silent.
# --------------------------------------------------------------------------- #
_MATE_CP = 100_000           # mate mapped onto a large centipawn magnitude
_BLUNDER_CP = 300            # centipawn loss thresholds (mover's POV)
_MISTAKE_CP = 120
_ONLY_MOVE_GAP = 150         # best is this much better than 2nd-best -> "great"
_CLEAR_BEST_GAP = 60         # clearly best (worth a nod) but not the only move
_BEST_TOLERANCE = 15         # treat near-zero loss as "played the best move"


@dataclass
class MoveQuality:
    label: str                      # blunder|mistake|best|great|brilliant|normal|forced
    cp_loss: int | None = None   # centipawns lost vs the engine's best move
    is_sacrifice: bool = False
    only_good_move: bool = False

    def noteworthy(self) -> bool:
        return self.label in {"blunder", "mistake", "best", "great", "brilliant"}


def _to_cp_mover(res: AnalysisResult | None, mover_white: bool) -> int | None:
    """Collapse an AnalysisResult (White's POV) to centipawns from the mover's POV."""
    if res is None:
        return None
    if res.mate_in is not None:
        cp = _MATE_CP - abs(res.mate_in) * 100
        signed = cp if res.mate_in > 0 else -cp
    elif res.score_cp is not None:
        signed = res.score_cp
    else:
        return None
    return signed if mover_white else -signed


def _offers_material(board_before: chess.Board, move: chess.Move) -> bool:
    """True if `move` puts material at risk (the basis for a 'sacrifice')."""
    piece = board_before.piece_at(move.from_square)
    if piece is None:
        return False
    pv = PIECE_VALUES[piece.piece_type]
    captured = board_before.piece_at(move.to_square)
    cap_val = PIECE_VALUES[captured.piece_type] if captured else 0
    if cap_val >= pv:                       # captured equal/greater: a trade or win, not a sac
        return False
    after = board_before.copy()
    after.push(move)
    mover = board_before.turn
    to = move.to_square
    defenders = after.attackers(mover, to)
    # A king can't capture a defended piece, so it isn't a real threat then.
    attacker_vals = []
    for s in after.attackers(not mover, to):
        attacker = after.piece_at(s)
        assert attacker is not None  # attacker squares are occupied by definition
        if not (attacker.piece_type == chess.KING and defenders):
            attacker_vals.append(PIECE_VALUES[attacker.piece_type])
    if not attacker_vals:
        return False
    min_attacker = min(attacker_vals)
    if not defenders:
        return pv >= 3 or min_attacker < pv     # left a real piece en prise
    return min_attacker < pv                     # attacked by something cheaper


def classify_move_quality(engine: ChessEngine, board_before: chess.Board,
                          move: chess.Move) -> MoveQuality:
    """Judge `move` against the engine's evaluation of `board_before`.

    Needs an evaluating engine (Stockfish); two analyses per move. Returns a
    `normal`/`forced` (silent) verdict when there's nothing worth saying.
    """
    if len(list(board_before.legal_moves)) <= 1:
        return MoveQuality("forced", 0)

    mover_white = board_before.turn == chess.WHITE
    tops = engine.top_moves(board_before, 2)
    if not tops or tops[0].best_move is None:
        return MoveQuality("normal", None)

    best_cp = _to_cp_mover(tops[0], mover_white)
    second_cp = _to_cp_mover(tops[1], mover_white) if len(tops) > 1 else None

    after = board_before.copy()
    after.push(move)
    if after.is_checkmate():
        played_cp: int | None = _MATE_CP
    else:
        # The multi-PV search already scored its lines; if the played move is one
        # of them, reuse that score rather than paying for a second search. The
        # engine's own move is always the top line, so this skips the extra
        # analysis on every machine move (and on any human move that was best).
        played_cp = next(
            (_to_cp_mover(r, mover_white) for r in tops if r.best_move == move),
            None,
        )
        if played_cp is None:
            played_cp = _to_cp_mover(engine.analyse(after), mover_white)
    if best_cp is None or played_cp is None:
        return MoveQuality("normal", None)

    cp_loss = max(0, best_cp - played_cp)
    gap = (best_cp - second_cp) if second_cp is not None else 0
    is_best = move == tops[0].best_move or cp_loss <= _BEST_TOLERANCE
    sac = _offers_material(board_before, move)
    only_good = gap >= _ONLY_MOVE_GAP

    if is_best:
        if sac and played_cp >= -30:                  # sound sacrifice -> brilliant
            return MoveQuality("brilliant", cp_loss, is_sacrifice=True, only_good_move=only_good)
        if only_good:
            return MoveQuality("great", cp_loss, only_good_move=True)
        if gap >= _CLEAR_BEST_GAP and board_before.fullmove_number > 3:
            return MoveQuality("best", cp_loss)        # clearly best; book moves stay quiet
        return MoveQuality("normal", cp_loss)          # many moves were fine -> nothing to say
    if cp_loss >= _BLUNDER_CP:
        return MoveQuality("blunder", cp_loss)
    if cp_loss >= _MISTAKE_CP:
        return MoveQuality("mistake", cp_loss)
    return MoveQuality("normal", cp_loss)


# --------------------------------------------------------------------------- #
# Tactical motifs — computed (never invented) so the SLM can name them safely.
# --------------------------------------------------------------------------- #
def _is_outpost(board: chess.Board, sq: int, color: bool) -> bool:
    """A knight on `sq` defended by a pawn that no enemy pawn can challenge."""
    rank, file = chess.square_rank(sq), chess.square_file(sq)
    if color == chess.WHITE and rank < 3:
        return False
    if color == chess.BLACK and rank > 4:
        return False
    back = rank - 1 if color == chess.WHITE else rank + 1
    defended = any(
        0 <= file + df <= 7 and 0 <= back <= 7
        and (p := board.piece_at(chess.square(file + df, back)))
        and p.color == color and p.piece_type == chess.PAWN
        for df in (-1, 1)
    )
    if not defended:
        return False
    ahead = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
    for df in (-1, 1):
        f = file + df
        if not 0 <= f <= 7:
            continue
        for r in ahead:
            p = board.piece_at(chess.square(f, r))
            if p and p.color != color and p.piece_type == chess.PAWN:
                return False
    return True


def _pin_against_king(board: chess.Board, from_sq: int, color: bool) -> int | None:
    """Square of an enemy piece the slider on `from_sq` pins to its king, if any."""
    king_sq = board.king(not color)
    if king_sq is None:
        return None
    for e in board.attacks(from_sq):
        p = board.piece_at(e)
        if not p or p.color == color or p.piece_type == chess.KING:
            continue
        if king_sq not in chess.SquareSet(chess.ray(from_sq, e)):
            continue
        if e not in chess.SquareSet(chess.between(from_sq, king_sq)):
            continue
        if any(board.piece_at(s) for s in chess.SquareSet(chess.between(e, king_sq))):
            continue
        return e
    return None


def find_tactics(board_after: chess.Board, mover: bool) -> list[str]:
    """Grounded tactical motifs the just-played move created for `mover`.

    `board_after` is the position after the move; `mover` is the side that moved.
    Returns short, factual phrases (e.g. "the knight on e5 forks ...").
    """
    motifs: list[str] = []
    opp = not mover
    last = board_after.peek() if board_after.move_stack else None
    to = last.to_square if last is not None else None
    mp = board_after.piece_at(to) if to is not None else None
    moved_ours = mp is not None and mp.color == mover

    def name(square: int) -> str:
        piece = board_after.piece_at(square)
        assert piece is not None  # name() is only called for occupied squares
        return f"the {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"

    # Fork: the moved piece attacks >= 2 valuable enemy pieces (or king + piece).
    if moved_ours:
        assert mp is not None and to is not None   # moved_ours implies both are set
        targets = [
            s for s in board_after.attacks(to)
            if (p := board_after.piece_at(s)) and p.color == opp
            and (PIECE_VALUES[p.piece_type] >= 3 or p.piece_type == chess.KING)
        ]
        if len(targets) >= 2:
            motifs.append(
                f"the {chess.piece_name(mp.piece_type)} on {chess.square_name(to)} "
                f"forks {' and '.join(name(s) for s in targets[:3])}"
            )

    # Most valuable hanging enemy piece (attacked by mover, undefended).
    hanging = [
        (PIECE_VALUES[p.piece_type], s, p)
        for s, p in board_after.piece_map().items()
        if p.color == opp and p.piece_type != chess.KING
        and PIECE_VALUES[p.piece_type] >= 3
        and board_after.attackers(mover, s) and not board_after.attackers(opp, s)
    ]
    if hanging:
        _, s, p = max(hanging)
        motifs.append(f"{name(s)} is hanging")

    if board_after.is_check() and not any("forks" in m for m in motifs):
        motifs.append("it gives check")

    if moved_ours:
        assert mp is not None and to is not None
        if mp.piece_type == chess.KNIGHT and _is_outpost(board_after, to, mover):
            motifs.append(f"the knight on {chess.square_name(to)} sits on an outpost")
        elif mp.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            pinned = _pin_against_king(board_after, to, mover)
            if pinned is not None:
                motifs.append(f"{name(pinned)} is pinned to the king")

    return motifs


# --------------------------------------------------------------------------- #
# Difficulty-scaled "should we speak, and what do we say" helpers.
# --------------------------------------------------------------------------- #
_LABEL_PHRASE = {
    "blunder": "That looks like a blunder.",
    "mistake": "That's a mistake.",
    "best": "That's the best move.",
    "great": "Great move — practically the only one that holds.",
    "brilliant": "Brilliant — a sacrifice that works.",
}
_TEACH = {
    "fork": "A fork is one piece attacking two at once.",
    "pinned": "A pinned piece can't move without exposing the king.",
    "outpost": "An outpost is a square a knight can't be kicked off by a pawn.",
    "hanging": "A hanging piece is undefended and free to take.",
}


def difficulty_tier(difficulty: str) -> str:
    """Collapse a difficulty label ('easy'/'1600 Elo'/...) to easy|medium|hard."""
    d = (difficulty or "").lower()
    if "easy" in d:
        return "easy"
    if "hard" in d:
        return "hard"
    if "medium" in d:
        return "medium"
    m = re.search(r"(\d{3,4})", d)
    if m:
        elo = int(m.group(1))
        return "easy" if elo < 1200 else "hard" if elo >= 2000 else "medium"
    return "medium"


def should_comment(quality: MoveQuality, motifs: list, difficulty: str) -> bool:
    """Decide whether a move is worth narrating at this difficulty.

    easy: chatty (any motif, plus every quality label). hard: terse (only the
    big swings). medium: quality swings, plus the sharp motifs (fork / hanging).
    """
    tier = difficulty_tier(difficulty)
    label = quality.label
    if tier == "easy":
        return label in {"blunder", "mistake", "best", "great", "brilliant"} or bool(motifs)
    if tier == "hard":
        return label in {"blunder", "brilliant"}
    strong_motif = any("fork" in m or "hanging" in m for m in motifs)
    return label in {"blunder", "mistake", "great", "brilliant"} or strong_motif


def _join_motifs(motifs: list) -> str:
    text = "; ".join(motifs)
    return text[0].upper() + text[1:] if text else ""


def move_comment_summary(quality: MoveQuality, motifs: list, difficulty: str,
                         mover: str = "", san: str = "") -> str:
    """Deterministic spoken comment (SLM-free fallback / rule-based backend)."""
    tier = difficulty_tier(difficulty)
    parts: list[str] = []
    phrase = _LABEL_PHRASE.get(quality.label)
    if phrase:
        parts.append(phrase)
    if motifs:
        parts.append(_join_motifs(motifs) + ".")
        if tier == "easy":
            for key, lesson in _TEACH.items():
                if any(key in m for m in motifs):
                    parts.append(lesson)
                    break
    return " ".join(p for p in parts if p).strip()
