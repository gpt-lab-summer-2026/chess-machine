"""Tests for chess_engine.analysis.

The move-quality and position-description logic needs an *evaluating* engine.
Rather than launch Stockfish, we inject a `FakeEngine` that returns scripted
`AnalysisResult`s, so every case is deterministic and hardware-free.
"""
import chess
import pytest

from chessmachine.chess_engine.analysis import (
    MoveQuality,
    _offers_material,
    classify_move_quality,
    describe_position,
    difficulty_tier,
    eval_verdict,
    facts_to_summary,
    find_tactics,
    game_phase,
    material_balance,
    move_comment_summary,
    should_comment,
)
from chessmachine.chess_engine.engine import AnalysisResult, ChessEngine


class FakeEngine(ChessEngine):
    """Engine stand-in returning scripted evaluations.

    `top` drives `top_moves` (the multi-PV used to judge a move); `after` is what
    `analyse` returns (the eval of the position *after* a non-top move is played).
    """

    def __init__(self, top=None, after=None):
        self._top = top or []
        self._after = after if after is not None else AnalysisResult()

    def set_difficulty(self, preset):  # pragma: no cover - unused
        pass

    def best_move(self, board):
        return self._top[0].best_move if self._top else None

    def analyse(self, board):
        return self._after

    def top_moves(self, board, n=2):
        return self._top[:n]

    @property
    def provides_evaluation(self):
        return True


def _after_move(fen: str, uci: str) -> chess.Board:
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(uci))
    return board


# --------------------------------------------------------------------------- #
# material / phase / verdict
# --------------------------------------------------------------------------- #
def test_material_balance_even_at_start():
    mat = material_balance(chess.Board())
    assert mat["white"] == mat["black"] == 39   # 8 + 6 + 6 + 10 + 9
    assert mat["diff"] == 0 and mat["leader"] == "even"


def test_material_balance_counts_a_missing_piece():
    board = chess.Board()
    board.remove_piece_at(chess.B8)             # take a black knight off
    mat = material_balance(board)
    assert mat["diff"] == 3 and mat["leader"] == "white"


@pytest.mark.parametrize("fen,expected", [
    (chess.STARTING_FEN, "opening"),
    ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 30", "endgame"),
    ("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 9", "middlegame"),
])
def test_game_phase(fen, expected):
    assert game_phase(chess.Board(fen)) == expected


@pytest.mark.parametrize("cp,mate,expect", [
    (None, None, "unclear"),
    (0, None, "roughly equal"),
    (150, None, "White is slightly better"),
    (-300, None, "Black is clearly better"),
    (600, None, "White is winning"),
    (None, 3, "White has a forced mate in 3"),
    (None, -2, "Black has a forced mate in 2"),
])
def test_eval_verdict(cp, mate, expect):
    assert expect in eval_verdict(cp, mate)


# --------------------------------------------------------------------------- #
# difficulty tiers / should_comment / deterministic phrasing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,tier", [
    ("easy", "easy"), ("hard", "hard"), ("medium", "medium"),
    ("1600 Elo", "medium"), ("800", "easy"), ("2400", "hard"),
])
def test_difficulty_tier(label, tier):
    assert difficulty_tier(label) == tier


def test_should_comment_scales_with_difficulty():
    blunder = MoveQuality("blunder")
    mistake = MoveQuality("mistake")
    normal = MoveQuality("normal")
    # hard: only the big swings
    assert should_comment(blunder, [], "hard")
    assert not should_comment(mistake, [], "hard")
    # easy: chatty — any motif counts, even on a 'normal' move
    assert should_comment(normal, ["the knight forks the king and queen"], "easy")
    # medium: quality swings and sharp motifs
    assert should_comment(mistake, [], "medium")
    assert not should_comment(normal, [], "medium")


def test_facts_to_summary_is_grounded():
    facts = {
        "verdict": "White is clearly better", "in_check": True, "turn": "black",
        "material": {"white": 39, "black": 36, "diff": 3, "leader": "white"},
        "best_move_san": "Qd5", "pv_sans": ["Qd5", "Kf8"], "game_over": False,
    }
    out = facts_to_summary(facts)
    assert "White is clearly better" in out
    assert "up 3 points" in out
    assert "Black is in check" in out
    assert "Qd5 Kf8" in out


def test_facts_to_summary_game_over_short_circuits():
    assert facts_to_summary({"game_over": True}) == "The game is over."


def test_move_comment_summary_includes_label_and_motif():
    out = move_comment_summary(
        MoveQuality("blunder"), ["the knight on e5 forks the king and the queen"], "easy"
    )
    assert "blunder" in out.lower()
    assert "forks" in out


# --------------------------------------------------------------------------- #
# _offers_material (the sacrifice heuristic)
# --------------------------------------------------------------------------- #
def test_offers_material_true_for_piece_en_prise():
    # Knight jumps to e5 where only a pawn (d6) attacks it and nothing defends.
    board = chess.Board("rnbqkb1r/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    assert _offers_material(board, chess.Move.from_uci("f3e5")) is True


def test_offers_material_false_for_safe_move():
    assert _offers_material(chess.Board(), chess.Move.from_uci("e2e4")) is False


# --------------------------------------------------------------------------- #
# find_tactics (motifs are computed, never invented)
# --------------------------------------------------------------------------- #
def test_find_tactics_fork():
    board = _after_move("3k1q2/8/8/8/5N2/8/8/4K3 w - - 0 1", "f4e6")
    motifs = find_tactics(board, chess.WHITE)
    assert any("forks" in m for m in motifs)


def test_find_tactics_hanging_piece():
    board = _after_move("k7/6r1/8/8/8/8/1B6/6K1 w - - 0 1", "g1f1")
    motifs = find_tactics(board, chess.WHITE)
    assert any("hanging" in m for m in motifs)


def test_find_tactics_check():
    board = _after_move("4k3/8/8/8/8/8/8/4R1K1 w - - 0 1", "e1e7")
    motifs = find_tactics(board, chess.WHITE)
    assert any("check" in m for m in motifs)


def test_find_tactics_outpost():
    board = _after_move("4k3/8/8/8/4PN2/8/8/4K3 w - - 0 1", "f4d5")
    motifs = find_tactics(board, chess.WHITE)
    assert any("outpost" in m for m in motifs)


def test_find_tactics_pin():
    board = _after_move("k7/8/4p3/3n4/8/8/8/4KB2 w - - 0 1", "f1g2")
    motifs = find_tactics(board, chess.WHITE)
    assert any("pinned" in m for m in motifs)


# --------------------------------------------------------------------------- #
# describe_position contract
# --------------------------------------------------------------------------- #
def test_describe_position_returns_the_full_facts_contract():
    eng = FakeEngine(after=AnalysisResult(
        score_cp=25, best_move=chess.Move.from_uci("e2e4"),
        pv=[chess.Move.from_uci("e2e4")],
    ))
    facts = describe_position(chess.Board(), eng)
    assert set(facts) == {
        "turn", "fullmove", "phase", "in_check", "legal_moves", "material",
        "score_cp", "mate_in", "verdict", "best_move_san", "pv_sans",
        "last_move_san", "game_over",
    }
    assert facts["turn"] == "white"
    assert facts["best_move_san"] == "e4"
    assert facts["material"]["diff"] == 0


# --------------------------------------------------------------------------- #
# classify_move_quality (label boundaries, via the fake engine)
# --------------------------------------------------------------------------- #
def test_classify_forced_when_one_legal_move():
    board = chess.Board("8/8/8/8/8/8/6q1/7K w - - 0 1")
    assert board.legal_moves.count() == 1
    move = next(iter(board.legal_moves))
    assert classify_move_quality(FakeEngine(), board, move).label == "forced"


def test_classify_best_move():
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]:
        board.push_san(san)
    move = board.parse_san("Ba4")
    eng = FakeEngine(top=[
        AnalysisResult(score_cp=80, best_move=move),
        AnalysisResult(score_cp=10, best_move=chess.Move.from_uci("e1g1")),
    ])
    assert classify_move_quality(eng, board, move).label == "best"


def test_classify_brilliant_sacrifice():
    board = chess.Board("rnbqkb1r/ppp1pppp/3p4/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    move = chess.Move.from_uci("f3e5")   # sound knight sac (best line)
    eng = FakeEngine(top=[
        AnalysisResult(score_cp=40, best_move=move),
        AnalysisResult(score_cp=30, best_move=chess.Move.from_uci("d2d4")),
    ])
    quality = classify_move_quality(eng, board, move)
    assert quality.label == "brilliant" and quality.is_sacrifice


def test_classify_blunder():
    board = chess.Board()
    move = chess.Move.from_uci("f2f3")   # not a top move; tanks the eval
    eng = FakeEngine(
        top=[
            AnalysisResult(score_cp=30, best_move=chess.Move.from_uci("e2e4")),
            AnalysisResult(score_cp=20, best_move=chess.Move.from_uci("d2d4")),
        ],
        after=AnalysisResult(score_cp=-320),   # White's POV after the blunder
    )
    assert classify_move_quality(eng, board, move).label == "blunder"
