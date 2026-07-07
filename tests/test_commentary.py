"""Move-quality classification, grounded tactics, and difficulty-scaled comments."""
import chess

from chessmachine.chess_engine.analysis import (
    MoveQuality,
    _offers_material,
    classify_move_quality,
    difficulty_tier,
    find_tactics,
    move_comment_summary,
    should_comment,
)
from chessmachine.chess_engine.engine import AnalysisResult, ChessEngine


# --- a stub engine with scripted evaluations (no Stockfish needed) ---------- #
class StubEngine(ChessEngine):
    """Returns canned multi-PV + post-move evaluation so we can test the bands."""

    def __init__(self, top, after_cp):
        self._top = top          # list[AnalysisResult], White's POV
        self._after = after_cp   # score_cp after the played move, White's POV

    def set_difficulty(self, preset):  # pragma: no cover - unused
        pass

    def best_move(self, board):
        return self._top[0].best_move if self._top else None

    def analyse(self, board):
        return AnalysisResult(score_cp=self._after)

    def top_moves(self, board, n=2):
        return self._top[:n]

    @property
    def provides_evaluation(self):
        return True


def _ar(cp, uci=None):
    mv = chess.Move.from_uci(uci) if uci else None
    return AnalysisResult(score_cp=cp, best_move=mv, pv=[mv] if mv else [])


# --- move quality ----------------------------------------------------------- #
def test_blunder_detected():
    board = chess.Board()
    eng = StubEngine([_ar(25, "e2e4"), _ar(20, "d2d4")], after_cp=-320)
    q = classify_move_quality(eng, board, chess.Move.from_uci("a2a3"))
    assert q.label == "blunder" and q.cp_loss >= 300


def test_great_when_only_move():
    board = chess.Board()
    eng = StubEngine([_ar(50, "e2e4"), _ar(-120, "d2d4")], after_cp=50)
    q = classify_move_quality(eng, board, chess.Move.from_uci("e2e4"))
    assert q.label == "great" and q.only_good_move


def test_clear_best_in_midgame():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")
    eng = StubEngine([_ar(40, "f1b5"), _ar(-40, "f1c4")], after_cp=40)
    q = classify_move_quality(eng, board, chess.Move.from_uci("f1b5"))
    assert q.label == "best"


def test_opening_best_move_stays_quiet():
    # e4 is the engine's pick but many moves are fine -> nothing worth saying.
    board = chess.Board()
    eng = StubEngine([_ar(40, "e2e4"), _ar(35, "d2d4")], after_cp=40)
    q = classify_move_quality(eng, board, chess.Move.from_uci("e2e4"))
    assert q.label == "normal" and not q.noteworthy()


def test_brilliant_sacrifice():
    board = chess.Board("4k3/8/2p1p3/8/8/8/8/3QK3 w - - 0 10")
    eng = StubEngine([_ar(20, "d1d5"), _ar(-30, "d1d3")], after_cp=20)
    q = classify_move_quality(eng, board, chess.Move.from_uci("d1d5"))   # Qd5, hit by ...cxd5/exd5
    assert q.label == "brilliant" and q.is_sacrifice


def test_forced_move_is_silent():
    # King must recapture the checking queen: only one legal move.
    board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 30")
    assert board.legal_moves.count() == 1
    eng = StubEngine([_ar(0, "e1e2")], after_cp=0)
    q = classify_move_quality(eng, board, chess.Move.from_uci("e1e2"))
    assert q.label == "forced" and not q.noteworthy()


# --- sacrifice heuristic ---------------------------------------------------- #
def test_offers_material_true_for_queen_into_pawn():
    board = chess.Board("4k3/8/2p1p3/8/8/8/8/3QK3 w - - 0 10")
    assert _offers_material(board, chess.Move.from_uci("d1d5"))


def test_offers_material_false_for_development():
    assert not _offers_material(chess.Board(), chess.Move.from_uci("g1f3"))


# --- tactics (computed, never invented) ------------------------------------- #
def test_find_fork():
    board = chess.Board("2r1k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    board.push(chess.Move.from_uci("b5d6"))      # Nd6 forks Rc8 and Ke8
    motifs = find_tactics(board, chess.WHITE)
    assert any("fork" in m for m in motifs)


def test_find_hanging():
    board = chess.Board("5k2/3n4/8/8/8/8/8/3RK3 b - - 0 1")  # Rd1 eyes an undefended Nd7
    motifs = find_tactics(board, chess.WHITE)
    assert any("hanging" in m for m in motifs)


def test_find_check():
    board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    board.push(chess.Move.from_uci("f1f8"))      # Qf8+ (no fork, no hang)
    motifs = find_tactics(board, chess.WHITE)
    assert "it gives check" in motifs


def test_find_outpost():
    board = chess.Board("4k3/8/8/8/3P4/3N4/8/4K3 w - - 0 1")
    board.push(chess.Move.from_uci("d3e5"))      # Ne5, pawn-supported, unkickable
    motifs = find_tactics(board, chess.WHITE)
    assert any("outpost" in m for m in motifs)


def test_find_pin():
    board = chess.Board("4k3/3n4/8/8/8/8/8/4KB2 w - - 0 1")
    board.push(chess.Move.from_uci("f1b5"))      # Bb5 pins Nd7 to Ke8
    motifs = find_tactics(board, chess.WHITE)
    assert any("pinned" in m for m in motifs)


# --- difficulty scaling ----------------------------------------------------- #
def test_difficulty_tier_mapping():
    assert difficulty_tier("easy") == "easy"
    assert difficulty_tier("1600 Elo") == "medium"
    assert difficulty_tier("2400 Elo") == "hard"


def test_hard_only_speaks_for_blunders_and_brilliancies():
    assert should_comment(MoveQuality("blunder", 400), [], "hard")
    assert should_comment(MoveQuality("brilliant", 0, is_sacrifice=True), [], "hard")
    assert not should_comment(MoveQuality("best", 0), [], "hard")
    assert not should_comment(MoveQuality("mistake", 150), [], "hard")


def test_medium_speaks_on_sharp_motif_but_not_quiet_positional_ones():
    fork = ["the knight on e5 forks the rook on c7 and the king on g8"]
    assert should_comment(MoveQuality("normal", 0), fork, "medium")          # fork -> speak
    outpost = ["the knight on e5 sits on an outpost"]
    assert not should_comment(MoveQuality("normal", 0), outpost, "medium")   # positional -> quiet
    assert should_comment(MoveQuality("normal", 0), outpost, "easy")         # but easy is chatty


def test_easy_teaches_the_term():
    out = move_comment_summary(
        MoveQuality("brilliant", 0, is_sacrifice=True),
        ["the knight on e5 forks the rook on c7 and the queen on g6"],
        difficulty="easy",
    )
    assert "fork" in out.lower() and "two" in out.lower()   # the teaching line fired
