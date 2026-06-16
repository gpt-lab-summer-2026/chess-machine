import chess
import pytest

from chessmachine.nlu.move_parsing import parse_move, normalize_spoken


def ucis(text, board=None):
    return [m.uci() for m in parse_move(text, board or chess.Board())]


@pytest.mark.parametrize("text,expected", [
    ("e2e4", "e2e4"),
    ("e4", "e2e4"),
    ("Nf3", "g1f3"),
    ("knight to f3", "g1f3"),
    ("horse f3", "g1f3"),
    ("e two to e four", "e2e4"),
    ("e2 e4", "e2e4"),
])
def test_unambiguous_from_start(text, expected):
    assert ucis(text) == [expected]


def test_capture_phrasings():
    b = chess.Board()
    b.push_san("e4"); b.push_san("d5")
    assert ucis("pawn takes d5", b) == ["e4d5"]
    assert ucis("exd5", b) == ["e4d5"]


def test_castling():
    b = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    assert ucis("castle kingside", b) == ["e1g1"]
    assert ucis("castle queenside", b) == ["e1c1"]


def test_promotion_default_and_explicit():
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    assert ucis("a8", b) == ["a7a8q"]                 # defaults to queen
    assert ucis("a7 a8 rook", b) == ["a7a8r"]
    assert ucis("promote to knight on a8", b) == ["a7a8n"]


def test_ambiguous_returns_multiple():
    # two black knights (b1, f3) can both reach d2, black to move
    b = chess.Board("4k3/8/8/8/8/5n2/8/1n2K3 b - - 0 1")
    res = parse_move("knight to d2", b)
    assert {m.uci() for m in res} == {"b1d2", "f3d2"}


def test_nonsense_and_illegal_return_empty():
    assert ucis("fly to the moon") == []
    assert ucis("e2e5") == []          # illegal from the start position


def test_normalize_spoken():
    assert normalize_spoken("Knight to E, four") == "knight to e4"
    assert normalize_spoken("echo four") == "e4"
