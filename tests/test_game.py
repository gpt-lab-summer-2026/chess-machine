import chess

from chessmachine.chess_engine.game import (
    GameState,
    MoveKind,
    classify_move,
    speak_san,
)


def test_classify_normal():
    c = classify_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert c.kind == MoveKind.NORMAL
    assert not c.is_capture and not c.is_castle


def test_classify_capture():
    b = chess.Board()
    b.push_san("e4"); b.push_san("d5")
    c = classify_move(b, chess.Move.from_uci("e4d5"))
    assert c.is_capture and c.captured_square == chess.D5
    assert c.kind == MoveKind.CAPTURE


def test_classify_en_passant():
    b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    c = classify_move(b, chess.Move.from_uci("e5f6"))
    assert c.is_en_passant and c.is_capture
    assert c.captured_square == chess.F5    # the captured pawn, not the landing square
    assert c.kind == MoveKind.EN_PASSANT


def test_classify_castle():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    c = classify_move(b, chess.Move.from_uci("e1g1"))
    assert c.is_castle and c.castle_side == "king"
    assert c.rook_from == chess.H1 and c.rook_to == chess.F1


def test_classify_promotion():
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    c = classify_move(b, chess.Move.from_uci("a7a8q"))
    assert c.promotion == chess.QUEEN
    assert c.kind == MoveKind.PROMOTION


def test_speak_san():
    assert speak_san("O-O") == "castles kingside"
    assert speak_san("O-O-O") == "castles queenside"
    assert "check" in speak_san("Qxh7+")
    assert "checkmate" in speak_san("Qxf7#")
    assert "knight" in speak_san("Nf3")
    assert "promote to queen" in speak_san("e8=Q")


def test_scholars_mate_result():
    g = GameState()
    for mv in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]:
        g.push(g.board.parse_san(mv))
    assert g.is_game_over()
    assert "Checkmate" in g.result_text()
    assert "White" in g.result_text()


def test_undo():
    g = GameState()
    g.push(g.board.parse_san("e4"))
    assert len(g.history) == 1
    g.undo()
    assert len(g.history) == 0
    assert g.board == chess.Board()
