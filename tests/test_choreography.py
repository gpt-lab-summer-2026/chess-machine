import chess
import pytest

from chessmachine.chess_engine.game import MoveKind
from chessmachine.config import GeometryConfig, MagnetConfig, SpeedsConfig
from chessmachine.motion.choreography import Choreographer
from chessmachine.motion.geometry import BoardGeometry
from chessmachine.motion.graveyard import Graveyard
from chessmachine.motion.mock import MockMotion


def _rig(geo_cfg=None):
    geo = BoardGeometry(geo_cfg or GeometryConfig())
    ctl = MockMotion()
    ctl.connect()
    gy = Graveyard(geo.graveyard_slots())
    ch = Choreographer(ctl, geo, gy, SpeedsConfig(settle_ms=0), MagnetConfig(settle_ms=0))
    return ctl, gy, ch, geo


@pytest.fixture
def rig():
    return _rig()


def _sq(geo, name):
    """A square's planar point as the mock records it (rounded x,z tuple)."""
    p = geo.name_to_point(name)
    return (round(p.x, 3), round(p.z, 3))


def _slot(geo, i):
    p = geo.graveyard_slots()[i]
    return (round(p.x, 3), round(p.z, 3))


def test_normal_move_is_pick_then_place(rig):
    ctl, gy, ch, geo = rig
    r = ch.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert r.kind == MoveKind.NORMAL and r.transfers == 1
    assert ctl.moves() == [_sq(geo, "e2"), _sq(geo, "e4")]  # pick e2, place e4
    # magnet on between pick and place, off after
    assert ("magnet", True) in ctl.ops and ("magnet", False) in ctl.ops


def test_capture_routes_to_graveyard_first(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    r = ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert r.kind == MoveKind.CAPTURE and gy.occupied() == 1
    moves = ctl.moves()
    # first transfer takes the d5 pawn to the first graveyard slot
    assert moves[0] == _sq(geo, "d5")
    assert moves[1] == _slot(geo, 0)


def test_castle_moves_king_then_rook(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    r = ch.execute_move(b, chess.Move.from_uci("e1g1"))
    assert r.kind == MoveKind.CASTLE and r.transfers == 2
    assert ctl.moves() == [_sq(geo, "e1"), _sq(geo, "g1"),   # king e1->g1
                           _sq(geo, "h1"), _sq(geo, "f1")]    # rook h1->f1


def test_en_passant_removes_correct_pawn(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    r = ch.execute_move(b, chess.Move.from_uci("e5f6"))
    assert r.kind == MoveKind.EN_PASSANT and gy.occupied() == 1
    # captured pawn lifted from f5 (not f6)
    assert ctl.moves()[0] == _sq(geo, "f5")


def test_promotion_without_spare_prompts_manual(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    r = ch.execute_move(b, chess.Move.from_uci("a7a8q"))
    assert r.kind == MoveKind.PROMOTION
    assert r.notes and "queen" in r.notes[0].lower()


def test_begin_move_discards_capture_before_the_body(rig):
    # The concurrency split: the captured piece is cleared synchronously, and
    # only `complete()` performs the capturing piece's own travel.
    ctl, gy, ch, geo = rig
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    complete, report = ch.begin_move(b, chess.Move.from_uci("e4d5"))
    assert gy.occupied() == 1                       # captured pawn already stored
    assert ctl.moves() == [_sq(geo, "d5"), _slot(geo, 0)]  # lift d5 -> graveyard, nothing else
    complete()
    assert ctl.moves()[2:] == [_sq(geo, "e4"), _sq(geo, "d5")]  # only now does e4 travel to d5


def test_begin_move_matches_execute_move(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    complete, _ = ch.begin_move(b, chess.Move.from_uci("e1g1"))
    complete()
    via_begin = ctl.moves()
    ctl2, _, ch2, _ = _rig()
    ch2.execute_move(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
                     chess.Move.from_uci("e1g1"))
    assert via_begin == ctl2.moves()                # identical actuation either path


def test_reverse_capture_restores_piece(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert gy.occupied() == 1
    ch.reverse_move(b, chess.Move.from_uci("e4d5"))   # b = position before the move
    assert gy.occupied() == 0                          # captured pawn returned


def test_reset_of_start_position_is_a_noop(rig):
    # Resetting a board that's already in the start position moves nothing.
    ctl, gy, ch, geo = rig
    notes = ch.setup_starting_position(chess.Board())
    assert notes == []
    assert ctl.moves() == []


def test_reset_uses_on_board_pieces_directly(rig):
    # The dish holds plenty: misplaced pieces go straight to their home squares
    # without being staged in storage.
    ctl, gy, ch, geo = rig
    assert gy.capacity == 30
    b = chess.Board(); b.push_san("e4"); b.push_san("e5"); b.push_san("Nf3")
    notes = ch.setup_starting_position(b)
    assert notes == []
    assert gy.occupied() == 0                  # nothing parked in storage
    # 3 displaced pieces (e4-pawn, e5-pawn, f3-knight) -> 3 transfers = 6 moves
    assert len(ctl.moves()) == 6


def test_reset_pulls_captured_pieces_back_from_the_graveyard(rig):
    ctl, gy, ch, geo = rig
    pre = chess.Board(); pre.push_san("e4"); pre.push_san("d5")
    ch.execute_move(pre, chess.Move.from_uci("e4d5"))   # captures the d5 pawn -> graveyard
    assert gy.occupied() == 1
    b = pre.copy(); b.push(chess.Move.from_uci("e4d5"))
    ctl.reset_log()
    notes = ch.setup_starting_position(b)
    assert notes == []
    assert gy.occupied() == 0                  # captured pawn returned to d7
    # white pawn d5 -> e2, black pawn graveyard -> d7 = 2 transfers = 4 moves
    assert len(ctl.moves()) == 4


def test_dish_graveyard_is_not_retrievable():
    # A dump dish piles pieces randomly, so retrieve() always fails: reversing a
    # capture can't fish the piece back out and asks for a manual placement instead.
    geo = BoardGeometry(GeometryConfig())
    ctl = MockMotion(); ctl.connect()
    gy = Graveyard(geo.graveyard_slots(), retrievable=False)
    ch = Choreographer(ctl, geo, gy, SpeedsConfig(settle_ms=0), MagnetConfig(settle_ms=0))
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert gy.occupied() == 1
    report = ch.reverse_move(b, chess.Move.from_uci("e4d5"))
    assert gy.occupied() == 1                       # NOT restored — can't retrieve from the pile
    assert report.notes and "couldn't find" in report.notes[0].lower()


def test_capture_with_full_storage_aborts_cleanly(rig):
    # Once storage is full, a capture must abort without touching the board so
    # the logical and physical positions can't drift apart.
    ctl, gy, ch, geo = rig
    for _ in range(gy.capacity):
        gy.store(chess.Piece(chess.PAWN, chess.WHITE))
    assert gy.free() == 0
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    ctl.reset_log()
    r = ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert r.aborted and r.transfers == 0
    assert ctl.moves() == []                  # nothing actuated
    assert r.notes and "storage" in r.notes[0].lower()
