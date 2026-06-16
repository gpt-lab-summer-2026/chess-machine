import chess
import pytest

from chessmachine.config import GeometryConfig, SpeedsConfig, MagnetConfig
from chessmachine.motion.geometry import BoardGeometry
from chessmachine.motion.mock import MockMotion
from chessmachine.motion.graveyard import Graveyard
from chessmachine.motion.choreography import Choreographer
from chessmachine.chess_engine.game import MoveKind


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


def test_normal_move_is_pick_then_place(rig):
    ctl, gy, ch, geo = rig
    r = ch.execute_move(chess.Board(), chess.Move.from_uci("e2e4"))
    assert r.kind == MoveKind.NORMAL and r.transfers == 1
    assert ctl.moves() == [(70.0, 25.0), (70.0, 55.0)]  # pick e2, place e4
    # magnet on between pick and place, off after
    assert ("magnet", True) in ctl.ops and ("magnet", False) in ctl.ops


def test_capture_routes_to_graveyard_first(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    r = ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert r.kind == MoveKind.CAPTURE and gy.occupied() == 1
    moves = ctl.moves()
    # first transfer takes the d5 pawn to the first graveyard slot
    assert moves[0] == (55.0, 70.0)
    assert moves[1] == (127.5, 10.0)


def test_castle_moves_king_then_rook(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    r = ch.execute_move(b, chess.Move.from_uci("e1g1"))
    assert r.kind == MoveKind.CASTLE and r.transfers == 2
    assert ctl.moves() == [(70.0, 10.0), (100.0, 10.0),   # king e1->g1
                           (115.0, 10.0), (85.0, 10.0)]    # rook h1->f1


def test_en_passant_removes_correct_pawn(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    r = ch.execute_move(b, chess.Move.from_uci("e5f6"))
    assert r.kind == MoveKind.EN_PASSANT and gy.occupied() == 1
    # captured pawn lifted from f5 (not f6)
    assert ctl.moves()[0] == (85.0, 70.0)


def test_promotion_without_spare_prompts_manual(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    r = ch.execute_move(b, chess.Move.from_uci("a7a8q"))
    assert r.kind == MoveKind.PROMOTION
    assert r.notes and "queen" in r.notes[0].lower()


def test_reverse_capture_restores_piece(rig):
    ctl, gy, ch, geo = rig
    b = chess.Board(); b.push_san("e4"); b.push_san("d5")
    ch.execute_move(b, chess.Move.from_uci("e4d5"))
    assert gy.occupied() == 1
    ch.reverse_move(b, chess.Move.from_uci("e4d5"))   # b = position before the move
    assert gy.occupied() == 0                          # captured pawn returned


def test_reset_clears_then_rebuilds_with_enough_storage():
    # A full reset needs 32 transient slots; give it a 4-column graveyard.
    ctl, gy, ch, geo = _rig(GeometryConfig(graveyard_columns=4))
    assert gy.capacity == 32
    b = chess.Board(); b.push_san("e4"); b.push_san("e5"); b.push_san("Nf3")
    notes = ch.setup_starting_position(b)
    assert notes == []
    assert gy.occupied() == 0                 # everything placed back on the board
    # 32 pieces cleared to storage + 32 placed back = 64 transfers = 128 moves
    assert len(ctl.moves()) == 128


def test_reset_with_default_16_slots_asks_for_manual():
    # The real hardware default (16 slots) can't stage a full 32-piece reset.
    ctl, gy, ch, geo = _rig()
    assert gy.capacity == 16
    notes = ch.setup_starting_position(chess.Board())
    assert notes and "storage" in notes[0].lower()
    assert ctl.moves() == []                  # nothing moved; reported instead
