"""End-to-end orchestration with everything mocked (no hardware, no models)."""
import chess

from chessmachine.config import Config, SlmConfig
from chessmachine.pipeline import ChessMachine
from chessmachine.factory import build_choreographer
from chessmachine.nlu import create_nlu
from chessmachine.chess_engine.engine import RandomEngine
from chessmachine.voice.tts import TTS
from chessmachine.voice.stt import StdinSTT


class CaptureTTS(TTS):
    def __init__(self):
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)


def make(auto_reply=True, play_as="black"):
    cfg = Config()
    cfg.app.auto_reply = auto_reply
    cfg.app.play_as = play_as
    cfg.motion.backend = "mock"
    cfg.motion.speeds.settle_ms = 0
    cfg.motion.magnet.settle_ms = 0
    tts = CaptureTTS()
    machine = ChessMachine(
        config=cfg, stt=StdinSTT(), tts=tts,
        nlu=create_nlu(SlmConfig(backend="rule_based")),
        engine=RandomEngine(seed=1),
        choreographer=build_choreographer(cfg.motion),
    )
    machine.start(home=True)
    return machine, tts


def test_opponent_move_actuated_with_autoreply():
    m, _ = make(auto_reply=True)
    m.handle("e4")
    assert m.game.history[0][1] == "e4"
    assert len(m.game.history) == 2                       # engine auto-replied
    assert any(op[0] == "move" for op in m.choreo.ctl.ops)  # gantry moved
    assert m.choreo.ctl.homed                              # start() homed it


def test_no_autoreply_plays_single_ply():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    assert len(m.game.history) == 1


def test_illegal_move_asks_to_repeat():
    m, tts = make(auto_reply=False)
    m.handle("e2 to e5")          # illegal jump
    assert len(m.game.history) == 0
    assert any("again" in l.lower() for l in tts.lines)


def test_scholars_mate_announced():
    m, tts = make(auto_reply=False)
    for mv in ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]:
        m.handle(mv)
    assert m.game.is_game_over()
    assert "Checkmate" in m.game.result_text()
    assert any("checkmate" in l.lower() for l in tts.lines)


def test_difficulty_change():
    m, tts = make(auto_reply=False)
    m.handle("set difficulty to hard")
    assert m.difficulty == "hard"
    assert any("hard" in l.lower() for l in tts.lines)


def test_analysis_speaks_something():
    m, tts = make(auto_reply=False)
    before = len(tts.lines)
    m.handle("who is winning")
    assert len(tts.lines) > before


def test_new_game_resets_board():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    m.handle("e5")
    m.handle("new game")
    assert m.game.board == chess.Board()


def test_undo_takes_back():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    assert len(m.game.history) == 1
    m.handle("undo")
    assert len(m.game.history) == 0
