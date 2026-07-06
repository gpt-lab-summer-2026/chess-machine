"""End-to-end orchestration with everything mocked (no hardware, no models)."""
import chess

from chessmachine.chess_engine.engine import RandomEngine
from chessmachine.config import Config, SlmConfig
from chessmachine.factory import build_choreographer
from chessmachine.nlu import create_nlu
from chessmachine.nlu.intents import Intent
from chessmachine.pipeline import ChessMachine
from chessmachine.voice.stt import StdinSTT
from chessmachine.voice.tts import TTS


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
    m.handle("yes")              # confirm the reset
    assert m.game.board == chess.Board()


def test_undo_takes_back():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    assert len(m.game.history) == 1
    m.handle("undo")
    assert len(m.game.history) == 0


def test_opponent_move_prefers_transcript_over_hallucinated_slm_move():
    # Playing Black, the SLM hallucinates a White move (illegal now) in `move`.
    # The user's literal transcript must win, so their real move still registers
    # and the machine auto-replies.
    from chessmachine.nlu.intents import Intent
    m, _ = make(auto_reply=True, play_as="white")   # machine opened as White
    assert m.game.turn() == chess.BLACK and len(m.game.history) == 1
    m.nlu.interpret = lambda t, c: [Intent("opponent_move", move="e2e4", text="e5")]
    m.handle("e5")
    assert m.game.history[1][1] == "e5"              # the real Black move was played
    assert len(m.game.history) == 3                  # ... and the machine auto-replied


def test_set_side_switches_color_midgame():
    # Default: machine plays black (user is white). Mid-game the user hands the
    # white side to the machine; it should adopt white AND move immediately.
    m, _ = make(auto_reply=True, play_as="black")
    assert m.machine_color == chess.BLACK
    m.handle("you take white")
    assert m.machine_color == chess.WHITE
    assert m.game.machine_color == chess.WHITE
    # it was white to move and the machine is now white -> it played a ply
    assert len(m.game.history) == 1


def test_switch_sides_swaps_and_takes_the_turn():
    # "switch sides" hands the side-to-move (White, on a fresh board) to the machine.
    m, _ = make(auto_reply=True, play_as="black")
    m.handle("switch sides")
    assert m.machine_color == chess.WHITE
    assert len(m.game.history) == 1       # machine took White's move


def test_compound_command_runs_each_clause():
    # "give me black and set difficulty to hard" must do BOTH, not drop one.
    m, _ = make(auto_reply=True, play_as="black")
    assert m.machine_color == chess.BLACK and m.difficulty == "medium"
    m.handle("give me black and set difficulty to hard")
    assert m.machine_color == chess.WHITE     # user wants black -> machine plays white
    assert m.difficulty == "hard"


def test_set_side_to_current_color_is_noop():
    m, tts = make(auto_reply=True, play_as="black")
    m.handle("you play black")            # already Black
    assert m.machine_color == chess.BLACK
    assert len(m.game.history) == 0       # nothing happened
    assert any("already" in l.lower() for l in tts.lines)


def test_undo_takes_back_full_move_pair():
    # With the machine auto-replying, "undo" should reverse BOTH its reply and
    # the user's move, handing the move back to the user.
    import chess
    m, _ = make(auto_reply=True)
    m.handle("e4")                       # user e4 + machine's auto-reply = 2 plies
    assert len(m.game.history) == 2
    m.handle("undo")
    assert len(m.game.history) == 0
    assert m.game.board == chess.Board()
    assert m.game.turn() == chess.WHITE   # user (White) is back on move


def test_engine_move_refuses_when_not_machine_turn():
    # Machine plays black; at the start it's the human's (white's) move.
    m, tts = make(auto_reply=False, play_as="black")
    before = len(m.game.history)
    m.handle("your move")
    assert len(m.game.history) == before      # didn't move out of turn
    assert any("your move" in l.lower() for l in tts.lines)


def test_resign_ends_game_and_blocks_further_moves():
    m, tts = make(auto_reply=False)
    m.handle("e4")
    m.handle("I resign")
    assert m.game.is_game_over()
    assert any("resign" in l.lower() for l in tts.lines)
    n = len(m.game.history)
    m.handle("e5")                            # game is over -> not applied
    assert len(m.game.history) == n


def test_capture_aborts_cleanly_when_storage_full():
    # Filling storage must not crash a capture or desync the board: the move is
    # simply refused with a spoken prompt to clear the captured pieces.
    m, tts = make(auto_reply=False)
    gy = m.choreo.graveyard
    for _ in range(gy.capacity):
        gy.store(chess.Piece(chess.PAWN, chess.WHITE))
    m.handle("e4"); m.handle("d5")
    n_before = len(m.game.history)
    tts.lines.clear()
    m.handle("exd5")
    assert len(m.game.history) == n_before    # capture NOT applied (no desync)
    assert any("storage" in l.lower() for l in tts.lines)


def test_new_game_clears_stale_storage():
    m, _ = make(auto_reply=False)
    gy = m.choreo.graveyard
    gy.store(chess.Piece(chess.PAWN, chess.WHITE))
    assert gy.occupied() == 1
    m.handle("new game")
    assert gy.occupied() == 0


def test_autoreply_failure_keeps_opponent_move_and_warns():
    # If the engine's auto-reply blows up, the human's move (already played and
    # announced) must stick, and we warn instead of making it look like the
    # human's move failed.
    m, tts = make(auto_reply=True, play_as="black")

    def boom(board):
        raise RuntimeError("engine crashed")

    m.engine.best_move = boom            # make the auto-reply fail
    tts.lines.clear()
    m.handle("e4")                       # human move succeeds; reply blows up
    assert [san for _, san in m.game.history] == ["e4"]   # human's move stuck, no reply
    spoken = " ".join(tts.lines).lower()
    assert "reply" in spoken and "check the board" in spoken


def test_new_game_requires_confirmation_midgame():
    m, tts = make(auto_reply=False)
    m.handle("e4")
    n = len(m.game.history)
    tts.lines.clear()
    m.handle("new game")                       # should ASK, not reset
    assert len(m.game.history) == n            # nothing reset yet
    assert any("yes" in line.lower() for line in tts.lines)
    m.handle("no, keep playing")               # decline
    assert len(m.game.history) == n            # still intact
    assert m.game.history[0][1] == "e4"


def test_set_color_switch_lets_machine_open():
    # Machine is Black and it's the human's (White's) move; asking to play Black
    # ourselves hands White to the machine, which then opens.
    m, _ = make(auto_reply=True, play_as="black")
    assert len(m.game.history) == 0
    m.handle("let me play black")
    assert m.machine_color == chess.WHITE
    assert len(m.game.history) == 1            # machine opened as White


def test_opponent_move_prefers_spoken_text_over_slm_guess():
    # If the SLM hallucinates a wrong/illegal move but the transcript is correct,
    # play what was actually said.
    m, _ = make(auto_reply=False)

    def fake_interpret(transcript, context):
        return Intent(action="opponent_move", move="a1a8", text="knight to f3")

    m.nlu.interpret = fake_interpret
    m.handle("whatever whisper produced")
    assert m.game.history and m.game.history[0][1] == "Nf3"


def test_difficulty_by_elo_number():
    # A numeric difficulty ("set difficulty to 1200") is synthesized via
    # preset_from_elo; regression for the missing import that made it crash.
    m, tts = make(auto_reply=False)
    m.handle("set difficulty to 1200")
    assert m.difficulty == "1200 Elo"
    assert any("1200" in l for l in tts.lines)


def test_actuation_failure_is_surfaced_not_swallowed():
    # If the crane fails mid-move on the worker thread, the machine must warn the
    # user (the logical move is already applied) instead of silently desyncing.
    from chessmachine.motion.choreography import ExecutionReport
    from chessmachine.chess_engine.game import MoveKind

    m, tts = make(auto_reply=False)               # concurrent_actuation defaults True
    def boom():
        raise RuntimeError("motor jam")
    m.choreo.begin_move = lambda b, mv: (boom, ExecutionReport(kind=MoveKind.NORMAL))
    m.handle("e4")
    assert m.game.history[-1][1] == "e4"          # move applied logically (no rollback)
    assert any("check the board" in l.lower() for l in tts.lines)
