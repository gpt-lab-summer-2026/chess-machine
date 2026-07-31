"""End-to-end orchestration with everything mocked (no hardware, no models)."""
import chess

from chessmachine.chess_engine.engine import AnalysisResult, RandomEngine
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
    cfg.tts.keep_alive = False        # no real speaker here; don't spawn a pw-play stream
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


def test_illegal_move_is_explained():
    m, tts = make(auto_reply=False)
    m.handle("e2 to e5")          # illegal jump
    assert len(m.game.history) == 0
    spoken = " ".join(tts.lines).lower()
    assert "isn't legal" in spoken               # says why, not a bare re-ask
    assert "e3" in spoken and "e4" in spoken     # ... and offers the pawn's real moves


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
    m.handle("yes")              # confirm the take-back
    assert len(m.game.history) == 0


def test_undo_requires_confirmation():
    m, tts = make(auto_reply=False)
    m.handle("e4")
    tts.lines.clear()
    m.handle("undo")                       # should ASK, not take back
    assert len(m.game.history) == 1        # nothing undone yet
    assert any("yes" in l.lower() for l in tts.lines)
    m.handle("no, keep it")                # decline
    assert len(m.game.history) == 1        # move still there
    assert m.game.history[0][1] == "e4"
    m.handle("undo")
    m.handle("yes")                        # confirm this time
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


def test_set_side_uses_transcript_perspective_over_slm_color():
    # Small models flip the perspective: the user says "let me play black" (they
    # want black -> the MACHINE takes white), but the SLM fills color="black" (the
    # colour the user named). The literal transcript must win, so the machine
    # switches to white instead of no-opping on its current colour.
    m, _ = make(auto_reply=True, play_as="black")     # machine starts Black
    m.nlu.interpret = lambda t, c: [Intent("set_side", color="black", text="let me play black")]
    m.handle("let me play black")
    assert m.machine_color == chess.WHITE
    assert len(m.game.history) == 1                    # switched and opened as White


def test_undo_takes_back_full_move_pair():
    # With the machine auto-replying, "undo" should reverse BOTH its reply and
    # the user's move, handing the move back to the user.
    import chess
    m, _ = make(auto_reply=True)
    m.handle("e4")                       # user e4 + machine's auto-reply = 2 plies
    assert len(m.game.history) == 2
    m.handle("undo")
    m.handle("yes")                      # confirm the take-back
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
        return [Intent(action="opponent_move", move="a1a8", text="knight to f3")]

    m.nlu.interpret = fake_interpret
    m.handle("whatever whisper produced")
    assert m.game.history and m.game.history[0][1] == "Nf3"


def test_ambiguous_transcript_disambiguated_by_slm_move():
    # Two white knights (b5, f5) both reach d4, so the spoken "knight to d4" is
    # ambiguous on its own. The SLM's full-coordinate move picks one, and since
    # it's one of the spoken candidates we play it instead of re-asking.
    m, _ = make(auto_reply=False)                 # human is White, machine Black
    m.game.reset(start_fen="4k3/8/8/1N3N2/8/8/8/4K3 w - - 0 1")
    m.nlu.interpret = lambda t, c: [Intent("opponent_move", move="b5d4", text="knight to d4")]
    m.handle("knight to d4")
    assert m.game.history and m.game.history[-1][1] == "Nbd4"


def test_ambiguous_transcript_without_slm_help_still_asks():
    # Same ambiguous position, but the SLM offers no usable move -> we must still
    # ask the user to clarify rather than guessing.
    m, tts = make(auto_reply=False)
    m.game.reset(start_fen="4k3/8/8/1N3N2/8/8/8/4K3 w - - 0 1")
    m.nlu.interpret = lambda t, c: [Intent("opponent_move", move=None, text="knight to d4")]
    m.handle("knight to d4")
    assert not m.game.history
    assert any("ambiguous" in l.lower() for l in tts.lines)


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
    from chessmachine.chess_engine.game import MoveKind
    from chessmachine.motion.choreography import ExecutionReport

    m, tts = make(auto_reply=False)               # concurrent_actuation defaults True
    def boom():
        raise RuntimeError("motor jam")
    m.choreo.begin_move = lambda b, mv, mover_is_machine=False: (boom, ExecutionReport(kind=MoveKind.NORMAL))
    m.handle("e4")
    assert m.game.history[-1][1] == "e4"          # move applied logically (no rollback)
    assert any("check the board" in l.lower() for l in tts.lines)


def test_recalibrate_command_rehomes():
    m, _ = make(auto_reply=False)
    m.choreo.ctl.ops.clear()                      # drop the start() home + startup ops
    reply = m.handle("recalibrate the crane")
    assert ("home",) in m.choreo.ctl.ops          # re-homed on command
    assert "re-homed" in reply.lower()


def test_capture_triggers_rehome():
    m, _ = make(auto_reply=False)
    m.handle("e4")                                # white (human)
    m.handle("d5")                                # black
    m.choreo.ctl.ops.clear()
    m.handle("exd5")                              # white captures -> re-home
    assert m.game.history[-1][1] == "exd5"        # capture applied
    assert ("home",) in m.choreo.ctl.ops          # re-homed after the capture


def test_every_move_rehomes_by_default():
    m, _ = make(auto_reply=False)
    m.choreo.ctl.ops.clear()
    m.handle("e4")                                # quiet move
    assert ("home",) in m.choreo.ctl.ops          # re-homed after every move


def test_non_capture_move_does_not_rehome_when_after_move_off():
    m, _ = make(auto_reply=False)
    m.cfg.app.rehome_after_move = False           # fall back to capture-only
    m.choreo.ctl.ops.clear()
    m.handle("e4")                                # quiet move
    assert ("home",) not in m.choreo.ctl.ops


def test_rehome_on_capture_can_be_disabled():
    m, _ = make(auto_reply=False)
    m.cfg.app.rehome_after_move = False           # else every move would re-home
    m.cfg.app.rehome_on_capture = False
    m.handle("e4")
    m.handle("d5")
    m.choreo.ctl.ops.clear()
    m.handle("exd5")
    assert m.game.history[-1][1] == "exd5"
    assert ("home",) not in m.choreo.ctl.ops


def test_opponent_move_confirms_when_only_slm_resolves():
    m, _ = make(auto_reply=False)
    # Transcript lost its destination ("knight"); the SLM recovered a legal move.
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move="g1f3"))
    assert m._pending == "confirm_move:g1f3"
    assert len(m.game.history) == 0              # nothing played yet
    m.handle("yes")
    assert m.game.history[0][1] == "Nf3"


def test_opponent_move_declined_confirmation_plays_nothing():
    m, _ = make(auto_reply=False)
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move="g1f3"))
    m.handle("no")
    assert len(m.game.history) == 0
    assert m._pending is None


def test_opponent_move_ungrounded_without_slm_move_reasks():
    m, tts = make(auto_reply=False)
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move=None))
    assert m._pending is None
    assert len(m.game.history) == 0
    assert tts.lines                              # it said something (a re-ask)


def test_normal_move_still_plays_directly():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    assert m.game.history[0][1] == "e4"
    assert m._pending is None


def test_analyze_answers_piece_question_with_square_facts():
    # B: a piece/square question gets a grounded, square-specific answer, not a
    # generic whole-board summary. Machine plays black, so the human is white and
    # "my knight" resolves to white's knight.
    m, tts = make(auto_reply=False, play_as="black")
    m.game.board.set_fen("4k3/8/3p4/4N3/8/8/8/4K3 w - - 0 1")  # white Ne5 hanging to pd6
    m._do_analyze(Intent(action="analyze", question="what is threatening my knight"))
    said = " ".join(tts.lines).lower()
    assert "e5" in said and "hanging" in said


class _FixedEval:
    """Minimal evaluating engine for the deferred-self-critique tests: analyse()
    returns a fixed White-POV centipawn score regardless of position."""
    provides_evaluation = True

    def __init__(self, cp):
        self.cp = cp

    def analyse(self, board):
        return AnalysisResult(score_cp=self.cp)


def _pending_blunder():
    return {"info": {"label": "blunder", "motifs": [], "difficulty": "medium",
                     "mover_is_machine": True, "san": "Qd1"},
            "cp_after": -320}


def test_deferred_self_critique_spoken_when_punished():
    m, _ = make(auto_reply=False, play_as="white")   # machine = white
    m.engine = _FixedEval(-350)                        # machine still badly worse -> punished
    m._pending_self_critique = _pending_blunder()
    bb = chess.Board(); bb.push_san("e4")   # black (human) to move = the opponent's reply
    out = m._resolve_self_critique(bb)
    assert "blunder" in out.lower()
    assert m._pending_self_critique is None


def test_deferred_self_critique_dropped_when_not_punished():
    m, _ = make(auto_reply=False, play_as="white")
    m.engine = _FixedEval(60)                          # machine recovered -> opponent let it go
    m._pending_self_critique = _pending_blunder()
    bb = chess.Board(); bb.push_san("e4")
    assert m._resolve_self_critique(bb) == ""
    assert m._pending_self_critique is None


def test_deferred_self_critique_waits_on_machines_own_move():
    m, _ = make(auto_reply=False, play_as="white")
    m._pending_self_critique = _pending_blunder()
    bb = chess.Board()   # white (machine) to move -> not the opponent yet
    assert m._resolve_self_critique(bb) == ""
    assert m._pending_self_critique is not None        # still held, waiting for the human's reply


# -- sequential interaction (speak, THEN move) ------------------------------- #
def test_sequential_actuation_speaks_before_the_crane_moves():
    """Default flow is one-thing-at-a-time: the machine says the move fully before
    any head motion, so speech and the crane never overlap."""
    m, _ = make(auto_reply=False)
    assert m.cfg.app.concurrent_actuation is False        # the hardened default
    timeline: list[str] = []
    orig_say, orig_move = m.tts.say, m.choreo.ctl.move_xz
    m.tts.say = lambda t: (timeline.append("say"), orig_say(t))[1]
    m.choreo.ctl.move_xz = lambda *a, **k: (timeline.append("move"), orig_move(*a, **k))[1]
    m.handle("e4")                                        # a plain move: nothing to discard first
    assert "say" in timeline and "move" in timeline
    assert timeline.index("say") < timeline.index("move")  # spoke before moving


def test_promotion_note_spoken_after_the_piece_is_placed():
    """A promotion with no spare piece asks the human to swap it in; that note must
    still be spoken (now AFTER the crane places the pawn, not before)."""
    m, tts = make(auto_reply=False)
    m.game.reset(start_fen="4k3/P7/8/8/8/8/8/4K3 w - - 0 1")   # white pawn on a7
    m.handle("a7 to a8")
    assert m.game.history[-1][1].startswith("a8=Q")
    assert any("replace the pawn" in l.lower() for l in tts.lines)


def test_concurrent_actuation_still_available_as_opt_in():
    m, _ = make(auto_reply=False)
    m.cfg.app.concurrent_actuation = True
    m.handle("e4")
    assert m.game.history[-1][1] == "e4"                  # opt-in path still actuates + applies
    assert any(op[0] == "move" for op in m.choreo.ctl.ops)
