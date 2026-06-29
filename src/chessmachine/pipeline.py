"""The orchestrator: turn an utterance into speech + (optionally) board motion.

    transcript --(NLU)--> Intent --(dispatch)--> {engine, choreographer} --> spoken text

This is the one place that holds game state and coordinates the subsystems. It
is transport-agnostic: `handle()` takes a string and returns the spoken reply,
so it can be driven by real speech, typed text, or tests identically.
"""
from __future__ import annotations

import contextlib
import logging
import re

import chess

from .chess_engine import ChessEngine, GameState, analysis
from .chess_engine.game import speak_san
from .config import Config, DifficultyPreset, preset_from_elo
from .motion import Choreographer
from .nlu import NLU, describe_candidates, parse_move
from .nlu.intents import Intent
from .voice.stt import STT
from .voice.tts import TTS

log = logging.getLogger(__name__)

HELP_TEXT = (
    "You can tell me your move, like 'knight to f3' or 'e2 to e4'. "
    "Say 'your move' for me to play, ask 'who's winning' or 'what's the best move' "
    "for analysis, set difficulty to easy, medium, or hard, take a move back, "
    "switch sides with 'let me play white', or say 'new game'."
)
QUIT_WORDS = {"quit", "exit", "stop", "goodbye", "q"}
_AFFIRM_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "affirmative", "please",
}


def _is_affirmative(text: str) -> bool:
    """True for a spoken 'yes' in answer to a confirmation prompt."""
    t = re.sub(r"[^a-z ]", "", text.lower())
    return bool(set(t.split()) & _AFFIRM_WORDS) or any(p in t for p in ("do it", "go ahead"))


class ChessMachine:
    def __init__(self, config: Config, stt: STT, tts: TTS, nlu: NLU,
                 engine: ChessEngine, choreographer: Choreographer):
        self.cfg = config
        self.stt = stt
        self.tts = tts
        self.nlu = nlu
        self.engine = engine
        self.choreo = choreographer

        self.machine_color = chess.WHITE if config.app.play_as.lower() == "white" else chess.BLACK
        self.game = GameState(machine_color=self.machine_color)
        self.difficulty = config.engine.default_difficulty
        self._last_spoken = ""
        self._pending: str | None = None   # a destructive action awaiting confirmation

    # -- lifecycle ----------------------------------------------------------- #
    def start(self, home: bool = True) -> None:
        self.choreo.connect()
        if home:
            self.choreo.home()
        self._say(f"Chess machine ready. I'm playing "
                  f"{GameState.color_name(self.machine_color)} at {self.difficulty} difficulty.")
        if self.cfg.app.auto_reply and self.game.is_machine_turn() and not self.game.is_game_over():
            self._do_engine_move("I'll open with")

    def run(self) -> None:
        try:
            while True:
                transcript = self.stt.listen()
                if not transcript:
                    continue
                if transcript.strip().lower() in QUIT_WORDS:
                    self._say("Goodbye.")
                    break
                try:
                    self.handle(transcript)
                except Exception:  # noqa: BLE001 - keep the loop alive
                    log.exception("Error handling utterance %r", transcript)
                    self._safe_park()   # release the magnet + lift if a piece was mid-move
                    self._say("Something went wrong handling that. If a piece was "
                              "mid-move, please check the board before we continue.")
        finally:
            self.close()

    def close(self) -> None:
        for fn in (self._park, self.choreo.close, self.engine.close, self.nlu.close):
            with contextlib.suppress(Exception):
                fn()

    def _park(self) -> None:
        self.choreo.park()

    def _safe_park(self) -> None:
        """Best-effort safe state after an error: lift the magnet and release it,
        so a half-finished move can't leave a piece stuck to the electromagnet."""
        try:
            self.choreo.park()
        except Exception:  # noqa: BLE001 - recovery must never raise
            log.exception("Failed to park after an error")

    # -- top-level dispatch -------------------------------------------------- #
    def handle(self, transcript: str) -> str:
        if self._pending is not None:
            return self._resolve_pending(transcript)
        intent = self.nlu.interpret(transcript, self._context())
        log.info("intent=%s move=%s diff=%s color=%s",
                 intent.action, intent.move, intent.difficulty, intent.color)
        return self._dispatch(intent)

    def _resolve_pending(self, transcript: str) -> str:
        """Resolve a yes/no answer to a pending confirmation (e.g. a new game)."""
        pending, self._pending = self._pending, None
        if _is_affirmative(transcript) and pending == "new_game":
            return self._really_new_game()
        return self._say("Okay, keeping the current game.")

    def _dispatch(self, intent: Intent) -> str:
        a = intent.action
        if a == "set_difficulty":
            return self._do_difficulty(intent)
        if a == "set_color":
            return self._do_set_color(intent)
        if a == "analyze":
            return self._do_analyze(intent)
        if a == "status":
            return self._do_status()
        if a == "engine_move":
            return self._do_engine_move("I'll play")
        if a == "opponent_move":
            return self._do_opponent_move(intent)
        if a == "new_game":
            return self._do_new_game()
        if a == "undo":
            return self._do_undo()
        if a == "resign":
            self.game.resign(not self.machine_color)   # the human (our opponent) resigns
            return self._say("You resigned. Good game! Say 'new game' to play again.")
        if a == "repeat":
            return self._say(self._last_spoken or "I haven't said anything yet.")
        if a == "help":
            return self._say(HELP_TEXT)
        if a == "chitchat":
            return self._say(self.nlu.small_talk(intent.text or "", self._context()))
        return self._say("Sorry, I didn't understand. Say a move, ask for analysis, "
                         "or change the difficulty.")

    # -- actions ------------------------------------------------------------- #
    def _do_difficulty(self, intent: Intent) -> str:
        resolved = self._resolve_difficulty(intent.difficulty or "")
        if resolved is None:
            return self._say("I didn't catch the difficulty. Try easy, medium, hard, "
                             "or an Elo number like 1500.")
        label, preset = resolved
        self.engine.set_difficulty(preset)
        self.difficulty = label
        return self._say(f"Difficulty set to {label}.")

    def _do_set_color(self, intent: Intent) -> str:
        want = (intent.color or intent.text or "").lower()
        if "white" in want:
            human = chess.WHITE
        elif "black" in want:
            human = chess.BLACK
        else:
            return self._say("Which colour would you like to play — white or black?")
        self.machine_color = not human
        self.game.machine_color = self.machine_color
        spoken = self._say(f"Okay, you play {GameState.color_name(human)}; "
                           f"I'll take {GameState.color_name(self.machine_color)}.")
        # If it's now my turn (e.g. I'm White on a fresh board), make my move.
        if (self.cfg.app.auto_reply and self.game.is_machine_turn()
                and not self.game.is_game_over()):
            return spoken + " " + self._do_engine_move("I'll open with")
        return spoken

    def _do_analyze(self, intent: Intent) -> str:
        facts = analysis.describe_position(self.game.board, self.engine, self._last_san())
        answer = self.nlu.phrase_analysis(intent.question or intent.text or "", facts)
        return self._say(answer)

    def _do_status(self) -> str:
        facts = analysis.describe_position(self.game.board, self.engine, self._last_san())
        turn = GameState.color_name(self.game.turn())
        return self._say(f"{turn} to move. {facts['verdict'].capitalize()}.")

    def _do_engine_move(self, prefix: str) -> str:
        if self.game.is_game_over():
            return self._say(self.game.result_text())
        if not self.game.is_machine_turn():
            return self._say("It's your move — tell me your move and I'll reply.")
        move = self.engine.best_move(self.game.board)
        if move is None:
            return self._say("I have no legal move to make.")
        return self._say(self._play_move(move, prefix))

    def _do_opponent_move(self, intent: Intent) -> str:
        if self.game.is_game_over():
            return self._say(self.game.result_text())
        board = self.game.board
        # Resolve from what was actually SAID first (deterministic, and filtered
        # against legal moves); only fall back to the SLM's guessed move if the
        # transcript yields nothing — a small model can hallucinate the wrong
        # square (e.g. "d5 takes e6" -> "d2e6").
        candidates = parse_move(intent.text or "", board)
        if not candidates and intent.move and intent.move != intent.text:
            candidates = parse_move(intent.move, board)
        if not candidates:
            return self._say("I couldn't read that move. Could you say it again?")
        if len(candidates) > 1:
            return self._say(f"That move is ambiguous — did you mean "
                             f"{describe_candidates(candidates, board)}?")
        spoken = self._say(self._play_move(candidates[0], "Okay,"))
        # Auto-reply with the engine's move if it's now our turn. Guard it so a
        # failure in our reply isn't reported as the human's move failing — their
        # move has already been played and announced.
        if (self.cfg.app.auto_reply and not self.game.is_game_over()
                and self.game.is_machine_turn()):
            try:
                reply = self._do_engine_move("My move:")
            except Exception:  # noqa: BLE001 - the opponent's move already stuck
                log.exception("Auto-reply failed after the opponent's move")
                self._safe_park()
                note = self._say("I couldn't make my reply just now; please check the board.")
                return spoken + " " + note
            return spoken + " " + reply
        return spoken

    def _do_new_game(self) -> str:
        # A reset discards a game in progress, so confirm before doing it.
        if self.game.board.move_stack:
            self._pending = "new_game"
            return self._say("Start a new game? That clears the current one. "
                             "Say yes to confirm.")
        return self._really_new_game()

    def _really_new_game(self) -> str:
        notes = self.choreo.setup_starting_position(self.game.board)
        self.game.reset()
        if notes:
            # Couldn't fully reset the pieces — don't claim it's done, and don't
            # start playing on a board the human still has to set up.
            return self._say("New game. " + " ".join(notes))
        spoken = self._say("New game. The board is reset.")
        if self.cfg.app.auto_reply and self.game.is_machine_turn():
            return spoken + " " + self._do_engine_move("I'll open with")
        return spoken

    def _do_undo(self) -> str:
        move = self.game.undo()
        if move is None:
            return self._say("There's no move to take back.")
        notes = list(self.choreo.reverse_move(self.game.board, move).notes)
        count = 1
        # With auto-reply, a turn is the opponent's move plus our reply. If we
        # just undid our reply, take the opponent's move back too so it's their
        # turn again rather than leaving the game mid-turn.
        if (self.cfg.app.auto_reply and self.game.is_machine_turn()
                and self.game.board.move_stack):
            move2 = self.game.undo()
            if move2 is not None:
                notes += self.choreo.reverse_move(self.game.board, move2).notes
                count = 2
        text = "Both moves taken back." if count == 2 else "Move taken back."
        if notes:
            text += " " + " ".join(notes)
        return self._say(text)

    # -- helpers ------------------------------------------------------------- #
    def _play_move(self, move: chess.Move, prefix: str) -> str:
        board_before = self.game.board.copy()
        report = self.choreo.execute_move(board_before, move)
        if report.aborted:
            # Nothing was actuated; keep the logical board in sync by not applying it.
            return " ".join(report.notes) or "I can't make that move right now."
        san = self.game.push(move)
        text = f"{prefix} {speak_san(san)}."
        if report.notes:
            text += " " + " ".join(report.notes)
        comment = self._move_comment(board_before, move, san)
        if comment:
            text += " " + comment
        if self.game.is_game_over():
            text += " " + self.game.result_text()
        return text

    def _move_comment(self, board_before: chess.Board, move: chess.Move, san: str) -> str:
        """Coach-style reaction to a noteworthy move (quality + grounded tactics).

        Scaled by difficulty; silent for normal/book/forced moves. Requires an
        evaluating engine (Stockfish) — a no-op for the dev RandomEngine.
        """
        if not getattr(self.engine, "provides_evaluation", False):
            return ""
        try:
            quality = analysis.classify_move_quality(self.engine, board_before, move)
            board_after = board_before.copy()
            board_after.push(move)
            mover = board_before.turn
            motifs = analysis.find_tactics(board_after, mover)
            if not analysis.should_comment(quality, motifs, self.difficulty):
                return ""
            info = {
                "label": quality.label,
                "cp_loss": quality.cp_loss,
                "is_sacrifice": quality.is_sacrifice,
                "only_good_move": quality.only_good_move,
                "motifs": motifs,
                "difficulty": self.difficulty,
                "tier": analysis.difficulty_tier(self.difficulty),
                "san": san,
                "mover": GameState.color_name(mover),
            }
            return self.nlu.comment_on_move(info)
        except Exception:  # noqa: BLE001 - commentary must never break a move
            log.exception("Move commentary failed")
            return ""

    def _resolve_difficulty(self, name: str) -> tuple[str, DifficultyPreset] | None:
        name = (name or "").strip().lower()
        if not name:
            return None
        presets = self.cfg.engine.presets
        if name in presets:
            return name, presets[name]
        m = re.search(r"(\d{3,4})", name)
        if m:
            elo = int(m.group(1))
            return f"{elo} Elo", preset_from_elo(elo)
        return None

    def _last_san(self) -> str | None:
        return self.game.history[-1][1] if self.game.history else None

    def _context(self) -> dict:
        return {
            "board": self.game.board,
            "machine_color": GameState.color_name(self.machine_color),
            "turn": GameState.color_name(self.game.turn()),
            "difficulty": self.difficulty,
            "last_san": self._last_san(),
        }

    def _say(self, text: str) -> str:
        self._last_spoken = text
        self.tts.say(text)
        return text
