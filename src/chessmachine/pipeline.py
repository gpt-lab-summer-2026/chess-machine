"""The orchestrator: turn an utterance into speech + (optionally) board motion.

    transcript --(NLU)--> Intent --(dispatch)--> {engine, choreographer} --> spoken text

This is the one place that holds game state and coordinates the subsystems. It
is transport-agnostic: `handle()` takes a string and returns the spoken reply,
so it can be driven by real speech, typed text, or tests identically.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import chess

from .config import Config, DifficultyPreset
from .chess_engine import GameState, ChessEngine, analysis
from .chess_engine.game import speak_san
from .motion import Choreographer
from .nlu import NLU, parse_move, describe_candidates
from .nlu.intents import Intent
from .voice.stt import STT
from .voice.tts import TTS

log = logging.getLogger(__name__)

HELP_TEXT = (
    "You can tell me your move, like 'knight to f3' or 'e2 to e4'. "
    "Say 'your move' for me to play, ask 'who's winning' or 'what's the best move' "
    "for analysis, set difficulty to easy, medium, or hard, or say 'new game'."
)
QUIT_WORDS = {"quit", "exit", "stop", "goodbye", "q"}


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
                    self._say("Something went wrong with that one. Let's try again.")
        finally:
            self.close()

    def close(self) -> None:
        for fn in (self._park, self.choreo.close, self.engine.close, self.nlu.close):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def _park(self) -> None:
        self.choreo.park()

    # -- top-level dispatch -------------------------------------------------- #
    def handle(self, transcript: str) -> str:
        intent = self.nlu.interpret(transcript, self._context())
        log.info("intent=%s move=%s diff=%s", intent.action, intent.move, intent.difficulty)
        return self._dispatch(intent)

    def _dispatch(self, intent: Intent) -> str:
        a = intent.action
        if a == "set_difficulty":
            return self._do_difficulty(intent)
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
        move = self.engine.best_move(self.game.board)
        if move is None:
            return self._say("I have no legal move to make.")
        return self._say(self._play_move(move, prefix))

    def _do_opponent_move(self, intent: Intent) -> str:
        board = self.game.board
        move_text = intent.move or intent.text or ""
        candidates = parse_move(move_text, board)
        if not candidates:
            return self._say("I couldn't read that move. Could you say it again?")
        if len(candidates) > 1:
            return self._say(f"That move is ambiguous — did you mean "
                             f"{describe_candidates(candidates, board)}?")
        spoken = self._say(self._play_move(candidates[0], "Okay,"))
        # Auto-reply with the engine's move if it's now our turn.
        if (self.cfg.app.auto_reply and not self.game.is_game_over()
                and self.game.is_machine_turn()):
            reply = self._do_engine_move("My move:")
            return spoken + " " + reply
        return spoken

    def _do_new_game(self) -> str:
        notes = self.choreo.setup_starting_position(self.game.board)
        self.game.reset()
        text = "New game. The board is reset."
        if notes:
            text += " " + " ".join(notes)
        spoken = self._say(text)
        if self.cfg.app.auto_reply and self.game.is_machine_turn():
            return spoken + " " + self._do_engine_move("I'll open with")
        return spoken

    def _do_undo(self) -> str:
        move = self.game.undo()
        if move is None:
            return self._say("There's no move to take back.")
        report = self.choreo.reverse_move(self.game.board, move)
        text = "Move taken back."
        if report.notes:
            text += " " + " ".join(report.notes)
        return self._say(text)

    # -- helpers ------------------------------------------------------------- #
    def _play_move(self, move: chess.Move, prefix: str) -> str:
        board_before = self.game.board.copy()
        report = self.choreo.execute_move(board_before, move)
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

    def _resolve_difficulty(self, name: str) -> Optional[tuple[str, DifficultyPreset]]:
        name = (name or "").strip().lower()
        if not name:
            return None
        presets = self.cfg.engine.presets
        if name in presets:
            return name, presets[name]
        m = re.search(r"(\d{3,4})", name)
        if m:
            elo = int(m.group(1))
            skill = max(0, min(20, round((elo - 800) / 110)))
            return f"{elo} Elo", DifficultyPreset(elo=elo, depth=16, movetime_ms=1000, skill=skill)
        return None

    def _last_san(self) -> Optional[str]:
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
