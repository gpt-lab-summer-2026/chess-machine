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
import threading
import time

import chess

from .chess_engine import ChessEngine, GameState, analysis
from .chess_engine.game import speak_san
from .clock import MatchClock
from .config import Config, DifficultyPreset, preset_from_elo
from .motion import Choreographer
from .nlu import (
    NLU,
    describe_candidates,
    explain_move_failure,
    parse_move,
    question_target_square,
    resolve_side_request,
)
from .nlu.intents import Intent
from .voice.stt import STT
from .voice.tts import TTS

log = logging.getLogger(__name__)

HELP_TEXT = (
    "You can tell me your move, like 'knight to f3' or 'e2 to e4'. "
    "Say 'your move' for me to play, ask 'who's winning' or 'what's the best move' "
    "for analysis, set difficulty to easy, medium, or hard, say 'switch sides' or "
    "'I'll play black' to change colors, say 'recalibrate' if my placement drifts, "
    "or say 'new game'."
)
QUIT_WORDS = {"quit", "exit", "stop", "goodbye", "q"}
# A real listen window blocks on the mic for up to audio.vad.max_utterance_s, so an
# empty one returning faster than this means the capture path failed rather than
# simply hearing nothing.
_FAST_EMPTY_WINDOW_S = 1.0
_EMPTY_WINDOW_BACKOFF_S = 1.0
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
        self._moves_since_home = 0         # finished machine moves since the last re-home (cadence)
        self._last_prompt_ply = -1         # ply we last spoke "Your move." at (one cue per human turn)
        # Switch the status LED to "thinking" the instant the mic closes, rather than
        # after whisper finishes (see DistilWhisperSTT.on_capture_done).
        if hasattr(stt, "on_capture_done"):
            stt.on_capture_done = lambda: self._led("blink")
        # A blunder/mistake the MACHINE just made, held back until we see whether
        # the opponent punishes it (C: only self-critique if it actually costs).
        self._pending_self_critique: dict | None = None
        self.clock = MatchClock()

    # -- lifecycle ----------------------------------------------------------- #
    def start(self, home: bool = True) -> None:
        # FIRST thing, before anything is spoken: wake the speaker's amplifier and
        # keep it awake. Homing and the SLM warm-up below take seconds, so by the
        # time we say "Chess machine ready" the amp is long since up and the
        # leading "Ch" survives.
        self._start_keep_alive()
        self.choreo.connect()
        if home:
            self.choreo.home()
        # Pay the SLM cold-start now (model load + prompt-prefix cache + grammar),
        # so the user's FIRST move doesn't race a cold server and time out. Passing
        # the live context makes the warm-up's cached prefix match the real one.
        log.info("Warming up the language model...")
        self.nlu.warmup(self._context())
        self._say(f"Chess machine ready. I'm playing "
                  f"{GameState.color_name(self.machine_color)} at {self.difficulty} difficulty.")
        if self.cfg.app.auto_reply and self.game.is_machine_turn() and not self.game.is_game_over():
            self._do_engine_move("I'll open with")

    def run(self) -> None:
        empty_windows = 0
        try:
            while True:
                window_start = time.monotonic()
                # Cue "Your move." right before we open the mic, so the prompt
                # finishes speaking just as recording begins (once per human turn).
                self._prompt_move_if_new_turn()
                # The user's clock runs only while we wait for their input — all
                # machine work (STT/SLM/actuation/speech) is off their clock.
                self.clock.start_user()
                self._led("on")               # solid = mic is open, speak now
                self._cue_listening()         # + an audible earcon the instant we open the mic
                transcript = self.stt.listen()  # on_capture_done -> "blink" mid-call
                self._led("blink")            # (idempotent) blinking = working on it
                self.clock.stop_user()
                if not transcript:
                    # A healthy silent window already cost ~25 s of blocking mic
                    # reads, so re-entering at once is fine. A BROKEN mic, though,
                    # returns instantly -- and this loop had no backoff, so it would
                    # spin the open/read/close cycle at 100% CPU forever. Throttle
                    # once the empty windows start coming back too fast to be real.
                    empty_windows += 1
                    if time.monotonic() - window_start < _FAST_EMPTY_WINDOW_S:
                        if empty_windows in (3, 30) or empty_windows % 300 == 0:
                            log.warning("%d empty mic windows, each under %.1fs -- is the "
                                        "microphone working? Backing off.",
                                        empty_windows, _FAST_EMPTY_WINDOW_S)
                        time.sleep(_EMPTY_WINDOW_BACKOFF_S)
                    continue
                empty_windows = 0
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
        for fn in (self._stop_keep_alive, self._park, self.choreo.close,
                   self.engine.close, self.nlu.close):
            with contextlib.suppress(Exception):
                fn()

    def _start_keep_alive(self) -> None:
        """Hold the Bluetooth speaker's amp awake for the whole session, so no
        utterance loses its first syllable. Best-effort: no pw-play (dev boxes) or
        no BT speaker simply means no keep-alive."""
        self._keep_alive = None
        # Nothing to keep awake unless we actually play audio through a speaker:
        # --dev/stdout TTS and the test fixtures must not spawn a pw-play stream.
        if not self.cfg.tts.keep_alive or self.cfg.tts.backend != "kokoro":
            return
        try:
            from .voice.audio import SpeakerKeepAlive
            ka = SpeakerKeepAlive(level=self.cfg.tts.keep_alive_level)
            if ka.start():
                self._keep_alive = ka
        except Exception:  # noqa: BLE001 - never block startup on a comfort stream
            log.debug("speaker keep-alive failed to start", exc_info=True)

    def _stop_keep_alive(self) -> None:
        ka, self._keep_alive = getattr(self, "_keep_alive", None), None
        if ka is not None:
            ka.stop()

    def _park(self) -> None:
        self.choreo.park()

    def _safe_park(self) -> None:
        """Best-effort safe state after an error: lift the magnet and release it,
        so a half-finished move can't leave a piece stuck to the electromagnet."""
        try:
            self.choreo.park()
        except Exception:  # noqa: BLE001 - recovery must never raise
            log.exception("Failed to park after an error")

    def _led(self, mode: str) -> None:
        """Set the status LED: 'on' = listening, 'blink' = thinking, 'off' = moving.

        Best-effort by design — backends without an LED (mock, relay) simply have no
        `led()`, and the serial one swallows unsupported firmware. A status light must
        never be able to interrupt a game.
        """
        fn = getattr(getattr(self.choreo, "ctl", None), "led", None)
        if fn is None:
            return
        try:
            fn(mode)
        except Exception:  # noqa: BLE001 - indicator only
            log.debug("status LED %s failed", mode, exc_info=True)

    def _cue_listening(self) -> None:
        """Play a short earcon the moment the mic opens, so the user has an
        unmistakable 'speak now' signal (the spoken prompt + LED were easy to
        miss). Best-effort: a cue must never break the loop, and it no-ops where
        there is no audio device (dev/CI) or when disabled in config."""
        if not self.cfg.audio.listen_beep:
            return
        try:
            from .voice.audio import make_beep, play
            sr = self.cfg.tts.sample_rate
            # The prompt was just spoken, so the amp is warm -> a tiny lead-in is
            # enough; then a short gap lets the (Bluetooth-buffered) beep finish
            # emitting before we record, so the mic doesn't capture its own echo.
            play(make_beep(sr), sr, lead_in_s=0.08)
            time.sleep(0.2)
        except Exception:  # noqa: BLE001 - a cue must never interrupt a game
            log.debug("listening cue failed", exc_info=True)

    def _prompt_move_if_new_turn(self) -> None:
        """Speak "Your move." once at the start of each human turn, right before
        we open the mic, so the cue finishes just as recording begins. Skipped
        when a yes/no confirmation is pending, the game is over, or it's the
        machine's turn; and only once per turn, so back-to-back questions during
        the same turn don't re-trigger it."""
        if (self._pending is not None or self.game.is_game_over()
                or self.game.is_machine_turn()):
            return
        ply = len(self.game.board.move_stack)
        if ply == self._last_prompt_ply:
            return
        self._last_prompt_ply = ply
        self._say("Your move.")

    # -- top-level dispatch -------------------------------------------------- #
    def handle(self, transcript: str) -> str:
        """Dispatch an utterance. The NLU turns it into one or more structured
        intents (the SLM does the language work, incl. splitting a compound
        request like "give me black and set difficulty to hard"); we just run
        each in order and join the spoken replies."""
        # A pending confirmation (e.g. "new game" mid-game) intercepts the next
        # utterance as a yes/no answer before any NLU dispatch.
        if self._pending:
            return self._resolve_pending(transcript)
        replies = []
        for intent in self.nlu.interpret(transcript, self._context()):
            log.info("intent=%s move=%s diff=%s color=%s", intent.action,
                     intent.move, intent.difficulty, intent.color)
            reply = self._dispatch(intent)
            if reply:
                replies.append(reply)
        return " ".join(replies)

    def _resolve_pending(self, transcript: str) -> str:
        """Resolve a yes/no answer to a pending confirmation (new game, undo, or
        a low-confidence move recovered from the SLM)."""
        pending, self._pending = self._pending, None
        if pending and pending.startswith("confirm_move:"):
            if not _is_affirmative(transcript):
                return self._say("Okay, ignoring that. Say your move again?")
            move = chess.Move.from_uci(pending.split(":", 1)[1])
            if move not in self.game.board.legal_moves:
                return self._say("That move isn't legal now; say it again?")
            return self._play_opponent_move(move)
        if _is_affirmative(transcript):
            if pending == "undo":
                return self._really_undo()
            return self._really_new_game()          # pending == "new_game"
        return self._say("Okay, keeping the move." if pending == "undo"
                         else "Okay, keeping the current game.")

    def _dispatch(self, intent: Intent) -> str:
        a = intent.action
        if a == "set_difficulty":
            return self._do_difficulty(intent)
        if a == "set_side":
            return self._do_set_side(intent)
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
        if a == "recalibrate":
            return self._do_recalibrate()
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

    def _do_set_side(self, intent: Intent) -> str:
        """Change which color the machine plays, keeping the current position.

        If the switch puts the machine on move, it plays immediately (auto-reply),
        so "you take white from here" hands the turn straight over.
        """
        # Resolve the machine's target colour from what the user actually said —
        # small models routinely flip the perspective on side requests (dropping
        # the colour the *user* wants where the *machine's* colour belongs, which
        # lands on the current colour and silently no-ops). Fall back to the
        # SLM's `color` only when the words don't clearly resolve.
        side = resolve_side_request(intent.text or "")
        if side == "swap":
            new_color = not self.machine_color
        elif side in ("white", "black"):
            new_color = chess.WHITE if side == "white" else chess.BLACK
        else:
            color = (intent.color or "").strip().lower()
            if color in ("white", "black"):
                new_color = chess.WHITE if color == "white" else chess.BLACK
            else:
                new_color = not self.machine_color    # no/blank color => swap sides
        if new_color == self.machine_color:
            return self._say(f"I'm already playing {GameState.color_name(new_color)}.")

        self.machine_color = new_color
        self.game.machine_color = new_color
        spoken = self._say(f"Okay, I'll play {GameState.color_name(new_color)} now. "
                           f"You're {GameState.color_name(not new_color)}.")
        if (self.cfg.app.auto_reply and not self.game.is_game_over()
                and self.game.is_machine_turn()):
            return spoken + " " + self._do_engine_move("My move:")
        return spoken

    def _do_analyze(self, intent: Intent) -> str:
        board = self.game.board
        question = intent.question or intent.text or ""
        facts = analysis.describe_position(board, self.engine, self._last_san())
        # If the question is about a specific piece/square ("is my knight on c4
        # good", "what threatens my knight"), attach that square's grounded facts
        # so the answer is specific instead of a generic whole-board summary.
        # "my" resolves to the human's side; the helper flips it on "your".
        target = question_target_square(question, board, prefer_color=not self.machine_color)
        if target is not None:
            facts["square"] = analysis.describe_square(board, target)
        answer = self.nlu.phrase_analysis(question, facts)
        return self._say(answer)

    def _do_status(self) -> str:
        facts = analysis.describe_position(self.game.board, self.engine, self._last_san())
        turn = GameState.color_name(self.game.turn())
        text = f"{turn} to move. {facts['verdict'].capitalize()}."
        if self.cfg.app.match_clock and self.clock.user_seconds >= 1:
            text += f" Your clock shows {self.clock.spoken()}."
        return self._say(text)

    def _do_engine_move(self, prefix: str) -> str:
        if self.game.is_game_over():
            return self._say(self.game.result_text())
        if not self.game.is_machine_turn():
            return self._say("It's your move — tell me your move and I'll reply.")
        move = self.engine.best_move(self.game.board)
        if move is None:
            return self._say("I have no legal move to make.")
        return self._play_move(move, prefix)   # _play_move speaks internally

    def _do_opponent_move(self, intent: Intent) -> str:
        if self.game.is_game_over():
            return self._say(self.game.result_text())
        board = self.game.board
        # Ground the move in what the user LITERALLY said. The SLM's `move` field
        # is only a tie-breaker among these candidates, never a source on its own,
        # so a mangled transcript or a hallucinated move can't actuate a piece.
        candidates = parse_move(intent.text or "", board)
        if not candidates:
            # Literal words didn't resolve. If the SLM proposed a single legal
            # move, CONFIRM it (STT may have dropped a word) rather than guessing.
            slm_moves = parse_move(intent.move, board) if intent.move else []
            if len(slm_moves) == 1:
                self._pending = f"confirm_move:{slm_moves[0].uci()}"
                return self._say(f"Did you mean {board.san(slm_moves[0])}? Say yes.")
            return self._say(explain_move_failure(intent.text or intent.move or "", board))
        # Ambiguous literal words: let the SLM's move break the tie, but only when
        # it resolves to exactly one of the candidates we already found.
        if len(candidates) > 1 and intent.move:
            slm_moves = parse_move(intent.move, board)
            if len(slm_moves) == 1 and slm_moves[0] in candidates:
                candidates = slm_moves
        if len(candidates) > 1:
            return self._say(f"ambiguous move: did you mean "
                             f"{describe_candidates(candidates, board)}?")
        return self._play_opponent_move(candidates[0])

    def _play_opponent_move(self, move: chess.Move) -> str:
        """Actuate the human's move, then auto-reply with the engine if it's our turn."""
        spoken = self._play_move(move, "Okay,")   # _play_move speaks internally
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
        # A take-back discards the position, so confirm first — like a new game.
        if not self.game.board.move_stack:
            return self._say("There's no move to take back.")
        self._pending = "undo"
        return self._say("Take back the last move? Say yes to confirm.")

    def _really_undo(self) -> str:
        first = self.game.undo()
        if first is None:
            return self._say("There's no move to take back.")
        # After the pop, board.turn is the colour that made `first`.
        first_by_machine = self.game.board.turn == self.game.machine_color
        notes = list(self.choreo.reverse_move(self.game.board, first, first_by_machine).notes)
        count = 1
        # With auto-reply, the move just popped was the machine's reply, so it is
        # now the machine's turn again — take back the user's move too, handing
        # the turn back to them. (If the popped move was the user's own, it is
        # already their turn and we stop at one.)
        if (self.cfg.app.auto_reply and self.game.board.move_stack
                and self.game.is_machine_turn()):
            second = self.game.undo()
            if second is not None:
                second_by_machine = self.game.board.turn == self.game.machine_color
                notes += self.choreo.reverse_move(self.game.board, second, second_by_machine).notes
                count += 1
        text = "Move taken back." if count == 1 else "Moves taken back."
        if notes:
            text += " " + " ".join(notes)
        return self._say(text)

    def _do_recalibrate(self) -> str:
        """Voice-triggered re-home to correct accumulated open-loop drift."""
        err = self._rehome()
        return err or self._say("Re-homed and re-centered.")

    # -- helpers ------------------------------------------------------------- #
    def _rehome(self) -> str:
        """Re-home the crane to zero out open-loop drift. Safe mid-game: it only
        re-references the head against the endstops (no piece is touched). Returns
        "" on success, or an already-spoken error line on failure."""
        try:
            self.choreo.home()
            self._moves_since_home = 0
            return ""
        except Exception:  # noqa: BLE001 - a failed re-home must not abort the turn
            log.exception("Re-home failed")
            return self._say("I couldn't re-home the crane; positions may have drifted.")

    def _play_move(self, move: chess.Move, prefix: str) -> str:
        """Actuate, narrate, and speak a move — speaking internally.

        SEQUENTIAL by default (app.concurrent_actuation off): the machine says what
        it is doing and only THEN drives the crane, so speech and motion never
        overlap and the turn reads as one step at a time. A capture's discard still
        runs up front — that is where a move can be refused for lack of storage —
        but for a plain move nothing moves before we speak. Set concurrent_actuation
        to speak WHILE the crane runs, trading legibility for speed."""
        board_before = self.game.board.copy()
        # LED off for the whole move: hands off the board until the crane is done.
        self._led("off")
        # Whose move this is (side to move vs. machine colour): the relay prototype
        # uses it to pick a motor direction; the crane ignores it.
        mover_is_machine = board_before.turn == self.game.machine_color
        # Clear any captured piece to storage NOW (blocking) and learn whether the
        # move is even possible, WITHOUT yet moving the piece itself. For a plain
        # move this touches no motor, so nothing moves before we speak.
        complete, report = self.choreo.begin_move(board_before, move, mover_is_machine)
        if report.aborted:
            # Refused before touching the board (e.g. storage full). Do NOT apply the
            # move, so the logical and physical boards stay in sync.
            return self._say(" ".join(report.notes) or "I can't make that move right now.")

        san = self.game.push(move)
        text = f"{prefix} {speak_san(san)}."
        comment = self._move_comment(board_before, move, san)
        if comment:
            text += " " + comment
        # If the opponent (this move) just punished a blunder the machine held
        # back, own it now — otherwise it stays unspoken.
        critique = self._resolve_self_critique(board_before)
        if critique:
            text += " " + critique
        if self.game.is_game_over():
            text += " " + self.game.result_text()

        # Promotions can add a manual-swap note DURING the crane's travel, so they
        # never run concurrently; that note is spoken afterwards (below).
        concurrent = self.cfg.app.concurrent_actuation and move.promotion is None
        finished_ok = True
        if concurrent:
            motion, motion_result = self._spawn_motion(complete)
            self._say(text)             # spoken while the crane is still moving
            motion.join()               # don't begin the next move until actuation is done
            finished_ok = motion_result["error"] is None
        else:
            self._say(text)             # say the move first...
            try:
                complete()              # ...then run the crane to completion (blocking)
            except Exception:  # noqa: BLE001 - reported below, never crashes the turn
                log.exception("Actuation failed during the move")
                finished_ok = False
        # A note produced during actuation (e.g. a promotion with no spare piece:
        # "replace the pawn on a8 with a queen") is spoken now, AFTER the crane has
        # placed the pawn, so it refers to a state that actually exists.
        if finished_ok and report.notes:
            note = " ".join(report.notes)
            self._say(note)
            text += " " + note
        if not finished_ok:
            # The move is already applied logically, but the crane didn't finish —
            # release the magnet and flag the possible desync rather than hiding it.
            self._safe_park()
            self._say("I couldn't finish moving that piece — please check the "
                      "board matches the position before we continue.")
        # Re-home to re-zero open-loop drift. Skip it if actuation didn't finish, so
        # we don't drag a stuck piece. Triggers (any fires): every finished move
        # (rehome_after_move); after a capture (rehome_on_capture); or every N finished
        # moves (rehome_every_n_moves).
        if finished_ok:
            self._moves_since_home += 1
            due = (self.cfg.app.rehome_after_move
                   or (self.cfg.app.rehome_on_capture and board_before.is_capture(move))
                   or (self.cfg.app.rehome_every_n_moves > 0
                       and self._moves_since_home >= self.cfg.app.rehome_every_n_moves))
            if due:
                self._rehome()          # silent unless it fails (resets the counter)
        # The "Your move." hand-back is spoken by the run loop right before the mic
        # opens (_prompt_move_if_new_turn), so there is exactly ONE listening cue and
        # it lands immediately before recording — not here, where it used to double up.
        return text

    def _spawn_motion(self, complete) -> tuple[threading.Thread, dict]:
        """Run the rest of a move's actuation on a worker thread.

        Returns the thread and a `result` holder whose "error" is set if
        actuation raised, so the caller can surface a physical/logical desync
        after joining instead of silently swallowing it. The turn never crashes.
        """
        result: dict = {"error": None}

        def run():
            try:
                complete()
            except Exception as exc:  # noqa: BLE001 - reported via `result`, not raised
                log.exception("Actuation failed during concurrent move")
                result["error"] = exc

        thread = threading.Thread(target=run, name="crane", daemon=True)
        thread.start()
        return thread, result

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
            mover_is_machine = mover == self.machine_color
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
                "mover_is_machine": mover_is_machine,
                # Grounded specifics so the reaction can be about the POSITION rather
                # than a restatement of the verdict (which made every comment sound
                # the same). All free: no extra engine call.
                "best_san": quality.best_san,
                "move_kind": self._describe_move(board_before, move, board_after),
                "phase": analysis.game_phase(board_after),
            }
            mat = analysis.material_balance(board_after)
            if mat and mat["leader"] != "even":
                info["material"] = (f"{mat['leader']} is up {abs(mat['diff'])} "
                                    f"point{'s' if abs(mat['diff']) != 1 else ''}")
            elif mat:
                info["material"] = "level"
            # C: don't flag the machine's OWN blunder/mistake right away — hold it
            # back and only voice it if the opponent actually punishes it (checked
            # after their reply in _resolve_self_critique). Machine's good moves
            # and all of the human's moves comment immediately, with attribution.
            if (mover_is_machine and self.cfg.app.self_critique_only_if_punished
                    and quality.label in ("blunder", "mistake")):
                res = self.engine.analyse(board_after)
                self._pending_self_critique = {
                    "info": info,
                    "cp_after": analysis.machine_pov_cp(res, self.machine_color == chess.WHITE),
                }
                return ""
            return self.nlu.comment_on_move(info)
        except Exception:  # noqa: BLE001 - commentary must never break a move
            log.exception("Move commentary failed")
            return ""

    @staticmethod
    def _describe_move(board_before: chess.Board, move: chess.Move,
                       board_after: chess.Board) -> str:
        """Plain description of what the move DID (capture, check, castle, ...).

        Purely mechanical, straight off the board -- no engine, no invention. Gives
        the commentary something concrete to talk about besides the verdict, which
        is what made every reaction sound identical.
        """
        parts = []
        if board_before.is_castling(move):
            parts.append("castles")
        elif board_before.is_capture(move):
            taken = board_before.piece_at(move.to_square)
            # en passant has no piece on the destination square
            name = chess.piece_name(taken.piece_type) if taken else "pawn"
            parts.append(f"captures a {name}")
        if move.promotion:
            parts.append(f"promotes to a {chess.piece_name(move.promotion)}")
        if board_after.is_checkmate():
            parts.append("delivers checkmate")
        elif board_after.is_check():
            parts.append("gives check")
        if not parts:
            piece = board_before.piece_at(move.from_square)
            if piece is not None:
                parts.append(f"a quiet {chess.piece_name(piece.piece_type)} move")
        return ", ".join(parts)

    def _resolve_self_critique(self, board_before: chess.Board) -> str:
        """Voice a held-back self-critique iff the move just played (by the
        opponent) punished the machine's earlier blunder. `board_before` is the
        position before this move; `self.game.board` is already after it. Returns
        the spoken critique, or "" (drop it silently if the opponent let it go)."""
        pending = self._pending_self_critique
        if pending is None:
            return ""
        # Only the opponent's reply can "punish" it; if this move is the machine's
        # own, keep waiting for the human's response.
        if board_before.turn == self.machine_color:
            return ""
        self._pending_self_critique = None
        if not getattr(self.engine, "provides_evaluation", False):
            return ""
        try:
            res = self.engine.analyse(self.game.board)
            cp_now = analysis.machine_pov_cp(res, self.machine_color == chess.WHITE)
            if analysis.blunder_punished(pending["cp_after"], cp_now):
                info = {**pending["info"], "punished": True}
                return self.nlu.comment_on_move(info)
        except Exception:  # noqa: BLE001 - commentary must never break a move
            log.exception("Deferred self-critique failed")
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
