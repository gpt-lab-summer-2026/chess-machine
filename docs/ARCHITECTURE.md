# Architecture

## Data flow

```
AudioCapture.record_utterance()        voice/audio.py     (mic + WebRTC VAD)
        │ float32 @ 16 kHz
        ▼
DistilWhisperSTT.transcribe()          voice/stt.py       (faster-whisper, int8)
        │ "knight to f three"
        ▼
NLU.interpret(transcript, context)     nlu/slm.py         → [Intent{action, move, ...}]
        │                              (llama.cpp; rule_based fallback)   one or more
        ▼
ChessMachine.handle → _dispatch(each)  pipeline.py        ← holds GameState
        ├── set_difficulty → engine.set_difficulty()
        ├── analyze        → analysis.describe_position() → NLU.phrase_analysis()
        ├── engine_move    → engine.best_move()  ┐
        └── opponent_move  → parse_move()         ├─▶ Choreographer.execute_move()
                                                  ┘        │ low-level ops
        │ spoken text                                      ▼
        ▼                                          MotionController (serial/mock)
KokoroTTS.say()                        voice/tts.py        (x,z)→(r,θ) in serial
        │ float32 @ 24 kHz                                        ▼
        ▼                                            ESP32 firmware (rotary+radial)
   speaker (sounddevice)
```

Mostly synchronous: `ChessMachine.run()` loops listen → handle → speak, and each
motion command blocks until the ESP32 acknowledges, so the host keeps no
motor-state bookkeeping. The one concurrency: when `concurrent_actuation` is on,
`_play_move` runs the piece's travel on a short-lived worker thread so the machine
can narrate the move while the crane is still moving. It always `join()`s that
thread before the next turn, and a worker-thread actuation failure is surfaced to
the user ("please check the board") rather than swallowed.

## Layers (inner → outer)

1. **`chess_engine/`** — pure chess. `GameState` (python-chess board + history),
   `classify_move` (the bridge to motion: capture / en passant / castle /
   promotion + the squares involved), `ChessEngine` (Stockfish or a `random`
   dev stand-in), and `analysis` (material, evaluation, best line → facts dict).
   No I/O, fully unit-tested.

2. **`motion/`** — physical actuation. `BoardGeometry` maps squares → planar
   (x,z) mm in the crane-pivot frame; `Graveyard` tracks captured pieces;
   `MotionController` is the transport interface with `SerialMotion` (ESP32,
   which converts each (x,z) to polar r/θ for the crane) and `MockMotion`
   (records planar ops); `Choreographer` turns a `chess.Move` into
   pick-and-place sequences. Knows chess shapes but not *why* a move was chosen.

3. **`nlu/`** — language only. `move_parsing` resolves speech → a legal move
   (exact SAN/UCI first, then fuzzy square/piece extraction, always filtered by
   legality). `SlmNLU` calls llama.cpp to turn an utterance into a *list* of
   intent objects (so "give me black and make it harder" becomes two) plus answer
   phrasing; `RuleBasedNLU` is the keyword fallback (it splits compounds on
   conjunctions). The pipeline just dispatches whatever list comes back. Prompts
   live in `prompts.py`.

4. **`voice/`** — speech I/O. STT/TTS/audio with all heavy deps imported lazily,
   so the package loads even where faster-whisper/kokoro/sounddevice aren't
   installed.

5. **`pipeline.py` + `app.py` + `factory.py`** — orchestration, CLI, and wiring.

## Why the SLM is kept on a short leash

3B models are great at language and unreliable at multi-step chess reasoning. So:

- **Understanding:** the SLM emits a small JSON list of intents. The move itself
  is resolved from the user's literal words first (`parse_move`, legality-filtered)
  and the SLM's move field is only a fallback — so a hallucinated or wrong-color
  move can't be played. Bad JSON → rule-based fallback.
- **Truth:** material, evaluation, and best moves are computed by Stockfish +
  python-chess and handed to the SLM as *facts*. The phrasing prompt forbids
  inventing numbers or moves.
- **Generation:** the SLM only rewords those facts into natural speech.

This keeps a small, fine-tunable model (QLoRA-friendly) useful without letting it
make illegal or wrong chess decisions.

## Failure handling

- SLM unreachable / bad output → `RuleBasedNLU` keeps the machine playable.
- Stockfish missing → clear error, or `engine.backend: random` for bring-up.
- Unplayable move → the machine explains *why* — the piece can't reach the
  square, the move would leave the king in check, it's the wrong colour, or
  nothing legal reaches the target — and suggests the real moves. A genuinely
  ambiguous spoken move → "did you mean X or Y?" (and the SLM's move breaks the
  tie when it agrees with what you said).
- Destructive actions (new game, take-back) ask for a spoken "yes" first.
- Promotion with no spare piece, or a too-small graveyard for reset → the
  machine does what it can and speaks the manual step needed.
- Storage full on a capture → the move is **refused without touching the board**
  (so the logical and physical positions can't drift apart) and the machine asks
  you to clear the captured pieces. The reference 16-slot graveyard fills after 16
  captures; see [DESIGN.md §4.1](DESIGN.md).
- Out-of-turn `engine_move` ("your move" when it's *your* turn) → declined, not
  played for the wrong colour.

### The actuation/state-sync contract

`pipeline` is the single owner of game state, and it keeps the logical board in
lockstep with the physical one: `_play_move` calls `Choreographer.execute_move`
*before* `GameState.push`, and only pushes if actuation succeeded.
`execute_move` therefore does its feasibility checks up front and returns an
`ExecutionReport` — `aborted=True` (nothing moved; caller must not apply the
move) plus human-readable `notes` for any manual step — rather than raising
mid-sequence. A new `MotionController` transport need only honour the same
"complete the op or fail before moving" expectation; nothing upstream changes.

## Extending it

- New transport (e.g. WiFi): implement `MotionController`, register in
  `factory.create_motion_controller`. Nothing upstream changes.
- New STT/TTS/engine: implement the interface + add to the relevant factory.
- New voice command: add an action in `nlu/intents.py`, handle it in
  `pipeline._dispatch`, and add an example to `nlu/prompts.py`.
