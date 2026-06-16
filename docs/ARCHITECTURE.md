# Architecture

## Data flow

```
AudioCapture.record_utterance()        voice/audio.py     (mic + WebRTC VAD)
        │ float32 @ 16 kHz
        ▼
DistilWhisperSTT.transcribe()          voice/stt.py       (faster-whisper, int8)
        │ "knight to f three"
        ▼
NLU.interpret(transcript, context)     nlu/slm.py         → Intent{action, move, ...}
        │                              (llama.cpp; rule_based fallback)
        ▼
ChessMachine._dispatch(intent)         pipeline.py        ← holds GameState
        ├── set_difficulty → engine.set_difficulty()
        ├── analyze        → analysis.describe_position() → NLU.phrase_analysis()
        ├── engine_move    → engine.best_move()  ┐
        └── opponent_move  → parse_move()         ├─▶ Choreographer.execute_move()
                                                  ┘        │ low-level ops
        │ spoken text                                      ▼
        ▼                                          MotionController (serial/mock)
KokoroTTS.say()                        voice/tts.py               │ protocol
        │ float32 @ 24 kHz                                        ▼
        ▼                                                   ESP32 firmware
   speaker (sounddevice)
```

Single-threaded and synchronous: `ChessMachine.run()` loops listen → handle →
speak. Each motion command blocks until the ESP32 acknowledges, so there is no
motor-state bookkeeping on the host and no concurrency to reason about.

## Layers (inner → outer)

1. **`chess_engine/`** — pure chess. `GameState` (python-chess board + history),
   `classify_move` (the bridge to motion: capture / en passant / castle /
   promotion + the squares involved), `ChessEngine` (Stockfish or a `random`
   dev stand-in), and `analysis` (material, evaluation, best line → facts dict).
   No I/O, fully unit-tested.

2. **`motion/`** — physical actuation. `BoardGeometry` maps squares → (x,z) mm;
   `Graveyard` tracks captured pieces; `MotionController` is the transport
   interface with `SerialMotion` (ESP32) and `MockMotion` (records ops)
   implementations; `Choreographer` turns a `chess.Move` into pick-and-place
   sequences. Knows chess shapes but not *why* a move was chosen.

3. **`nlu/`** — language only. `move_parsing` resolves speech → a legal move
   (exact SAN/UCI first, then fuzzy square/piece extraction, always filtered by
   legality). `SlmNLU` calls llama.cpp for intent JSON + answer phrasing;
   `RuleBasedNLU` is the keyword fallback. Prompts live in `prompts.py`.

4. **`voice/`** — speech I/O. STT/TTS/audio with all heavy deps imported lazily,
   so the package loads even where faster-whisper/kokoro/sounddevice aren't
   installed.

5. **`pipeline.py` + `app.py` + `factory.py`** — orchestration, CLI, and wiring.

## Why the SLM is kept on a short leash

3B models are great at language and unreliable at multi-step chess reasoning. So:

- **Understanding:** the SLM emits a tiny JSON intent. Even if it hallucinates a
  move, `parse_move` validates it against `board.legal_moves` before anything
  moves. Bad JSON → rule-based fallback.
- **Truth:** material, evaluation, and best moves are computed by Stockfish +
  python-chess and handed to the SLM as *facts*. The phrasing prompt forbids
  inventing numbers or moves.
- **Generation:** the SLM only rewords those facts into natural speech.

This keeps a small, fine-tunable model (QLoRA-friendly) useful without letting it
make illegal or wrong chess decisions.

## Failure handling

- SLM unreachable / bad output → `RuleBasedNLU` keeps the machine playable.
- Stockfish missing → clear error, or `engine.backend: random` for bring-up.
- Unreadable move → "say that again"; ambiguous move → "did you mean X or Y?".
- Promotion with no spare piece, or a too-small graveyard for reset → the
  machine does what it can and speaks the manual step needed.

## Extending it

- New transport (e.g. WiFi): implement `MotionController`, register in
  `factory.create_motion_controller`. Nothing upstream changes.
- New STT/TTS/engine: implement the interface + add to the relevant factory.
- New voice command: add an action in `nlu/intents.py`, handle it in
  `pipeline._dispatch`, and add an example to `nlu/prompts.py`.
