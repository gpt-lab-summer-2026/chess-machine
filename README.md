# chess-machine

A voice-controlled, self-actuating chess board. You speak; the machine listens,
talks back, decides on a move (consulting Stockfish at the chosen difficulty),
and physically moves the magnetic pieces with a gantry + electromagnet on a
pulley. It runs entirely onboard a Raspberry Pi 5, with an ESP32 driving the
motors.

```
 mic ─▶ distil-whisper (STT) ─▶ llama.cpp 3B SLM ─▶ intent (JSON)
                                                       │
        ┌──────────────────────────────────────────────┼───────────────────────────────┐
   set difficulty                board analysis                    make / relay a move
        │                              │                                    │
  Stockfish config        Stockfish + python-chess              python-chess (validate)
                                       │                                    │
                              facts ─▶ SLM phrasing                  Choreographer
                                       │                                    │ serial
                                       └──────────▶ Kokoro (TTS) ◀── ESP32 (X / Z / pulley / magnet)
```

**Design principle: the language model is never trusted for chess truth.** The
SLM only does language — classifying what you want (intent + slots) and phrasing
answers. Every move is validated by [python-chess](https://python-chess.readthedocs.io/),
and every evaluation comes from Stockfish. A noisy transcript can only ever
resolve to a *legal* move, or trigger a "say that again."

## Deliverables

| Deliverable | Where |
|---|---|
| Difficulty voice control | `set difficulty hard` → Stockfish Skill/Elo/limit ([pipeline.py](src/chessmachine/pipeline.py)) |
| On-demand board analysis | `who's winning?`, `best move?` → grounded answer ([analysis.py](src/chessmachine/chess_engine/analysis.py)) |
| Actuation of moves (self + opponent) | pick-and-place incl. captures, castling, en passant, promotion ([choreography.py](src/chessmachine/motion/choreography.py)) |
| STT → model → (Stockfish & motors) → TTS | the full loop ([pipeline.py](src/chessmachine/pipeline.py)) |

## Quickstart (dev mode — no hardware, no models)

Runs the entire pipeline with typed input, printed speech, a random-move engine,
and a mock motor, on any machine with Python 3.10+:

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m chessmachine --dev
# you> e4
# [speaker] Okay, pawn to e 4.
# [speaker] My move: knight to f 6.
# you> who's winning?
# you> set difficulty to hard
# you> new game
# you> quit
```

`--dev` swaps every subsystem for a no-hardware variant. Mix and match for
partial testing: `--text` (typed I/O, real engine/motor), `--mock` (mock motor),
`--engine random`, `--slm rule_based`.

## Real deployment (Raspberry Pi 5)

Full instructions in **[docs/SETUP_PI.md](docs/SETUP_PI.md)**. In brief:

```bash
sudo apt install stockfish portaudio19-dev
pip install -r requirements-pi.txt
bash scripts/fetch_models.sh            # Kokoro + distil-whisper + SLM hints
llama-server -m models/slm/<model>.gguf -c 4096 --port 8080 &
cp config/config.example.yaml config/config.yaml   # edit serial port, geometry
python -m chessmachine --config config/config.yaml
```

Flash the ESP32 first — see **[firmware/esp32_chess/](firmware/esp32_chess/)**.

## Voice commands

| You say | Action |
|---|---|
| "knight to f3", "e2 to e4", "castle kingside", "pawn takes d5" | play your move (machine actuates it) |
| "your move", "okay, go" | machine plays its move |
| "who's winning?", "what's the best move?", "any threats?" | spoken analysis |
| "set difficulty to easy / medium / hard", "set elo to 1600" | change strength |
| "new game", "take that back", "whose turn is it?" | reset / undo / status |

With `auto_reply` on (default), stating your move makes the machine immediately
reply with its own — so a turn is one spoken sentence from you.

## Repo layout

```
src/chessmachine/
  config.py            dataclass config + YAML deep-merge
  pipeline.py          orchestrator: transcript -> intent -> action -> speech
  app.py               CLI entrypoint (--dev/--text/--mock/...)
  factory.py           build subsystems from config
  chess_engine/        game state, Stockfish wrapper, analysis  (pure logic)
  nlu/                 SLM client, intents, prompts, speech->move parsing
  motion/              geometry, controller (serial/mock), pick&place choreography
  voice/               distil-whisper STT, Kokoro TTS, mic/VAD capture
firmware/esp32_chess/  Arduino firmware: steppers + pulley + magnet + protocol
scripts/               calibrate.py, serial_console.py, fetch_models.sh
tests/                 pytest suite for the hardware-free core (51 tests)
docs/                  ARCHITECTURE, HARDWARE, PROTOCOL, SETUP_PI
```

## Testing

```bash
pip install pytest && python -m pytest      # 51 tests, no hardware needed
```

Covers config merging, board geometry, move classification, the speech→move
parser, the full pick-and-place choreography (asserted op-by-op against the mock
motor), intent routing, and an end-to-end scripted game (including Scholar's mate).

## Known limitations

- **Promotions** need a spare piece pre-stocked in storage to auto-swap;
  otherwise the machine carries the pawn and asks you to swap it by hand.
- **Undo of a promotion** asks for a manual pawn swap (the rest of undo is automatic).
- Auto board **reset** requires enough storage slots for a full set (the default
  geometry provides 32).
- The `rule_based` SLM fallback is keyword-based; the llama.cpp SLM handles
  nuance far better. The fallback keeps the machine usable if the model is down.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## License

MIT
