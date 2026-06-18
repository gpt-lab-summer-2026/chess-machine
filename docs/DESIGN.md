# Design notes & decisions

> Living design log for the chess machine. It records *why* things are the way
> they are and the open hardware questions — context that isn't obvious from the
> code. If you're an AI assistant picking this project up in a later session,
> **read this first**, then [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. What it is

A voice-controlled, self-actuating chess board. The user speaks; the machine
listens, replies, decides on a move (Stockfish at the chosen difficulty), and
physically moves magnetic pieces with an X/Z gantry + an electromagnet on a
pulley. Everything runs onboard a Raspberry Pi 5 (8 GB); an ESP32 drives the
motors over USB serial.

Pipeline: `mic → STT → SLM (intent) → {Stockfish, motion} → TTS → speaker`.

## 2. Component decisions (from the build conversation)

| Concern | Choice | Why |
|---|---|---|
| Host | Raspberry Pi 5, 8 GB | onboard; fits the whole stack with headroom |
| STT | **distil-whisper** (faster-whisper, int8) | fast on the Pi CPU; user's pick |
| TTS | **Kokoro** (ONNX) | small, natural, CPU-friendly; user's pick |
| SLM | **llama.cpp**, 3B, server mode | user's pick; QLoRA-finetunable; runs as `llama-server` |
| Chess strength/eval | **Stockfish** + python-chess | all chess truth (legality, eval, best move) |
| Motion controller | **ESP32 over USB serial** | user's pick; firmware does steppers + magnet |

**Cardinal rule — the SLM is never trusted for chess truth.** It only does
language: classify intent (+ slots) and phrase answers. Every move is validated
against `board.legal_moves`; every evaluation comes from Stockfish. A garbled
transcript can only ever resolve to a *legal* move or trigger "say that again".
This is what makes a small, finetunable model safe to use here.

Every subsystem also has a dev/mock backend, so `python -m chessmachine --dev`
runs the entire pipeline with no hardware and no models (used for the test suite
and for development on a laptop).

## 3. Software architecture

See [ARCHITECTURE.md](ARCHITECTURE.md). Layers, inner → outer: `chess_engine`
(pure chess) → `motion` (geometry + choreography + transport) → `nlu` (SLM +
move parsing) → `voice` (STT/TTS) → `pipeline`/`app` (orchestration). Move
classification (`classify_move`) is the bridge from chess rules to physical ops.

## 4. Physical constraints & open questions

### 4.1 Geometry budget — stroke vs board vs storage  ⚠️ binding constraint

The two linear motors have a **150 mm stroke each**, so the gantry can only reach
a **150 mm × 150 mm** square. The physical board is also 15 cm. **Both the
playable 8×8 grid and the captured-piece storage must fit inside that 150 mm
reach** — there is no "off-board" beyond it.

Therefore the playable grid must be **smaller than 150 mm**, with storage in the
leftover strip:

```
        x=0 ─────────────────────── x=150 (stroke limit)
   z=0 ┌───────────────────────────┬─────────┐
       │  a8 b8 c8 d8 e8 f8 g8 h8   │ ▢ ▢     │  ← storage strip
       │  ..  playable 8x8 grid ..  │ ▢ ▢     │    (2 columns ×
       │  15 mm squares, 120 mm     │ ▢ ▢     │     8 rows = 16 slots,
       │  a1 ........... h1         │ ▢ ▢     │     "2 per row")
 z=120 └───────────────────────────┴─────────┘
       grid centers: 10 → 115 mm     127.5 / 142.5 mm   (all ≤ 150 ✓)
```

Recommended numbers (now the defaults in `config.py` / `config.example.yaml`):

| Param | Value | Note |
|---|---|---|
| `square_pitch_mm` | 15.0 | 120 mm playable grid / 8 |
| `origin_x_mm`, `origin_z_mm` | 10.0, 10.0 | center of a1; leaves a low-side margin |
| grid square centers | 10 → 115 mm | 7 × 15 mm span, well inside 150 |
| `graveyard_x_mm` | 127.5 | first storage column, ~12 mm past the h-file |
| `graveyard_x_pitch_mm` | 15.0 | second column at 142.5 mm (≤ 150 ✓) |
| `graveyard_columns` × `slots_per_column` | 2 × 8 = **16** | matches "2 pieces per row" |

**Capture ordering** is already implemented: on a capture the taken piece is
lifted to storage *first*, then the capturing piece moves
([choreography.py](../src/chessmachine/motion/choreography.py) `execute_move`).

**Storage capacity caveat.** 16 slots covers a typical game, but two cases exceed
it, and both **degrade gracefully rather than crash**:

- **Mid-game (>16 captures).** Once all 16 slots are full, the next capture is
  *refused without touching the board* (no logical/physical desync) and the
  machine asks you to clear the captured pieces off the storage strip; play then
  continues. A full game can produce up to 30 captures, so this is reachable.
- **Board reset (needs 32 transient slots).** "New game" stages every piece in
  storage at once, which needs a free slot per piece on the board. With 16 the
  machine says it can't reset itself and asks you to set the pieces up by hand
  (it does *not* claim a reset it didn't perform), then drops its captured-piece
  bookkeeping to match the hand re-setup.

To make either automatic you need ~32 slots, which means a smaller grid (see the
open decision below).

**Open decision — 16 vs 32 storage slots.** Storage on *one* edge (2 columns)
gives 16 slots with a comfortable 15 mm grid. Storage on *both* edges (4 columns
total) gives 32 (auto-reset works) but shrinks the grid to ~90 mm → ~11.25 mm
squares, which is tight for ~10 mm piece magnets. Recommendation: stick with 16
unless auto-reset is a must-have.

### 4.2 Magnetics — is a ≤10 mm piece magnet workable?

Piece magnet: ≤5 mm radius (≤10 mm diameter), a small NdFeB disc (~2–3 mm thick)
in a light 3D-printed piece. Three sub-questions:

**(a) Will pieces pull on their neighbors?** Square centers are 15 mm apart;
magnet edges ~5 mm apart. For magnets this small the inter-piece force at that
gap is small (order 0.1–0.3 N). Crucially, if **every piece is mounted with the
same pole up** (required so the overhead electromagnet treats them all alike),
side-by-side pieces *mildly repel* rather than attract — so they won't clump.
Mitigation for drift: print **shallow conical wells at each square center** so
pieces self-center and resist small lateral forces (this also fixes placement
error from cable swing). Verdict: not a problem with small magnets + recessed
squares.

**(b) Will the electromagnet pick up cleanly without grabbing neighbors?**
Pickup is the easy direction. A modest electromagnet lowered to ~2–4 mm above the
target lifts a few-gram piece with margin (the piece's own NdFeB magnet bridges
to the EM's iron core and adds strong adhesion). The field is concentrated under
the pole, so a neighbor 15 mm away sees a much weaker field and stays put —
**provided you only energize once centered over the target and lowered close.**

**(c) Releasing is the hard part.** ⚠️ An iron-cored electromagnet (plus residual
magnetism) keeps attracting the *permanent* piece magnet even when de-energized,
so a light piece may not drop on "MAG OFF". Mitigations, in order of preference:
  1. Drive the EM through an **H-bridge** so firmware can *actively repel* —
     a brief reverse-polarity pulse pushes the piece off. (The protocol/firmware
     currently expose only `MAG ON/OFF` via a MOSFET; adding a `MAG REPEL`
     reverse drive is a small, well-isolated extension — noted as future work.)
  2. Use a **low-remanence soft-iron core** and a thin non-magnetic spacer/cap so
     the piece never seats directly on the core (reduces residual stiction).
  3. Worst case, a tiny mechanical stripper/knock-off, but prefer (1)+(2).

**Travel height & swing.** Keep `travel_height_mm` generous (default 60 mm) so the
carried magnet is far enough above the board (~≥30 mm) that it doesn't perturb
standing pieces — small magnets decouple quickly with distance. `settle_ms` adds a
pause after each move to damp cable swing; the recessed wells capture placement
despite residual swing.

**Net verdict.** Feasible with ≤10 mm magnets, *if*: (1) recess each square,
(2) plan for **active (reverse-drive) magnet release** via an H-bridge,
(3) keep travel height generous, (4) mount all piece magnets same-pole-up.
The one thing to bench-test early with your real magnets + EM: pull force vs air
gap at the pick height, and whether "off" alone releases or you need the reverse
pulse.

### 4.3 Motor / actuator choice for X and Z

The X/Z axes only **position a light head** (pulley + electromagnet + at most one
carried piece — a few grams). They never push pieces or bear a holding load. So
optimize for **speed, low cost, and low friction**, not force or holding torque.

- ✅ **Recommended: belt-driven carriage on a linear rail** — a stepper turns a
  GT2 pulley that drags a low-friction cart (MGN12 rail, or V-wheels on
  aluminium extrusion). This is exactly the "cart on a rail that doesn't have to
  push" idea. It's fast, cheap, and matches the firmware's `STEPS_PER_MM` model:
  a 20-tooth GT2 pulley at 1/16 microstepping ≈ **80 steps/mm**, which is the
  firmware default. Two such axes (simple Cartesian, or an H-bot/CoreXY if you
  want both motors fixed to the frame) cover the 150 mm × 150 mm area.
- ⚠️ **Lead-screw "linear actuators"** work but are the wrong tool: high push
  force you don't need, slow, and self-locking — they hurt responsiveness.
- ⚠️ **True magnetic linear motors** are precise and fast but expensive overkill
  for a few-gram head.

**Guide vs drive — keep the MGN12.** The MGN12 rail + carriage is the *guide*
(it bears the load and keeps motion straight and low-friction); it is *not* the
*drive*. Pair it with a belt drive — they're complementary, not alternatives.

**Per-axis stack (belt on MGN12):** MGN12 rail + MGN12C block (light load → the C
block is plenty); a NEMA17 at one end with a 20T GT2 pulley; a flanged idler at
the far end; a 6 mm GT2 belt with both ends clamped to the carriage, looped
motor↔idler (printer-style); a slotted/spring idler mount for tension; a
microswitch endstop at the home end. Resolution: 20T GT2 = 40 mm/rev; NEMA17
(200 steps) at 1/16 µstep = **80 steps/mm → 0.0125 mm/step**, ~1000× finer than a
15 mm cell needs. Belt compliance is irrelevant for parking a magnet over a
square, and homing resets any drift.

**Why belt over the alternatives at 15–40 cm travel:** lead screws get slow and
whip (critical speed) past ~30 cm and add holding force you don't want;
rack-and-pinion adds backlash and cost; belt is light, fast, cheap, and scales.

**Two-axis topology:**
- *Stacked Cartesian* (recommended): the Z rail rides on the X carriage, the head
  rides on Z. Dead simple, and the firmware drives X/Z as independent steppers —
  **no software change**.
- *CoreXY / H-bot*: both motors fixed to the frame (less moving mass, faster) but
  needs belt-mixing kinematics added to the firmware (X=(A+B)/2, Z=(A−B)/2).

Stroke: **150 mm is fine** — just size the playable grid to ~120 mm (§4.1). The
moving mass is tiny, so you can run fast feeds; keep acceleration moderate so the
hanging electromagnet doesn't swing (firmware `ACCEL`, host `settle_ms` are
tunable). Steppers are open-loop — fine at this scale; if a belt ever skips,
re-home. The firmware/protocol are motor-agnostic (STEP/DIR + steps/mm), so belt
vs screw vs linear motor is purely a `*_STEPS_PER_MM` change.

## 5. Open questions / risks

- **Magnet release method** — confirm whether MOSFET ON/OFF suffices or you need
  an H-bridge for reverse-drive release. If H-bridge: extend firmware + protocol
  with a 3-state magnet command.
- **Storage count** — 16 (one edge, 15 mm grid) vs 32 (both edges, ~11 mm grid).
- **Magnet spec** — exact NdFeB grade/size and the electromagnet's pull curve;
  bench-test at the real air gaps before finalizing pick/travel heights.
- **Placement precision** — 15 mm cells demand tighter placement than the cable
  alone gives; recessed wells are the planned mitigation.
- **STT move accuracy** — distil-whisper may mishear move words; mitigated by the
  legality-filtered parser, which rejects anything that isn't a legal move.

## 6. Pointers

- [ARCHITECTURE.md](ARCHITECTURE.md) — modules and data flow
- [HARDWARE.md](HARDWARE.md) — wiring, mechanics, calibration
- [PROTOCOL.md](PROTOCOL.md) — host ↔ ESP32 serial protocol
- [SETUP_PI.md](SETUP_PI.md) — Raspberry Pi deployment
- Calibrate geometry with `python scripts/calibrate.py` (jog to a1 & h8, `calc`).
