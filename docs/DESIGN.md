# Design notes & decisions

> Living design log for the chess machine. It records *why* things are the way
> they are and the open hardware questions — context that isn't obvious from the
> code. If you're an AI assistant picking this project up in a later session,
> **read this first**, then [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. What it is

A voice-controlled, self-actuating chess board. The user speaks; the machine
listens, replies, decides on a move (Stockfish at the chosen difficulty), and
physically moves magnetic pieces with a **polar crane** (rotary base + a railcart
on the arm) + a 6 V electromagnet on a pulley. Everything runs onboard a
Raspberry Pi 5 (8 GB); an ESP32 drives the motors over USB serial.

Pipeline: `mic → STT → SLM (intent) → {Stockfish, motion} → TTS → speaker`.

> **Frame change (2026-06):** the original X/Z linear gantry was replaced by the
> polar crane. The chess grid is still computed in planar (x,z) mm; the
> Cartesian→polar conversion happens at the transport (see §4.1, §4.3).

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

### 4.1 Geometry budget — fitting a square board in the crane's sector  ⚠️ binding constraint

The crane reaches an **annular sector**: radius `r ∈ [r_min, r_max]` swept over an
angle. `r_max ≈ 300 mm` is the arm/railcart reach; `r_min ≈ 80 mm` is the dead
zone around the crane body (it can't reach its own base). The playable board is a
**square centered on the sector's bisector**, near edge at `r_min`; the leftover
wedges hold the graveyards.

```
                 · · ·  ← graveyard arc (r≈293)
              ·          \
      ┌───────────────┐   ·   r_max = 300
      │  a8 ...... h8  │    ·
      │   216 mm 8x8   │     ·
      │  square board  │     ·  sweep ≈ 107°
      │  a1 ...... h1  │    ·
      └───────────────┘   ·
        near edge          /
        at r_min=80   · · ·
                  ╳  ← crane pivot (origin; dead zone r<80)
```

**Sizing.** Only the **square centers** must be reachable — a magnet never visits
the empty board corners — so the binding constraint is the far *centers* on
`r_max`. The h8/a8 centers sit at `hypot(r_min + 7.5·pitch, 3.5·pitch)`; keeping
that ≤ `r_max` gives a maximum side of ~218 mm for `r_min=80, r_max=300`. We use
**216 mm** (a hair under the max, matching the physical board). The board's
*physical* far corners then reach ~315 mm and overhang the annulus, but carry no
piece, so that's harmless. The resulting numbers are the defaults in `config.py` /
`config.example.yaml`:

| Param | Value | Note |
|---|---|---|
| `r_min_mm`, `r_max_mm` | 80, 300 | reachable annulus |
| `board_size_mm` | 216 | square side (≤ ~218 max, far *centers* on r_max) |
| `square_pitch_mm` | 27 | 216 mm / 8 |
| `origin_x_mm`, `origin_z_mm` | 93.5, −94.5 | center of a1 (near edge at r_min, board on the bisector) |
| far square centers | ~298 mm | `hypot(282.5, 94.5)`, inside r_max ✓ |
| physical far corners | ~315 mm | `hypot(296, 108)` — overhangs r_max, but no piece there |
| sweep | ~107° | `2·atan((board/2)/r_min)` to the near corners; centers span ±45° |
| `graveyard_radius_mm` | 293 | side arcs, clear of the board (below) |
| `graveyard_slots_per_side` × 2 | 8 × 2 = **16** | two symmetric side arcs |

**Why the graveyard arcs clear the board.** A point is on the board only if its
`|z| ≤ 108 mm` (half the 216 mm side). The graveyard slots sit at `θ ≥ 26°` on a
293 mm arc, so `|z| = 293·sinθ ≥ 128 mm > 108` — every slot clears the board's
z-extent (and so never collides with a piece), even though 293 mm is inside the
~315 mm corner radius.

**Capture ordering** is already implemented: on a capture the taken piece is
lifted to storage *first*, then the capturing piece moves
([choreography.py](../src/chessmachine/motion/choreography.py) `execute_move`).

**Storage capacity caveat.** 16 slots is enough for a typical game's captures,
but a full **board reset needs 32 transient slots**. With 16, the machine reports
"not enough storage to auto-reset; reset by hand" rather than failing — so "new
game" with the default geometry is a manual re-setup. For automatic resets, widen
the sweep and add a second pair of arcs (more `graveyard_slots_per_side`, or a
second radius) — there is room in the side wedges at larger angles.

### 4.2 Magnetics — is a ≤10 mm piece magnet workable?

Piece magnet: ≤5 mm radius (≤10 mm diameter), a small NdFeB disc (~2–3 mm thick)
in a light 3D-printed piece. Three sub-questions:

**(a) Will pieces pull on their neighbors?** Square centers are 27 mm apart
(roomier than the old 15 mm grid), so magnet edges are well separated. For magnets
this small the inter-piece force at that gap is tiny. Crucially, if **every piece is mounted with the
same pole up** (required so the overhead electromagnet treats them all alike),
side-by-side pieces *mildly repel* rather than attract — so they won't clump.
Mitigation for drift: print **shallow conical wells at each square center** so
pieces self-center and resist small lateral forces (this also fixes placement
error from cable swing). Verdict: not a problem with small magnets + recessed
squares.

**(b) Will the electromagnet pick up cleanly without grabbing neighbors?**
Pickup is the easy direction. The electromagnet runs at **6 V**; its rated hold is
**2.5 kg at 12 V**, so even derated at 6 V it lifts a ≤**70 g** piece with enormous
margin (and the piece's own NdFeB magnet bridges to the EM core for extra
adhesion). The field is concentrated under the pole, so a neighbor 27 mm away
sees a much weaker field and stays put — **provided you only energize once
centered over the target and lowered close.**

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

### 4.3 Motor / actuator choice for the crane axes

Both positioning axes only **position a light head** (the cart carries the winch,
pulley, and electromagnet + at most one piece). They never push pieces or bear a
holding load. The real build drives them with **geared brushed DC motors**, each
through a **2-relay SPDT H-bridge**, positioned **open-loop by time** (no encoders)
— cheap and simple for a light head. The winch is the exception: a stepper, so it
can hold a precise height.

- **Rotary base (θ → protocol `A`).** Geared DC motor via an H-bridge. The firmware
  times each move from a measured slew rate — **0.1887 rad/s (~92.5 ms/deg)** with a
  ~50 ms start deadtime — so `A_MS_PER_DEG` replaces the old `A_STEPS_PER_DEG`. A
  rotary endstop defines the home angle (`A_HOME_DEG`); soft limits bound the ~107°
  sweep.
- **Radial railcart (r → protocol `R`).** Geared DC motor via an H-bridge, dragging a
  low-friction cart along the arm (the cart also carries the winch stepper). Timed
  from the measured traverse — **315 mm in 1020 ms of motion (~3.24 ms/mm)** plus a
  30 ms deadzone — so `R_MS_PER_MM` replaces `R_STEPS_PER_MM`. A microswitch at the
  inner end defines `R_HOME_MM ≈ r_min`.
  - ⚠️ Relays are bang-bang (full speed / off): **no speed control**, so the
    protocol's feed `F` is ignored on these axes — travel time is fixed by the motor.
- **Winch (pulley `H`).** A **28BYJ-48 stepper via a ULN2003**, step-counted, with a
  top endstop for homing. It **holds its height between moves** (coils energized) so
  the hanging load can't back-drive it.

**Guide vs drive.** A linear rail/carriage on the arm is the *guide* (bears load,
keeps motion straight and low-friction); the DC motor is the *drive*. Complementary.

**Open-loop accuracy.** Timed DC positioning has no feedback, so error accumulates
across a game (relay lag, coasting, wheel slip). Mitigations: **re-home between
phases** to zero it out, keep the per-move rates/deadzones calibrated, and lean on
the recessed square wells (§4.2) to capture the last few mm. If drift becomes a
problem, the natural upgrade is wheel encoders (closed-loop) — but bench-tune the
timing first.

**Why polar fits this build.** A crane gives a long reach (~30 cm arm) from a small
footprint, and the two natural axes (rotate + extend) map directly to (θ, r). The
host converts the planar board coordinate to polar (`r = hypot(x,z)`,
`θ = atan2(z,x)`), so the firmware stays a dumb two-axis positioner. Keep moves
gentle so the hanging electromagnet doesn't swing (firmware deadzones/rates, host
`settle_ms` are tunable).

## 5. Open questions / risks

- **Magnet release method** — confirm whether MOSFET ON/OFF suffices or you need
  an H-bridge for reverse-drive release. If H-bridge: extend firmware + protocol
  with a 3-state magnet command.
- **Storage count** — 16 (two side arcs) vs 32 (wider sweep + a second pair of
  arcs) for auto-reset. See §4.1.
- **Crane dimensions** — `r_min` (dead radius around the base) and `r_max` (arm
  reach) are estimates (80 / 300 mm); measure the real crane and re-run
  `scripts/calibrate.py`. Every board/graveyard coordinate follows from them.
- **DC axis calibration** — measure each axis's rate and start deadtime (deg/s + ms
  for rotary, mm/s + ms for the cart) and set `A_MS_PER_DEG` / `R_MS_PER_MM`; re-check
  after any gearing or supply-voltage change.
- **Magnet spec** — the 6 V electromagnet's pull curve at the real air gap;
  bench-test before finalizing pick/travel heights.
- **Placement precision** — 27 mm cells are forgiving, but cable swing still
  argues for recessed wells at square centers (planned mitigation).
- **STT move accuracy** — distil-whisper may mishear move words; mitigated by the
  legality-filtered parser, which rejects anything that isn't a legal move.

## 6. Pointers

- [ARCHITECTURE.md](ARCHITECTURE.md) — modules and data flow
- [HARDWARE.md](HARDWARE.md) — wiring, mechanics, calibration
- [PROTOCOL.md](PROTOCOL.md) — host ↔ ESP32 serial protocol
- [SETUP_PI.md](SETUP_PI.md) — Raspberry Pi deployment
- Calibrate geometry with `python scripts/calibrate.py` (jog to a1 & h8, `calc`).
