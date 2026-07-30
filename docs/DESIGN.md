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
angle. `r_max ≈ 455 mm` is the crane's max **radius** (pivot → end of reach — *not*
the 355 mm cart-travel span, which is a common mix-up); `r_min ≈ 80 mm` is the dead
zone around the crane body. The board sits **corner-first (diagonal)**: the **h8
corner is in the near deadzone**, the **a1 corner is farthest**, and the h8→a1
diagonal runs along the sector's bisector (a8/h1 are the side corners; a8 on the
right). `board_angle_deg = 135°` realizes it; the wedges beyond ±28° hold the graveyards.

```
                          · a1  (far corner, ~380 mm)
                       ·     ·
                 h1 ·   8×8     · a8            ·  r_max = 455
                       · board ·             ·
                          · ·             ·  sweep ≈ ±28°
                          h8           ·
                    (near, ~120 mm)  ·  graveyard (θ 37–52°)
              ╳ pivot   (dead zone r < 80)
```

**Sizing.** Corner-first, the **board diagonal** is the binding radial dimension
(not the side): near corner toward the deadzone, far corner out toward reach. The
board's centre-to-centre diagonal is `7·pitch·√2 ≈ 260 mm`, so with h8's centre at
the measured **~120 mm**, a1 lands at **~380 mm** — comfortably inside the 455 mm
radius (~75 mm to spare). The winch's 33 mm lateral offset only shrinks the cart
radius slightly (`cart = √(M²−33²)`), so the near corner still clears the 80 mm
deadzone (h8 cart ≈ 115 mm). These live in `config.example.yaml` / `config.yaml`
(the code defaults in `config.py` stay a generic axis-aligned board):

| Param | Value | Note |
|---|---|---|
| `r_min_mm`, `r_max_mm` | 80, 455 | reachable annulus (455 = max radius, not the 355 travel span) |
| `board_angle_deg` | 135 | corner-first diagonal (h8 near, a1 far, a8 right) |
| `board_size_mm` | 210.4 | square side = 297.5 mm diagonal / √2 |
| `square_pitch_mm` | 26.30 | 210.4 / 8 |
| `origin_x_mm`, `origin_z_mm` | 380, 0 | centre of a1 (far corner, on the bisector) |
| h8 centre / a1 centre | ~120 / ~380 mm | near / far, both on the bisector |
| sweep | ±28° | side corners a8/h1 at ~`atan2(130, 250)`; well inside ±55° |
| `graveyard_*` | 293 mm, 37–52° | side arcs beyond the ±28° sweep (**TODO: rework**) |

**Note.** The graveyard still needs a proper pass for this diagonal layout (angles
are set outside the sweep as an interim). The rotary axis only sweeps ±28° for the
board, so the winch offset (which adds ≤ ~asin(33/120) ≈ 16° at the nearest square)
stays well inside the ±55° firmware limit.

**Capture ordering** is already implemented: on a capture the taken piece is
lifted to storage *first*, then the capturing piece moves
([choreography.py](../src/chessmachine/motion/choreography.py) `execute_move`).

**One-way dish caveat.** Captured pieces are dumped into a single dish and pile up
randomly, so they can't be individually retrieved (`graveyard_retrievable: false`).
Promotion (no spare to fetch), undo of a capture, and "new game" reset therefore ask
you to place the affected pieces by hand rather than the crane fishing them out. The
dish holds `graveyard_capacity` (default 30) before a capture is refused — enough for
any game's total captures.

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
holding load. The build is now a **mixed drivetrain**: the radial cart is a geared
brushed DC motor (cheap, simple, open-loop by time); the rotary base and the winch
are both **28BYJ-48 steppers via ULN2003 drivers** — step-counted, so neither
accumulates the timing drift a relay-timed DC axis does.

- **Rotary base (θ → protocol `A`).** **28BYJ-48 stepper via a ULN2003**, moved by
  exact step count (`A_STEPS_PER_DEG`) rather than a timed slew rate — this
  replaced an earlier DC + H-bridge base, which under-rotated under load/backlash.
  No rotary endstop is wired; `HOME` instead sweeps a **down-looking ultrasound
  sensor** left/right until it sees a box placed at the home bearing, and zeros
  `A` there. Soft limits (`A_MIN_DEG`/`A_MAX_DEG`) bound the sweep — note the
  ultrasound seek itself does not clamp to them, so the home box must be in place
  before homing.
- **Radial railcart (r → protocol `R`).** Still a **geared DC motor via a 2-relay
  H-bridge**, dragging a low-friction cart along the arm (the cart also carries
  the winch stepper). Timed from the measured traverse — **320 mm end to end**,
  ~1150 ms out / ~950 ms in, plus a 30 ms start deadzone — with a load-correction
  factor applied on top (bench-unloaded timing under-extends the real, loaded
  cart). No endstop is wired; `HOME` re-zeros `R` by driving inward for a capped
  time, ramming the inner mechanical stop (safe for this DC mechanism, unlike the
  steppers).
  - ⚠️ Relays are bang-bang (full speed / off): **no speed control**, so the
    protocol's feed `F` is ignored on this axis — travel time is fixed by the motor.
- **Winch (pulley `H`).** Same stepper type as the base (28BYJ-48 / ULN2003), but
  simpler: it has exactly **two fixed positions** — TRAVEL (up, rest) and PICK
  (down) — a measured step stroke apart, no top endstop, no continuous mm
  positioning. It **holds its position between moves** (coils energized) so the
  hanging load can't back-drive it.

**Guide vs drive.** A linear rail/carriage on the arm is the *guide* (bears load,
keeps motion straight and low-friction); the motor (DC or stepper) is the *drive*.
Complementary.

**Open-loop accuracy.** Neither drive type has feedback, so error can still
accumulate: the DC radial axis from relay lag/coasting/wheel slip (mitigated by
re-homing — ramming its stop is a real, if blunt, re-reference); the stepper axes
from skipped steps if ever stalled or under-torqued (mitigated by *not* forcing
them against a hard limit — the ultrasound seek and the winch's fixed stroke are
the closed-enough references). Re-home between phases, keep the per-axis
rates/step-counts calibrated, and lean on the recessed square wells (§4.2) to
capture the last few mm. If drift becomes a problem, the natural upgrade is real
endstops/encoders — but bench-tune the open-loop numbers first.

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
  reach) are measured on the real crane (~118 / ~390–410 mm depending on config);
  re-run `scripts/calibrate.py` if the mechanism changes. Every board/graveyard
  coordinate follows from them.
- **Radial (DC) axis calibration** — measure the cart's rate and start deadzone
  (mm/s + ms, per direction) and set `R_MS_PER_MM_OUT`/`R_MS_PER_MM_IN`; re-check
  after any supply-voltage or load change (a `R_LOAD_FACTOR` already corrects for
  loaded vs. bench-unloaded speed — re-derive it from a `JOG R<ms>` test rather
  than assuming it still holds after a mechanical change).
- **Rotary (stepper) axis calibration** — `A_STEPS_PER_DEG` is a placeholder
  assuming direct drive; calibrate with `JOG A<steps>` (rotate a known step
  count, measure the swept angle) and re-check if the base is geared.
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
