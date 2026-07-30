# Hardware & calibration

## Mechanism

- **Board:** 3D-printed, **~210 mm** square 8×8 grid (**~26 mm pitch**, 297.5 mm
  diagonal), sitting inside the crane's reach. Checkers-style pieces (≤70 g), each
  with a magnet inside. See [DESIGN.md §4.1](DESIGN.md) for the geometry budget.
- **Crane:** a rotary base carries an arm; a **railcart runs along the arm**.
  Rotation is the **angular** coordinate (θ); the railcart is the **radial**
  coordinate (r). The reachable workspace is an **annular sector** (r ∈ [80,
  455] mm — 455 is the max radius, not the 355 mm cart travel). The board sits
  **corner-first (diagonal)** — h8 in the near deadzone, a1 farthest — so the sweep
  is only **~±28°**; the wedges beyond it hold the
  graveyards.
- **Pulley:** a winch raises/lowers a **6 V electromagnet on a cable** to pick
  up and place pieces from above.
- **Storage ("graveyard"):** two symmetric arcs in the leftover sector, beyond
  the board's far corners (default 2 × 8 = 16 slots) holding captured pieces;
  also the source for promotions. 16 slots can't stage a full 32-piece
  reset — see DESIGN.md.
- **Controller:** ESP32 drives the **radial** cart as a **timed DC motor via a
  relay H-bridge**, and the **rotary base + pulley winch as ULN2003 steppers**
  (the base homes by sweeping a down-looking ultrasound sensor for a box at the
  home bearing, rather than an endstop switch), plus the magnet relay, talking
  to the Pi over USB serial. See [PROTOCOL.md](PROTOCOL.md) and
  [../firmware/esp32_chess/README.md](../firmware/esp32_chess/README.md).

## Why lift-and-carry

Pieces are lifted straight up to a **travel height that clears the tallest
piece**, carried over the top, and lowered onto the target. This handles any
move — including knights jumping over pieces — without sliding between squares.
`speeds.settle_ms` adds a pause after each move to damp cable swing.

> ⚠️ Drive the electromagnet through a logic-level MOSFET (or relay) from its own
> **6 V** supply, with a **flyback diode** across the coil. Do not source coil
> current from an ESP32 pin.

## Coordinate model

The board is a flat grid, so square centers are computed in **planar (x, z) mm**
by [`geometry.py`](../src/chessmachine/motion/geometry.py), in a frame whose
**origin is the crane pivot**:

```
point(file, rank) = origin + Rot(board_angle) · pitch·(file, rank indices)
```

- `origin_x_mm`, `origin_z_mm` — center of **a1** in the pivot frame
  (real config: 380, 0 — a1 is the **far** corner on the bisector; see DESIGN §4.1).
- `square_pitch_mm` — ~210 mm board / 8 ≈ **26.3** (from the 297.5 mm diagonal).
- `r_min_mm` / `r_max_mm` — the reachable annulus (80 / 455 mm; 455 = max radius,
  not the 355 mm cart travel).
- `board_angle_deg` — rotates the board about a1 (**135°** = the corner-first
  diagonal); `file_axis` / `rank_axis` + `invert_file` / `invert_rank` cover the
  remaining axis assignment / mirroring.
- `travel_height_mm` — magnet raised (must clear pieces, stay below the pulley's
  `P_MAX_HEIGHT_MM`).
- `pick_height_mm` — magnet lowered to contact a piece.

The transport ([`serial_esp32.py`](../src/chessmachine/motion/serial_esp32.py))
converts each planar point to polar — `r = hypot(x,z)`, `θ = atan2(z,x)` — and
sends `MOVE R… A…`. Nothing above the transport knows the machine is a crane.

### Graveyard

Captured pieces are **dumped into a single circular dish** on the h1/−θ side (the
a8/+θ side is blocked by the base limit switch). The dish sits at
`graveyard_center_r_mm` / `graveyard_center_deg` (**more negative = further off the
board's g/h edge** — the one tuning knob), and drops scatter within
`graveyard_jitter_mm` of the centre so they don't all stack on one point. Pieces
**pile up randomly**, so the dish is **one-way** (`graveyard_retrievable: false`):
the machine can't fish a specific piece back out, so promotion, undo, and reset ask
you to place those pieces by hand. See [DESIGN.md §4.1](DESIGN.md).

## Calibration workflow

```bash
python scripts/calibrate.py --config config/config.yaml
```

Jog the head (in planar x/z) to the **center of a1**, `mark a1`; jog to the
**center of h8**, `mark h8`; then `calc` prints an `origin_x_mm` /
`origin_z_mm` / `square_pitch_mm` / inversion block to paste into `config.yaml`.
Use `goto e4` to verify, and `mag on` / `h -2` to tune `pick_height_mm`.
Rehearse without hardware using `--mock`.

On the firmware side, set `R_MS_PER_MM_OUT`/`R_MS_PER_MM_IN` (radial DC timing),
`A_STEPS_PER_DEG` (base stepper), direction inversions (swap the R H-bridge
pins, or flip `A_STEP_DIR`/`P_UP_STEP_DIR`), `R_HOME_MM`/`A_HOME_DEG`, the
`R`/`A` soft limits, and the winch's `WINCH_STROKE_STEPS` at the top of the
sketch to match your drivers and mechanics.
