# Hardware & calibration

## Mechanism

- **Board:** 3D-printed, **190 mm** square 8×8 grid (**23.75 mm square pitch**),
  sitting inside the crane's reach. Checkers-style pieces (≤70 g), each with a
  magnet inside. See [DESIGN.md §4.1](DESIGN.md) for the geometry budget.
- **Crane:** a ~48 cm-high rotary base carries a ~30 cm arm; a **railcart runs
  along the arm**. Rotation is the **angular** coordinate (θ); the railcart is
  the **radial** coordinate (r). The reachable workspace is an **annular sector**
  (r ∈ [80, 300] mm, sweep ~100°). The board is a square centered on the sector;
  the leftover wedges hold the graveyards.
- **Pulley:** a winch raises/lowers a **6 V electromagnet on a cable** to pick
  up and place pieces from above.
- **Storage ("graveyard"):** two symmetric arcs in the leftover sector, beyond
  the board's far corners (default 2 × 8 = 16 slots) holding captured pieces;
  also the source for promotions. 16 slots can't stage a full 32-piece
  reset — see DESIGN.md.
- **Controller:** ESP32 drives three stepper axes (radial, rotary, pulley) + the
  magnet, talking to the Pi over USB serial. See [PROTOCOL.md](PROTOCOL.md) and
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
point(file, rank) = origin + pitch * (file_or_rank index along each axis)
```

- `origin_x_mm`, `origin_z_mm` — center of **a1** in the pivot frame
  (default 91.875, −83.125: near edge at r_min, board centered on the bisector).
- `square_pitch_mm` — 190 mm board / 8 = **23.75**.
- `r_min_mm` / `r_max_mm` — the reachable annulus (80 / 300 mm).
- `file_axis` / `rank_axis` (`x`/`z`) + `invert_file` / `invert_rank` — orient the
  logical board to however it physically sits in the sector.
- `travel_height_mm` — magnet raised (must clear pieces, stay below the pulley's
  `P_MAX_HEIGHT_MM`).
- `pick_height_mm` — magnet lowered to contact a piece.

The transport ([`serial_esp32.py`](../src/chessmachine/motion/serial_esp32.py))
converts each planar point to polar — `r = hypot(x,z)`, `θ = atan2(z,x)` — and
sends `MOVE R… A…`. Nothing above the transport knows the machine is a crane.

### Graveyard

Slots sit on **two symmetric arcs** at `graveyard_radius_mm` (default 293 mm,
beyond the board's ~286 mm far corners so the arc always clears the board),
spread between `graveyard_inner_deg` and `graveyard_outer_deg` on each side of
the bisector. The default `graveyard_slots_per_side = 8` gives **16 slots**. A
full board reset needs 32 transient slots, so with 16 the machine asks for a
manual reset. See [DESIGN.md §4.1](DESIGN.md).

## Calibration workflow

```bash
python scripts/calibrate.py --config config/config.yaml
```

Jog the head (in planar x/z) to the **center of a1**, `mark a1`; jog to the
**center of h8**, `mark h8`; then `calc` prints an `origin_x_mm` /
`origin_z_mm` / `square_pitch_mm` / inversion block to paste into `config.yaml`.
Use `goto e4` to verify, and `mag on` / `h -2` to tune `pick_height_mm`.
Rehearse without hardware using `--mock`.

On the firmware side, set `R_STEPS_PER_MM`, `A_STEPS_PER_DEG`, direction
inversions, `*_HOME_DIR`, `R_HOME_MM` / `A_HOME_DEG`, the `R`/`A` soft limits,
and `P_MAX_HEIGHT_MM` at the top of the sketch to match your drivers and
mechanics.
