# Hardware & calibration

## Mechanism

- **Board:** 3D-printed, ~150 mm physical. Because the gantry stroke is only
  150 mm, the **playable 8×8 grid is ~120 mm (15 mm square pitch)** and the
  remaining edge holds the storage strip — everything must fit inside the
  150 mm reach. See [DESIGN.md §4.1](DESIGN.md) for the geometry budget.
  Checkers-style pieces, each with a magnet inside.
- **Gantry:** two linear motors (**15 cm stroke**) form an X/Z grid. X carries
  files a–h, Z carries ranks 1–8 (swappable in config).
- **Pulley:** a winch raises/lowers an **electromagnet on a cable** to pick up
  and place pieces from above.
- **Storage ("graveyard"):** an in-reach strip beside the grid (default 2×8 =
  16 slots, "2 per row") holding captured pieces; also the source for
  promotions. Note: 16 slots can't stage a full 32-piece reset — see DESIGN.md.
- **Controller:** ESP32 drives three stepper axes + the magnet, talking to the
  Pi over USB serial. See [PROTOCOL.md](PROTOCOL.md) and
  [../firmware/esp32_chess/README.md](../firmware/esp32_chess/README.md).

## Why lift-and-carry

Pieces are lifted straight up to a **travel height that clears the tallest
piece**, carried over the top, and lowered onto the target. This handles any
move — including knights jumping over pieces — without sliding between squares.
`speeds.settle_ms` adds a pause after each move to damp cable swing.

> ⚠️ Drive the electromagnet through a logic-level MOSFET (or relay) from its own
> supply, with a **flyback diode** across the coil. Do not source coil current
> from an ESP32 pin.

## Coordinate model

Square centers are computed by [`geometry.py`](../src/chessmachine/motion/geometry.py):

```
point(file, rank) = origin + pitch * (file_or_rank index along each axis)
```

- `origin_x_mm`, `origin_z_mm` — center of **a1**, from the homed (0,0) corner
  (default 10, 10).
- `square_pitch_mm` — 120 mm playable grid / 8 = **15.0** (must fit the 150 mm stroke).
- `file_axis` / `rank_axis` (`x`/`z`) + `invert_file` / `invert_rank` — orient the
  logical board to however it physically sits under the gantry.
- `travel_height_mm` — magnet raised (must clear pieces, stay below the pulley's
  `P_MAX_HEIGHT_MM`).
- `pick_height_mm` — magnet lowered to contact a piece.

### Graveyard

Slots are generated column-major from `graveyard_x_mm` / `graveyard_z_start_mm`
with `graveyard_*_pitch_mm`, `graveyard_slots_per_column`, and
`graveyard_columns`. The default 2×8 = **16 slots** (at x = 127.5 / 142.5 mm)
sits inside the **150 mm reach**. A full board reset needs 32 transient slots, so
with 16 the machine asks for a manual reset; bump to 4 columns (32) only if you
shrink the grid to keep everything ≤ 150 mm. See [DESIGN.md §4.1](DESIGN.md).

## Calibration workflow

```bash
python scripts/calibrate.py --config config/config.yaml
```

Jog the gantry to the **center of a1**, `mark a1`; jog to the **center of h8**,
`mark h8`; then `calc` prints an `origin_x_mm` / `origin_z_mm` /
`square_pitch_mm` / inversion block to paste into `config.yaml`. Use
`goto e4` to verify, and `mag on` / `h -2` to tune `pick_height_mm`. Rehearse the
flow without hardware using `--mock`.

On the firmware side, set `*_STEPS_PER_MM`, direction inversions, `*_HOME_DIR`,
and `P_MAX_HEIGHT_MM` at the top of the sketch to match your drivers and
mechanics.
