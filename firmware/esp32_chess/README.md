# esp32_chess firmware

Motor controller for the chess machine. Receives high-level commands from the
host over USB serial and drives the polar crane with a mixed drivetrain:

- **R axis** (radial, mm from the pivot) — the **linear cart** on the arm. Brushed
  DC motor via a **2-relay H-bridge**, positioned by **time** (open-loop). No
  endstop wired; `HOME` re-zeros it by timed ram into the inner mechanical stop
  (safe for this mechanism).
- **A axis** (angle, degrees) — the **rotating base**. **28BYJ-48 stepper via a
  ULN2003** driver, positioned by **step count** (open-loop, but exact — no
  timing drift or backlash the way a timed DC motor has). `HOME` finds the
  reference angle by sweeping a down-looking ultrasound sensor until it sees a
  box placed at the home bearing — **not** by ramming a stop (a stepper's
  gearbox is not safe to stall against a hard limit the way the DC cart is).
- **Pulley H** (mm) — the **winch**. Same stepper type as the base (28BYJ-48 /
  ULN2003). Two fixed positions — TRAVEL (up, rest) and PICK (down) — a
  measured step stroke apart; no top endstop.
- **Magnet** — one relay, on/off.

The wire protocol is unchanged regardless of which motor type sits behind each
axis (see [../../docs/PROTOCOL.md](../../docs/PROTOCOL.md)).

## Hardware

| Function                    | Default pin(s)     | Notes                                                        |
|------------------------------|--------------------|---------------------------------------------------------------|
| Radial H-bridge FWD/REV (relay 3) | 22 / 23      | cart out / in along the arm (r, mm). Swap to invert.         |
| Radial endstop               | 32                 | placeholder, **not wired** — `HOME` rams the inner stop instead |
| Base stepper ULN2003 IN1–IN4 | 27 / 26 / 25 / 33  | 28BYJ-48 half-step. Swap IN2↔IN3 if it only vibrates.         |
| Winch stepper ULN2003 IN1–IN4| 5 / 21 / 18 / 17   | 28BYJ-48 half-step. Same swap note as above.                  |
| Ultrasound trig / echo       | 13 / 12            | down-looking HC-SR04-style sensor, used to home the base      |
| Electromagnet (relay 1)      | 19                 | relay / MOSFET; `MAGNET_ACTIVE_LOW` sets polarity             |

The radial H-bridge is **2 SPDT relays**: energize one for forward, the other
for reverse, both released = coast (STOP). We pass through STOP with a 30 ms
dead-time on every reversal so the supply is never shorted. Power the DC motor
and both ULN2003 drivers from **their own supply**, common ground with the
ESP32. Put a **flyback diode / snubber** across the magnet coil and the DC
motor terminals, and add **external ~10 kΩ pull-ups** to the de-energized
level on the relay control pins if your relay board is active-low (GPIOs float
at boot / during a brownout reset, and a floating H-bridge input pair can spin
the motor unattended — see the `resetReasonStr()` boot log below).

## Tuning (top of the sketch)

- `R_MS_PER_MM_OUT` / `R_MS_PER_MM_IN` — the radial cart's core calibration: ms
  per mm in each direction, derived from bench numbers and an empirical
  `R_LOAD_FACTOR` (the loaded cart is slower than the unloaded bench test).
  **Re-measure with `JOG R<ms>` and adjust the factor.**
- `R_DEADZONE_MS` — dead-time before the cart actually moves; added to every
  timed run so short moves aren't undershot.
- `A_STEPS_PER_DEG` — the base's steps-per-degree; a **placeholder** assuming
  direct drive. **Calibrate with `JOG A<steps>`**: rotate a known step count,
  measure the swept angle, `steps/deg = steps / degrees`.
- Wrong direction? DC axis: swap the FWD/REV pins. Stepper axis: flip
  `A_STEP_DIR` (or `P_UP_STEP_DIR` for the winch).
- ⚠️ **The pin comment and the calibration-constant names (`_OUT`/`_IN`,
  "FORWARD = +r") describe the ORIGINAL wiring assumption, but `dcStartMove`'s
  direction logic no longer matches it** (`FORWARD` is chosen when the target
  is *smaller* than the current position). If the two disagree, the code's
  behavior wins at runtime — but this means the names are misleading. Verify
  with `JOG R<ms>` which relay actually drives which way before trusting `HOME`
  / soft limits, and fix the comments/names to match reality once confirmed.
- `R_MIN_MM` / `R_MAX_MM`, `A_MIN_DEG` / `A_MAX_DEG` — soft limits of the
  reachable annular sector; the host stays inside these but they backstop a bad
  command.
- `US_HOME_THRESHOLD_CM` / `US_SEEK_DEG` — the base's ultrasound homing: how
  close a reading must be to count as "box below", and how far each way to
  sweep looking for it. `seekBoxSweep`/`baseStep` do **not** clamp to
  `A_MIN_DEG`/`A_MAX_DEG` — if the box is missing, the sweep can drive the base
  past its soft limits. Make sure the home box is in place before calling `HOME`.
- `WINCH_STROKE_STEPS` — the winch's measured TRAVEL↔PICK step count; the
  winch has no continuous mm positioning anymore, just these two positions.
- `MAGNET_ACTIVE_LOW` — relay/MOSFET polarity for the magnet (the only relay
  left in the system).

Neither the DC axis nor the steppers have real speed control, so the host's
feed `F` is **accepted and ignored** on every axis.

## Build & flash

No external libraries required.

**Arduino IDE / arduino-cli** — install the ESP32 board package, then:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/esp32_chess
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/esp32_chess
```

**PlatformIO** — from this folder: `pio run -t upload`.

## Test it

```bash
python scripts/serial_console.py --port /dev/ttyUSB0
> PING
    OK PONG
> HOME
    OK HOMED
> MOVE R150 A0
    OK
> PULLEY H4
    OK
> MAG ON
    OK
> STATUS
    OK R150.00 A0.00 H4.00 MAG1 ENDR0 ENDA0
```

See [../../docs/PROTOCOL.md](../../docs/PROTOCOL.md) for the full protocol spec.
