# esp32_chess firmware

Motor controller for the chess machine. Receives high-level commands from the
host over USB serial and drives the polar crane with the real hybrid drivetrain:

- **R axis** (radial, mm from the pivot) — the **linear cart** on the arm. Brushed
  DC motor via a **2-relay H-bridge**, positioned by **time** (open-loop).
- **A axis** (angle, degrees) — the **rotating base**. Brushed DC motor via a
  **2-relay H-bridge**, positioned by **time**.
- **Pulley H** (mm) — the **winch**. 28BYJ-48-style stepper via a **ULN2003** driver.
- **Magnet** — one relay, on/off.

The wire protocol is unchanged from the old all-stepper build (see
[../../docs/PROTOCOL.md](../../docs/PROTOCOL.md)); only the motor layer changed.

## Hardware

| Function              | Default pin(s) | Notes                                             |
|-----------------------|----------------|---------------------------------------------------|
| Radial H-bridge FWD/REV (relay 3) | 26 / 16 | cart out / in along the arm (r, mm). Swap to invert. |
| Rotary H-bridge FWD/REV (relay 2) | 25 / 27 | base + / − degrees. Swap to invert.               |
| Winch ULN2003 IN1–IN4 | 33 / 32 / 18 / 19 | 28BYJ-48 half-step. Swap IN2↔IN3 if it only vibrates. |
| Radial endstop        | 13             | `INPUT_PULLUP`, switch to GND, LOW = pressed (inner end) |
| Rotary endstop        | 14             | rotary home switch                                |
| Winch top endstop     | 15             | homes the magnet to its highest point             |
| Electromagnet (relay 1) | 23           | relay / MOSFET; `MAGNET_ACTIVE_LOW` sets polarity |
| *(free)*              | 5              | old shared enable; reserved for the winch's own power relay |

Each DC motor uses a **2-SPDT-relay H-bridge**: energize one relay for forward,
the other for reverse, both released = coast (STOP). We pass through STOP with a
30 ms dead-time on every reversal so the supply is never shorted. Power the motors
and the ULN2003 from **their own supply**, common ground with the ESP32. Put a
**flyback diode / snubber** across the magnet coil, and add **external ~10 kΩ
pull-ups** on GPIO22/23 if your relay board is active-low (those pins float at
boot and a motor/magnet could twitch before `setup()` runs).

## Tuning (top of the sketch)

- `R_MS_PER_MM` / `A_MS_PER_DEG` — the core calibration: how long the motor runs
  per mm / per degree. Defaults derived from the bench numbers (315 mm in 1020 ms
  of motion; 0.1887 rad/s). **Measure against a ruler / protractor and adjust.**
- `R_DEADZONE_MS` / `A_DEADZONE_MS` — dead-time before each axis actually moves
  (30 ms / 50 ms); added to every timed run.
- Wrong direction? **Swap the FWD/REV pins** for that axis (no invert flag).
- `R_HOME_MM` / `A_HOME_DEG` — the radius / angle at each endstop, so absolute
  moves are correct straight after homing. `*_HOME_BACKOFF_*` backs off the switch.
- `R_MIN_MM`/`R_MAX_MM`, `A_MIN_DEG`/`A_MAX_DEG` — soft limits of the reachable
  annular sector; the host stays inside these but they backstop a bad command.
  (The rail is mechanically ~315 mm; the board only uses the [80, 300] annulus —
  the ms/mm rate is range-independent, so that's fine.)
- `RELAY_ACTIVE_LOW` / `MAGNET_ACTIVE_LOW` — relay board polarity.
- **Winch (placeholder):** `P_STEPS_PER_MM` must be calibrated to the drum;
  `P_UP_STEP_DIR` sets which way raises the magnet; `P_MAX_HEIGHT_MM` is the
  height when the top endstop trips; `WINCH_STEP_DELAY_MS` sets winch speed. The
  winch **holds its height between moves** (coils stay energized — holding torque,
  runs warm), releasing only on ESTOP / boot. The winch is a first pass — its
  3.3 cm offset and the real piece heights come later.

The DC axes are relays (bang-bang), so the host's feed `F` is **accepted and
ignored** — travel time is fixed by each motor's speed.

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
