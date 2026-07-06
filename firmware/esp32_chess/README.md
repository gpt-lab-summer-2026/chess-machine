# esp32_chess firmware

Motor controller for the chess machine. Receives high-level commands from the
Raspberry Pi over USB serial and drives the polar crane (rotary base + radial
railcart), the pulley winch, and the electromagnet.

## Hardware

| Function            | Default pin | Notes                                        |
|---------------------|-------------|----------------------------------------------|
| Radial step / dir   | 26 / 16     | railcart along the arm (r, mm from pivot)    |
| Rotary step / dir   | 25 / 27     | crane base rotation (θ, degrees)             |
| Pulley step / dir   | 33 / 32     | winch raising/lowering the magnet            |
| Driver enable       | 5           | shared, active **LOW**                       |
| Radial endstop      | 13          | `INPUT_PULLUP`, switch to GND, LOW = pressed |
| Rotary endstop      | 14          | rotary home switch                           |
| Pulley top endstop  | 15          | homes the magnet to its highest point        |
| Electromagnet       | 23          | MOSFET gate or relay, HIGH = energized       |

Use proper stepper drivers (A4988 / DRV8825 / TMC2209) and a **flyback diode**
across the electromagnet coil. The magnet runs from its own **6 V** supply,
switched by a logic-level MOSFET — do not drive it from an ESP32 pin directly.

## Tuning (top of the sketch)

- `R_STEPS_PER_MM` — radial railcart, from microstepping + belt pitch.
- `A_STEPS_PER_DEG` — rotary base, from microstepping × gear ratio ÷ 360.
- `*_DIR_INVERT`, `*_HOME_DIR` — flip if an axis runs the wrong way / homes away
  from its switch.
- `R_HOME_MM` / `A_HOME_DEG` — the radius / pivot-frame angle at each endstop, so
  absolute moves are correct straight after homing.
- `R_MIN_MM`/`R_MAX_MM`, `A_MIN_DEG`/`A_MAX_DEG` — soft limits of the reachable
  annular sector; the host stays inside these but they backstop a bad command.
- `P_MAX_HEIGHT_MM` — magnet height above the board when the top endstop trips.
  Keep `travel_height_mm` (host config) below this and above the tallest piece.
- `MAX_FEED_MM_MIN`, `ROTARY_SPEED_DEG_S`, `ACCEL_MM_S2` — software stepping on
  the ESP32 tops out around 8–10k steps/s; lower microstepping if you need
  faster travel. The host's feed `F` is radial mm/min; the rotary axis slews at
  `ROTARY_SPEED_DEG_S`.

## Build & flash

**Arduino IDE / arduino-cli** — install the ESP32 board package and the
*AccelStepper* library, then:

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
> MOVE R150 A0 F4000
    OK
> PULLEY H4
    OK
> MAG ON
    OK
> STATUS
    OK R150.00 A0.00 H4.00 MAG1 ENDR0 ENDA0
```

See [../../docs/PROTOCOL.md](../../docs/PROTOCOL.md) for the full protocol spec.
