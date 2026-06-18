# esp32_chess firmware

Motor controller for the chess machine. Receives high-level commands from the
Raspberry Pi over USB serial and drives the gantry, pulley, and electromagnet.

## Hardware

| Function            | Default pin | Notes                                        |
|---------------------|-------------|----------------------------------------------|
| X step / dir        | 26 / 16     | gantry axis 1 (files a–h)                    |
| Z step / dir        | 25 / 27     | gantry axis 2 (ranks 1–8)                    |
| Pulley step / dir   | 33 / 32     | winch raising/lowering the magnet            |
| Driver enable       | 5           | shared, active **LOW**                       |
| X / Z endstop       | 13 / 14     | `INPUT_PULLUP`, switch to GND, LOW = pressed |
| Pulley top endstop  | 15          | homes the magnet to its highest point        |
| Electromagnet       | 23          | MOSFET gate or relay, HIGH = energized       |

Use proper stepper drivers (A4988 / DRV8825 / TMC2209) and a **flyback diode**
across the electromagnet coil. Power the magnet from its own supply, switched by
a logic-level MOSFET — do not drive it from an ESP32 pin directly.

## Tuning (top of the sketch)

- `*_STEPS_PER_MM` — set from your microstepping and belt/leadscrew pitch.
- `*_DIR_INVERT`, `*_HOME_DIR` — flip if an axis runs the wrong way / homes away
  from its switch.
- `P_MAX_HEIGHT_MM` — magnet height above the board when the top endstop trips.
  Keep `travel_height_mm` (host config) below this and above the tallest piece.
- `X_MAX_MM`, `Z_MAX_MM` — soft travel limits; `MOVE` targets are clamped into
  `[0, *_MAX_MM]` so a bad coordinate can't drive into the frame. Set to your
  usable axis length.
- `MAX_FEED_MM_MIN`, `ACCEL_MM_S2` — software stepping on the ESP32 tops out
  around 8–10k steps/s; lower microstepping if you need faster travel.

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
> MOVE X50 Z50 F4000
    OK
> PULLEY H4
    OK
> MAG ON
    OK
> STATUS
    OK X50.00 Z50.00 H4.00 MAG1 ENDX0 ENDZ0
```

See [../../docs/PROTOCOL.md](../../docs/PROTOCOL.md) for the full protocol spec.
