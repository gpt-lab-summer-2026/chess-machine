# Host ↔ ESP32 serial protocol

Plain-text, line-based, **115200 baud, 8N1**. The host sends one command per
line (`\n`-terminated); the ESP32 replies when the command completes. Every
motion command **blocks on the ESP32 until the move finishes**, so the host can
sequence moves just by waiting for `OK`.

Implemented by [`firmware/esp32_chess`](../firmware/esp32_chess/) (device) and
[`serial_esp32.py`](../src/chessmachine/motion/serial_esp32.py) (host).

## Commands

| Command | Reply | Meaning |
|---|---|---|
| `PING` | `OK PONG` | liveness check (used on connect) |
| `HOME` | `OK HOMED` | home X, Z, and the pulley against their endstops |
| `MOVE X<mm> Z<mm> [F<mm/min>]` | `OK` | move the gantry to (x, z); `F` optional feedrate |
| `PULLEY H<mm> [F<mm/min>]` | `OK` | set magnet height above the board |
| `MAG ON` / `MAG OFF` | `OK` | energize / release the electromagnet |
| `STATUS` | `OK X<f> Z<f> H<f> MAG<0\|1> ENDX<0\|1> ENDZ<0\|1>` | current state |
| `ESTOP` | `OK ESTOP` | disable motors + magnet; cleared by `HOME` |

## Conventions

- **Units:** millimetres for positions, mm/min for feedrates. Coordinates are
  absolute from the homed origin (X0, Z0 = the homed corner; height 0 = board
  surface, larger = magnet higher / cable retracted).
- **Argument order is free:** `MOVE Z50 X20 F3000` is valid. Missing axis args
  keep the current value; missing `F` uses the firmware default feed.
- **Acknowledgements:** a line starting with `OK` is success (optionally followed
  by a payload). A line starting with `ERR ` is a failure (e.g. `ERR estopped;
  send HOME to clear`, `ERR MAG expects ON or OFF`). The host raises on `ERR`.
- **Async / debug:** lines starting with `#` are debug logs; anything starting
  with `EVT` is reserved for future async events. The host logs and ignores both
  while waiting for `OK`/`ERR`.
- **Reset on connect:** opening the serial port resets the ESP32. The host waits
  `serial.connect_settle_s` (≈2 s) and flushes input before the first `PING`.

## Example session

```
-> PING
<- # esp32_chess ready
<- OK PONG
-> HOME
<- OK HOMED
-> MOVE X95.00 Z38.75 F4000     # over square e2
<- OK
-> PULLEY H4.00 F1500           # lower onto the piece
<- OK
-> MAG ON                       # grab it
<- OK
-> PULLEY H60.00 F1500          # lift to travel height
<- OK
-> MOVE X95.00 Z76.25 F4000     # over square e4
<- OK
-> PULLEY H4.00 F1500
<- OK
-> MAG OFF                      # release
<- OK
-> PULLEY H60.00 F1500
<- OK
```

That nine-command sequence is one `_transfer` in
[`choreography.py`](../src/chessmachine/motion/choreography.py); a full move is
one or more transfers depending on captures/castling/promotion.

## Timeouts

The host uses `serial.timeout_s` for normal commands and `serial.home_timeout_s`
for `HOME` (homing sweeps can be slow). A timeout raises `TimeoutError`.
