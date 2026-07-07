#!/usr/bin/env python3
"""Bring-up tool for the DC-motor relay backend (firmware/esp32_dc_prototype).

Drives the SAME `RelayMotion` backend the chess stack uses, but with nothing else
in the way — so you can prove out the ESP32 firmware, the relay H-bridge wiring,
and the motor with plain forward / reverse before any chess logic is involved.

    python scripts/relay_jog.py --port /dev/ttyUSB0          # interactive
    python scripts/relay_jog.py --config config/config.yaml  # port from config
    python scripts/relay_jog.py --port COM3 --selftest       # F 500ms, R 500ms, stop

Interactive commands (one per line):
    f [ms]     run forward  (default: config pulse_ms)
    r [ms]     run reverse
    s          stop now
    status     print motor state
    estop      emergency stop + latch
    clear      release the estop latch
    quit
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config  # noqa: E402
from chessmachine.motion.relay_esp32 import RelayMotion  # noqa: E402


def _drive(ctl: RelayMotion, forward: bool, ms: int) -> None:
    print(f"    {'FWD' if forward else 'REV'} {ms} ms ...", flush=True)
    ctl.drive(forward, ms)
    print("    stopped.")


def main() -> int:
    ap = argparse.ArgumentParser(description="DC-motor relay bring-up tool")
    ap.add_argument("--config", help="YAML config (reads motion.relay.* )")
    ap.add_argument("--port", help="serial port (overrides config; e.g. /dev/ttyUSB0, COM3)")
    ap.add_argument("--baud", type=int, help="baud rate (overrides config)")
    ap.add_argument("--pulse-ms", type=int, help="default run duration per pulse (overrides config)")
    ap.add_argument("--selftest", action="store_true",
                    help="run forward then reverse once and exit")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    relay = cfg.motion.relay
    if args.port:
        relay.port = args.port
    if args.baud:
        relay.baud = args.baud
    if args.pulse_ms:
        relay.pulse_ms = args.pulse_ms

    ctl = RelayMotion(relay)
    print(f"Connecting to {relay.port} @ {relay.baud} ...", flush=True)
    ctl.connect()
    print(f"Connected. Default pulse = {relay.pulse_ms} ms.")

    try:
        if args.selftest:
            _drive(ctl, True, relay.pulse_ms)
            _drive(ctl, False, relay.pulse_ms)
            print("Self-test done.")
            return 0

        print("Type a command (f/r/s/status/estop/clear), or 'quit'.")
        for line in sys.stdin:
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd in ("f", "fwd", "forward"):
                _drive(ctl, True, int(arg) if arg else relay.pulse_ms)
            elif cmd in ("r", "rev", "reverse"):
                _drive(ctl, False, int(arg) if arg else relay.pulse_ms)
            elif cmd in ("s", "stop"):
                ctl.stop()
                print("    stopped.")
            elif cmd == "status":
                print("   ", ctl.status())
            elif cmd == "estop":
                ctl.estop()
                print("    ESTOP latched.")
            elif cmd == "clear":
                ctl.clear()
                print("    cleared.")
            else:
                print(f"    ? unknown command {cmd!r}")
    except KeyboardInterrupt:
        pass
    finally:
        ctl.close()  # sends STOP and releases the port
    return 0


if __name__ == "__main__":
    sys.exit(main())
