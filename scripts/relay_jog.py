#!/usr/bin/env python3
"""Bring-up tool for the DC-motor relay backend (firmware/esp32_dc_prototype).

Drives the SAME `RelayMotion` backend the chess stack uses, but with nothing else
in the way — so you can prove out the ESP32 firmware, the relay H-bridge wiring,
and each motor with plain forward / reverse before any chess logic is involved.

    python scripts/relay_jog.py --port /dev/ttyUSB0          # interactive
    python scripts/relay_jog.py --config config/config.yaml  # port from config
    python scripts/relay_jog.py --port COM3 --selftest       # cycle every motor

Actuators (see the firmware's tables):
    1  base relay (single, FWD only)   2  boom rail (H-bridge)   3  lift wire (H-bridge)
    stepper via ULN2003 (STEP command)

Interactive commands (one per line):
    <m> f [ms]     run DC motor m forward   (e.g. "1 f", "2 f 500")
    <m> r [ms]     run DC motor m reverse   (H-bridge motors only)
    <m> s          stop DC motor m
    step <n> [ms]  move stepper n steps (signed: +/- = direction), ms per step
    s              stop ALL (motors + stepper)
    status         print actuator state
    estop          emergency stop + latch    clear   release the latch
    quit
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config  # noqa: E402
from chessmachine.motion.relay_esp32 import RelayMotion  # noqa: E402

MOTOR_NAMES = {1: "base rotation", 2: "boom rail (horizontal)", 3: "lift wire (vertical)"}


def _drive(ctl: RelayMotion, motor: int, forward: bool, ms: int) -> None:
    name = MOTOR_NAMES.get(motor, f"motor {motor}")
    print(f"    M{motor} {name}: {'FWD' if forward else 'REV'} {ms} ms ...", flush=True)
    ctl.drive(forward, ms, motor)
    print("    stopped.")


def _run_interactive(ctl: RelayMotion, default_ms: int) -> None:
    print("Type '<motor> <dir> [ms]' e.g. '1 r' or '2 f 500', 'step <n> [ms]', "
          "'s' stops all, 'status', 'estop', 'clear', 'quit'.")
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        head = parts[0].lower()

        if head in ("quit", "exit", "q"):
            break
        elif head == "step":
            if len(parts) < 2 or not parts[1].lstrip("-+").isdigit():
                print("    ? usage: step <signed steps> [ms per step]")
                continue
            steps = int(parts[1])
            delay = int(parts[2]) if len(parts) > 2 else None
            print(f"    STEP {steps} ...", flush=True)
            ctl.step(steps, delay)
            print("    done.")
        elif head == "status":
            print("   ", ctl.status())
        elif head == "estop":
            ctl.estop()
            print("    ESTOP latched (all motors).")
        elif head == "clear":
            ctl.clear()
            print("    cleared.")
        elif head in ("s", "stop"):          # bare stop = all motors
            ctl.stop()
            print("    all stopped.")
        elif head.isdigit():                 # "<motor> <dir> [ms]"
            motor = int(head)
            direction = parts[1].lower() if len(parts) > 1 else ""
            ms = int(parts[2]) if len(parts) > 2 else default_ms
            if direction in ("f", "fwd", "forward"):
                _drive(ctl, motor, True, ms)
            elif direction in ("r", "rev", "reverse"):
                _drive(ctl, motor, False, ms)
            elif direction in ("s", "stop"):
                ctl.stop(motor)
                print(f"    M{motor} stopped.")
            else:
                print(f"    ? expected f/r/s after the motor number, got {direction!r}")
        else:
            print(f"    ? unknown command {head!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="DC-motor relay bring-up tool")
    ap.add_argument("--config", help="YAML config (reads motion.relay.* )")
    ap.add_argument("--port", help="serial port (overrides config; e.g. /dev/ttyUSB0, COM3)")
    ap.add_argument("--baud", type=int, help="baud rate (overrides config)")
    ap.add_argument("--pulse-ms", type=int, help="default run duration per pulse (overrides config)")
    ap.add_argument("--selftest", action="store_true",
                    help="run every motor forward then reverse once, then exit")
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
            for motor in sorted(MOTOR_NAMES):
                _drive(ctl, motor, True, relay.pulse_ms)
                _drive(ctl, motor, False, relay.pulse_ms)
            print("Self-test done.")
            return 0
        _run_interactive(ctl, relay.pulse_ms)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.close()  # sends STOP (all) and releases the port
    return 0


if __name__ == "__main__":
    sys.exit(main())
