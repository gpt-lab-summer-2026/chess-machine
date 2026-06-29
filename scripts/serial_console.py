#!/usr/bin/env python3
"""Raw serial REPL for testing the ESP32 firmware directly.

    python scripts/serial_console.py --port COM3
    > PING
    OK PONG
    > HOME
    OK HOMED
    > MOVE R150 A0 F4000
    OK

Type protocol lines; everything the board sends back (OK / ERR / # debug) is
printed as it arrives. Ctrl-C or 'quit' to exit.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Raw ESP32 serial console")
    ap.add_argument("--port", required=True, help="serial port (COM3, /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    import serial  # lazy

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(2.0)  # ESP32 reboots on connect
    ser.reset_input_buffer()
    print(f"Connected to {args.port} @ {args.baud}. Type a command, or 'quit'.")

    stop = threading.Event()

    def reader():
        while not stop.is_set():
            raw = ser.readline()
            if raw:
                sys.stdout.write("    " + raw.decode(errors="replace").rstrip() + "\n")
                sys.stdout.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for line in sys.stdin:
            line = line.strip()
            if line.lower() in ("quit", "exit", "q"):
                break
            ser.write((line + "\n").encode())
            ser.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.3)
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
