"""ESP32-over-USB-serial motion backend.

Speaks the line protocol implemented by firmware/esp32_chess. Every command
blocks until the controller acknowledges with `OK` (or raises on `ERR`/timeout),
so the host can sequence moves without tracking motor state itself.

Protocol (see docs/PROTOCOL.md):
    ->  PING                          <-  OK PONG
    ->  HOME                          <-  OK HOMED
    ->  MOVE X<mm> Z<mm> [F<mm/min>]  <-  OK
    ->  PULLEY H<mm> [F<mm/min>]      <-  OK
    ->  MAG ON|OFF                    <-  OK
    ->  STATUS                        <-  OK X<f> Z<f> H<f> MAG<0|1> ENDX<0|1> ENDZ<0|1>
    ->  ESTOP                         <-  OK ESTOP
Lines beginning with '#' (debug) or 'EVT' (async event) are logged and ignored.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..config import SerialConfig
from .base import MotionController

log = logging.getLogger(__name__)


class SerialMotion(MotionController):
    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self._ser: Any = None   # pyserial Serial handle (opened in connect())

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        import serial  # lazy: pyserial only needed for the real transport

        self._ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_s)
        time.sleep(self.cfg.connect_settle_s)  # ESP32 reboots when the port opens
        self._ser.reset_input_buffer()
        payload = self._command("PING", expect="PONG")
        log.info("ESP32 connected on %s (%s)", self.cfg.port, payload or "PONG")

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.close()
            finally:
                self._ser = None

    # -- transport ----------------------------------------------------------- #
    def _readline(self, deadline: float) -> str:
        while time.time() < deadline:
            raw = self._ser.readline()
            if raw:
                return raw.decode(errors="replace").strip()
        raise TimeoutError("Timed out waiting for ESP32 reply")

    def _command(self, line: str, expect: str | None = None,
                 timeout: float | None = None) -> str:
        if self._ser is None:
            raise RuntimeError("SerialMotion not connected; call connect() first")
        self._ser.write((line + "\n").encode())
        self._ser.flush()
        deadline = time.time() + (timeout if timeout is not None else self.cfg.timeout_s)
        while True:
            resp = self._readline(deadline)
            if not resp:
                continue
            if resp[0] == "#" or resp.startswith("EVT"):
                log.debug("esp32: %s", resp)
                continue
            if resp.startswith("OK"):
                payload = resp[2:].strip()
                if expect is not None and expect not in payload:
                    raise RuntimeError(f"Unexpected reply to {line!r}: {resp!r}")
                return payload
            if resp.startswith("ERR"):
                raise RuntimeError(f"ESP32 error for {line!r}: {resp}")
            log.debug("esp32(unparsed): %s", resp)

    @staticmethod
    def _feed(feed: int | None) -> str:
        return f" F{int(feed)}" if feed else ""

    # -- primitives ---------------------------------------------------------- #
    def home(self) -> None:
        self._command("HOME", expect="HOMED", timeout=self.cfg.home_timeout_s)

    def move_xz(self, x_mm: float, z_mm: float, feed: int | None = None) -> None:
        self._command(f"MOVE X{x_mm:.2f} Z{z_mm:.2f}{self._feed(feed)}")

    def set_pulley(self, height_mm: float, feed: int | None = None) -> None:
        self._command(f"PULLEY H{height_mm:.2f}{self._feed(feed)}")

    def magnet(self, on: bool) -> None:
        self._command(f"MAG {'ON' if on else 'OFF'}")

    def estop(self) -> None:
        self._command("ESTOP", expect="ESTOP")

    def status(self) -> dict:
        payload = self._command("STATUS")
        return self._parse_status(payload)

    @staticmethod
    def _parse_status(payload: str) -> dict:
        out: dict = {}
        for tok in payload.split():
            try:
                if tok.startswith("X"):
                    out["x"] = float(tok[1:])
                elif tok.startswith("Z"):
                    out["z"] = float(tok[1:])
                elif tok.startswith("H"):
                    out["height"] = float(tok[1:])
                elif tok.startswith("MAG"):
                    out["magnet"] = tok[3:] == "1"
                elif tok.startswith("ENDX"):
                    out["endstop_x"] = tok[4:] == "1"
                elif tok.startswith("ENDZ"):
                    out["endstop_z"] = tok[4:] == "1"
            except ValueError:
                continue
        return out
