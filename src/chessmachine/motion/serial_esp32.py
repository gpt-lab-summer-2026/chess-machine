"""ESP32-over-USB-serial motion backend.

Speaks the line protocol implemented by firmware/esp32_chess. Every command
blocks until the controller acknowledges with `OK` (or raises on `ERR`/timeout),
so the host can sequence moves without tracking motor state itself.

This is also where the planar (x,z) workspace is converted to the crane's polar
axes. `move_xz` maps the desired MAGNET position to the CART's polar target,
compensating for the winch's fixed lateral offset from the arm (see
`_magnet_to_cart`); the rotary base + radial railcart realize `r`/`theta`.

Protocol (see docs/PROTOCOL.md):
    ->  PING                          <-  OK PONG
    ->  HOME                          <-  OK HOMED
    ->  MOVE R<mm> A<deg> [F<mm/min>] <-  OK
    ->  PULLEY H<mm> [F<mm/min>]      <-  OK
    ->  MAG ON|OFF                    <-  OK
    ->  STATUS                        <-  OK R<f> A<f> H<f> MAG<0|1> ENDR<0|1> ENDA<0|1>
    ->  ESTOP                         <-  OK ESTOP
Lines beginning with '#' (debug) or 'EVT' (async event) are logged and ignored.
"""
from __future__ import annotations

import logging
import math
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
        log.info("TX %s", line)
        deadline = time.time() + (timeout if timeout is not None else self.cfg.timeout_s)
        while True:
            resp = self._readline(deadline)
            if not resp:
                continue
            log.info("RX %s", resp)
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
        # Desired MAGNET position (planar, pivot frame) -> CART polar target.
        r, a = self._magnet_to_cart(x_mm, z_mm, self.cfg.winch_offset_mm)
        self._command(f"MOVE R{r:.2f} A{a:.2f}{self._feed(feed)}")

    @staticmethod
    def _magnet_to_cart(x_mm: float, z_mm: float, offset_mm: float) -> tuple[float, float]:
        """Map a desired magnet position (planar mm) to the cart's polar target
        (r mm, angle degrees).

        The magnet hangs `offset_mm` perpendicular to the arm, so pivot->cart (r),
        cart->magnet (offset) and pivot->magnet (M) form a right triangle: the cart
        sits at r = sqrt(M^2 - offset^2) and the arm swings past the target angle by
        asin(offset/M). `offset_mm` is signed (picks the side); 0 disables it.
        Falls back to the plain conversion if the target is closer than the offset.
        """
        m = math.hypot(x_mm, z_mm)
        ang = math.atan2(z_mm, x_mm)
        if offset_mm and m > abs(offset_mm):
            r = math.sqrt(m * m - offset_mm * offset_mm)
            a = math.degrees(ang + math.asin(offset_mm / m))
        else:
            r = m
            a = math.degrees(ang)
        return r, a

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
                if tok.startswith("ENDR"):
                    out["endstop_r"] = tok[4:] == "1"
                elif tok.startswith("ENDA"):
                    out["endstop_a"] = tok[4:] == "1"
                elif tok.startswith("MAG"):
                    out["magnet"] = tok[3:] == "1"
                elif tok.startswith("R"):
                    out["r"] = float(tok[1:])
                elif tok.startswith("A"):
                    out["a"] = float(tok[1:])
                elif tok.startswith("H"):
                    out["height"] = float(tok[1:])
            except ValueError:
                continue
        return out
