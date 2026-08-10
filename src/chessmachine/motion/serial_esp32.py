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

    @property
    def serial(self) -> Any:
        """The live pyserial handle (or None before connect()). Exposed so the
        ESP32 mic capture can share this ONE port — the chess machine is turn-
        based, so the mic and the motors never use it at the same time."""
        return self._ser

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

    def led(self, mode: str) -> None:
        """Status LED on the ESP32: 'on' (listening), 'blink' (thinking), 'off' (moving).

        Best-effort and self-disabling: firmware built before the LED command answers
        `ERR unknown command`, so the first failure logs once and every later call is
        a no-op. An indicator must never take the game down.
        """
        if getattr(self, "_led_unsupported", False):
            return
        try:
            self._command(f"LED {mode.upper()}")
        except Exception as exc:  # noqa: BLE001 - indicator only
            self._led_unsupported = True
            log.warning("ESP32 LED not available (%s) — reflash firmware/esp32_chess "
                        "for the status LED; continuing without it", exc)

    def estop(self) -> None:
        self._command("ESTOP", expect="ESTOP")

    def jog_base(self, steps: int) -> None:
        """Raw base-stepper jog: `JOG A<steps>` (signed half-steps). Bypasses the
        soft-angle clamp and does NOT update the tracked angle, so send HOME
        afterwards. Blocks until the move finishes."""
        steps = int(steps)
        self._command(f"JOG A{steps}", timeout=abs(steps) * 0.012 + 3.0)

    def jog_rail(self, steps: int) -> None:
        """Raw radial-cart jog: `JOG R<steps>` (signed half-steps; + = out, - = in).
        The cart is a stepper now. Updates the step counter (HOME re-zeros it) but
        NOT the tracked mm, so send HOME afterwards. Blocks."""
        steps = int(steps)
        self._command(f"JOG R{steps}", timeout=abs(steps) * 0.02 + 3.0)

    def jog_winch(self, steps: int) -> None:
        """Raw winch jog: `JOG W<steps>` (signed half-steps; sign per P_UP_STEP_DIR).
        For sag/height calibration. Updates the winch step counter (HOME re-zeros
        it). Blocks."""
        steps = int(steps)
        self._command(f"JOG W{steps}", timeout=abs(steps) * 0.01 + 3.0)

    def goto_steps(self, base: int | None = None, rail: int | None = None,
                   winch: int | None = None) -> None:
        """Absolute step-count move (`GOTO`): drive each named axis to an absolute
        half-step count (boot-home = 0), the space `STEPS` reports. The stepmap
        backend's primitive — no geometry, no mm/deg. Blocks until the move finishes."""
        parts = []
        if base is not None:
            parts.append(f"A{int(base)}")
        if rail is not None:
            parts.append(f"R{int(rail)}")
        if winch is not None:
            parts.append(f"W{int(winch)}")
        if not parts:
            return
        # An absolute move can traverse the full range (base + rail); allow the same
        # headroom as homing rather than the short per-command timeout.
        self._command("GOTO " + " ".join(parts), timeout=self.cfg.home_timeout_s)

    def sync_steps(self, base: int | None = None, rail: int | None = None,
                   winch: int | None = None) -> None:
        """Concurrent absolute move (`SYNC`): base/rail/winch driven together, each at its
        own rate, finishing when the slowest does. For the winch RISE overlapped with a
        base/rail reposition — NOT a lower (the head must be positioned first). Blocks."""
        parts = []
        if base is not None:
            parts.append(f"A{int(base)}")
        if rail is not None:
            parts.append(f"R{int(rail)}")
        if winch is not None:
            parts.append(f"W{int(winch)}")
        if not parts:
            return
        # A concurrent move is bounded by the slowest axis (a full winch rise is slow);
        # allow the same headroom as homing rather than the short per-command timeout.
        self._command("SYNC " + " ".join(parts), timeout=self.cfg.home_timeout_s)

    def seek_base_switch(self) -> int:
        """Bench-measure the base limit-switch offset (`SEEK`): rotate to the switch
        and read its OUTPUT step count from the trusted zero. HOME the base first so
        the count starts at 0. Returns the offset to bake into A_ENDSTOP_STEPS and
        live-enables switch homing on the firmware for this session. Blocks; returns
        to the start pose afterwards."""
        payload = self._command("SEEK", timeout=self.cfg.home_timeout_s)
        for tok in payload.split():
            if tok.startswith("AEND"):
                try:
                    return int(tok[4:])
                except ValueError:
                    break
        raise RuntimeError(f"SEEK: no AEND offset in reply {payload!r}")

    # -- live calibration (CAL) --------------------------------------------- #
    _CAL_KEYS = {"aspd": "ASPD", "ahome": "AHOME", "rspm": "RSPM", "aend": "AEND"}

    def get_cal(self) -> dict:
        """Read the firmware's current calibration (base steps/deg + home offset,
        rail steps/mm) as {'aspd','ahome','rspm'}."""
        return self._parse_cal(self._command("CAL"))

    def set_cal(self, **kw: float) -> dict:
        """Set any of aspd/ahome/rspm live (no reflash) and return the firmware's
        read-back. Unknown keys are ignored; empty call just reads."""
        parts = [f"{self._CAL_KEYS[k]} {float(v):.4f}"
                 for k, v in kw.items() if k in self._CAL_KEYS and v is not None]
        return self._parse_cal(self._command("CAL " + " ".join(parts) if parts else "CAL"))

    @staticmethod
    def _parse_cal(payload: str) -> dict:
        out: dict = {}
        inv = {v: k for k, v in SerialMotion._CAL_KEYS.items()}
        for tok in payload.split():
            for tag, key in inv.items():
                if tok.startswith(tag):
                    try:
                        out[key] = float(tok[len(tag):])
                    except ValueError:
                        pass
        return out

    def get_steps(self) -> dict:
        """Raw physical step counts from boot-home, {'a','r','w'} (base, rail,
        winch). The authoritative position for step-space calibration."""
        payload = self._command("STEPS")
        out: dict = {}
        keys = {"A": "a", "R": "r", "W": "w"}
        for tok in payload.split():
            k = keys.get(tok[:1])
            if k and tok[1:].lstrip("-").isdigit():
                out[k] = int(tok[1:])
        return out

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
