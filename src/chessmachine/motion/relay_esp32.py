"""Relay/DC-motor prototype transport (firmware/esp32_dc_prototype).

Speaks the tiny line protocol of the prototype firmware, which drives THREE
brushed DC motors, each through a 2-channel relay H-bridge (<m> = motor 1..3):

    ->  PING          <-  OK PONG
    ->  FWD <m> <ms>  <-  OK      (motor m forward ms, then auto-stop; EVT DONE M<m>)
    ->  REV <m> <ms>  <-  OK      (motor m reverse ms, then auto-stop)
    ->  STOP [m]      <-  OK      (stop motor m, or all motors if omitted)
    ->  ESTOP         <-  OK ESTOP
    ->  STATUS        <-  OK M1:<F|R|S> M2:<F|R|S> M3:<F|R|S> ESTOP<0|1>

This is a bring-up backend, NOT the crane. It cannot position a head, so the
pick-and-place primitives (`move_xz`/`set_pulley`/`magnet`) are no-ops — the
actual motion is a timed pulse issued via `drive()`. `RelayChoreographer` drives
motor 1 by default; the jog tool (scripts/relay_jog.py) selects any motor.
Lines beginning with '#' (debug) or 'EVT' (async event) are logged and ignored.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from ..config import RelayConfig
from .base import MotionController

log = logging.getLogger(__name__)


class RelayMotion(MotionController):
    def __init__(self, cfg: RelayConfig):
        self.cfg = cfg
        self._ser = None

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        import serial  # lazy: pyserial only needed for the real transport

        self._ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=self.cfg.timeout_s)
        time.sleep(self.cfg.connect_settle_s)  # ESP32 reboots when the port opens
        self._ser.reset_input_buffer()
        payload = self._command("PING", expect="PONG")
        log.info("Relay ESP32 connected on %s (%s)", self.cfg.port, payload or "PONG")

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            try:
                self._command("STOP")  # leave the motor de-energized
            except Exception:  # noqa: BLE001 - closing anyway
                pass
            finally:
                self._ser.close()
                self._ser = None

    # -- transport ----------------------------------------------------------- #
    def _readline(self, deadline: float) -> str:
        while time.time() < deadline:
            raw = self._ser.readline()
            if raw:
                return raw.decode(errors="replace").strip()
        raise TimeoutError("Timed out waiting for relay ESP32 reply")

    def _command(self, line: str, expect: Optional[str] = None,
                 timeout: Optional[float] = None) -> str:
        if self._ser is None:
            raise RuntimeError("RelayMotion not connected; call connect() first")
        self._ser.write((line + "\n").encode())
        self._ser.flush()
        deadline = time.time() + (timeout if timeout is not None else self.cfg.timeout_s)
        while True:
            resp = self._readline(deadline)
            if not resp:
                continue
            if resp[0] == "#" or resp.startswith("EVT"):
                log.debug("relay: %s", resp)
                continue
            if resp.startswith("OK"):
                payload = resp[2:].strip()
                if expect is not None and expect not in payload:
                    raise RuntimeError(f"Unexpected reply to {line!r}: {resp!r}")
                return payload
            if resp.startswith("ERR"):
                raise RuntimeError(f"Relay ESP32 error for {line!r}: {resp}")
            log.debug("relay(unparsed): %s", resp)

    # -- prototype motor control --------------------------------------------- #
    def drive(self, forward: bool, ms: int, motor: int = 1) -> None:
        """Pulse `motor` (1-based) for `ms` and block until it has stopped.

        The firmware auto-stops after `ms` (it replies OK immediately), so we
        wait out the run here to keep the call blocking like the crane backend.
        """
        ms = max(0, int(ms))
        self._command(("FWD" if forward else "REV") + f" {int(motor)} {ms}")
        time.sleep(ms / 1000.0 + 0.05)  # let the timed run finish before returning

    def stop(self, motor: Optional[int] = None) -> None:
        """Stop one motor, or all motors (+ release the stepper) if `motor` is None."""
        self._command("STOP" if motor is None else f"STOP {int(motor)}")

    def step(self, steps: int, delay_ms: Optional[int] = None) -> None:
        """Move the stepper `steps` (signed; +/- selects direction) and block
        until it finishes. The firmware steps synchronously, so we allow a reply
        window proportional to the move."""
        steps = int(steps)
        per = int(delay_ms) if delay_ms is not None else 2
        line = f"STEP {steps}" + (f" {per}" if delay_ms is not None else "")
        self._command(line, timeout=abs(steps) * per / 1000.0 + 2.0)

    def estop(self) -> None:
        self._command("ESTOP", expect="ESTOP")

    def clear(self) -> None:
        """Release an ESTOP latch (firmware CLEAR), re-enabling motion."""
        self._command("CLEAR")

    def status(self) -> dict:
        payload = self._command("STATUS")
        out: dict = {"motors": {}}
        for tok in payload.split():
            if tok.startswith("STEP:"):                     # e.g. "STEP:-2048"
                try:
                    out["stepper_pos"] = int(tok[5:])
                except ValueError:
                    pass
            elif tok.startswith("M") and ":" in tok:         # e.g. "M2:F"
                idx, _, d = tok[1:].partition(":")
                if idx.isdigit():
                    out["motors"][int(idx)] = {"F": "forward", "R": "reverse"}.get(d, "stopped")
            elif tok.startswith("ESTOP"):
                out["estopped"] = tok[5:] == "1"
        return out

    # -- unused positioning primitives (this backend can't position a head) --- #
    def home(self) -> None:
        pass  # the prototype firmware has no homing

    def move_xz(self, x_mm: float, z_mm: float, feed: Optional[int] = None) -> None:
        log.debug("relay: move_xz ignored (no positioning on the DC prototype)")

    def set_pulley(self, height_mm: float, feed: Optional[int] = None) -> None:
        log.debug("relay: set_pulley ignored (no pulley on the DC prototype)")

    def magnet(self, on: bool) -> None:
        log.debug("relay: magnet ignored (no magnet on the DC prototype)")
