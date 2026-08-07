"""In-memory motion controller for development and tests.

Records every primitive call as a tuple in `self.ops`, so choreography can be
asserted without hardware, and prints a readable trace at debug log level.
"""
from __future__ import annotations

import logging

from .base import MotionController

log = logging.getLogger(__name__)


class MockMotion(MotionController):
    def __init__(self):
        self.ops: list[tuple] = []
        self.x = 0.0
        self.z = 0.0
        self.height = 0.0
        self.magnet_on = False
        self.connected = False
        self.homed = False
        self._cal = {"aspd": 43.284, "ahome": 1.063, "rspm": 66.8, "aend": 0}
        self._steps = {"a": 0, "r": 0, "w": 0}
        self._base_switch = False   # base limit switch (ENDA) pressed?
        self._rail_switch = False   # rail HOME limit switch (ENDR) pressed?

    def connect(self) -> None:
        self.connected = True
        self.ops.append(("connect",))
        log.debug("mock: connect")

    def close(self) -> None:
        self.connected = False
        self.ops.append(("close",))
        log.debug("mock: close")

    def home(self) -> None:
        self.x = self.z = self.height = 0.0
        self._steps = {"a": 0, "r": 0, "w": 0}
        self.homed = True
        self.ops.append(("home",))
        log.debug("mock: home")

    def move_xz(self, x_mm: float, z_mm: float, feed: int | None = None) -> None:
        self.x, self.z = x_mm, z_mm
        self.ops.append(("move", round(x_mm, 3), round(z_mm, 3)))
        log.debug("mock: move x=%.2f z=%.2f f=%s", x_mm, z_mm, feed)

    def set_pulley(self, height_mm: float, feed: int | None = None) -> None:
        self.height = height_mm
        self.ops.append(("pulley", round(height_mm, 3)))
        log.debug("mock: pulley h=%.2f f=%s", height_mm, feed)

    def magnet(self, on: bool) -> None:
        self.magnet_on = on
        self.ops.append(("magnet", bool(on)))
        log.debug("mock: magnet %s", "ON" if on else "OFF")

    def pick_dip(self, steps: int) -> None:
        self.ops.append(("pick_dip", int(steps)))
        log.debug("mock: pick_dip %d", int(steps))

    def status(self) -> dict:
        return {
            "x": self.x, "z": self.z, "height": self.height,
            "magnet": self.magnet_on, "homed": self.homed,
            "endstop_a": self._base_switch, "endstop_r": self._rail_switch,
        }

    def estop(self) -> None:
        self.ops.append(("estop",))
        log.debug("mock: estop")

    def get_cal(self) -> dict:
        return dict(self._cal)

    def set_cal(self, **kw: float) -> dict:
        for k, v in kw.items():
            if k in self._cal and v is not None:
                self._cal[k] = float(v)
        self.ops.append(("set_cal", dict(self._cal)))
        log.debug("mock: set_cal %s", self._cal)
        return dict(self._cal)

    def jog_base(self, steps: int) -> None:
        self._steps["a"] += int(steps)
        self.ops.append(("jog_base", int(steps)))
        log.debug("mock: jog_base %d steps", int(steps))

    def jog_rail(self, steps: int) -> None:
        self._steps["r"] += int(steps)
        self.ops.append(("jog_rail", int(steps)))
        log.debug("mock: jog_rail %d steps", int(steps))

    def jog_winch(self, steps: int) -> None:
        self._steps["w"] += int(steps)
        self.ops.append(("jog_winch", int(steps)))
        log.debug("mock: jog_winch %d steps", int(steps))

    def get_steps(self) -> dict:
        return dict(self._steps)

    def goto_steps(self, base: int | None = None, rail: int | None = None,
                   winch: int | None = None) -> None:
        if base is not None:
            self._steps["a"] = int(base)
        if rail is not None:
            self._steps["r"] = int(rail)
        if winch is not None:
            self._steps["w"] = int(winch)
        self.ops.append(("goto_steps", int(base) if base is not None else None,
                         int(rail) if rail is not None else None,
                         int(winch) if winch is not None else None))
        log.debug("mock: goto_steps a=%s r=%s w=%s", base, rail, winch)

    def seek_base_switch(self) -> int:
        """Pretend to seek the base limit switch: return a plausible a8 offset
        (~+32 deg * aspd) and live-enable switch homing, matching the firmware."""
        off = round(32.0 * self._cal.get("aspd", 43.284))
        self._cal["aend"] = off
        self.ops.append(("seek_base_switch", off))
        log.debug("mock: seek_base_switch -> %d", off)
        return off

    # -- test helpers -------------------------------------------------------- #
    def moves(self) -> list[tuple]:
        """Just the (x,z) head moves, for concise assertions."""
        return [(o[1], o[2]) for o in self.ops if o[0] == "move"]

    def reset_log(self) -> None:
        self.ops.clear()
