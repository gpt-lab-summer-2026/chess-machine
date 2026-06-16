"""In-memory motion controller for development and tests.

Records every primitive call as a tuple in `self.ops`, so choreography can be
asserted without hardware, and prints a readable trace at debug log level.
"""
from __future__ import annotations

import logging
from typing import Optional

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
        self.homed = True
        self.ops.append(("home",))
        log.debug("mock: home")

    def move_xz(self, x_mm: float, z_mm: float, feed: Optional[int] = None) -> None:
        self.x, self.z = x_mm, z_mm
        self.ops.append(("move", round(x_mm, 3), round(z_mm, 3)))
        log.debug("mock: move x=%.2f z=%.2f f=%s", x_mm, z_mm, feed)

    def set_pulley(self, height_mm: float, feed: Optional[int] = None) -> None:
        self.height = height_mm
        self.ops.append(("pulley", round(height_mm, 3)))
        log.debug("mock: pulley h=%.2f f=%s", height_mm, feed)

    def magnet(self, on: bool) -> None:
        self.magnet_on = on
        self.ops.append(("magnet", bool(on)))
        log.debug("mock: magnet %s", "ON" if on else "OFF")

    def status(self) -> dict:
        return {
            "x": self.x, "z": self.z, "height": self.height,
            "magnet": self.magnet_on, "homed": self.homed,
        }

    def estop(self) -> None:
        self.ops.append(("estop",))
        log.debug("mock: estop")

    # -- test helpers -------------------------------------------------------- #
    def moves(self) -> list[tuple]:
        """Just the (x,z) gantry moves, for concise assertions."""
        return [(o[1], o[2]) for o in self.ops if o[0] == "move"]

    def reset_log(self) -> None:
        self.ops.clear()
