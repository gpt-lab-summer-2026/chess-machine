"""Ground-truth step-map motion backend.

Plays BOARD squares from a hand-measured step map (scripts/mapboard.py) instead of
the geometry model: each square stores the exact firmware step counts (base, rail,
winch pick-depth) captured by jogging the head onto it, and a move drives straight
to those absolute counts via the firmware's `GOTO`. No origin/pitch/steps-per-unit,
no interpolation error — and because the base now homes against its a8 limit switch,
the map is repeatable across power cycles.

The choreographer still speaks in planar Points (it turns squares into (x,z) with the
SAME geometry), so this backend resolves each move_xz point back to its square by a
tight nearest-match and looks up the steps. Points that DON'T match a mapped square —
the off-board graveyard — fall back to the geometry/MOVE path on the same serial
transport (a storage bin needs far less precision than a playing square).

Winch: a mapped square carries its own pick depth (the crane sags more with the cart
extended), so `set_pulley(low)` lowers to THAT square's depth and `set_pulley(high)`
raises to travel (step 0). The fallback path uses the firmware's two-position winch.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Any

from ..config import MotionConfig
from .base import MotionController
from .geometry import BoardGeometry
from .serial_esp32 import SerialMotion

log = logging.getLogger(__name__)


class StepMapMotion(MotionController):
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self.geo = BoardGeometry(cfg.geometry)
        self.map_path = cfg.stepmap.map_path
        self.pick_below_mm = cfg.stepmap.pick_below_mm
        self.match_tol_mm = cfg.stepmap.match_tol_mm
        self._inner = SerialMotion(cfg.serial)   # the real transport (GOTO / MOVE / HOME / MAG)
        self._index: list[tuple[float, float, str, dict]] = []   # (x, z, name, entry)
        self._on_map = False        # was the last move_xz a mapped square (vs geometry fallback)?
        self._cur_winch: int | None = None   # mapped square's pick depth (steps), for set_pulley

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        self._inner.connect()
        self._load_map()

    def close(self) -> None:
        self._inner.close()

    def _load_map(self) -> None:
        p = pathlib.Path(self.map_path)
        if not p.exists():
            log.warning("stepmap: map file %s not found — EVERY move falls back to the "
                        "geometry path; run scripts/mapboard.py to build it", p)
            self._index = []
            return
        data = json.loads(p.read_text())
        squares = data.get("squares") or data.get("table") or {}
        self._index = []
        for name, entry in squares.items():
            try:
                pt = self.geo.name_to_point(name)   # same (x,z) the choreographer will pass
            except Exception:  # noqa: BLE001 - skip non-board keys, keep the rest
                log.debug("stepmap: skipping non-board map key %r", name)
                continue
            if "base" not in entry or "rail" not in entry:
                log.warning("stepmap: %s missing base/rail — skipped", name)
                continue
            self._index.append((pt.x, pt.z, name.lower(), entry))
        log.info("stepmap: loaded %d/64 board squares from %s", len(self._index), p)

    # -- resolution ---------------------------------------------------------- #
    def _match(self, x: float, z: float) -> dict | None:
        """Nearest mapped square to (x,z), or None if none within match_tol_mm."""
        best: dict | None = None
        best_d = None
        for px, pz, _name, entry in self._index:
            d = math.hypot(px - x, pz - z)
            if best_d is None or d < best_d:
                best_d, best = d, entry
        if best is not None and best_d is not None and best_d <= self.match_tol_mm:
            return best
        return None

    # -- primitives (MotionController) --------------------------------------- #
    def home(self) -> None:
        self._inner.home()
        self._on_map = False
        self._cur_winch = None

    def move_xz(self, x_mm: float, z_mm: float, feed: int | None = None) -> None:
        entry = self._match(x_mm, z_mm)
        if entry is None:
            # Off-board (graveyard) or an unmapped square: geometry/MOVE fallback.
            self._on_map = False
            self._cur_winch = None
            self._inner.move_xz(x_mm, z_mm, feed)
            return
        self._on_map = True
        self._cur_winch = int(entry["winch"]) if entry.get("winch") is not None else None
        self._inner.goto_steps(base=int(entry["base"]), rail=int(entry["rail"]))

    def set_pulley(self, height_mm: float, feed: int | None = None) -> None:
        if not self._on_map:
            self._inner.set_pulley(height_mm, feed)      # two-position winch on the fallback path
            return
        if height_mm < self.pick_below_mm:               # lower onto the piece
            if self._cur_winch is None:
                self._inner.set_pulley(height_mm, feed)  # square has no depth: two-position fallback
            else:
                self._inner.goto_steps(winch=self._cur_winch)
        else:                                            # raise to travel (step 0)
            self._inner.goto_steps(winch=0)

    def magnet(self, on: bool) -> None:
        self._inner.magnet(on)

    def status(self) -> dict:
        return self._inner.status()

    def estop(self) -> None:
        self._inner.estop()

    # -- pass-throughs (calibration tools / diagnostics) --------------------- #
    def get_steps(self) -> dict:
        return self._inner.get_steps()

    def get_cal(self) -> dict:
        return self._inner.get_cal()

    def set_cal(self, **kw: float) -> dict:
        return self._inner.set_cal(**kw)

    def goto_steps(self, base: Any = None, rail: Any = None, winch: Any = None) -> None:
        self._inner.goto_steps(base=base, rail=rail, winch=winch)

    def jog_base(self, steps: int) -> None:
        self._inner.jog_base(steps)

    def jog_rail(self, steps: int) -> None:
        self._inner.jog_rail(steps)

    def jog_winch(self, steps: int) -> None:
        self._inner.jog_winch(steps)

    def seek_base_switch(self) -> int:
        return self._inner.seek_base_switch()
