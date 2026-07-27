#!/usr/bin/env python3
"""Ground-truth step-map calibration for the polar crane.

WHY (vs tune.py): tune.py fits a 1-D scale+offset on the base ANGLE and a 1-D
scale on the rail RADIUS. In polar (r, theta) that can rotate/shift/stretch the
board but CANNOT move the pivot or fix a skew -- so aligning the a1-h8 diagonal
warps the a8-h1 one, forever. This tool throws the geometry model away: you JOG
the head to each real corner, GREENLIGHT it as ground truth (its raw base/rail/
winch step counts read straight from the firmware), and every other square is
bilinearly interpolated between the four measured corners. No origin, pitch,
steps/deg or steps/mm -- just measured truth at the corners.

It also captures WINCH depth per corner, so the crane's SAG is calibrated: the arm
droops when the cart is extended, so the magnet at a1 hangs lower than at h8 and
needs fewer winch steps to touch. Jog the winch down to touch at each corner and
it's baked into the map like base/rail.

Workflow:
  1. `home`             establish step 0 on all axes (start with the head AT home)
  2. jog to a corner:   `a <steps>` (base), `r <steps>` (rail)
  3. `w <steps>`        jog the winch DOWN until the magnet just touches the square
                        (`mag on` to feel it grab); the winch count is its depth
  4. `set a1`           greenlight: log (base, rail, winch) as a1's truth
  5. `wtop`             raise the winch back before moving to the next corner
  6. repeat for h1, a8, h8 (the four corners); optionally `set e4` as a mid check
  7. `map`              build + print the 64-square table, corner deltas, sag delta
  8. `goto <sq>`        drive to a square's interpolated base/rail (verify)
  9. `save map.json`    export anchors + table

Commands:
  a <n> / r <n> / w <n>   jog base / rail / winch by N raw steps (signed)
  wtop            raise the winch back to travel (step 0)
  pos             show current step position
  set <sq>        capture current pose as <sq>'s ground truth
  anchors         list captured anchors          del <sq>   remove one
  map             interpolate all squares; print deltas, sag, residuals
  goto <sq>       drive to <sq>'s interpolated base/rail (verify)
  mag on|off      electromagnet
  home | pos | save [path] | estop | quit

    python scripts/anchor.py --config config/config.yaml
    python scripts/anchor.py --mock
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config  # noqa: E402
from chessmachine import factory                     # noqa: E402

FILES = "abcdefgh"
CORNERS = ["a1", "h1", "a8", "h8"]
KEYS = ("base", "rail", "winch")
_SQ = re.compile(r"^([a-hA-H])([1-8])$")


def sq_uv(name: str) -> tuple[int, int]:
    """'a1'..'h8' -> 0-based (file, rank)."""
    m = _SQ.match(name.strip())
    if not m:
        raise ValueError(f"square must be a1..h8, got {name!r}")
    return FILES.index(m.group(1).lower()), int(m.group(2)) - 1


def bilinear(f: int, r: int, c: dict) -> float:
    """Interpolate a per-corner value at board square (file f, rank r), 0..7.
    Corners: a1=(0,0) h1=(7,0) a8=(0,7) h8=(7,7)."""
    u, v = f / 7.0, r / 7.0
    return (c["a1"] * (1 - u) * (1 - v) + c["h1"] * u * (1 - v)
            + c["a8"] * (1 - u) * v + c["h8"] * u * v)


class StepMap:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sp = cfg.motion.speeds
        self.ctl = factory.create_motion_controller(cfg.motion)
        self.anchors: dict[str, dict] = {}   # square -> {"base","rail","winch"}

    def connect(self) -> None:
        print(f"Connecting ({self.cfg.motion.backend}) ...", flush=True)
        self.ctl.connect()
        print("Ready. Start with the head at HOME, then `home` to zero the counters.")

    def close(self) -> None:
        try:
            self.ctl.magnet(False)
        except Exception:  # noqa: BLE001
            pass
        self.ctl.close()

    # -- position ----------------------------------------------------------- #
    def steps(self) -> tuple[int, int, int]:
        s = self.ctl.get_steps()
        return int(s.get("a", 0)), int(s.get("r", 0)), int(s.get("w", 0))

    def pos(self) -> None:
        a, r, w = self.steps()
        print(f"   pos: base={a}  rail={r}  winch={w}")

    def jog(self, axis: str, n: int) -> None:
        {"a": self.ctl.jog_base, "r": self.ctl.jog_rail, "w": self.ctl.jog_winch}[axis](n)
        self.pos()

    def wtop(self) -> None:
        _, _, w = self.steps()
        if w:
            self.ctl.jog_winch(-w)
        print("   winch raised to travel (0).")

    # -- anchors ------------------------------------------------------------ #
    def capture(self, sq: str) -> None:
        sq_uv(sq)                       # validate
        a, r, w = self.steps()
        self.anchors[sq.lower()] = {"base": a, "rail": r, "winch": w}
        print(f"   set {sq.lower()}: base={a} rail={r} winch={w}   (ground truth)")

    def _corner_vals(self) -> dict:
        return {k: {c: self.anchors[c][k] for c in CORNERS} for k in KEYS}

    def build(self) -> dict | None:
        missing = [c for c in CORNERS if c not in self.anchors]
        if missing:
            print(f"   need all 4 corners first; missing {missing}")
            return None
        cv = self._corner_vals()
        table: dict[str, dict] = {}
        for f in range(8):
            for r in range(8):
                table[FILES[f] + str(r + 1)] = {k: round(bilinear(f, r, cv[k])) for k in KEYS}
        return table

    def report(self) -> dict | None:
        table = self.build()
        if not table:
            return None

        def d(s1: str, s2: str, k: str) -> int:
            return self.anchors[s2][k] - self.anchors[s1][k]

        print("   corner step-distances (base, rail, winch):")
        for s1, s2 in [("a1", "a8"), ("h1", "h8"), ("a1", "h1"),
                       ("a8", "h8"), ("a1", "h8"), ("a8", "h1")]:
            print(f"     {s1}->{s2}:  base {d(s1, s2, 'base'):+6d}   "
                  f"rail {d(s1, s2, 'rail'):+6d}   winch {d(s1, s2, 'winch'):+6d}")
        wa1, wh8 = self.anchors["a1"]["winch"], self.anchors["h8"]["winch"]
        print(f"   SAG (winch touch depth):  a1={wa1}  h8={wh8}  -> a1 needs {wh8 - wa1:+d} "
              "steps less than h8 (arm droops when extended)")

        extra = [s for s in self.anchors if s not in CORNERS]
        if extra:
            cv = self._corner_vals()
            print("   bilinear residual at extra anchors (measured - interpolated):")
            for s in extra:
                f, r = sq_uv(s)
                print(f"     {s}:  " + "   ".join(
                    f"{k} {self.anchors[s][k] - round(bilinear(f, r, cv[k])):+5d}" for k in KEYS)
                    + "   (large -> corners alone miss the curvature; add mid anchors)")
        return table

    def goto(self, sq: str) -> None:
        table = self.build()
        if not table:
            return
        t = table.get(sq.lower())
        if not t:
            print(f"   {sq}: not a board square")
            return
        a, r, w = self.steps()
        if w != 0:
            self.ctl.jog_winch(-w)                 # raise winch first so nothing drags
        if t["base"] != a:
            self.ctl.jog_base(t["base"] - a)
        if t["rail"] != r:
            self.ctl.jog_rail(t["rail"] - r)
        na, nr, _ = self.steps()
        print(f"   goto {sq.lower()}: base {a}->{na} (tgt {t['base']}), rail {r}->{nr} (tgt {t['rail']}); "
              f"winch pick depth = {t['winch']} steps  (jog 'w {t['winch']}' to touch-test)")

    def save(self, path: str | None) -> None:
        table = self.build()
        out = {"anchors": self.anchors, "table": table}
        p = pathlib.Path(path or "stepmap.json")
        p.write_text(json.dumps(out, indent=2))
        print(f"   wrote {p} ({len(self.anchors)} anchors"
              + (", full 64-square table)" if table else ", no table yet — need 4 corners)"))


def _int(s: str | None) -> int:
    if s is None:
        raise ValueError("need a step count, e.g. 'a 200' or 'r -500'")
    return int(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground-truth step-map calibration")
    ap.add_argument("--config", help="YAML config (serial port)")
    ap.add_argument("--mock", action="store_true", help="mock backend (no hardware)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.mock:
        cfg.motion.backend = "mock"
    m = StepMap(cfg)
    m.connect()
    print("Commands: a/r/w <n> | wtop | pos | set <sq> | anchors | del <sq> | map "
          "| goto <sq> | mag on|off | home | save | quit")

    try:
        for line in _prompt():
            parts = line.split()
            if not parts:
                continue
            c = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            try:
                if c in ("quit", "exit", "q"):
                    break
                elif c in ("a", "r", "w"):
                    m.jog(c, _int(arg))
                elif c == "wtop":
                    m.wtop()
                elif c == "pos":
                    m.pos()
                elif c in ("set", "s") and arg:
                    m.capture(arg)
                elif c == "anchors":
                    for s, v in sorted(m.anchors.items()):
                        print(f"   {s}: base={v['base']} rail={v['rail']} winch={v['winch']}")
                    if not m.anchors:
                        print("   (none)")
                elif c == "del" and arg:
                    m.anchors.pop(arg.lower(), None); print(f"   removed {arg.lower()}")
                elif c in ("map", "build"):
                    m.report()
                elif c == "goto" and arg:
                    m.goto(arg)
                elif c in ("mag", "m") and arg:
                    on = arg.lower() in ("on", "1", "true")
                    m.ctl.magnet(on); print(f"   magnet {'ON' if on else 'OFF'}")
                elif c == "home":
                    m.ctl.home(); print("   homed (step counters zeroed).")
                elif c == "save":
                    m.save(arg)
                elif c == "estop":
                    m.ctl.estop(); print("   ESTOP — send 'home' to clear.")
                elif c in ("help", "h", "?"):
                    print(__doc__)
                else:
                    print(f"   ? unknown/incomplete: {line.strip()!r}")
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                print(f"   ! {e}")
    finally:
        m.close()
    return 0


def _prompt():
    while True:
        try:
            yield input("anchor> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == "__main__":
    sys.exit(main())
