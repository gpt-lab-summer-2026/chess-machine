#!/usr/bin/env python3
"""Full-board ground-truth step-map calibration.

Now that the base homes against its a8 limit switch (absolute, repeatable), the
step counts to reach a square are stable across power cycles — so we can hard-code
EVERY square instead of interpolating four corners. You jog the head onto each of
the 64 squares, GREENLIGHT it, and its exact firmware step counts (base, rail, and
winch pick-depth) are logged as ground truth. The result is written to a step-map
file that the `stepmap` motion backend reads at startup and plays straight from —
no geometry model, no interpolation error.

Speed-up: once the four corners are captured, `goto <sq>` / `next` PRE-POSITION the
head at a bilinear estimate of any square, so you only fine-tune the last few mm
before `set`. Capture the corners first (a1 h1 a8 h8), then walk the rest.

Workflow:
  1. `home`                 base + cart self-home to their switches; start the winch UP
  2. corners first:         jog to a1 (`a`/`r`/`w`), `w` down until the magnet touches,
                            `set a1`; repeat h1, a8, h8
  3. `next` (or `goto e4`)  drive to the estimated next square, fine-jog, `set e4`
  4. `coverage`             see what's captured / still missing
  5. `save`                 write the 64-square map (config/stepmap.json)

Commands:
  a <n> / r <n> / w <n>   jog base / rail / winch by N raw steps (signed)
  wtop            raise the winch to travel (step 0)
  set <sq>        capture the current pose as <sq>'s ground truth (a1..h8)
  goto <sq>       drive to <sq> (measured if captured, else a corner-estimate)
  next            goto the next uncaptured square (row-major)
  coverage        how many squares captured; list what's missing
  fill            fill any MISSING squares with corner estimates (marked estimated)
  del <sq>        drop a captured square      pos   show current step position
  seek            measure the base limit-switch offset -> A_ENDSTOP_STEPS (home first)
  mag on|off      electromagnet
  home | save [path] | estop | quit

    python scripts/mapboard.py --config config/config.yaml
    python scripts/mapboard.py --mock
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
ALL_SQUARES = [f + str(r + 1) for r in range(8) for f in FILES]   # row-major: a1..h1, a2..h2, ...
_SQ = re.compile(r"^([a-hA-H])([1-8])$")


def sq_uv(name: str) -> tuple[int, int]:
    """'a1'..'h8' -> 0-based (file, rank)."""
    m = _SQ.match(name.strip())
    if not m:
        raise ValueError(f"square must be a1..h8, got {name!r}")
    return FILES.index(m.group(1).lower()), int(m.group(2)) - 1


def bilinear(f: int, r: int, c: dict) -> float:
    """Interpolate a per-corner value at (file f, rank r), 0..7.
    Corners: a1=(0,0) h1=(7,0) a8=(0,7) h8=(7,7)."""
    u, v = f / 7.0, r / 7.0
    return (c["a1"] * (1 - u) * (1 - v) + c["h1"] * u * (1 - v)
            + c["a8"] * (1 - u) * v + c["h8"] * u * v)


class BoardMapper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ctl = factory.create_motion_controller(cfg.motion)
        self.squares: dict[str, dict] = {}   # name -> {"base","rail","winch"}

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        print(f"Connecting ({self.cfg.motion.backend}) ...", flush=True)
        self.ctl.connect()
        print("Ready. The base + cart self-home to their limit switches; only the winch "
              "is sensorless, so start it UP, then `home`.")

    def close(self) -> None:
        try:
            self.ctl.magnet(False)
        except Exception:  # noqa: BLE001
            pass
        self.ctl.close()

    # -- position ------------------------------------------------------------ #
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

    # -- capture ------------------------------------------------------------- #
    def capture(self, sq: str) -> None:
        sq_uv(sq)                       # validate
        a, r, w = self.steps()
        self.squares[sq.lower()] = {"base": a, "rail": r, "winch": w}
        n = len(self.squares)
        print(f"   set {sq.lower()}: base={a} rail={r} winch={w}   ({n}/64 captured)")

    def _corners(self) -> dict | None:
        if any(c not in self.squares for c in CORNERS):
            return None
        return {k: {c: self.squares[c][k] for c in CORNERS} for k in KEYS}

    def estimate(self, sq: str) -> dict | None:
        """Bilinear estimate of a square from the four corners (None if not all set)."""
        cv = self._corners()
        if not cv:
            return None
        f, r = sq_uv(sq)
        return {k: round(bilinear(f, r, cv[k])) for k in KEYS}

    # -- drive --------------------------------------------------------------- #
    def goto(self, sq: str) -> None:
        sq = sq.lower()
        t = self.squares.get(sq) or self.estimate(sq)
        if not t:
            print(f"   no target for {sq}: capture the 4 corners (a1 h1 a8 h8) first "
                  "for estimates, or jog there by hand.")
            return
        kind = "measured" if sq in self.squares else "estimate"
        self.ctl.goto_steps(winch=0)                       # raise before travelling
        self.ctl.goto_steps(base=t["base"], rail=t["rail"])
        a, r, _ = self.steps()
        print(f"   goto {sq} ({kind}): base->{a} rail->{r}; winch pick depth ~{t['winch']} "
              f"(jog `w {t['winch']}` to touch). Fine-tune, then `set {sq}`.")

    def next_square(self) -> None:
        remaining = [s for s in ALL_SQUARES if s not in self.squares]
        if not remaining:
            print("   all 64 squares captured.")
            return
        nxt = remaining[0]
        print(f"   next uncaptured: {nxt}")
        self.goto(nxt)

    # -- reporting ----------------------------------------------------------- #
    def coverage(self) -> None:
        missing = [s for s in ALL_SQUARES if s not in self.squares]
        print(f"   {len(self.squares)}/64 captured.")
        if missing:
            print("   missing: " + " ".join(missing))
        else:
            print("   full board mapped.")

    def fill(self) -> None:
        cv = self._corners()
        if not cv:
            print("   need the 4 corners first to estimate the rest.")
            return
        added = 0
        for s in ALL_SQUARES:
            if s not in self.squares:
                f, r = sq_uv(s)
                self.squares[s] = {k: round(bilinear(f, r, cv[k])) for k in KEYS}
                self.squares[s]["estimated"] = True
                added += 1
        print(f"   filled {added} missing square(s) with corner estimates (marked "
              "'estimated'; re-`set` any you refine).")

    def save(self, path: str | None) -> None:
        p = pathlib.Path(path or self.cfg.motion.stepmap.map_path)
        missing = [s for s in ALL_SQUARES if s not in self.squares]
        out = {"squares": {s: self.squares[s] for s in ALL_SQUARES if s in self.squares}}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"   wrote {p} ({len(out['squares'])}/64 squares).")
        if missing:
            print(f"   WARNING: {len(missing)} square(s) NOT mapped — the game will fall back to "
                  f"geometry for them: {' '.join(missing)}")
            print("   (run `fill` first to write corner estimates instead.)")


def _int(s: str | None) -> int:
    if s is None:
        raise ValueError("need a step count, e.g. 'a 200' or 'r -500'")
    return int(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Full-board ground-truth step-map calibration")
    ap.add_argument("--config", help="YAML config (serial port + map path)")
    ap.add_argument("--mock", action="store_true", help="mock backend (no hardware)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    # This is a bench tool: always drive the raw serial transport (or mock), never the
    # stepmap backend (which would try to LOAD the very map we're building).
    cfg.motion.backend = "mock" if args.mock else "serial"

    m = BoardMapper(cfg)
    m.connect()
    print("Commands: a/r/w <n> | wtop | set <sq> | goto <sq> | next | coverage | fill | "
          "del <sq> | pos | seek | mag on|off | home | save | quit")

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
                elif c == "goto" and arg:
                    m.goto(arg)
                elif c == "next":
                    m.next_square()
                elif c in ("coverage", "cov"):
                    m.coverage()
                elif c == "fill":
                    m.fill()
                elif c == "del" and arg:
                    m.squares.pop(arg.lower(), None); print(f"   removed {arg.lower()}")
                elif c in ("seek", "findsw"):
                    off = m.ctl.seek_base_switch()
                    print(f"   base limit switch at {off} steps -> bake A_ENDSTOP_STEPS={off} "
                          "into the .ino (live-enabled now).")
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
            yield input("map> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == "__main__":
    sys.exit(main())
