#!/usr/bin/env python3
"""A/B pickup test: try to grab the piece at the MAPPED coords, then at the PREDICTED
coords, and watch which one sticks.

    python scripts/verifymap.py --go              # interactive: hand-pick squares
    python scripts/verifymap.py                   # offline: whole-board plan, no hardware
    python scripts/verifymap.py b3 c3 b4 --go     # batch: run these squares and exit
    python scripts/verifymap.py --worst winch:6 --go

Put a piece on the square, then per square the crane runs two INDEPENDENT attempts,
returning the piece each time:

    A (mapped)     GOTO mapped base/rail -> lower to mapped depth -> magnet ON
                   -> raise (does it lift?) -> lower -> magnet OFF -> raise
    B (predicted)  the same, using the model's base/rail/depth

Nothing is asked mid-run — it holds at the raised position for --dwell seconds so you
can see whether the piece came up. If it sticks on B, the formula reaches that square;
if only A sticks, the map's hand-measured value is doing real work there.

Interactive commands (like scripts/anchor.py):
    <sq>            run the A/B test on a square (e.g. just type: b3)
    a <sq>          only attempt A (mapped)          b <sq>   only attempt B (predicted)
    show <sq>       print both coordinate sets, move nothing
    worst <axis> [n]  list the N biggest table-vs-model disagreements (base|rail|winch)
    dwell <s> | settle <s>    adjust timing live
    home | wtop | mag on|off  raw helpers (wtop = raise winch to travel)
    help | quit

SAFETY. Dry run is the default; nothing moves without --go. The magnet is released and
the winch raised in a finally block, so Ctrl-C cannot leave a piece stuck to the
electromagnet. Hand-measured map values are never written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config                      # noqa: E402
from chessmachine.motion.geometry import BoardGeometry           # noqa: E402
from chessmachine.motion.predict import KEYS, BoardModel         # noqa: E402

WINCH_TRAVEL = 0          # stepmap convention: winch step 0 == raised / travel
STEPS_PER_MM_WINCH = 7500 / 56.0     # WINCH_STROKE_STEPS over travel-to-pick mm


class Verifier:
    def __init__(self, cfg, map_path: pathlib.Path, dwell: float, settle: float):
        self.cfg = cfg
        self.geo = BoardGeometry(cfg.motion.geometry)
        data = json.loads(map_path.read_text())
        self.anchors = data.get("anchors", {})
        self.table = data.get("table") or data.get("squares") or {}
        self.model = BoardModel(self.geo, self.anchors)
        self.map_path = map_path
        self.dwell = dwell
        self.settle = settle
        self.ctl = None

    # -- coordinates --------------------------------------------------------- #
    def coords(self, sq: str) -> tuple[dict, dict] | None:
        sq = sq.lower()
        if sq not in self.table:
            print(f"   ! {sq} is not in the map table")
            return None
        return {k: int(self.table[sq][k]) for k in KEYS}, self.model.predict(sq)

    def show(self, sq: str) -> None:
        c = self.coords(sq)
        if not c:
            return
        a, b = c
        anchor = "  [hand-measured anchor]" if sq.lower() in self.anchors else ""
        print(f"   {sq}{anchor}")
        print(f"     A mapped     base {a['base']:+6d}  rail {a['rail']:6d}  winch {a['winch']:5d}")
        print(f"     B predicted  base {b['base']:+6d}  rail {b['rail']:6d}  winch {b['winch']:5d}")
        dw = b['winch'] - a['winch']
        print(f"     B-A          base {b['base']-a['base']:+6d}  rail {b['rail']-a['rail']:+6d}"
              f"  winch {dw:+5d}  ({dw/STEPS_PER_MM_WINCH:+.2f} mm)")

    def worst(self, axis: str, n: int = 6) -> None:
        if axis not in KEYS:
            print(f"   ! axis must be one of {', '.join(KEYS)}")
            return
        print(f"   biggest {axis} disagreements (table - predicted):")
        for s, v in self.model.disagreements(self.table, axis)[:n]:
            tag = " [anchor]" if s in self.anchors else ""
            extra = f"  ({v/STEPS_PER_MM_WINCH:+.2f} mm)" if axis == "winch" else ""
            print(f"     {s:>3}  {v:+8.0f}{extra}{tag}")

    # -- motion -------------------------------------------------------------- #
    def _attempt(self, label: str, c: dict) -> None:
        print(f"   {label}: base={c['base']} rail={c['rail']} winch={c['winch']}")
        self.ctl.goto_steps(winch=WINCH_TRAVEL)
        self.ctl.goto_steps(base=c["base"], rail=c["rail"])
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=c["winch"])       # lower onto the piece
        self.ctl.magnet(True)
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=WINCH_TRAVEL)     # lift — watch here
        print(f"      ...raised, holding {self.dwell:g}s — did it lift?")
        time.sleep(self.dwell)
        self.ctl.goto_steps(winch=c["winch"])       # put it back where it was
        time.sleep(self.settle)
        self.ctl.magnet(False)
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=WINCH_TRAVEL)

    def run(self, sq: str, which: str = "ab") -> None:
        if self.ctl is None:
            print("   ! not connected — re-run with --go to move the crane")
            return
        c = self.coords(sq)
        if not c:
            return
        a, b = c
        print(f"=== {sq} ===")
        if "a" in which:
            self._attempt("A mapped   ", a)
        if "b" in which:
            self._attempt("B predicted", b)

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        from chessmachine.motion.serial_esp32 import SerialMotion
        self.ctl = SerialMotion(self.cfg.motion.serial)
        self.ctl.connect()

    def close(self) -> None:
        if self.ctl is None:
            return
        # Never leave a piece stuck to the electromagnet.
        for fn in (lambda: self.ctl.magnet(False),
                   lambda: self.ctl.goto_steps(winch=WINCH_TRAVEL)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - best-effort recovery
                print(f"   (cleanup step failed: {exc})", file=sys.stderr)
        try:
            self.ctl.close()
        except Exception:  # noqa: BLE001
            pass


HELP = ("Commands: <sq> | a <sq> | b <sq> | show <sq> | worst <axis> [n] | "
        "dwell <s> | settle <s> | home | wtop | mag on|off | help | quit")


def _prompt():
    while True:
        try:
            yield input("verify> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("squares", nargs="*", help="squares to test; omit for interactive mode")
    ap.add_argument("--map", default=None, help="map file (default: config's map_path)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--go", action="store_true", help="actually move the crane")
    ap.add_argument("--no-home", action="store_true", help="skip homing on connect")
    ap.add_argument("--worst", metavar="AXIS:N", help="batch-test the N worst on AXIS")
    ap.add_argument("--settle", type=float, default=1.0, help="pause after each move (s)")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="hold at the raised position so you can see if it lifted (s)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    map_path = pathlib.Path(args.map or cfg.motion.stepmap.map_path)
    v = Verifier(cfg, map_path, dwell=args.dwell, settle=args.settle)
    if not v.anchors:
        print(f"{map_path}: no anchors to fit from", file=sys.stderr)
        return 1

    squares = [s.lower() for s in args.squares]
    if args.worst:
        axis, _, n = args.worst.partition(":")
        if axis not in KEYS:
            print(f"--worst axis must be one of {KEYS}", file=sys.stderr)
            return 1
        squares += [s for s, _x in v.model.disagreements(v.table, axis)[:int(n or 5)]]

    print(f"map {map_path}  ({len(v.anchors)} anchors, {len(v.table)} squares)   "
          f"winch ~{STEPS_PER_MM_WINCH:.0f} steps/mm")

    # Offline: no --go means show the plan and stop (safe default).
    if not args.go:
        for s in (squares or sorted(v.table)):
            v.show(s)
        print("\nDRY RUN — nothing moved. Re-run with --go to drive the crane"
              + (" (omit squares for interactive mode)." if squares else "."))
        return 0

    v.connect()
    try:
        if not args.no_home:
            print("homing (this can take a while)...")
            v.ctl.home()

        if squares:                                  # batch mode
            for s in squares:
                v.run(s)
            return 0

        print("Put a piece on a square, then type it. " + HELP)
        for line in _prompt():
            parts = line.split()
            if not parts:
                continue
            c, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
            try:
                if c in ("quit", "exit", "q"):
                    break
                elif c in ("help", "h", "?"):
                    print("   " + HELP)
                elif c == "show" and arg:
                    v.show(arg)
                elif c == "worst" and arg:
                    v.worst(arg, int(parts[2]) if len(parts) > 2 else 6)
                elif c == "dwell" and arg:
                    v.dwell = float(arg); print(f"   dwell = {v.dwell:g}s")
                elif c == "settle" and arg:
                    v.settle = float(arg); print(f"   settle = {v.settle:g}s")
                elif c == "home":
                    v.ctl.home()
                elif c == "wtop":
                    v.ctl.goto_steps(winch=WINCH_TRAVEL); print("   winch at travel")
                elif c == "mag" and arg in ("on", "off"):
                    v.ctl.magnet(arg == "on"); print(f"   magnet {arg}")
                elif c in ("a", "b") and arg:
                    v.run(arg, which=c)
                elif len(c) == 2 and c[0] in "abcdefgh" and c[1] in "12345678":
                    v.run(c)                          # bare square name
                else:
                    print("   ? " + HELP)
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                print(f"   ! {e}")
    finally:
        v.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
