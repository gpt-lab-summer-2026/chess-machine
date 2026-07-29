#!/usr/bin/env python3
"""A/B pickup test: try to grab the piece at the MAPPED coords, then at the PREDICTED
coords, and watch which one sticks.

    python scripts/verifymap.py b3                    # DRY RUN: print the plan only
    python scripts/verifymap.py b3 --go               # actually drive the crane
    python scripts/verifymap.py b3 c3 b4 --go         # several squares in one session
    python scripts/verifymap.py --worst winch:6 --go  # the 6 worst winch disagreements
    python scripts/verifymap.py b3 --go --no-home     # skip homing (already homed)

Put a piece on the square, then per square the crane runs two INDEPENDENT attempts and
returns the piece each time:

    attempt A (mapped)     GOTO mapped base/rail -> lower to mapped depth -> magnet ON
                           -> raise (does it lift?) -> lower -> magnet OFF -> raise
    attempt B (predicted)  same, using the model's base/rail/depth

Nothing is asked mid-run — it just executes, holding at the raised position for --dwell
seconds each time so you can see whether the piece came up. If it sticks on B, the
formula reaches that square; if only A works, the map's value is doing real work there.

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("squares", nargs="*", help="squares to test, e.g. b3 c3")
    ap.add_argument("--map", default=None, help="map file (default: config's map_path)")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--go", action="store_true", help="actually move the crane")
    ap.add_argument("--no-home", action="store_true", help="skip homing on connect")
    ap.add_argument("--worst", metavar="AXIS:N",
                    help="instead of naming squares, take the N worst disagreements "
                         "on AXIS (base|rail|winch)")
    ap.add_argument("--settle", type=float, default=1.0, help="pause after each move (s)")
    ap.add_argument("--dwell", type=float, default=3.0,
                    help="hold at the raised position so you can see if it lifted (s)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    geo = BoardGeometry(cfg.motion.geometry)
    map_path = pathlib.Path(args.map or cfg.motion.stepmap.map_path)
    data = json.loads(map_path.read_text())
    anchors, table = data.get("anchors", {}), data.get("table") or data.get("squares") or {}
    if not anchors:
        print(f"{map_path}: no anchors to fit from", file=sys.stderr)
        return 1
    model = BoardModel(geo, anchors)

    squares = [s.lower() for s in args.squares]
    if args.worst:
        axis, _, n = args.worst.partition(":")
        if axis not in KEYS:
            print(f"--worst axis must be one of {KEYS}", file=sys.stderr)
            return 1
        squares += [s for s, _v in model.disagreements(table, axis)[:int(n or 5)]]
    if not squares:
        ap.error("name at least one square, or use --worst AXIS:N")
    missing = [s for s in squares if s not in table]
    if missing:
        print(f"not in the map table: {missing}", file=sys.stderr)
        return 1

    print(f"map {map_path}  ({len(anchors)} anchors)   squares: {' '.join(squares)}")
    print(f"winch scale ~{STEPS_PER_MM_WINCH:.0f} steps/mm\n")
    print("  square      A = mapped (hand-measured)        B = predicted (model)        B-A")
    plan = []
    for s in squares:
        a = {k: int(table[s][k]) for k in KEYS}
        b = model.predict(s)
        plan.append((s, a, b))
        print(f"    {s:<4}  base {a['base']:+6d} rail {a['rail']:6d} w {a['winch']:5d}"
              f"   |  base {b['base']:+6d} rail {b['rail']:6d} w {b['winch']:5d}"
              f"   |  dbase {b['base']-a['base']:+5d} drail {b['rail']-a['rail']:+6d}"
              f" dw {b['winch']-a['winch']:+5d} ({(b['winch']-a['winch'])/STEPS_PER_MM_WINCH:+.2f}mm)")

    if not args.go:
        print("\nDRY RUN — nothing moved. Re-run with --go to drive the crane.")
        print("Put a piece on each square first.")
        return 0

    from chessmachine.motion.serial_esp32 import SerialMotion
    ctl = SerialMotion(cfg.motion.serial)
    print()
    ctl.connect()
    try:
        if not args.no_home:
            print("homing (this can take a while)...")
            ctl.home()
        for s, a, b in plan:
            print(f"\n=== {s} ===")
            for label, c in (("A mapped   ", a), ("B predicted", b)):
                print(f"  {label}: base={c['base']} rail={c['rail']} winch={c['winch']}")
                ctl.goto_steps(winch=WINCH_TRAVEL)
                ctl.goto_steps(base=c["base"], rail=c["rail"])
                time.sleep(args.settle)
                ctl.goto_steps(winch=c["winch"])      # lower onto the piece
                ctl.magnet(True)
                time.sleep(args.settle)
                ctl.goto_steps(winch=WINCH_TRAVEL)    # lift — watch here
                print(f"     ...raised, holding {args.dwell:g}s — did it lift?")
                time.sleep(args.dwell)
                ctl.goto_steps(winch=c["winch"])      # put it back where it was
                time.sleep(args.settle)
                ctl.magnet(False)
                time.sleep(args.settle)
                ctl.goto_steps(winch=WINCH_TRAVEL)
    except KeyboardInterrupt:
        print("\ninterrupted — releasing the magnet and raising the winch")
    finally:
        # Never leave a piece stuck to the electromagnet.
        for fn in (lambda: ctl.magnet(False), lambda: ctl.goto_steps(winch=WINCH_TRAVEL)):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - best-effort recovery
                print(f"  (cleanup step failed: {exc})", file=sys.stderr)
        try:
            ctl.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
