#!/usr/bin/env python3
"""A/B/C pickup test: try to grab the piece at the MAPPED coords, the PREDICTED coords,
or halfway between — and keep whichever wins.

    python scripts/verifymap.py --go              # interactive: hand-pick squares
    python scripts/verifymap.py                   # offline: whole-board plan, no hardware
    python scripts/verifymap.py b3 c3 b4 --go     # batch: run these squares and exit
    python scripts/verifymap.py --worst winch:6 --go

Put a piece on the square, then per attempt the crane lowers, grabs, lifts (so you can
see whether it came up), sets the piece back down and releases. Three candidates:

    A  mapped      the hand-measured value in the map's `table`
    B  predicted   the parametric model (chessmachine.motion.predict)
    C  halfway     round((A+B)/2) per axis — for when A undershoots and B overshoots

The winch only lifts --lift steps above the pick depth (default 2000, ~15 mm), not the
full stroke: enough to see the piece clear the board and to shuffle between candidates.

When a candidate wins, `accept` writes it into the map:
    accept b3 b            -> table[b3] = the predicted coords
    accept b3 c            -> table[b3] = the halfway coords
    accept b3 b anchor     -> ALSO replace the hand-measured anchors[b3]
A .bak copy is made before the first write of a session. Note the distinction: without
`anchor`, only the derived `table` changes, so a later `anchor.py map` refit — which
regenerates `table` from `anchors` — would undo it. If the anchor itself was mis-measured
(which is what a winning B usually means), pass `anchor` to fix it at the source.

Interactive commands (like scripts/anchor.py):
    <sq>            run A then B          a|b|c <sq>   run one candidate
    abc <sq>        run all three         show <sq>    print all three, move nothing
    accept <sq> [a|b|c] [anchor]          worst <axis> [n]
    lift <n> | dwell <s> | settle <s>     adjust live
    home | wtop | mag on|off              raw helpers (wtop = full raise to travel)
    help | quit

SAFETY. Dry run is the default; nothing moves without --go. The magnet is released and
the winch fully raised in a finally block, so Ctrl-C cannot leave a piece stuck to the
electromagnet.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config                      # noqa: E402
from chessmachine.motion.geometry import BoardGeometry           # noqa: E402
from chessmachine.motion.predict import KEYS, BoardModel         # noqa: E402

WINCH_TRAVEL = 0          # stepmap convention: winch step 0 == fully raised / travel
STEPS_PER_MM_WINCH = 7500 / 56.0     # WINCH_STROKE_STEPS over travel-to-pick mm
LABELS = {"a": "A mapped   ", "b": "B predicted", "c": "C halfway  "}


class Verifier:
    def __init__(self, cfg, map_path: pathlib.Path, dwell: float, settle: float, lift: int):
        self.cfg = cfg
        self.geo = BoardGeometry(cfg.motion.geometry)
        self.map_path = map_path
        self.dwell, self.settle, self.lift = dwell, settle, lift
        self.ctl = None
        self._backed_up = False
        self._load()

    def _load(self) -> None:
        data = json.loads(self.map_path.read_text())
        self.data = data
        self.anchors = data.get("anchors", {})
        self.table = data.get("table") or data.get("squares") or {}
        self.model = BoardModel(self.geo, self.anchors)

    # -- candidates ---------------------------------------------------------- #
    def candidates(self, sq: str) -> dict[str, dict] | None:
        sq = sq.lower()
        if sq not in self.table:
            print(f"   ! {sq} is not in the map table")
            return None
        a = {k: int(self.table[sq][k]) for k in KEYS}
        b = self.model.predict(sq)
        c = {k: int(round((a[k] + b[k]) / 2)) for k in KEYS}
        return {"a": a, "b": b, "c": c}

    def show(self, sq: str) -> None:
        cs = self.candidates(sq)
        if not cs:
            return
        tag = "  [hand-measured anchor]" if sq.lower() in self.anchors else ""
        print(f"   {sq}{tag}")
        for k in ("a", "b", "c"):
            v = cs[k]
            print(f"     {LABELS[k]}  base {v['base']:+6d}  rail {v['rail']:6d}"
                  f"  winch {v['winch']:5d}")
        dw = cs["b"]["winch"] - cs["a"]["winch"]
        print(f"     B-A           base {cs['b']['base']-cs['a']['base']:+6d}"
              f"  rail {cs['b']['rail']-cs['a']['rail']:+6d}"
              f"  winch {dw:+5d}  ({dw/STEPS_PER_MM_WINCH:+.2f} mm)")

    def worst(self, axis: str, n: int = 6) -> None:
        if axis not in KEYS:
            print(f"   ! axis must be one of {', '.join(KEYS)}")
            return
        print(f"   biggest {axis} disagreements (table - predicted):")
        for s, val in self.model.disagreements(self.table, axis)[:n]:
            t = " [anchor]" if s in self.anchors else ""
            extra = f"  ({val/STEPS_PER_MM_WINCH:+.2f} mm)" if axis == "winch" else ""
            print(f"     {s:>3}  {val:+8.0f}{extra}{t}")

    # -- motion -------------------------------------------------------------- #
    def _attempt(self, which: str, c: dict) -> None:
        lift_to = max(WINCH_TRAVEL, c["winch"] - self.lift)   # partial lift, never past travel
        print(f"   {LABELS[which]}: base={c['base']} rail={c['rail']} winch={c['winch']}"
              f"  (lift to {lift_to})")
        self.ctl.goto_steps(base=c["base"], rail=c["rail"])
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=c["winch"])        # lower onto the piece
        self.ctl.magnet(True)
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=lift_to)           # lift — watch here
        print(f"      ...lifted {c['winch']-lift_to} steps, holding {self.dwell:g}s — did it lift?")
        time.sleep(self.dwell)
        self.ctl.goto_steps(winch=c["winch"])        # set it back down where it was
        time.sleep(self.settle)
        self.ctl.magnet(False)
        time.sleep(self.settle)
        self.ctl.goto_steps(winch=lift_to)           # clear of the board for the next candidate

    def run(self, sq: str, which: str = "ab") -> None:
        if self.ctl is None:
            print("   ! not connected — re-run with --go to move the crane")
            return
        cs = self.candidates(sq)
        if not cs:
            return
        print(f"=== {sq} ===")
        # Full raise before the first candidate: the base/rail hop to a NEW square can
        # be long, and only a full lift is guaranteed to clear other pieces.
        self.ctl.goto_steps(winch=WINCH_TRAVEL)
        for k in which:
            if k in cs:
                self._attempt(k, cs[k])
        self.ctl.goto_steps(winch=WINCH_TRAVEL)

    # -- accepting a candidate ----------------------------------------------- #
    def accept(self, sq: str, which: str = "b", also_anchor: bool = False) -> None:
        sq = sq.lower()
        cs = self.candidates(sq)
        if not cs:
            return
        if which not in cs:
            print("   ! pick one of a, b, c")
            return
        new = cs[which]
        if not self._backed_up:
            bak = self.map_path.with_suffix(self.map_path.suffix + ".bak")
            shutil.copy2(self.map_path, bak)
            self._backed_up = True
            print(f"   backed up {self.map_path} -> {bak}")
        old = dict(self.table[sq])
        self.table[sq] = dict(new)
        note = ""
        if also_anchor:
            if sq in self.anchors:
                self.anchors[sq] = dict(new)
                note = " + anchor"
            else:
                self.anchors[sq] = dict(new)
                note = " + NEW anchor"
        self.map_path.write_text(json.dumps(self.data, indent=2) + "\n")
        print(f"   {sq}: table{note} = {LABELS[which].strip()}  {old} -> {dict(new)}")
        if also_anchor:
            self._load()          # anchors changed -> refit so later predictions follow
            print("   refitted the model on the updated anchors")
        elif sq in self.anchors:
            print(f"   NOTE anchors[{sq}] still holds the old value; an `anchor.py map`"
                  f" refit would undo this. Use `accept {sq} {which} anchor` to fix the source.")

    # -- lifecycle ----------------------------------------------------------- #
    def connect(self) -> None:
        from chessmachine.motion.serial_esp32 import SerialMotion
        self.ctl = SerialMotion(self.cfg.motion.serial)
        self.ctl.connect()

    def close(self) -> None:
        if self.ctl is None:
            return
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


HELP = ("Commands: <sq> | a|b|c <sq> | abc <sq> | show <sq> | accept <sq> [a|b|c] [anchor] | "
        "worst <axis> [n] | lift <n> | dwell <s> | settle <s> | home | wtop | mag on|off | quit")


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
                    help="hold at the lifted position so you can see if it lifted (s)")
    ap.add_argument("--lift", type=int, default=2000,
                    help="steps to lift above the pick depth (default 2000, ~15mm)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    map_path = pathlib.Path(args.map or cfg.motion.stepmap.map_path)
    v = Verifier(cfg, map_path, dwell=args.dwell, settle=args.settle, lift=args.lift)
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
          f"winch ~{STEPS_PER_MM_WINCH:.0f} steps/mm   lift {v.lift} steps")

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
        if squares:
            for s in squares:
                v.run(s)
            return 0

        print("Put a piece on a square, then type it. " + HELP)
        for line in _prompt():
            parts = line.split()
            if not parts:
                continue
            c, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
            rest = [p.lower() for p in parts[2:]]
            try:
                if c in ("quit", "exit", "q"):
                    break
                elif c in ("help", "h", "?"):
                    print("   " + HELP)
                elif c == "show" and arg:
                    v.show(arg)
                elif c == "accept" and arg:
                    which = rest[0] if rest and rest[0] in LABELS else "b"
                    v.accept(arg, which, also_anchor="anchor" in rest)
                elif c == "worst" and arg:
                    v.worst(arg, int(parts[2]) if len(parts) > 2 else 6)
                elif c == "lift" and arg:
                    v.lift = int(arg); print(f"   lift = {v.lift} steps "
                                             f"(~{v.lift/STEPS_PER_MM_WINCH:.1f} mm)")
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
                elif c in ("a", "b", "c", "ab", "abc") and arg:
                    v.run(arg, which=c)
                elif len(c) == 2 and c[0] in "abcdefgh" and c[1] in "12345678":
                    v.run(c)
                else:
                    print("   ? " + HELP)
            except Exception as e:  # noqa: BLE001 - keep the REPL alive
                print(f"   ! {e}")
    finally:
        v.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
