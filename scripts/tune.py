#!/usr/bin/env python3
"""Active, closed-loop motion calibration for the crane's steppers.

Touch a target square, tell the tool where the magnet ACTUALLY landed (a real
square, off-board like h9 / i8 / -a8, or between files/ranks like ab8 / e4.5), and
it solves for the corrected calibration and pushes it to the firmware live over
serial (CAL) — no reflash. Both positioning axes are steppers now, so the fit is an
exact ratio: it converges in ONE round, not the iterate-and-creep the old DC cart
needed. `save` writes the final numbers to bake into the .ino.

(Tunes the MOTION constants for the two positioning axes. Board geometry — origin,
pitch, axis mapping — is a separate tool, scripts/calibrate.py. The winch is a
fixed two-position stroke set in firmware (WINCH_STROKE_STEPS) — nothing here.)

What it tunes (held in firmware RAM via CAL):
    ASPD   base steps per degree   (rotary scale)
    AHOME  base home / zero offset  (rotary offset — the sensorless base zero)
    RSPM   rail steps per mm        (radial scale)
The rail homes to its inner hard stop (step 0 = R_MIN_MM), so it needs no offset
term — just the scale.

The math (per axis, from >=2 touches at different targets): fit the landings to
    landed = m * commanded + c
then the values that make landed == commanded are
    ASPD_new = ASPD_cur / m       AHOME_new = c + AHOME_cur * m
    RSPM_new = RSPM_cur / m_rail
A single touch falls back to scale-only (holds the offset — fine for the rail, and
for the base once a theta=0 endstop makes AHOME moot).

    python scripts/tune.py --config config/config.yaml
    python scripts/tune.py --mock          # dry run, no hardware

REPL:
    t <sq>        touch a square, then report where it landed
    auto [sqs..]  touch a preset set (default a1 h8 a8 h1), reporting each
    fit           show accumulated touches + the suggested calibration
    apply         push the suggested calibration to the firmware, re-home, new round
    cal           print the firmware's current live calibration
    obs / reset   list / clear the accumulated touches
    save [path]   write the current calibration as a firmware paste-block
    home | status | estop | quit
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config          # noqa: E402
from chessmachine import factory                             # noqa: E402
from chessmachine.motion.geometry import BoardGeometry       # noqa: E402
from chessmachine.motion.serial_esp32 import SerialMotion    # noqa: E402

DEFAULT_AUTO = ["a1", "h8", "a8", "h1"]
_SQ_RE = re.compile(r"^(-?)([a-zA-Z])([a-zA-Z]?)(-?\d+(?:\.\d+)?)$")


def _file_index(letter: str, neg: bool) -> int:
    step = ord(letter.lower()) - ord("a")            # a->0, b->1, ... i->8, ...
    return -(step + 1) if neg else step               # '-a'->-1, '-b'->-2, ...


def parse_square(name: str) -> tuple[float, float]:
    """Square name -> 0-based (file_idx, rank_idx); off-board in ANY direction OK.

    Files:  a..h = 0..7 (the board).  Past the h edge:  i, j, k = 8, 9, 10.
            Past the a edge (leading '-'):  -a, -b, -c = -1, -2, -3.
            BETWEEN two files, name both: two letters = their midpoint, so
            ab = 0.5, bc = 1.5, hi = 7.5 (composes with '-': -ab = -1.5).
    Ranks:  the trailing number, so 1..8 = 0..7; off-board 9 / 0 / -1 fine, and
            fractions allowed (e4.5).
    Examples:  e4, h9, i8, e4.5, -a8 (past a), ab8 (between a and b), bc4.5.
    """
    m = _SQ_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad square {name!r} (want like e4, h9, i8, -a8, ab8, e4.5)")
    neg, l1, l2, num = m.group(1), m.group(2), m.group(3), m.group(4)
    file_idx: float = _file_index(l1, bool(neg))
    if l2:                                            # two letters -> midpoint between the files
        file_idx = (file_idx + _file_index(l2, bool(neg))) / 2.0
    rank_idx = float(num) - 1.0
    return float(file_idx), rank_idx


def linfit(pts: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares (m, c) for y = m*x + c; None if <2 points spanning some x."""
    n = len(pts)
    xs = [p[0] for p in pts]
    if n < 2 or (max(xs) - min(xs)) < 1e-6:
        return None
    sx = sum(xs)
    sy = sum(p[1] for p in pts)
    sxx = sum(x * x for x in xs)
    sxy = sum(p[0] * p[1] for p in pts)
    d = n * sxx - sx * sx
    if abs(d) < 1e-9:
        return None
    m = (n * sxy - sx * sy) / d
    c = (sy - m * sx) / n
    return m, c


class Calibrator:
    def __init__(self, cfg: Config, r_min: float):
        self.cfg = cfg
        self.geo = BoardGeometry(cfg.motion.geometry)
        self.g = cfg.motion.geometry
        self.sp = cfg.motion.speeds
        self.r_min = r_min
        ser = getattr(cfg.motion, "serial", None)
        self.offset = getattr(ser, "winch_offset_mm", 0.0) if ser else 0.0
        self.ctl = factory.create_motion_controller(cfg.motion)
        self.cur: dict = {}
        # observations for THIS round (under self.cur): (name_t, name_a, r_t,a_t, r_a,a_a)
        self.obs: list[tuple] = []

    # -- geometry helpers --------------------------------------------------- #
    def cart(self, name: str) -> tuple[float, float]:
        """Square name -> the CART polar (r_mm, angle_deg) the firmware controls,
        via the same magnet->cart transform the game uses."""
        fi, ri = parse_square(name)
        p = self.geo.point_from_indices(fi, ri)
        return SerialMotion._magnet_to_cart(p.x, p.z, self.offset)

    def point(self, name: str):
        fi, ri = parse_square(name)
        return self.geo.point_from_indices(fi, ri)

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self) -> None:
        print(f"Connecting ({self.cfg.motion.backend}) ...", flush=True)
        self.ctl.connect()
        self.cur = self.ctl.get_cal()
        print("Firmware calibration:", self._fmt_cal(self.cur))
        print("Assuming all steppers are AT HOME (0). Park them there before touching;"
              " use 'home' only to re-zero by hand.")

    def close(self) -> None:
        try:
            self.ctl.set_pulley(self.g.travel_height_mm, self.sp.lift_feed)
            self.ctl.magnet(False)
        except Exception:  # noqa: BLE001
            pass
        self.ctl.close()

    # -- touch + record ----------------------------------------------------- #
    def touch(self, name: str) -> None:
        # No re-home between touches: the steppers are exact, so each move goes
        # straight from the current tracked pose to the next target (absolute).
        # (Re-homing every time would just waste travel and slam nothing useful.)
        r_t, a_t = self.cart(name)
        p = self.point(name)
        print(f"== touch {name}: cart r={r_t:.1f} mm  theta={a_t:+.2f} deg ==", flush=True)
        self.ctl.move_xz(p.x, p.z, self.sp.travel_feed)
        self.ctl.set_pulley(self.g.pick_height_mm, self.sp.lift_feed)   # lower to touch
        try:
            ans = input(f"   landed on {name}? [Enter = yes, or type where it hit "
                        f"(h9 / i8 / -a8 / ab8=between a,b / e4.5)]: ").strip()
        finally:
            self.ctl.set_pulley(self.g.travel_height_mm, self.sp.lift_feed)  # raise
        actual = name if ans == "" or ans.lower() in ("y", "yes") else ans
        try:
            r_a, a_a = self.cart(actual)
        except ValueError as e:
            print(f"   ! {e} — touch not recorded")
            return
        self.obs.append((name, actual, r_t, a_t, r_a, a_a))
        if actual == name:
            print("   noted: on target.")
        else:
            print(f"   noted: {name} -> {actual}  "
                  f"(base off {a_a - a_t:+.2f} deg, rail off {r_a - r_t:+.1f} mm)")
        self.show_fit()

    # -- fit + suggestion --------------------------------------------------- #
    def suggest(self) -> dict | None:
        if not self.obs:
            return None
        base_pts = [(o[3], o[5]) for o in self.obs]      # (a_t, a_a)
        rail_pts = [(o[2], o[4]) for o in self.obs]      # (r_t, r_a)
        new = dict(self.cur)
        notes: list[str] = []

        bf = linfit(base_pts)
        if bf:
            m, c = bf
            if abs(m) > 1e-6:
                new["aspd"] = self.cur["aspd"] / m
                new["ahome"] = c + self.cur["ahome"] * m
                notes.append(f"base: fit slope {m:.3f}, intercept {c:+.2f} deg")
        else:
            # scale-only fallback (1 point / all same angle): ignore small angles
            rr = [t / a for t, a in base_pts if abs(a) > 1.0 and abs(t) > 1.0]
            if rr:
                new["aspd"] = self.cur["aspd"] * (sum(rr) / len(rr))
                notes.append("base: scale-only (1 pt) — AHOME held; add a 2nd angle for offset")

        rf = linfit(rail_pts)
        if rf:
            mr, cr = rf
            if abs(mr) > 1e-6:
                new["rspm"] = self.cur["rspm"] / mr
                notes.append(f"rail: fit slope {mr:.3f}, offset {cr:+.1f} mm"
                             + (f"  (~R_MIN drift {cr:+.1f} mm — nudge R_MIN if it persists)" if abs(cr) > 3 else ""))
        else:
            rr = [(t - self.r_min) / (a - self.r_min)
                  for t, a in rail_pts if abs(a - self.r_min) > 1.0]
            if rr:
                f = sum(rr) / len(rr)
                new["rspm"] = self.cur["rspm"] * f
                notes.append("rail: scale-only (1 pt)")
        new["_notes"] = notes
        return new

    def show_fit(self) -> None:
        s = self.suggest()
        if not s:
            print("   (no touches yet)")
            return
        for n in s["_notes"]:
            print("   " + n)
        print(f"   suggested:  ASPD {self.cur['aspd']:.3f} -> {s['aspd']:.3f}"
              f"   AHOME {self.cur['ahome']:+.3f} -> {s['ahome']:+.3f}"
              f"   RSPM {self.cur['rspm']:.3f} -> {s['rspm']:.3f}")
        print(f"   ({len(self.obs)} touch(es) — 'apply' to push these live, then a fresh round)")

    def apply(self) -> None:
        s = self.suggest()
        if not s:
            print("   nothing to apply (no touches).")
            return
        self.cur = self.ctl.set_cal(aspd=s["aspd"], ahome=s["ahome"], rspm=s["rspm"])
        print("   applied ->", self._fmt_cal(self.cur))
        self.obs.clear()
        # No re-home: the new cal only remaps degrees/mm <-> steps; the steppers'
        # tracked counts stay valid. Just touch again to verify (fresh round).
        print("   ready — touch again to verify (fresh round).")

    # -- reporting ---------------------------------------------------------- #
    @staticmethod
    def _fmt_cal(c: dict) -> str:
        return (f"ASPD={c.get('aspd', float('nan')):.3f}  AHOME={c.get('ahome', float('nan')):+.3f}  "
                f"RSPM={c.get('rspm', float('nan')):.3f}")

    def save_block(self, path: str | None) -> None:
        c = self.cur
        block = (
            "// --- calibrated by scripts/tune.py — paste into esp32_chess.ino ---\n"
            f"const float         A_STEPS_PER_DEG = {c['aspd']:.3f}f;\n"
            f"const float A_HOME_DEG = {c['ahome']:.3f}f;\n"
            f"const float         R_STEPS_PER_MM  = {c['rspm']:.3f}f;\n"
        )
        print("\n" + block)
        if path:
            pathlib.Path(path).write_text(block)
            print(f"   wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Active closed-loop crane motion calibration")
    ap.add_argument("--config", help="YAML config (geometry + serial port)")
    ap.add_argument("--mock", action="store_true", help="mock backend (no hardware)")
    ap.add_argument("--r-min", type=float, default=117.5,
                    help="cart R at the inner stop (mm), for rail scale math")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.mock:
        cfg.motion.backend = "mock"
    cal = Calibrator(cfg, args.r_min)
    cal.connect()
    print("Commands: t <sq> | auto | fit | apply | cal | obs | reset | save | home | status | quit")

    try:
        for line in _prompt_lines():
            parts = line.split()
            if not parts:
                continue
            c = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None
            try:
                if c in ("quit", "exit", "q"):
                    break
                elif c in ("t", "touch") and arg:
                    cal.touch(arg)
                elif c == "auto":
                    for sq in (parts[1:] or DEFAULT_AUTO):
                        cal.touch(sq)
                    cal.show_fit()
                elif c in ("fit", "show"):
                    cal.show_fit()
                elif c == "apply":
                    cal.apply()
                elif c == "cal":
                    print("   ", cal._fmt_cal(cal.ctl.get_cal()))
                elif c == "obs":
                    for o in cal.obs:
                        print(f"   {o[0]} -> {o[1]}   (theta {o[3]:+.1f}->{o[5]:+.1f}, "
                              f"r {o[2]:.0f}->{o[4]:.0f})")
                    if not cal.obs:
                        print("   (none)")
                elif c == "reset":
                    cal.obs.clear(); print("   cleared.")
                elif c == "save":
                    cal.save_block(arg)
                elif c in ("home", "zero"):
                    cal.ctl.magnet(False); cal.ctl.home(); print("   homed.")
                elif c == "status":
                    print("   ", cal.ctl.status())
                elif c == "estop":
                    cal.ctl.estop(); print("   ESTOP — send 'home' to clear.")
                elif c in ("help", "h", "?"):
                    print(__doc__)
                else:
                    print(f"   ? unknown/incomplete: {line.strip()!r}")
            except Exception as e:  # noqa: BLE001 - keep the REPL alive on a bad line
                print(f"   ! {e}")
    finally:
        cal.close()
    return 0


def _prompt_lines():
    while True:
        try:
            yield input("cal> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == "__main__":
    sys.exit(main())
