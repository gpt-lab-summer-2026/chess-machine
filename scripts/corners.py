#!/usr/bin/env python3
"""Touch the four board corners in turn (a1 -> h8 -> a8 -> h1) to reveal the
board's rotation / offset.

It uses the SAME geometry and transport as the running game (including the winch
offset), so wherever the head lowers is exactly where the machine believes each
corner is. Compare that to the real square to work out how the board model is
rotated/mirrored/shifted, then fix motion.geometry (board_angle_deg, invert_*,
origin_*) in config.yaml.

It re-homes (recalibrates) before EACH corner, so open-loop drift can't build up
between corners and each reading reflects the board geometry alone.

    python scripts/corners.py --config config/config.yaml
    python scripts/corners.py --config config/config.yaml --auto 3   # 3 s per corner
    python scripts/corners.py --mock --no-home --auto 0              # dry print only
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config  # noqa: E402
from chessmachine import factory  # noqa: E402
from chessmachine.motion.geometry import BoardGeometry  # noqa: E402

CORNERS = ["a1", "h8", "a8", "h1"]   # order requested for calibration
# Base-stepper spin test: rotate +90 one way, then 180 the other, then back to 0.
SPIN_SEQUENCE = [(+90, "+90 to one side"),
                 (-180, "180 to the other side (now -90)"),
                 (+90, "return to 0")]


def _pause(auto: float | None, prompt: str) -> None:
    if auto is not None:
        time.sleep(auto)
    else:
        input(prompt)


def _run_spin(ctl, steps_per_deg: float, auto: float | None) -> None:
    """Spin the base stepper through SPIN_SEQUENCE via the raw JOG (bypasses the
    soft-angle clamp so we can exceed +/-55 deg)."""
    if not hasattr(ctl, "jog_base"):
        print("--spin needs the serial backend (SerialMotion); this backend has no jog_base.")
        return
    for deg, label in SPIN_SEQUENCE:
        steps = int(round(deg * steps_per_deg))
        print(f"\n== spin {label}: {deg:+d} deg = {steps:+d} steps ==", flush=True)
        ctl.jog_base(steps)
        _pause(auto, "   done — measure the angle, press Enter for the next spin...")
    print("\nSpin test done. JOG doesn't track the angle — send HOME (or re-run) to re-zero.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Board-corner + base-spin calibration tool")
    ap.add_argument("--config", help="YAML config (geometry + serial port)")
    ap.add_argument("--mock", action="store_true", help="use the mock backend (prints targets, no hardware)")
    ap.add_argument("--no-home", action="store_true", help="skip homing first")
    ap.add_argument("--auto", type=float, metavar="S",
                    help="dwell S seconds at each step instead of waiting for Enter")
    ap.add_argument("--spin", action="store_true",
                    help="base-stepper spin test (+90, -180, back to 0) instead of the corners")
    ap.add_argument("--a-steps-per-deg", type=float, default=25.38,
                    help="base steps/deg for --spin (match firmware A_STEPS_PER_DEG)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.mock:
        cfg.motion.backend = "mock"
    geo = BoardGeometry(cfg.motion.geometry)
    g = cfg.motion.geometry
    sp = cfg.motion.speeds
    ctl = factory.create_motion_controller(cfg.motion)

    print(f"Connecting ({cfg.motion.backend}) ...", flush=True)
    ctl.connect()
    if args.spin:
        try:
            if not args.no_home:
                print("Homing (re-zero the base) ...", flush=True)
                ctl.home()
            _run_spin(ctl, args.a_steps_per_deg, args.auto)
        finally:
            ctl.close()
        return 0

    try:
        for sq in CORNERS:
            # Re-home (recalibrate) BEFORE every corner so each is measured from a
            # freshly-zeroed pose — open-loop drift can't accumulate corner-to-corner,
            # so the readings reflect the board geometry alone, not travel error.
            if not args.no_home:
                print("  re-homing (recalibrate) ...", flush=True)
                ctl.home()
            p = geo.name_to_point(sq)
            r = math.hypot(p.x, p.z)
            th = math.degrees(math.atan2(p.z, p.x))
            print(f"\n== {sq}:  x={p.x:.1f} z={p.z:.1f}   (magnet r={r:.1f} mm, theta={th:+.1f} deg) ==",
                  flush=True)
            ctl.move_xz(p.x, p.z, sp.travel_feed)
            ctl.set_pulley(g.pick_height_mm, sp.lift_feed)      # lower to touch the square
            if args.auto is not None:
                time.sleep(args.auto)
            else:
                input("   touching — note WHERE it landed, then press Enter for the next corner...")
            ctl.set_pulley(g.travel_height_mm, sp.lift_feed)    # raise before moving on
        print("\nDone. Report which real square each of a1/h8/a8/h1 actually touched.")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ctl.set_pulley(g.travel_height_mm, sp.lift_feed)    # leave the head parked up
            ctl.magnet(False)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        ctl.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
