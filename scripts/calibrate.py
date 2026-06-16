#!/usr/bin/env python3
"""Interactive geometry calibration.

Jog the gantry over the real board, mark the centers of a1 and h8, and print a
ready-to-paste `motion.geometry` block. Line-based commands (one per line):

    x +5 / x -5      jog X by mm        z +5 / z -1      jog Z by mm
    h +2 / h -2      jog pulley height  mag on / mag off  electromagnet
    home             re-home            status            print position
    goto e4          move to a square using the CURRENT geometry (verify)
    mark a1          record this spot as a1's center
    mark h8          record this spot as h8's center
    calc             compute + print geometry from the two marks
    quit

Run with --mock to rehearse without hardware.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import Config, load_config  # noqa: E402
from chessmachine import factory  # noqa: E402
from chessmachine.motion.geometry import BoardGeometry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive gantry calibration")
    ap.add_argument("--config", help="YAML config (for serial port + current geometry)")
    ap.add_argument("--mock", action="store_true", help="use the mock backend")
    ap.add_argument("--step", type=float, default=5.0, help="default jog step (mm)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.mock:
        cfg.motion.backend = "mock"

    ctl = factory.create_motion_controller(cfg.motion)
    geo = BoardGeometry(cfg.motion.geometry)
    ctl.connect()
    ctl.home()

    x = z = 0.0
    h = cfg.motion.geometry.travel_height_mm
    ctl.set_pulley(h)
    marks: dict[str, tuple[float, float]] = {}
    print(__doc__)

    def move():
        ctl.move_xz(x, z, cfg.motion.speeds.travel_feed)

    for line in sys.stdin:
        cmd = line.strip().lower()
        if not cmd:
            continue
        try:
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd == "home":
                ctl.home(); x = z = 0.0
            elif cmd == "status":
                print(f"  x={x:.2f} z={z:.2f} h={h:.2f} marks={marks}")
            elif cmd.startswith(("x ", "z ", "h ")):
                axis, delta = cmd.split()
                d = float(delta)
                if axis == "x":
                    x += d; move()
                elif axis == "z":
                    z += d; move()
                else:
                    h += d; ctl.set_pulley(h, cfg.motion.speeds.lift_feed)
            elif cmd.startswith("mag"):
                ctl.magnet(cmd.endswith("on"))
            elif cmd.startswith("goto "):
                sq = cmd.split()[1]
                p = geo.name_to_point(sq)
                x, z = p.x, p.z; move()
                print(f"  -> {sq} at ({x:.2f}, {z:.2f})")
            elif cmd.startswith("mark "):
                sq = cmd.split()[1]
                marks[sq] = (x, z)
                print(f"  marked {sq} = ({x:.2f}, {z:.2f})")
            elif cmd == "calc":
                _print_geometry(marks)
            else:
                print(f"  ? unknown command: {cmd}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! error: {exc}")

    ctl.close()
    return 0


def _print_geometry(marks: dict[str, tuple[float, float]]) -> None:
    if "a1" not in marks or "h8" not in marks:
        print("  need both 'mark a1' and 'mark h8' first")
        return
    (ax, az), (hx, hz) = marks["a1"], marks["h8"]
    pitch_x = abs(hx - ax) / 7.0
    pitch_z = abs(hz - az) / 7.0
    print("\n# --- paste into config.yaml under motion.geometry ---")
    print(f"    origin_x_mm: {ax:.2f}")
    print(f"    origin_z_mm: {az:.2f}")
    print(f"    square_pitch_mm: {(pitch_x + pitch_z) / 2:.3f}   "
          f"# x-pitch={pitch_x:.3f} z-pitch={pitch_z:.3f}")
    print(f"    invert_file: {hx < ax}")
    print(f"    invert_rank: {hz < az}")
    print("# ----------------------------------------------------\n")


if __name__ == "__main__":
    sys.exit(main())
