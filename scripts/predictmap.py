#!/usr/bin/env python3
"""Fit the parametric board model and emit an AUXILIARY predicted step map.

    python scripts/predictmap.py                          # report only (default, safe)
    python scripts/predictmap.py --write                   # + add a "predicted" section
    python scripts/predictmap.py --write -o config/withpred.json
    python scripts/predictmap.py --map bestmap.json        # analyse the newer root map
    python scripts/predictmap.py --top 15                  # longer disagreement lists

Reads a map file's hand-measured `anchors`, fits the stiff kinematic model in
`chessmachine.motion.predict`, and reports where the spline-fitted `table` disagrees
with it. With `--write` it adds a top-level `predicted` section (all 64 squares).

HAND-MEASURED VALUES ARE NEVER MODIFIED. `anchors` and `table` are copied through
byte-for-byte; only the new `predicted` key is added. The game ignores it — the
stepmap backend reads `table`/`squares` only — so writing it cannot affect play.

Use the disagreement report to pick squares to physically re-check with
scripts/verifymap.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config                      # noqa: E402
from chessmachine.motion.geometry import BoardGeometry           # noqa: E402
from chessmachine.motion.predict import KEYS, BoardModel         # noqa: E402

ALL_SQUARES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="config/bestmap.json", help="map file to read")
    ap.add_argument("--config", default="config/config.yaml", help="config (for board geometry)")
    ap.add_argument("--write", action="store_true", help="add the 'predicted' section")
    ap.add_argument("-o", "--out", help="write here instead of in place (implies --write)")
    ap.add_argument("--top", type=int, default=8, help="how many disagreements to list")
    ap.add_argument("--sigma", type=float, default=3.0, help="outlier rejection threshold")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    geo = BoardGeometry(cfg.motion.geometry)
    path = pathlib.Path(args.map)
    data = json.loads(path.read_text())
    anchors = data.get("anchors")
    if not anchors:
        print(f"{path}: no 'anchors' section — nothing to fit from.", file=sys.stderr)
        return 1
    table = data.get("table") or data.get("squares") or {}

    model = BoardModel(geo, anchors, reject_sigma=args.sigma)

    print(f"map: {path}   anchors: {len(anchors)}   table: {len(table)}")
    print("=" * 78)
    print("FITTED MODEL  (R = radius mm, Theta = base angle deg, from BoardGeometry)")
    print("=" * 78)
    for key in KEYS:
        f = model.fits[key]
        loo, _ = model.leave_one_out(key)
        print(f"  {f.formula()}")
        print(f"      anchors used {f.n_used}/{len(anchors)}   in-sample RMS {f.rms:7.1f}"
              f"   leave-one-out RMS {loo:7.1f}  steps")
        if f.rejected:
            det = ", ".join(f"{s} ({v:+.0f})" for s, v in
                            sorted(f.rejected.items(), key=lambda t: -abs(t[1])))
            print(f"      rejected as outliers: {det}")
    print("\n  Reading the numbers: leave-one-out is the honest error for an UNMEASURED")
    print("  square. base/winch are usable as a cross-check; rail scatters ~475 steps")
    print("  (~7 mm) because the rail anchors themselves disagree that much, so the")
    print("  measured table stays the source of truth for play.")

    if table:
        print("\n" + "=" * 78)
        print(f"TABLE vs MODEL — biggest disagreements (table - predicted), top {args.top}")
        print("=" * 78)
        for key in KEYS:
            d = model.disagreements(table, key)
            if not d:
                continue
            anchor_note = lambda s: " [anchor]" if s in anchors else ""      # noqa: E731
            print(f"\n  {key}:")
            for s, v in d[:args.top]:
                print(f"     {s:>3}  {v:+8.0f}{anchor_note(s)}")

        # The regional question: does a neighbourhood read systematically high/low?
        print("\n" + "=" * 78)
        print("REGIONAL BIAS in winch (table - predicted; + = table lowers the magnet more)")
        print("=" * 78)
        dw = dict(model.disagreements(table, "winch"))
        def mean(v): return sum(v) / len(v) if v else float("nan")
        print("   by file: " + "  ".join(
            f"{f}:{mean([dw[s] for s in dw if s[0] == f]):+6.0f}" for f in "abcdefgh"))
        print("   by rank: " + "  ".join(
            f"{r}:{mean([dw[s] for s in dw if s[1] == r]):+6.0f}" for r in "12345678"))
        abc = [dw[s] for s in dw if s[0] in "abc"]
        rest = [dw[s] for s in dw if s[0] not in "abc"]
        r34 = [dw[s] for s in dw if s[1] in "34"]
        o34 = [dw[s] for s in dw if s[1] not in "34"]
        print(f"   files a-c {mean(abc):+.0f}  vs  d-h {mean(rest):+.0f}")
        print(f"   ranks 3-4 {mean(r34):+.0f}  vs  others {mean(o34):+.0f}")

    if args.write or args.out:
        out = dict(data)                       # preserve key order + every original value
        out["predicted"] = {s: model.predict(s) for s in ALL_SQUARES}
        out["predicted_model"] = {
            "note": "AUXILIARY cross-check, generated by scripts/predictmap.py. "
                    "Not read by the game; anchors/table are authoritative.",
            "source_map": str(path),
            "reject_sigma": args.sigma,
            "axes": {k: {"formula": model.fits[k].formula(),
                         "coeffs": model.fits[k].coeffs,
                         "anchors_used": model.fits[k].n_used,
                         "in_sample_rms": round(model.fits[k].rms, 2),
                         "leave_one_out_rms": round(model.leave_one_out(k)[0], 2),
                         "rejected": {s: round(v, 1) for s, v in model.fits[k].rejected.items()}}
                     for k in KEYS},
        }
        dest = pathlib.Path(args.out) if args.out else path
        # Safety: verify we are not about to change a hand-measured value.
        if dest == path:
            before = json.loads(path.read_text())
            for sec in ("anchors", "table", "squares"):
                if before.get(sec) != out.get(sec):
                    print(f"REFUSING to write: '{sec}' would change.", file=sys.stderr)
                    return 2
        dest.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {dest}  (+{len(out['predicted'])} predicted squares; "
              f"anchors/table untouched)")
    else:
        print("\n(report only — pass --write to add the 'predicted' section)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
