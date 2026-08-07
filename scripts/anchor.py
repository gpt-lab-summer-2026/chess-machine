#!/usr/bin/env python3
"""Ground-truth step-map calibration for the polar crane.

WHY (vs tune.py): tune.py fits a 1-D scale+offset on the base ANGLE and a 1-D
scale on the rail RADIUS. In polar (r, theta) that can rotate/shift/stretch the
board but CANNOT move the pivot or fix a skew -- so aligning the a1-h8 diagonal
warps the a8-h1 one, forever. This tool throws the geometry model away: you JOG
the head to real squares, GREENLIGHT each as ground truth (its raw base/rail/winch
step counts read straight from the firmware), and a THIN-PLATE SPLINE is fit through
ALL of them -- the four corners plus any mid-board anchors -- to fill the 64-square
table. The spline honors every anchor EXACTLY and bends smoothly between them, so
each mid anchor you add corrects the curvature a 4-corner fit misses (with just the
four corners it stays within ~2 mm of the old bilinear). No origin, pitch, steps/deg
or steps/mm -- measured truth plus a smooth fit.

It also captures WINCH depth per corner, so the crane's SAG is calibrated: the arm
droops when the cart is extended, so the magnet at a1 hangs lower than at h8 and
needs fewer winch steps to touch. Jog the winch down to touch at each corner and
it's baked into the map like base/rail.

Workflow:
  1. `home`             establish step 0 on all axes. The BASE (a8) and CART (inner home)
                        each self-home to a limit switch (no hand-placing); only the WINCH
                        is sensorless, so start it UP. One-time: run `seek` to measure the
                        base switch offset -> A_ENDSTOP_STEPS (see below).
  2. jog to a corner:   `a <steps>` (base), `r <steps>` (rail)
  3. `w <steps>`        jog the winch DOWN until the magnet just touches the square
                        (`mag on` to feel it grab); the winch count is its depth
  4. `set a1`           greenlight: log (base, rail, winch) as a1's truth
  5. `wtop`             raise the winch back before moving to the next corner
  6. repeat h1, a8, h8, then add mid-board anchors (e4, c6, ...) to pin curvature
  7. `map`              fit the spline; print corner deltas, sag, + leave-one-out accuracy
  8. `goto <sq>`        drive to a square's fitted base/rail (verify)
  9. `save map.json`    export anchors + the fitted 64-square table

Commands:
  a <n> / r <n> / w <n>   jog base / rail / winch by N raw steps (signed)
  wtop            raise the winch back to travel (step 0)
  pos             show current step position
  set <sq>        capture current pose as <sq>'s ground truth
  anchors         list captured anchors          del <sq>   remove one
  load <path>     load anchors from a saved map file (to re-fit / add more)
  offset <n>      add n base steps to EVERY square (board-wide rotation drift fix); `offset`
                  alone shows the total, `offset reset` zeroes it. `save` bakes it in.
  fit model|spline  choose the interpolation: `spline` = thin-plate (exact, honors every
                  anchor but chases noise); `model` = stiff parametric least-squares
                  (motion.predict) that resists noise so MORE anchors help. `map`/`goto`/`save` follow it.
  map             fit over ALL anchors; print deltas/sag/LOO (spline) or formulas/LOO (model)
  goto <sq>       drive to <sq>'s fitted base/rail (verify)
  pawns [n]       physical test: push every pawn fwd one rank x2 (2->3->4, 7->6->5),
                  re-homing every n transfers (default 3). Set up pawns on ranks 2 & 7.
  backrank [n]    physical test: move rank 1 -> rank 2, then rank 8 -> rank 7 (in order),
                  re-homing every n transfers (default 3). Set up pieces on ranks 1 & 8.
  seek            measure the base limit-switch offset -> A_ENDSTOP_STEPS (HOME first)
  mag on|off      electromagnet
  home | pos | save [path] | estop | quit

    python scripts/anchor.py --config config/config.yaml
    python scripts/anchor.py --mock
    python scripts/anchor.py --refit map.json [--out new.json]   # offline re-fit, no hardware
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
import time

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


def _solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve A x = b (A square) by Gaussian elimination w/ partial pivoting.
    Returns x, or None if the system is singular. Pure Python (no numpy)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-9:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] / pv
            if factor:
                for k in range(col, n + 1):
                    M[r][k] -= factor * M[col][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _phi(r2: float) -> float:
    """Thin-plate radial basis phi(r) = r^2 * ln(r), taking r^2 (phi(0) = 0)."""
    return 0.0 if r2 <= 1e-12 else 0.5 * r2 * math.log(r2)


class _TPS:
    """Thin-plate spline through scattered 2-D points -> a smooth interpolant.

    Fits f(u,v) = a0 + a1*u + a2*v + sum_i w_i*phi(|p - p_i|): passes through EVERY
    anchor exactly and minimizes bending between them. With coplanar values the
    weights vanish and it's just the affine plane; in general each extra anchor adds
    exactly the curvature it measures (4 corners stay within ~2 mm of bilinear)."""
    def __init__(self, pts: list[tuple[float, float]], vals: list[float]):
        n = len(pts)
        A = [[0.0] * (n + 3) for _ in range(n + 3)]
        b = [0.0] * (n + 3)
        for i in range(n):
            for j in range(n):
                du, dv = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
                A[i][j] = _phi(du * du + dv * dv)
            A[i][n], A[i][n + 1], A[i][n + 2] = 1.0, pts[i][0], pts[i][1]
            A[n][i], A[n + 1][i], A[n + 2][i] = 1.0, pts[i][0], pts[i][1]
            b[i] = vals[i]
        sol = _solve(A, b)
        if sol is None:
            raise ValueError("spline system singular (anchors collinear or duplicated)")
        self.pts, self.w, self.a = pts, sol[:n], sol[n:]

    def __call__(self, u: float, v: float) -> float:
        s = self.a[0] + self.a[1] * u + self.a[2] * v
        for (pu, pv), wi in zip(self.pts, self.w):
            du, dv = u - pu, v - pv
            s += wi * _phi(du * du + dv * dv)
        return s


def _fit(anchors: dict, names: list[str]) -> dict | None:
    """Fit one TPS per key (base/rail/winch) over the named anchors, in normalized
    board coords (file/7, rank/7). None if too few or a degenerate (collinear) set."""
    if len(names) < 4:
        return None
    pts = []
    for s in names:
        f, r = sq_uv(s)
        pts.append((f / 7.0, r / 7.0))
    fits: dict = {}
    for k in KEYS:
        try:
            fits[k] = _TPS(pts, [anchors[s][k] for s in names])
        except ValueError:
            return None
    return fits


class StepMap:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sp = cfg.motion.speeds
        self.ctl = factory.create_motion_controller(cfg.motion)
        self.anchors: dict[str, dict] = {}   # square -> {"base","rail","winch"}
        self._base_offset = 0                # board-wide base-rotation offset (steps), added to EVERY
                                             # square by build(); baked into the anchors on save. Corrects a
                                             # steady drift after the crane structure shifts (`offset` cmd).
        self._loaded_path: str | None = None  # last file `load`ed; save() defaults back to it
        self.fit_mode = "spline"             # "spline" = thin-plate (exact interpolant) | "model" = stiff
                                             # parametric least-squares fit (motion.predict.BoardModel):
                                             # can't bend to one anchor, so MORE anchors reduce distortion

    def connect(self) -> None:
        print(f"Connecting ({self.cfg.motion.backend}) ...", flush=True)
        self.ctl.connect()
        print("Ready. The base + cart self-home to their limit switches; only the winch "
              "is sensorless, so start it UP, then `home` to zero the counters.")

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

    def build(self) -> dict | None:
        if self.fit_mode == "model":
            return self._build_model()
        names = list(self.anchors)
        fits = _fit(self.anchors, names)
        if fits is None:
            print(f"   need >=4 non-collinear anchors to fit a table (have {len(names)})")
            return None
        table: dict[str, dict] = {}
        for f in range(8):
            for r in range(8):
                u, v = f / 7.0, r / 7.0
                entry = {k: round(fits[k](u, v)) for k in KEYS}
                entry["base"] += self._base_offset   # board-wide base-rotation correction
                table[FILES[f] + str(r + 1)] = entry
        return table

    def _build_model(self) -> dict | None:
        """Stiff parametric fit (motion.predict.BoardModel): a low-order least-squares
        model of the crane kinematics with outlier rejection. It CAN'T bend to a single
        anchor, so more anchors AVERAGE OUT measurement noise instead of adding wiggles
        — the opposite of the exact spline. Best when anchors are noisy / you're fighting
        'fix one square, distort another'."""
        from chessmachine.motion.geometry import BoardGeometry
        from chessmachine.motion.predict import BoardModel
        if len(self.anchors) < 4:
            print(f"   model fit needs >=4 anchors (have {len(self.anchors)})")
            return None
        try:
            bm = BoardModel(BoardGeometry(self.cfg.motion.geometry), self.anchors)
        except ValueError as e:  # noqa: BLE001 - degenerate anchors
            print(f"   model fit failed ({e}). The stiff model needs anchors that span R and"
                  " theta — 4 symmetric corners alone are degenerate for its curvature terms."
                  " Add a CENTRE anchor (d4/e5) + an edge midpoint or two, then `fit model` again."
                  " (With only the 4 corners, use `fit spline`.)")
            return None
        table = bm.predict_all()
        if self._base_offset:
            for entry in table.values():
                entry["base"] += self._base_offset
        return table

    def report(self) -> dict | None:
        table = self.build()
        if not table:
            return None
        if self.fit_mode == "model":
            self._report_model()
            return table
        A = self.anchors

        def have(*sqs: str) -> bool:
            return all(s in A for s in sqs)

        print(f"   fitted a thin-plate spline through {len(A)} anchors "
              "(each honored exactly; smooth between).")
        print("   corner step-distances (base, rail, winch):")
        for s1, s2 in [("a1", "a8"), ("h1", "h8"), ("a1", "h1"),
                       ("a8", "h8"), ("a1", "h8"), ("a8", "h1")]:
            if have(s1, s2):
                print(f"     {s1}->{s2}:  base {A[s2]['base']-A[s1]['base']:+6d}   "
                      f"rail {A[s2]['rail']-A[s1]['rail']:+6d}   winch {A[s2]['winch']-A[s1]['winch']:+6d}")
        if have("a1", "h8"):
            wa1, wh8 = A["a1"]["winch"], A["h8"]["winch"]
            print(f"   SAG (winch touch depth):  a1={wa1}  h8={wh8}  -> a1 needs {wh8 - wa1:+d} "
                  "steps less than h8 (arm droops when extended)")

        # Leave-one-out: drop each anchor, refit, predict it. At anchors the spline
        # is exact, so this is the HONEST accuracy BETWEEN measurements — a big LOO
        # error at a square means the curvature there wants another anchor nearby.
        names = list(A)
        if len(names) >= 5:
            print("   leave-one-out residuals (measured - predicted-without-it):")
            worst = 0
            for s in names:
                fits = _fit(A, [o for o in names if o != s])
                if fits is None:
                    continue
                f, r = sq_uv(s)
                u, v = f / 7.0, r / 7.0
                errs = {k: A[s][k] - round(fits[k](u, v)) for k in KEYS}
                worst = max(worst, abs(errs["base"]), abs(errs["rail"]))
                print(f"     {s}:  " + "   ".join(f"{k} {errs[k]:+5d}" for k in KEYS))
            print(f"   worst base/rail LOO error: {worst} steps (rail ~{worst/50.0:.1f} mm). "
                  "Add an anchor near the worst square to shrink it.")
        else:
            print("   (add >=5 anchors for a leave-one-out accuracy check)")
        return table

    def _report_model(self) -> None:
        from chessmachine.motion.geometry import BoardGeometry
        from chessmachine.motion.predict import BoardModel
        bm = BoardModel(BoardGeometry(self.cfg.motion.geometry), self.anchors)
        n = len(self.anchors)
        print(f"   STIFF parametric model over {n} anchors (least-squares of the crane "
              "kinematics; no single anchor can bend it):")
        for key in KEYS:
            fit = bm.fits[key]
            loo, _ = bm.leave_one_out(key)
            mm = f" (~{loo / 50.0:.1f} mm)" if key == "rail" else ""
            rej = ("  dropped " + ",".join(f"{s}({r:+.0f})" for s, r in fit.rejected.items())
                   if fit.rejected else "")
            print(f"     {fit.formula()}")
            print(f"        used {fit.n_used}/{n}   in-RMS {fit.rms:.0f}   "
                  f"leave-one-out {loo:.0f}{mm}{rej}")
        print("   LOO is the honest accuracy. Shrink it with SPANNING anchors (4 corners +"
              " a centre, then edge mids) — clustered / bad-square anchors don't help a stiff fit.")

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

    def offset(self, n: int | None) -> None:
        """Board-wide base-rotation offset control. `offset <n>` adds n steps to EVERY
        square (corrects a steady drift after the crane structure shifts); repeat to
        fine-tune. `offset` alone reports the running total; `offset reset` zeroes it.
        `goto`/`map` reflect it live; `save` bakes it into the anchors."""
        if n is None:
            print(f"   base offset = {self._base_offset:+d} steps "
                  "(added to every square; `save` bakes it in)")
            return
        self._base_offset += int(n)
        print(f"   base offset now {self._base_offset:+d} steps ({int(n):+d} this call) — "
              "verify with `goto`, then `save`.")

    def save(self, path: str | None) -> None:
        if self._base_offset:                       # bake the live offset into the anchors, then clear it
            for v in self.anchors.values():
                v["base"] += self._base_offset
            print(f"   baked base offset {self._base_offset:+d} into the anchors")
            self._base_offset = 0
        table = self.build()
        out = {"anchors": self.anchors, "table": table}
        p = pathlib.Path(path or self._loaded_path or "stepmap.json")
        p.write_text(json.dumps(out, indent=2))
        print(f"   wrote {p} ({len(self.anchors)} anchors"
              + (", refitted 64-square table)" if table else ", no table yet — need >=4 anchors)"))

    def load(self, path: str) -> None:
        """Pull the raw anchors out of a saved map file so they can be re-fit (and
        more added). Reads the `anchors` section — the game only ever used `table`,
        so every measurement you took is preserved there."""
        data = json.loads(pathlib.Path(path).read_text())
        anchors = data.get("anchors")
        if not anchors:
            print(f"   {path}: no 'anchors' section to re-fit (only a derived table?).")
            return
        loaded: dict[str, dict] = {}
        for name, v in anchors.items():
            try:
                sq_uv(name)
            except ValueError:
                continue                                  # skip non-board keys
            if all(k in v for k in KEYS):
                loaded[name.lower()] = {k: int(v[k]) for k in KEYS}
        self.anchors = loaded
        self._loaded_path = path                      # `save` with no arg writes back here
        print(f"   loaded {len(loaded)} anchors from {path}: {' '.join(sorted(loaded))}")

    # -- physical test loop -------------------------------------------------- #
    def pawn_sweep(self, rehome_every: int = 3) -> None:
        """Physical shakedown: push every pawn forward one rank, TWICE, driving the
        fitted map and re-homing every `rehome_every` transfers exactly like gameplay.
        Loop 1: White a2->a3 .. h2->h3, then Black a7->a6 .. h7->h6.
        Loop 2: White a3->a4 .. h3->h4, then Black a6->a5 .. h6->h5.
        Set up 8 White pawns on rank 2 and 8 Black on rank 7 first. Ctrl-C stops it."""
        loop1 = [(f + "2", f + "3") for f in FILES] + [(f + "7", f + "6") for f in FILES]
        loop2 = [(f + "3", f + "4") for f in FILES] + [(f + "6", f + "5") for f in FILES]
        self._sweep("pawn sweep", loop1 + loop2, rehome_every)

    def backrank_sweep(self, rehome_every: int = 3) -> None:
        """Physical shakedown: move every back-rank piece off its start square — all
        of rank 1 to rank 2 (a1->a2 .. h1->h2), then rank 8 to rank 7 (a8->a7 .. h8->h7),
        re-homing every `rehome_every` transfers like gameplay. Set up 8 pieces on rank 1
        and 8 on rank 8 (and clear ranks 2 & 7) first. Ctrl-C stops it."""
        seq = [(f + "1", f + "2") for f in FILES] + [(f + "8", f + "7") for f in FILES]
        self._sweep("back-rank sweep", seq, rehome_every)

    def _sweep(self, label: str, seq: list, rehome_every: int) -> None:
        """Drive a list of (src, dst) transfers with the gameplay pick/place, re-homing
        every `rehome_every` like gameplay. Auto-loads the configured map if none is
        loaded. Ctrl-C stops it (drops the magnet, raises the winch)."""
        if not self.anchors:
            self.load(self.cfg.motion.stepmap.map_path)   # fall back to the configured map
        table = self.build()
        if not table:
            print("   no map to sweep — `load <file>` a step map first.")
            return
        dip = int(self.cfg.motion.magnet.pick_dip_steps)
        hover_s = self.cfg.motion.magnet.hover_ms / 1000.0
        print(f"   {label}: {len(seq)} transfers, re-home every {rehome_every} "
              f"(dip={dip}, hover={hover_s:.1f}s). Ctrl-C to stop.")
        done = 0
        try:
            for src, dst in seq:
                print(f"   [{done + 1}/{len(seq)}] {src} -> {dst}", flush=True)
                self._transfer_squares(table[src], table[dst], dip, hover_s)
                done += 1
                if done % rehome_every == 0 and done < len(seq):
                    print(f"   re-homing after {done} moves (as gameplay does)...", flush=True)
                    self.ctl.home()
        except KeyboardInterrupt:
            print("\n   interrupted — releasing the magnet and raising the winch.")
            self.ctl.magnet(False)
            self.ctl.goto_steps(winch=0)
            return
        print(f"   {label} complete ({done} transfers).")

    def _transfer_squares(self, ts: dict, td: dict, dip: int, hover_s: float) -> None:
        """One pick-and-place between two mapped squares, mirroring the gameplay
        choreography: magnet ON before the dip and OFF only at the drop; a dip past
        the pick depth on the PICK, none on the drop."""
        self.ctl.goto_steps(winch=0)                           # ensure raised before travel
        self.ctl.goto_steps(base=ts["base"], rail=ts["rail"])  # over the source pawn
        self.ctl.magnet(True)                                  # energize (stays on until the drop)
        if hover_s > 0:
            time.sleep(hover_s)                                # hover a moment over the piece
        self.ctl.goto_steps(winch=ts["winch"])                 # lower to the calibrated pick depth
        if dip:
            self.ctl.goto_steps(winch=ts["winch"] + dip)       # dip past it for sure contact (pick only)
        self.ctl.goto_steps(winch=0)                           # lift to travel
        self.ctl.goto_steps(base=td["base"], rail=td["rail"])  # carry to the destination
        release = int(self.cfg.motion.magnet.release_above_steps)
        self.ctl.goto_steps(winch=td["winch"] - release)       # release ABOVE the mapped depth (don't press in)
        self.ctl.magnet(False)                                 # release — the only magnet-off
        self.ctl.goto_steps(winch=0)                           # lift to travel


def _int(s: str | None) -> int:
    if s is None:
        raise ValueError("need a step count, e.g. 'a 200' or 'r -500'")
    return int(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ground-truth step-map calibration")
    ap.add_argument("--config", help="YAML config (serial port)")
    ap.add_argument("--mock", action="store_true", help="mock backend (no hardware)")
    ap.add_argument("--load", metavar="MAP", help="pre-load anchors from a saved map file")
    ap.add_argument("--refit", metavar="MAP",
                    help="OFFLINE: re-fit a saved map's anchors and rewrite its table (no hardware)")
    ap.add_argument("--rebase", type=int, metavar="DELTA",
                    help="with --refit: add DELTA to every anchor's BASE step, then re-fit. Use to shift a "
                         "map to a new home origin (e.g. switch-home: DELTA = -A_ENDSTOP_STEPS)")
    ap.add_argument("--out", help="output path for --refit (default: overwrite the input)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()

    # Offline re-fit: no serial, no connect() — just load anchors, fit, report, write.
    if args.refit:
        cfg.motion.backend = "mock"           # controller built but never connected
        m = StepMap(cfg)
        m.load(args.refit)
        if not m.anchors:
            return 1
        if args.rebase:
            for v in m.anchors.values():
                v["base"] += args.rebase
            print(f"   rebased every anchor's base by {args.rebase:+d} steps (home-origin shift)")
        m.report()
        m.save(args.out or args.refit)
        return 0

    # Bench tool: drive the raw serial transport (or mock), never the stepmap backend.
    cfg.motion.backend = "mock" if args.mock else "serial"
    m = StepMap(cfg)
    m.connect()
    if args.load:
        m.load(args.load)
    print("Commands: a/r/w <n> | wtop | pos | set <sq> | anchors | del <sq> | load <path> | offset <n> | "
          "fit model|spline | "
          "map | goto <sq> | pawns [n] | backrank [n] | seek | mag on|off | home | save | quit")

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
                elif c == "load" and arg:
                    m.load(arg)
                elif c in ("offset", "rebase"):
                    if arg and arg.lower() in ("reset", "zero"):
                        m._base_offset = 0; print("   base offset reset to 0")
                    else:
                        m.offset(int(arg) if arg else None)
                elif c in ("map", "build"):
                    m.report()
                elif c == "fit":
                    if arg and arg.lower() in ("model", "spline"):
                        m.fit_mode = arg.lower()
                    elif arg:
                        print("   ! fit model | fit spline");
                    stiff = m.fit_mode == "model"
                    print(f"   fit mode = {m.fit_mode}  "
                          + ("(stiff least-squares — resists noise, more anchors help)" if stiff
                             else "(thin-plate spline — exact through each anchor)"))
                elif c == "goto" and arg:
                    m.goto(arg)
                elif c == "pawns":
                    m.pawn_sweep(int(arg) if arg else 3)
                elif c == "backrank":
                    m.backrank_sweep(int(arg) if arg else 3)
                elif c in ("seek", "findsw"):
                    off = m.ctl.seek_base_switch()
                    print(f"   base limit switch at {off} steps from home -> bake "
                          f"A_ENDSTOP_STEPS={off} into the .ino (switch homing live-enabled now).")
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
