"""Parametric predictive model for board step coordinates.

The step map (`scripts/anchor.py`) fits a thin-plate spline through the hand-measured
anchors. A TPS is an *exact interpolator*: it reproduces every anchor perfectly, so a
mis-measured anchor is not just preserved — it is smeared over that anchor's whole
neighbourhood. This module fits an INDEPENDENT, deliberately stiff model from the
crane's actual kinematics instead:

    R, Theta = the square's polar coordinates from BoardGeometry (radius mm, angle deg)

    base  = b0 + b1*Theta + b2*Theta^2 + b3*R     rotary axis; the R term is the
                                                  winch_offset correction (the magnet
                                                  hangs to the SIDE of the arm, so the
                                                  angle needed depends on reach)
    rail  = r0 + r1*R                             purely radial cart
    winch = w0 + w1*R                             cable sag grows with reach

Because it is low-order and fitted with outlier rejection, no single anchor can bend
it. That is the whole point: where this model and the TPS table disagree sharply, the
table is usually being pulled by a suspect anchor — which is exactly what you want to
go re-measure (or physically double-check with scripts/verifymap.py).

It is a CROSS-CHECK, not a replacement. Leave-one-out accuracy on the 23-anchor map is
~36 steps for base and ~80 for winch, but ~475 for rail (~7 mm) — the rail anchors
themselves scatter that much, so the map stays the source of truth for play.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from .geometry import BoardGeometry

KEYS = ("base", "rail", "winch")

# Design matrix per axis: each entry maps (R, Theta) -> the model's basis row.
_BASIS = {
    "base":  lambda R, T: [1.0, T, T * T, R],
    "rail":  lambda R, T: [1.0, R],
    "winch": lambda R, T: [1.0, R],
}
_TERMS = {
    "base":  ("1", "Theta", "Theta^2", "R"),
    "rail":  ("1", "R"),
    "winch": ("1", "R"),
}


def _solve(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Least-squares solve via normal equations + Gaussian elimination.

    Kept dependency-free (no numpy) so the motion package stays importable on a
    bare Pi. The systems here are tiny (<=4 unknowns) and well-conditioned once
    R/Theta are centred, so normal equations are perfectly adequate.
    """
    n = len(A[0])
    # Normal equations: (A^T A) x = A^T b
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] + [
        sum(A[k][i] * b[k] for k in range(len(A)))] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular system (degenerate anchors)")
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= p
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def _median(v: Sequence[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


@dataclass
class AxisFit:
    """One axis' fitted coefficients plus the anchors it rejected."""
    key: str
    coeffs: list[float]
    rejected: dict[str, float] = field(default_factory=dict)   # square -> residual
    rms: float = 0.0        # in-sample RMS over the KEPT anchors
    n_used: int = 0

    def formula(self) -> str:
        parts = []
        for c, t in zip(self.coeffs, _TERMS[self.key]):
            parts.append(f"{c:+.4f}" + ("" if t == "1" else f"*{t}"))
        return f"{self.key} = " + " ".join(parts)


class BoardModel:
    """Fits the parametric model from a map file's `anchors` section.

    Hand-measured values are only ever READ here; nothing in this class mutates a
    map. Use `predict()` for a single square or `predict_all()` for the full board.
    """

    def __init__(self, geo: BoardGeometry, anchors: dict[str, dict],
                 reject_sigma: float = 3.0):
        self.geo = geo
        self.anchors = anchors
        self.reject_sigma = reject_sigma
        self.fits: dict[str, AxisFit] = {}
        for key in KEYS:
            self.fits[key] = self._fit_axis(key)

    # -- geometry ------------------------------------------------------------ #
    def polar(self, square: str) -> tuple[float, float]:
        """(radius mm, angle deg) for a square name, via the SAME geometry the
        choreographer uses — so the model shares the board's frame of reference."""
        pt = self.geo.name_to_point(square)
        return math.hypot(pt.x, pt.z), math.degrees(math.atan2(pt.z, pt.x))

    # -- fitting ------------------------------------------------------------- #
    def _fit_axis(self, key: str) -> AxisFit:
        basis = _BASIS[key]
        names = [s for s in sorted(self.anchors) if key in self.anchors[s]]
        rows = {s: basis(*self.polar(s)) for s in names}
        vals = {s: float(self.anchors[s][key]) for s in names}

        keep = list(names)
        rejected: dict[str, float] = {}
        coeffs: list[float] = []
        # Iteratively drop the worst anchor while it sits beyond reject_sigma of a
        # robust (MAD-based) spread. Bounded passes; always keep enough to fit.
        for _ in range(len(names)):
            coeffs = _solve([rows[s] for s in keep], [vals[s] for s in keep])
            res = {s: vals[s] - sum(c * x for c, x in zip(coeffs, rows[s])) for s in keep}
            if len(keep) <= len(coeffs) + 2:
                break
            med = _median(list(res.values()))
            mad = _median([abs(r - med) for r in res.values()])
            scale = 1.4826 * mad          # MAD -> sigma for a normal distribution
            if scale <= 1e-9:
                break
            worst = max(keep, key=lambda s: abs(res[s] - med))
            if abs(res[worst] - med) <= self.reject_sigma * scale:
                break
            rejected[worst] = res[worst]
            keep.remove(worst)

        res = {s: vals[s] - sum(c * x for c, x in zip(coeffs, rows[s])) for s in keep}
        rms = math.sqrt(sum(r * r for r in res.values()) / len(res)) if res else 0.0
        return AxisFit(key=key, coeffs=coeffs, rejected=rejected, rms=rms, n_used=len(keep))

    # -- prediction ---------------------------------------------------------- #
    def predict(self, square: str) -> dict[str, int]:
        R, T = self.polar(square)
        out = {}
        for key in KEYS:
            row = _BASIS[key](R, T)
            out[key] = int(round(sum(c * x for c, x in zip(self.fits[key].coeffs, row))))
        return out

    def predict_all(self, squares: Iterable[str] | None = None) -> dict[str, dict[str, int]]:
        if squares is None:
            squares = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
        return {s: self.predict(s) for s in squares}

    # -- diagnostics --------------------------------------------------------- #
    def leave_one_out(self, key: str) -> tuple[float, dict[str, float]]:
        """Refit without each anchor in turn and predict it. Returns (RMS, errors).
        This is the honest accuracy estimate — in-sample RMS flatters the fit."""
        basis = _BASIS[key]
        names = [s for s in sorted(self.anchors) if key in self.anchors[s]]
        errs: dict[str, float] = {}
        for held in names:
            tr = [s for s in names if s != held]
            if len(tr) <= len(basis(0.0, 0.0)):
                continue
            try:
                c = _solve([basis(*self.polar(s)) for s in tr],
                           [float(self.anchors[s][key]) for s in tr])
            except ValueError:
                continue
            row = basis(*self.polar(held))
            errs[held] = sum(cc * x for cc, x in zip(c, row)) - float(self.anchors[held][key])
        rms = math.sqrt(sum(e * e for e in errs.values()) / len(errs)) if errs else float("nan")
        return rms, errs

    def disagreements(self, table: dict[str, dict], key: str) -> list[tuple[str, float]]:
        """(square, table - predicted) for every mapped square, worst first.
        Large values flag table entries worth physically re-checking."""
        out = []
        for s, entry in table.items():
            if key not in entry:
                continue
            try:
                p = self.predict(s)
            except Exception:  # noqa: BLE001 - skip non-board keys
                continue
            out.append((s, float(entry[key]) - p[key]))
        out.sort(key=lambda t: -abs(t[1]))
        return out
