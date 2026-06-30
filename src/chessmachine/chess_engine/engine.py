"""Chess engine abstraction.

`StockfishEngine` drives a real UCI Stockfish via python-chess. `RandomEngine`
is a dependency-free stand-in for development on machines without Stockfish.
Difficulty maps onto Stockfish's Skill Level / UCI_Elo plus a search limit.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Optional

import chess

from ..config import DifficultyPreset, EngineConfig

log = logging.getLogger(__name__)

# Modern Stockfish refuses UCI_Elo below this; below it we lean on Skill Level.
_ELO_FLOOR = 1320


def _wellknown_stockfish_paths() -> list[str]:
    """Per-platform locations where Stockfish commonly lands, so the same config
    works on every machine without a hardcoded absolute path. Glob patterns are
    expanded by the caller; literal paths are tried as-is."""
    home = Path.home()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        return [
            # winget (how it's installed on the dev laptops)
            str(Path(local) / "Microsoft" / "WinGet" / "Packages"
                / "Stockfish.Stockfish*" / "**" / "stockfish*.exe"),
            # scoop / chocolatey
            str(home / "scoop" / "apps" / "stockfish" / "**" / "stockfish*.exe"),
            str(Path(local) / "Programs" / "stockfish" / "**" / "stockfish*.exe"),
            str(Path(program_files) / "Stockfish" / "**" / "stockfish*.exe"),
        ]
    # Linux (Pi/apt) + macOS (homebrew)
    return [
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        str(home / ".local" / "bin" / "stockfish"),
    ]


def resolve_stockfish_path(configured: str) -> str:
    """Find a usable Stockfish binary, so one config runs on every machine.

    Resolution order:
      1. `configured` if it points at an existing file (an explicit path).
      2. `configured` resolved on PATH (the default "stockfish", or any command).
      3. Well-known per-platform install locations (winget/scoop on Windows,
         apt/brew on Linux/macOS) — lets the config just say "stockfish".

    Returns the first match. If nothing is found, returns `configured` unchanged
    so the caller still raises a clear error naming what was looked for.
    """
    expanded = Path(configured).expanduser()
    if expanded.is_file():
        return str(expanded)

    on_path = shutil.which(configured)
    if on_path:
        return on_path

    for pattern in _wellknown_stockfish_paths():
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob(pattern, recursive=True))
            hit = next((m for m in matches if Path(m).is_file()), None)
        else:
            hit = pattern if Path(pattern).is_file() else None
        if hit:
            log.info("Auto-detected Stockfish at %s (config requested %r)",
                     hit, configured)
            return hit

    return configured


@dataclass
class AnalysisResult:
    """Evaluation normalized to White's point of view."""
    score_cp: Optional[int] = None       # centipawns, +ve = good for White
    mate_in: Optional[int] = None        # +ve = White mates, -ve = Black mates
    best_move: Optional[chess.Move] = None
    pv: list[chess.Move] = field(default_factory=list)
    depth: Optional[int] = None


class ChessEngine(ABC):
    @abstractmethod
    def set_difficulty(self, preset: DifficultyPreset) -> None: ...

    @abstractmethod
    def best_move(self, board: chess.Board) -> Optional[chess.Move]: ...

    @abstractmethod
    def analyse(self, board: chess.Board) -> AnalysisResult: ...

    @property
    def provides_evaluation(self) -> bool:
        """True if `analyse`/`top_moves` give real evaluations (drives move
        commentary). False for stand-ins like RandomEngine."""
        return False

    def top_moves(self, board: chess.Board, n: int = 2) -> list[AnalysisResult]:
        """Best `n` moves with evaluations (multi-PV). Default: just the best."""
        res = self.analyse(board)
        return [res] if res.best_move else []

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
class StockfishEngine(ChessEngine):
    def __init__(self, cfg: EngineConfig):
        import chess.engine  # local import: only needed for the real engine

        self.cfg = cfg
        self.stockfish_path = resolve_stockfish_path(cfg.stockfish_path)
        self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self._configure_base()
        self._limit = chess.engine.Limit(time=0.8)
        self.set_difficulty(cfg.presets[cfg.default_difficulty])

    def _configure_base(self) -> None:
        opts = {}
        if "Threads" in self._engine.options:
            opts["Threads"] = self.cfg.threads
        if "Hash" in self._engine.options:
            opts["Hash"] = self.cfg.hash_mb
        if opts:
            self._engine.configure(opts)

    def set_difficulty(self, preset: DifficultyPreset) -> None:
        import chess.engine

        opts: dict[str, object] = {}
        if "Skill Level" in self._engine.options:
            opts["Skill Level"] = max(0, min(20, preset.skill))
        if "UCI_LimitStrength" in self._engine.options:
            if preset.elo >= _ELO_FLOOR:
                opts["UCI_LimitStrength"] = True
                opts["UCI_Elo"] = min(preset.elo, 3190)
            else:
                # Too weak for the Elo limiter — rely on Skill Level instead.
                opts["UCI_LimitStrength"] = False
        if opts:
            self._engine.configure(opts)
        self._limit = chess.engine.Limit(
            time=preset.movetime_ms / 1000.0, depth=preset.depth
        )
        log.info("Difficulty set: elo=%s skill=%s depth=%s movetime=%sms",
                 preset.elo, preset.skill, preset.depth, preset.movetime_ms)

    @property
    def provides_evaluation(self) -> bool:
        return True

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if board.is_game_over():
            return None
        result = self._engine.play(board, self._limit)
        return result.move

    def top_moves(self, board: chess.Board, n: int = 2) -> list[AnalysisResult]:
        import chess.engine

        if board.is_game_over():
            return []
        # Analyse at honest strength (independent of play difficulty) so move
        # quality is judged fairly. multipv lets us see how forced a move was.
        infos = self._engine.analyse(
            board, chess.engine.Limit(depth=16, time=1.0), multipv=max(1, n)
        )
        if isinstance(infos, dict):   # python-chess returns a bare dict for multipv=1
            infos = [infos]
        return [_analysis_from_info(i) for i in infos]

    def analyse(self, board: chess.Board) -> AnalysisResult:
        import chess.engine

        if board.is_game_over():
            return AnalysisResult()
        # Analyse at honest strength regardless of play difficulty.
        info = self._engine.analyse(
            board, chess.engine.Limit(depth=16, time=1.5)
        )
        return _analysis_from_info(info)

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


def _analysis_from_info(info) -> AnalysisResult:
    score = info.get("score")
    res = AnalysisResult(depth=info.get("depth"))
    if score is not None:
        white = score.white()
        if white.is_mate():
            res.mate_in = white.mate()
            res.score_cp = None
        else:
            res.score_cp = white.score()
    pv = info.get("pv") or []
    res.pv = list(pv)
    if pv:
        res.best_move = pv[0]
    return res


# --------------------------------------------------------------------------- #
class RandomEngine(ChessEngine):
    """Plays a random legal move. For dev/CI only — no chess strength."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def set_difficulty(self, preset: DifficultyPreset) -> None:
        log.info("RandomEngine ignores difficulty (%s elo)", preset.elo)

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return self._rng.choice(moves) if moves else None

    def analyse(self, board: chess.Board) -> AnalysisResult:
        mv = self.best_move(board)
        return AnalysisResult(score_cp=0, best_move=mv, pv=[mv] if mv else [])


# --------------------------------------------------------------------------- #
def create_engine(cfg: EngineConfig) -> ChessEngine:
    """Build the engine selected in config, with a clear error if unavailable."""
    if cfg.backend == "random":
        return RandomEngine()
    if cfg.backend == "stockfish":
        try:
            return StockfishEngine(cfg)
        except FileNotFoundError as exc:
            if cfg.allow_random_fallback:
                log.warning("Stockfish not found at '%s'; falling back to RandomEngine "
                            "(dev mode — no real evaluation or move commentary).",
                            cfg.stockfish_path)
                return RandomEngine()
            raise RuntimeError(
                f"Stockfish not found (config requested '{cfg.stockfish_path}', "
                "and auto-detection on PATH and the usual winget/apt/brew locations "
                "came up empty). Install it (e.g. `winget install Stockfish.Stockfish` "
                "or `apt install stockfish`), set engine.stockfish_path, or use "
                "engine.backend=random."
            ) from exc
    raise ValueError(f"Unknown engine backend: {cfg.backend!r}")
