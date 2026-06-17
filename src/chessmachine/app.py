"""Command-line entrypoint.

    chessmachine --config config/config.yaml             # full Pi deployment
    chessmachine --dev                                    # typed I/O, mock motor, real brain
    chessmachine --dev --config config/config.yaml        # + Stockfish path / SLM server URL
    chessmachine --dev --once "knight to f3"               # run one command and exit

`--dev` swaps only the hardware-facing parts for stubs — typed input, printed
output, mock motor — but KEEPS the real brain (llama SLM + Stockfish engine), so
dev sessions exercise the actual NLU and move commentary. Both degrade
gracefully: the SLM falls back to rule-based if no llama-server is running, and
the engine falls back to a random mover if Stockfish isn't found — so `--dev`
still runs on a bare machine. Pass `--config` so Stockfish/SLM are found, or
`--engine random --slm rule_based` to force the old fully-offline behaviour.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config, Config
from . import factory
from .pipeline import ChessMachine


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="chessmachine", description=__doc__)
    p.add_argument("--config", help="path to a YAML config file")
    p.add_argument("--dev", action="store_true",
                   help="dev mode: typed I/O + mock motor, but keep the real SLM/engine "
                        "(both fall back gracefully if the server/binary is missing)")
    p.add_argument("--text", action="store_true", help="typed input + printed output (keeps real engine/motor)")
    p.add_argument("--mock", action="store_true", help="use the mock motion backend (no hardware)")
    p.add_argument("--engine", choices=["stockfish", "random"], help="override engine backend")
    p.add_argument("--slm", choices=["llama_cpp", "rule_based"], help="override SLM backend")
    p.add_argument("--play-as", choices=["white", "black"], help="color the machine plays")
    p.add_argument("--difficulty", help="starting difficulty preset (easy/medium/hard)")
    p.add_argument("--once", metavar="TEXT", help="handle one utterance then exit")
    p.add_argument("--no-home", action="store_true", help="skip homing on startup")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    return p.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.dev:
        # Stub only the hardware-facing parts; keep the real brain (it degrades
        # gracefully when the SLM server / Stockfish binary aren't available).
        cfg.stt.backend = "stdin"
        cfg.tts.backend = "stdout"
        cfg.motion.backend = "mock"
        cfg.engine.allow_random_fallback = True
    if args.text:
        cfg.stt.backend = "stdin"
        cfg.tts.backend = "stdout"
    if args.mock:
        cfg.motion.backend = "mock"
    if args.engine:
        cfg.engine.backend = args.engine
    if args.slm:
        cfg.slm.backend = args.slm
    if args.play_as:
        cfg.app.play_as = args.play_as
    if args.difficulty:
        if args.difficulty in cfg.engine.presets:
            cfg.engine.default_difficulty = args.difficulty
        else:
            logging.warning("Unknown difficulty preset %r; keeping %s",
                            args.difficulty, cfg.engine.default_difficulty)
    return cfg


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config) if args.config else Config()
    cfg = _apply_overrides(cfg, args)

    logging.basicConfig(
        level=(args.log_level or cfg.app.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    machine = ChessMachine(
        config=cfg,
        stt=factory.build_stt(cfg),
        tts=factory.build_tts(cfg),
        nlu=factory.build_nlu(cfg),
        engine=factory.build_engine(cfg),
        choreographer=factory.build_choreographer(cfg.motion),
    )

    machine.start(home=not args.no_home)
    if args.once is not None:
        try:
            machine.handle(args.once)
        finally:
            machine.close()
        return 0
    machine.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
