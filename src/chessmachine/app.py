"""Command-line entrypoint.

    chessmachine --config config/config.yaml      # full Pi deployment
    chessmachine --dev                             # typed input, mock motor, random engine
    chessmachine --dev --once "knight to f3"        # run one command and exit

`--dev` swaps every subsystem for its no-hardware/no-model variant, so the whole
pipeline runs on any machine for development and testing.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import yaml

from .config import load_config, Config, preset_from_elo
from . import factory
from .pipeline import ChessMachine


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="chessmachine", description=__doc__)
    p.add_argument("--config", help="path to a YAML config file")
    p.add_argument("--dev", action="store_true",
                   help="dev mode: stdin STT, stdout TTS, rule-based NLU, random engine, mock motor")
    p.add_argument("--text", action="store_true", help="typed input + printed output (keeps real engine/motor)")
    p.add_argument("--mock", action="store_true", help="use the mock motion backend (no hardware)")
    p.add_argument("--engine", choices=["stockfish", "random"], help="override engine backend")
    p.add_argument("--slm", choices=["llama_cpp", "rule_based"], help="override SLM backend")
    p.add_argument("--play-as", choices=["white", "black"], help="color the machine plays")
    p.add_argument("--difficulty", help="starting difficulty: a preset (easy/medium/hard) or an Elo number, e.g. 1600")
    p.add_argument("--once", metavar="TEXT", help="handle one utterance then exit")
    p.add_argument("--no-home", action="store_true", help="skip homing on startup")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    return p.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.dev:
        cfg.stt.backend = "stdin"
        cfg.tts.backend = "stdout"
        cfg.slm.backend = "rule_based"
        cfg.engine.backend = "random"
        cfg.motion.backend = "mock"
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
        d = args.difficulty.strip().lower()
        m = re.fullmatch(r"\d{3,4}", d)
        if d in cfg.engine.presets:
            cfg.engine.default_difficulty = d
        elif m:
            # An ad-hoc Elo: synthesize a preset and select it (matches the
            # "set elo to 1600" voice command).
            cfg.engine.presets[d] = preset_from_elo(int(d))
            cfg.engine.default_difficulty = d
        else:
            logging.warning("Unknown difficulty %r; keeping %s",
                            args.difficulty, cfg.engine.default_difficulty)
    return cfg


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config) if args.config else Config()
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
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
