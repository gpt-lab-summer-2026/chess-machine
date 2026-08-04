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
import re
import sys

import yaml

from . import factory
from .config import Config, load_config, preset_from_elo
from .pipeline import ChessMachine


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="chessmachine", description=__doc__)
    p.add_argument("--config", help="path to a YAML config file")
    p.add_argument("--dev", action="store_true",
                   help="dev mode: typed I/O + mock motor, but keep the real SLM/engine "
                        "(both fall back gracefully if the server/binary is missing)")
    p.add_argument("--text", action="store_true",
                   help="typed input + printed output (keeps real engine/motor)")
    p.add_argument("--mock", action="store_true", help="use the mock motion backend (no hardware)")
    p.add_argument("--relay", action="store_true",
                   help="use the relay DC-motor prototype backend (firmware/esp32_dc_prototype)")
    p.add_argument("--engine", choices=["stockfish", "random"], help="override engine backend")
    p.add_argument("--slm", choices=["llama_cpp", "rule_based"], help="override SLM backend")
    p.add_argument("--play-as", choices=["white", "black"], help="color the machine plays")
    p.add_argument("--difficulty",
                   help="starting difficulty: a preset (easy/medium/hard) or an "
                        "Elo number, e.g. 1600")
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
    if args.relay:
        cfg.motion.backend = "relay"
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

    # Build the choreographer first so the esp32_whisper STT can share the motion
    # controller's serial link (the ESP32 mic streams over the same port).
    choreo = factory.build_choreographer(cfg.motion)
    machine = ChessMachine(
        config=cfg,
        stt=factory.build_stt(cfg, motion=getattr(choreo, "ctl", None)),
        tts=factory.build_tts(cfg),
        nlu=factory.build_nlu(cfg),
        engine=factory.build_engine(cfg),
        choreographer=choreo,
    )

    machine.start(home=not args.no_home)
    if args.once is not None:
        try:
            machine.handle(args.once)
        finally:
            machine.close()
        return 0
    try:
        machine.run()
    except KeyboardInterrupt:
        # run()'s own try/finally already parked the crane and closed everything;
        # this just swaps the default raw traceback for a clean exit, since Ctrl-C
        # is the documented way to stop the autostart loop (deploy/boot_chessmachine.sh).
        print("\nStopped.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
