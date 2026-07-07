"""Build subsystems from a Config. Keeps wiring in one place."""
from __future__ import annotations

from .config import Config, MotionConfig
from .chess_engine import create_engine, ChessEngine
from .motion import (
    MotionController, MockMotion, BoardGeometry, Graveyard, Choreographer,
    RelayMotion, RelayChoreographer,
)
from .nlu import create_nlu, NLU


def create_motion_controller(cfg: MotionConfig) -> MotionController:
    if cfg.backend == "mock":
        return MockMotion()
    if cfg.backend == "serial":
        from .motion.serial_esp32 import SerialMotion
        return SerialMotion(cfg.serial)
    if cfg.backend == "relay":
        return RelayMotion(cfg.relay)
    raise ValueError(f"Unknown motion backend: {cfg.backend!r}")


def build_choreographer(cfg: MotionConfig) -> "Choreographer | RelayChoreographer":
    controller = create_motion_controller(cfg)
    if cfg.backend == "relay":
        # DC-motor prototype: no geometry/storage, just a directional pulse per move.
        return RelayChoreographer(controller, cfg.relay.pulse_ms)
    geometry = BoardGeometry(cfg.geometry)
    graveyard = Graveyard(geometry.graveyard_slots())
    return Choreographer(controller, geometry, graveyard, cfg.speeds, cfg.magnet)


def build_engine(cfg: Config) -> ChessEngine:
    return create_engine(cfg.engine)


def build_nlu(cfg: Config) -> NLU:
    return create_nlu(cfg.slm)


def build_stt(cfg: Config):
    from .voice import create_stt
    return create_stt(cfg.stt, cfg.audio)


def build_tts(cfg: Config):
    from .voice import create_tts
    return create_tts(cfg.tts, output_device=cfg.audio.output_device)
