"""Build subsystems from a Config. Keeps wiring in one place."""
from __future__ import annotations

from .chess_engine import ChessEngine, create_engine
from .config import Config, MotionConfig
from .motion import BoardGeometry, Choreographer, Graveyard, MockMotion, MotionController, RelayChoreographer, RelayMotion
from .nlu import NLU, create_nlu


def create_motion_controller(cfg: MotionConfig) -> MotionController:
    if cfg.backend == "mock":
        return MockMotion()
    if cfg.backend == "serial":
        from .motion.serial_esp32 import SerialMotion
        return SerialMotion(cfg.serial)
    if cfg.backend == "stepmap":
        from .motion.stepmap import StepMapMotion
        return StepMapMotion(cfg)
    if cfg.backend == "relay":
        return RelayMotion(cfg.relay)
    raise ValueError(f"Unknown motion backend: {cfg.backend!r}")


def build_choreographer(cfg: MotionConfig) -> "Choreographer | RelayChoreographer":
    controller = create_motion_controller(cfg)
    if cfg.backend == "relay":
        # DC-motor prototype: no geometry/storage, just a directional pulse per move.
        return RelayChoreographer(controller, cfg.relay.pulse_ms)
    geometry = BoardGeometry(cfg.geometry)
    graveyard = Graveyard(geometry.graveyard_slots(),
                          retrievable=cfg.geometry.graveyard_retrievable)
    return Choreographer(controller, geometry, graveyard, cfg.speeds, cfg.magnet)


def build_engine(cfg: Config) -> ChessEngine:
    return create_engine(cfg.engine)


def build_nlu(cfg: Config) -> NLU:
    return create_nlu(cfg.slm)


def build_stt(cfg: Config, motion=None):
    from .voice import create_stt
    return create_stt(cfg.stt, cfg.audio, motion=motion)


def build_tts(cfg: Config):
    from .voice import create_tts
    return create_tts(cfg.tts, output_device=cfg.audio.output_device)
