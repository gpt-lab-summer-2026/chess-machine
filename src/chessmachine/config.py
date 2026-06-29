"""Configuration model + YAML loader.

Plain dataclasses (no pydantic) keep dependencies light enough to run on a Pi.
`load_config()` deep-merges a YAML file over the dataclass defaults, so a config
file only needs to specify the values it wants to override.

NOTE: deliberately NOT using `from __future__ import annotations` — the YAML
merger inspects `field.type` via `is_dataclass`, which needs real type objects,
not stringized annotations.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


# --------------------------------------------------------------------------- #
# Audio / speech I/O
# --------------------------------------------------------------------------- #
@dataclass
class VadConfig:
    enabled: bool = True
    aggressiveness: int = 2          # webrtcvad 0..3 (3 = most aggressive)
    silence_ms: int = 800            # trailing silence that ends an utterance
    frame_ms: int = 30               # webrtcvad frame size (10/20/30)
    max_utterance_s: float = 15.0


@dataclass
class AudioConfig:
    input_device: str | None = None      # None = system default
    output_device: str | None = None
    sample_rate: int = 16000                 # whisper expects 16 kHz mono
    channels: int = 1
    push_to_talk: bool = False               # if True, record while a key is held
    vad: VadConfig = field(default_factory=VadConfig)


@dataclass
class SttConfig:
    backend: str = "distil_whisper"          # distil_whisper | stdin
    model: str = "distil-small.en"          # faster-whisper alias -> CTranslate2 build
    device: str = "cpu"                      # cpu | cuda | auto
    compute_type: str = "int8"               # int8 is fast on the Pi 5 CPU
    language: str = "en"
    beam_size: int = 1


@dataclass
class TtsConfig:
    backend: str = "kokoro"                  # kokoro | stdout
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000                 # Kokoro native rate
    model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "models/kokoro/voices-v1.0.bin"


# --------------------------------------------------------------------------- #
# Small language model (intent + phrasing)
# --------------------------------------------------------------------------- #
@dataclass
class SlmConfig:
    backend: str = "llama_cpp"               # llama_cpp | rule_based (dev/fallback)
    mode: str = "server"                     # server (llama-server HTTP) | inproc
    server_url: str = "http://127.0.0.1:8080"
    model_path: str = "models/slm/model.gguf"   # used in inproc mode
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.2
    max_tokens: int = 256
    request_timeout_s: float = 30.0


# --------------------------------------------------------------------------- #
# Chess engine
# --------------------------------------------------------------------------- #
@dataclass
class DifficultyPreset:
    elo: int = 1500          # UCI_Elo (with UCI_LimitStrength)
    depth: int = 12          # search depth cap
    movetime_ms: int = 800   # wall-clock cap per move
    skill: int = 10          # Stockfish "Skill Level" 0..20


def preset_from_elo(elo: int) -> DifficultyPreset:
    """Synthesize a difficulty preset from a target Elo (used for ad-hoc
    'set elo to 1600' requests, by voice or via --difficulty)."""
    skill = max(0, min(20, round((elo - 800) / 110)))
    return DifficultyPreset(elo=elo, depth=16, movetime_ms=1000, skill=skill)


@dataclass
class EngineConfig:
    backend: str = "stockfish"               # stockfish | random (dev/fallback)
    stockfish_path: str = "stockfish"        # binary path or command on PATH
    threads: int = 2
    hash_mb: int = 256
    default_difficulty: str = "medium"
    presets: dict[str, DifficultyPreset] = field(
        default_factory=lambda: {
            "easy": DifficultyPreset(elo=800, depth=6, movetime_ms=300, skill=3),
            "medium": DifficultyPreset(elo=1500, depth=12, movetime_ms=800, skill=10),
            "hard": DifficultyPreset(elo=2400, depth=18, movetime_ms=1500, skill=20),
        }
    )


# --------------------------------------------------------------------------- #
# Motion / hardware
# --------------------------------------------------------------------------- #
@dataclass
class SerialConfig:
    port: str = "COM3"                # Windows: COMx ; Pi/Linux: /dev/ttyUSB0
    baud: int = 115200
    timeout_s: float = 5.0
    connect_settle_s: float = 2.0     # ESP32 auto-resets when the port opens
    home_timeout_s: float = 60.0      # homing can be slow


@dataclass
class GeometryConfig:
    # Origin = (X,Z) of the center of square a1, measured from the homed corner.
    # Defaults assume 150 mm linear-motor stroke: a 120 mm PLAYABLE 8x8 grid
    # (15 mm squares) plus a ~30 mm storage strip, all within reach. See docs/DESIGN.md.
    origin_x_mm: float = 10.0
    origin_z_mm: float = 10.0
    square_pitch_mm: float = 15.0     # 120 mm playable grid / 8 squares
    board_size_mm: float = 120.0      # playable area; physical board incl. storage ~150
    # Which physical axis the files (a..h) and ranks (1..8) run along.
    file_axis: str = "x"              # "x" or "z"
    rank_axis: str = "z"              # the other one
    invert_file: bool = False         # flip a..h direction
    invert_rank: bool = False         # flip 1..8 direction
    # Pulley (vertical) travel.
    travel_height_mm: float = 60.0    # cable retracted: clears the tallest piece
    pick_height_mm: float = 4.0       # cable lowered: magnet contacts a piece
    # Off-board storage for captured pieces (a.k.a. graveyard): a strip beside
    # the grid, still inside the 150 mm reach. 2 columns x 8 rows = 16 slots
    # ("2 per row"). A full board reset needs 32 slots, so with 16 the machine
    # asks for a manual reset (see Choreographer.setup_starting_position).
    graveyard_x_mm: float = 127.5
    graveyard_z_start_mm: float = 10.0
    graveyard_z_pitch_mm: float = 15.0
    graveyard_x_pitch_mm: float = 15.0
    graveyard_slots_per_column: int = 8
    graveyard_columns: int = 2


@dataclass
class SpeedsConfig:
    travel_feed: int = 4000           # mm/min, XZ gantry moves
    lift_feed: int = 1500             # mm/min, pulley up/down
    settle_ms: int = 250              # pause after a move to damp cable swing


@dataclass
class MagnetConfig:
    settle_ms: int = 300              # pause after toggling the electromagnet


@dataclass
class MotionConfig:
    backend: str = "serial"           # serial | mock (dev)
    serial: SerialConfig = field(default_factory=SerialConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    speeds: SpeedsConfig = field(default_factory=SpeedsConfig)
    magnet: MagnetConfig = field(default_factory=MagnetConfig)


# --------------------------------------------------------------------------- #
# Application behaviour
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    play_as: str = "black"            # color the machine plays ("white"/"black")
    auto_reply: bool = True           # auto-make the engine move after opponent's
    confirm_moves: bool = True        # speak each move as it is executed
    require_legal_confirmation: bool = True  # re-ask on illegal/ambiguous moves
    log_level: str = "INFO"


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    slm: SlmConfig = field(default_factory=SlmConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        return load_config(path)


# --------------------------------------------------------------------------- #
# YAML deep-merge into dataclasses
# --------------------------------------------------------------------------- #
def _merge_into(obj, data: dict):
    """Recursively overwrite fields of dataclass `obj` with values from `data`."""
    if data is None:
        return obj
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {type(obj).__name__}, got {type(data).__name__}")
    valid = {f.name: f for f in fields(obj)}
    for key, val in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' in section '{type(obj).__name__}'")
        f = valid[key]
        cur = getattr(obj, f.name)
        if is_dataclass(f.type):
            if not isinstance(val, dict):
                raise TypeError(
                    f"Config section '{key}' (in '{type(obj).__name__}') expects a "
                    f"mapping, got {type(val).__name__}"
                )
            setattr(obj, f.name, _merge_into(cur, val))
        elif f.name == "presets" and isinstance(val, dict):
            merged = dict(cur)
            for name, preset in val.items():
                base = merged.get(name, DifficultyPreset())
                merged[name] = _merge_into(base, preset)
            setattr(obj, f.name, merged)
        else:
            setattr(obj, f.name, val)
    return obj


def _validate(cfg: Config) -> None:
    """Sanity-check a (merged) config so a calibration typo fails fast and clearly.

    Run after the YAML merge rather than in __post_init__: the merger assigns
    fields via setattr after construction, so __post_init__ would only ever see
    the defaults, never the overridden values.
    """
    g = cfg.motion.geometry
    if g.square_pitch_mm <= 0:
        raise ValueError(f"geometry.square_pitch_mm must be > 0 (got {g.square_pitch_mm})")
    if g.pick_height_mm < 0 or g.travel_height_mm < 0:
        raise ValueError("geometry pick_height_mm / travel_height_mm must be >= 0")
    if g.travel_height_mm < g.pick_height_mm:
        raise ValueError(
            f"geometry.travel_height_mm ({g.travel_height_mm}) must be >= "
            f"pick_height_mm ({g.pick_height_mm})"
        )
    s = cfg.motion.speeds
    if s.travel_feed <= 0 or s.lift_feed <= 0:
        raise ValueError("motion.speeds feeds (travel_feed, lift_feed) must be > 0")
    ser = cfg.motion.serial
    if ser.timeout_s <= 0 or ser.home_timeout_s <= 0:
        raise ValueError("motion.serial timeouts must be > 0")


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, merging the YAML at `path` over built-in defaults."""
    cfg = Config()
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = _merge_into(cfg, data)
    _validate(cfg)
    return cfg
