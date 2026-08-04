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
    aggressiveness: int = 3          # webrtcvad 0..3; 3 = most aggressive at REJECTING
                                     # non-speech. MEASURED on this mic over 6 s of an empty
                                     # room: agg 2 called 38/200 frames "speech", agg 3 only
                                     # 9/200. Those false positives are what kept resetting
                                     # the end-of-utterance timer, so 3 is the right default.
    silence_ms: int = 800            # trailing silence that ends an utterance
    end_tolerance: float = 0.15      # fraction of the trailing silence window allowed to be
                                     # (mis)labelled speech and still end the utterance. The old
                                     # rule needed silence_ms of CONSECUTIVE silence, so a single
                                     # noise blip reset it and the mic stayed open to the cap --
                                     # Whisper then got the whole 7 s window every time.
                                     # 0.0 = strict consecutive run (the old behaviour).
    min_speech_ms: int = 250         # ignore blips with less actual speech than this
    frame_ms: int = 30               # webrtcvad frame size (10/20/30)
    pre_roll_ms: int = 240           # audio kept from BEFORE the VAD triggered. webrtcvad
                                     # spends a frame or two deciding, and discarding those
                                     # ate the leading plosive: "pawn to e4" -> "on to e4".
                                     # 0 disables. Doesn't count toward min_speech_ms.
    max_utterance_s: float = 7.0     # HARD cap on the whole listen window (wait-for-speech +
                                     # speech). The mic cuts off here even if you keep talking,
                                     # so each turn is bounded and predictable. VAD still ends it
                                     # earlier on trailing silence; this is just the ceiling.


@dataclass
class AudioConfig:
    input_device: str | None = None      # None = system default
    output_device: str | None = None
    sample_rate: int = 16000                 # rate handed to WHISPER. Fixed by the model
                                             # (it is trained at 16 kHz mono) -- don't change it.
    capture_rate: int = 48000                # rate the MIC is opened at. The AK5370 USB mic runs
                                             # natively at 44.1/48 kHz and REJECTS 16 kHz outright
                                             # (PaErrorCode -9997), so asking for 16 kHz meant
                                             # PipeWire silently resampled for us. Capture at the
                                             # native rate instead and downsample in-process
                                             # (voice/audio.resample, anti-aliased), so the
                                             # conversion is ours to control and measure.
                                             # webrtcvad only accepts 8/16/32/48 kHz; anything else
                                             # falls back to sample_rate. 0 = capture at sample_rate.
    channels: int = 1
    push_to_talk: bool = False               # if True, record while a key is held
    listen_beep: bool = True                 # play a short earcon the instant the mic opens, so
                                             # the user has an unambiguous "speak now" cue (the
                                             # spoken prompt + LED alone were easy to miss)
    vad: VadConfig = field(default_factory=VadConfig)


# Priming Whisper with chess vocabulary makes NATURAL speech ("knight to f3",
# "e4", "castle") transcribe far more reliably at the mic's 8 kHz — the user does
# NOT have to speak phonetically ("bravo four"). Kept short so it biases, not
# dominates. Set stt.prompt: "" to disable.
#
# DO NOT LENGTHEN THIS. Measured on distil-small.en via a TTS->STT round trip
# (Kokoro synthesises the phrase, whisper transcribes it):
#     181 chars (this)  -> "recalibrate" and "knight to f3" both transcribe
#     ~300 chars        -> short phrases start returning EMPTY
#     ~400 chars        -> every phrase returns EMPTY
# A long initial_prompt fills the decoder's context and the model predicts an
# immediate end-of-transcript on a short utterance, so the mic goes deaf. It is
# deliberately limited to NOTATION, which is the part whisper gets wrong; the
# spoken commands ("your move", "take back", "recalibrate") transcribe correctly
# with no prompt at all, so adding them costs accuracy and buys nothing.
_CHESS_STT_PROMPT = (
    "Chess moves and squares: e4, d5, Nf3, Bc4, Qxd5, O-O, castle kingside, "
    "knight to f3, bishop takes e5, pawn e4, rook a1, queen d1, king e2, "
    "files a b c d e f g h, ranks one to eight."
)


@dataclass
class SttConfig:
    backend: str = "distil_whisper"          # distil_whisper | esp32_whisper | network_whisper | stdin
    model: str = "distil-small.en"           # faster-whisper alias -> CTranslate2 build.
                                             # distil-large-v3.5 measures 26-29 s/utterance on
                                             # the Pi 5 CPU vs 6.2 s here -- unusable for a
                                             # conversational loop. Its extra accuracy was on
                                             # notation that move_parsing already repairs.
    device: str = "cpu"                      # cpu | cuda | auto
    compute_type: str = "int8"               # CPU supports int8 / int8_float32 / float32 only
                                             # (int8_float16 is CUDA-only and RAISES on the Pi)
    cpu_threads: int = 3                     # MEASURED on the 4-core Pi 5: 1->9.0 s, 2->6.4 s,
                                             # 3->6.2 s, 4->6.4 s. 3 wins because the 4th thread
                                             # fights the main thread; leaving a core free also
                                             # keeps llama-server/Kokoro from being starved.
                                             # 0 = let CTranslate2 grab every core.
    language: str = "en"
    beam_size: int = 5                       # beam search, NOT greedy (=1). On a short chess
                                             # command the encoder dominates and the decode is
                                             # ~10 tokens, so beam 5 is within noise of beam 1 on
                                             # latency (MEASURED 2026-07-30: 6.0-6.7 s either way)
                                             # while being markedly more robust on quiet/degraded
                                             # audio -- greedy is what turned "d2 to d4" into
                                             # "to do four".
    prompt: str = _CHESS_STT_PROMPT          # bias Whisper toward chess vocab (no NATO needed)
    # These four together bound Whisper's worst case. The problem: on noisy or
    # near-silent input (e.g. the mic dropping out) Whisper falls into a repetition
    # hallucination ("four, four, four, ...") and by default retries that losing
    # decode at each of SIX rising temperatures -- one observed case took 48s to
    # "transcribe" 2.9s of audio, producing garbage the SLM then acted on.
    #
    # temperature=0.0 collapses those six attempts to one greedy pass (the big
    # latency win). But it also removes the fallback ladder, which is what whisper
    # normally uses to RECOVER from a repetitive decode -- so on its own, greedy
    # LOOPS instead. Measured (A/B on synthesized chess speech, 2026-07-30): with
    # temperature=0.0 and no penalty, "take it back take it back" ran away to ~9.7s
    # of "take it back, take it back, x12"; WITH the penalty it stopped at ~5.8s.
    # So repetition_penalty is not optional decoration here -- it is what replaces
    # the recovery that temperature=0.0 removed. It cost nothing on the 5 clean
    # phrases in that A/B (byte-identical output). 1.15 matched 1.3 in testing; 1.3
    # is kept for margin against the worse hallucinations seen on real hardware.
    # max_new_tokens caps a single pass (~48 tokens >> any real chess command, but
    # far below the ~448-token runaway); it must stay well under 448 minus the
    # initial_prompt length (~72 tokens) or faster-whisper raises.
    max_new_tokens: int = 48
    temperature: float = 0.0
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3            # hard ban on repeated 3-grams; belt-and-suspenders
    max_chars: int = 200                     # final text-level backstop, independent of tokens
    # -- audio conditioning before Whisper (DistilWhisperSTT._prep_audio) --------- #
    # The single biggest real-vs-synthetic gap. faster-whisper does NO amplitude
    # normalization, and Whisper hallucinates confident text on quiet input -- so a
    # loud TTS round-trip transcribes fine while a real (quieter) mic mangles the
    # same words. Measured on distil-small.en (2026-07-30) over degraded chess
    # audio: normalize + pad + beam search + vad_filter OFF turned "to do four"
    # back into "d2 to d4" and "to-f3" into "knight to f3", at no extra latency.
    normalize: bool = True                   # level-normalize each utterance toward target_rms
    target_rms: float = 0.12                 # RMS-normalize toward this; gain is capped so the
                                             # peak stays < 1.0 (never clips, and one click can't
                                             # drag quiet speech up on its own)
    pad_ms: int = 200                        # frame the clip with this much silence each side, so
                                             # a short, tightly VAD-gated command gets clean
                                             # onset/offset instead of being clipped
    vad_filter: bool = False                 # Whisper's INTERNAL Silero VAD. Off by default: the
                                             # capture layer already gates with webrtcvad (+ the
                                             # saw_signal/min_speech guards), and double-VADing an
                                             # already-tight ~1 s clip re-trimmed and mangled it
                                             # ("pawn to e4" -> "Pond to E4"). Flip back to true only
                                             # if a false VAD trigger starts transcribing room noise.
    # network_whisper only: an HTTP/RTSP audio stream (e.g. the IP Webcam app on a
    # phone, http://<phone-ip>:8080/audio.opus). Opened per listen window via PyAV.
    stream_url: str = ""
    stream_timeout_s: float = 5.0            # network open/read timeout before giving up a window


@dataclass
class TtsConfig:
    backend: str = "kokoro"                  # kokoro | stdout
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000                 # Kokoro native rate
    model_path: str = "models/kokoro/kokoro-v1.0.onnx"
    voices_path: str = "models/kokoro/voices-v1.0.bin"
    intra_op_threads: int = 2                # onnxruntime would otherwise take all 4 cores
                                             # and busy-wait between utterances, fighting
                                             # whisper (3) and llama-server on a 4-core Pi.
    # Silence prepended to every spoken clip so the Bluetooth speaker's amplifier
    # is awake before the first syllable. The BTL-324 powers its amp down between
    # utterances (a chess turn is longer than any idle timeout), and the first
    # ~350 ms after it wakes is swallowed ("Chess machine ready" -> "ess machine
    # ready"). Disabling PipeWire node-suspend (50-bt-no-suspend.conf) stopped the
    # link renegotiation but NOT this amp wake, so the lead-in has to cover it.
    # Tune by ear with scripts/tts_leadin_test.py: use the smallest clean value.
    lead_in_s: float = 0.5
    lead_in_primer: bool = False             # if the amp only wakes on SIGNAL (not on a
                                             # silent-but-active link), fill the lead-in with
                                             # inaudible low-level noise instead of pure silence
    # The real fix for the clipped first syllable. A silent lead-in CANNOT wake a
    # signal-detect amplifier -- silence is what put it to sleep -- so the pad passes
    # unheard and the amp still wakes on the first syllable. Streaming continuously
    # at an inaudible level means it never sleeps, at zero per-utterance latency.
    # Once this is confirmed working you can drop lead_in_s back to ~0.15.
    keep_alive: bool = True                  # hold the BT speaker's amp awake (needs pw-play)
    keep_alive_level: float = 0.002          # ~-54 dBFS. RAISE if the first syllable is still
                                             # clipped (the amp isn't noticing it); LOWER if you
                                             # can hear hiss between utterances. 0 = silence.


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
    warmup_timeout_s: float = 180.0          # cold model load on a Pi can take a while; warm-up absorbs it


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
    allow_random_fallback: bool = False      # dev: fall back to RandomEngine if no Stockfish
    presets: dict[str, DifficultyPreset] = field(
        default_factory=lambda: {
            "easy": DifficultyPreset(elo=800, depth=6, movetime_ms=600, skill=6),
            "medium": DifficultyPreset(elo=1500, depth=12, movetime_ms=1100, skill=10),
            "hard": DifficultyPreset(elo=2400, depth=18, movetime_ms=1800, skill=20),
        }
    )


# --------------------------------------------------------------------------- #
# Motion / hardware
# --------------------------------------------------------------------------- #
@dataclass
class SerialConfig:
    port: str = "COM3"                # Windows: COMx ; Pi/Linux: /dev/ttyUSB0
    baud: int = 460800                # must match the firmware's Serial.begin — 460800
                                      # carries the 8 kHz MAX4466 mic audio + commands
    timeout_s: float = 5.0
    connect_settle_s: float = 2.0     # ESP32 auto-resets when the port opens
    home_timeout_s: float = 120.0     # homing is slow: base seeks its switch (full sweep worst case)
                                      # + sensorless winch/cart count-home
    # The electromagnet hangs a fixed distance to the SIDE of the arm (the winch's
    # mounting offset), so the transport aims the CART past the target square by
    # asin(offset/reach). Signed: flip the sign if placements land on the wrong
    # side; 0 disables it. See serial_esp32._magnet_to_cart.
    winch_offset_mm: float = 33.0


@dataclass
class RelayConfig:
    """Prototype DC-motor controller (firmware/esp32_dc_prototype).

    Drives ONE brushed DC motor through a 2-channel relay H-bridge. There is no
    positioning: every move just pulses the motor for `pulse_ms`. The pipeline
    picks the direction — the user's moves spin one way, the machine's the other
    — as a hardware bring-up test, not real piece placement.
    """
    port: str = "/dev/ttyUSB0"        # Windows: COMx ; Pi/Linux: /dev/ttyUSB0
    baud: int = 115200
    timeout_s: float = 5.0
    connect_settle_s: float = 2.0     # ESP32 auto-resets when the port opens
    pulse_ms: int = 800               # how long to run the motor per move


@dataclass
class StepMapConfig:
    """Ground-truth per-square step map (scripts/mapboard.py) consumed by the
    `stepmap` motion backend. Board squares drive from measured absolute step
    counts (no geometry model); off-board points (the graveyard) fall back to the
    geometry/MOVE path on the same transport."""
    map_path: str = "config/stepmap.json"   # JSON written by scripts/mapboard.py
    pick_below_mm: float = 30.0              # a set_pulley height below this = lower to the
                                             # square's calibrated pick depth; above = raise to travel
    match_tol_mm: float = 1.0                # how close a move_xz point must be to a mapped square


@dataclass
class GeometryConfig:
    # Coordinates are planar (X,Z) millimetres in a frame whose ORIGIN IS THE
    # CRANE PIVOT. The transport (serial_esp32) converts each (x,z) to polar
    # (r=hypot, theta=atan2) for the rotary base + radial railcart. The reachable
    # workspace is an annular sector: r in [r_min_mm, r_max_mm], swept by rotation.
    r_min_mm: float = 80.0            # inner reachable radius (crane body dead zone)
    r_max_mm: float = 300.0           # arm / railcart maximum reach
    # Board: a square centered on the sector bisector (+X axis), near edge at
    # r_min. 216 mm side -> far SQUARE CENTERS reach ~298 mm (<= r_max); the
    # physical corners overhang the annulus but hold no piece. See docs/DESIGN.md.
    square_pitch_mm: float = 27.0     # 216 mm board / 8 squares
    board_size_mm: float = 216.0      # physical 8x8 side
    # Origin = (X,Z) of the center of square a1. Default places the board square
    # symmetric about the bisector with its near edge at r_min:
    #   a1 = (r_min + pitch/2, -board/2 + pitch/2) = (93.5, -94.5).
    origin_x_mm: float = 93.5
    origin_z_mm: float = -94.5
    # Which physical (planar) axis the files (a..h) and ranks (1..8) run along.
    file_axis: str = "x"              # "x" or "z"
    rank_axis: str = "z"              # the other one
    invert_file: bool = False         # flip a..h direction
    invert_rank: bool = False         # flip 1..8 direction
    board_angle_deg: float = 0.0      # rotate the whole board about a1 in the plane.
                                      # 0 = axis-aligned (edge-first); 135 = the real
                                      # machine's corner-first diagonal (h8 near the
                                      # deadzone, a1 far). The real values live in
                                      # config.example.yaml / config.yaml, not here.
    # Pulley (vertical) travel.
    travel_height_mm: float = 60.0    # cable retracted: clears the tallest piece
    pick_height_mm: float = 4.0       # cable lowered: magnet contacts a piece
    # Off-board storage (graveyard): captured pieces are DUMPED into a single circular
    # dish on the h1/-theta side (the a8/+theta side is blocked by the base limit
    # switch). Pieces pile up randomly, so the dish is a ONE-WAY container — the machine
    # can't fish a specific piece back out, so promotion/undo/reset ask you to place
    # those by hand (see graveyard_retrievable + Choreographer).
    graveyard_center_r_mm: float = 270.0  # radial distance to the dish CENTRE (mm from the pivot)
    graveyard_center_deg: float = -48.0   # angle to the dish centre. MORE NEGATIVE = further off the
                                          # board's g/h edge. THE tuning knob (~43 steps/deg, so ~400 steps
                                          # ~= 9 deg): nudge if drops catch the board or overshoot the dish.
    graveyard_diameter_mm: float = 100.0  # physical dish size (10 cm), bounds the drop scatter
    graveyard_jitter_mm: float = 15.0     # scatter radius for drops, so pieces don't all stack on one point
    graveyard_capacity: int = 30          # max pieces before a capture is refused (a game caps ~30 total)
    graveyard_retrievable: bool = False   # a random pile can't be retrieved from -> promotion/undo/reset
                                          # prompt for a manual placement instead of fishing a piece out


@dataclass
class SpeedsConfig:
    travel_feed: int = 4000           # mm/min, radial axis feed for head moves
    lift_feed: int = 1500             # mm/min, pulley up/down
    settle_ms: int = 250              # pause after a move to damp cable swing


@dataclass
class MagnetConfig:
    settle_ms: int = 300              # pause after toggling the electromagnet
    hover_ms: int = 500               # dwell over a piece (at travel height, magnet already on) before
                                      # dipping to pick it up
    pick_dip_steps: int = 38          # extra winch half-steps to dip PAST the calibrated pick depth on a
                                      # PICK, so a slightly-too-high predetermined height still makes contact
                                      # (stepmap backend only; 0 disables). Drops release at the height, no dip.
    release_above_steps: int = 40     # DROP a piece from this many winch steps ABOVE the mapped touch depth,
                                      # so the winch sets it down instead of pressing it into the board
                                      # (stepmap backend only; 0 = release at the mapped depth)


@dataclass
class MotionConfig:
    backend: str = "relay"            # relay (DC prototype) | serial (crane, geometry) | stepmap
                                      # (crane, ground-truth step map) | mock (dev)
    serial: SerialConfig = field(default_factory=SerialConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    stepmap: StepMapConfig = field(default_factory=StepMapConfig)
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
    concurrent_actuation: bool = False  # OFF = sequential/legible: speak the move fully, THEN move
                                        # the crane (one thing at a time). ON = speak while the crane
                                        # runs, trading legibility for speed.
    match_clock: bool = True          # track the human's thinking time (their clock only)
    rehome_after_move: bool = True    # re-home after EVERY finished move to zero open-loop drift
    rehome_on_capture: bool = True    # re-home after a capture (subsumed by rehome_after_move when on)
    rehome_every_n_moves: int = 0     # re-home every N finished machine moves (0 = off). With the base
                                      # limit switch, drift is bounded, so periodic homing beats per-move.
    self_critique_only_if_punished: bool = True  # voice own blunder only if opponent punishes it
    two_player: bool = False          # human-vs-human: the machine actuates + comments but makes no moves
                                      # (prompts each player in turn instead of playing a side)
    comment_openings: bool = True     # name recognized openings once or twice as the game develops
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
