"""The shared VAD end-of-utterance state machine (`voice.audio.collect_utterance`).

Every mic backend — local sounddevice (USB/built-in), ESP32 serial, network
stream — funnels through this one function, so these pure tests cover the
end-of-utterance rules for all of them. A scripted fake VAD stands in for
webrtcvad, so no microphone, model, or hardware is needed.
"""
import numpy as np

from chessmachine.config import VadConfig
from chessmachine.voice.audio import _lead_in, collect_utterance

SR = 16000


def test_lead_in_is_silence_by_default():
    z = _lead_in(240, channels=1, primer=False)
    assert z.shape == (240,) and z.dtype == np.dtype("<i2") and not z.any()


def test_lead_in_primer_is_nonzero_but_inaudible():
    """A signal-detect amp needs actual signal to wake; the primer supplies it, but
    quiet enough (~-62 dBFS) to be inaudible under speech."""
    p = _lead_in(240, channels=1, primer=True)
    assert p.shape == (240,) and p.any()               # signal present
    assert int(np.abs(p).max()) <= 24                  # but bounded / inaudible


def test_lead_in_zero_length_is_empty():
    assert _lead_in(0, channels=1, primer=True).shape == (0,)


def _cfg(**kw) -> VadConfig:
    # frame_ms=30 -> 480 samples/frame; silence_ms=90 -> 3 silent frames ends it;
    # min_speech_ms=60 -> at least 2 speech frames or it's rejected as a blip.
    base = {"frame_ms": 30, "silence_ms": 90, "min_speech_ms": 60, "aggressiveness": 2}
    base.update(kw)
    return VadConfig(**base)


def _frames(n: int, value: int = 1000):
    """n identical int16 frames of 480 samples."""
    return [np.full(480, value, dtype="<i2") for _ in range(n)]


class ScriptedVad:
    """Returns speech/silence from a fixed script, then silence forever."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def is_speech(self, frame_bytes, sr):
        assert sr == SR
        assert isinstance(frame_bytes, (bytes, bytearray))
        i, self.calls = self.calls, self.calls + 1
        return self.script[i] if i < len(self.script) else False


def test_captures_speech_then_ends_on_trailing_silence():
    # 4 speech frames, then 3 silent frames end the utterance.
    vad = ScriptedVad([True] * 4 + [False] * 3)
    out = collect_utterance(_frames(10), vad, _cfg(), SR)
    assert out.dtype == np.float32
    assert out.shape[0] == 7 * 480          # 4 speech + 3 trailing silence, then stop
    assert np.all(np.abs(out) <= 1.0)       # normalized into [-1, 1]


def test_stops_early_and_does_not_consume_the_whole_stream():
    vad = ScriptedVad([True] * 2 + [False] * 3)
    collect_utterance(_frames(50), vad, _cfg(), SR)
    assert vad.calls == 5                   # ended at the silence run, not 50 frames in


def test_pre_roll_keeps_the_speech_onset():
    """webrtcvad spends a frame or two latching on, and dropping those frames ate
    the leading plosive ("pawn to e4" -> "on to e4"). Up to pre_roll_ms of
    pre-trigger audio is prepended instead."""
    vad = ScriptedVad([False] * 5 + [True] * 3 + [False] * 3)
    out = collect_utterance(_frames(20), vad, _cfg(), SR)
    # 5 available pre-roll (< the 8-frame cap at 240 ms) + 3 speech + 3 trailing
    assert out.shape[0] == 11 * 480


def test_pre_roll_is_bounded_by_pre_roll_ms():
    # 20 silent frames precede speech, but only 240/30 == 8 may be kept.
    vad = ScriptedVad([False] * 20 + [True] * 3 + [False] * 3)
    out = collect_utterance(_frames(40), vad, _cfg(), SR)
    assert out.shape[0] == (8 + 3 + 3) * 480


def test_pre_roll_disabled_skips_leading_silence():
    vad = ScriptedVad([False] * 5 + [True] * 3 + [False] * 3)
    out = collect_utterance(_frames(20), vad, _cfg(pre_roll_ms=0), SR)
    assert out.shape[0] == 6 * 480          # 3 speech + 3 trailing silence only


def test_pre_roll_does_not_satisfy_min_speech_ms():
    """Pre-roll is context, not speech: a 1-frame blip preceded by plenty of
    pre-roll must still be rejected by min_speech_ms."""
    vad = ScriptedVad([False] * 5 + [True] + [False] * 3)
    out = collect_utterance(_frames(20), vad, _cfg(), SR)
    assert out.shape[0] == 0


def test_digital_silence_is_reported_as_a_dead_device(caplog):
    """All-zero samples mean the capture device is not delivering (e.g. the USB mic
    dropped off and `default` resolved to a source-less PipeWire graph). Whisper
    hallucinates confident text from silence, so this must be refused loudly."""
    frames = [np.zeros(480, dtype="<i2") for _ in range(10)]
    vad = ScriptedVad([True] * 4 + [False] * 3)     # VAD claims speech anyway
    with caplog.at_level("ERROR"):
        out = collect_utterance(frames, vad, _cfg(), SR)
    assert out.shape[0] == 0
    assert "ZERO audio" in caplog.text


def test_quiet_but_nonzero_audio_is_still_accepted():
    """Only exact digital silence is treated as a dead device; a genuinely quiet
    room must still transcribe."""
    vad = ScriptedVad([True] * 4 + [False] * 3)
    out = collect_utterance(_frames(10, value=2), vad, _cfg(), SR)
    assert out.shape[0] > 0


def test_blip_is_rejected_as_too_little_speech():
    # A single 30 ms speech frame is below min_speech_ms=60 -> nothing returned.
    vad = ScriptedVad([True] + [False] * 3)
    out = collect_utterance(_frames(10), vad, _cfg(), SR)
    assert out.shape[0] == 0 and out.dtype == np.float32


def test_pure_silence_returns_empty():
    out = collect_utterance(_frames(10), ScriptedVad([]), _cfg(), SR)
    assert out.shape[0] == 0


def test_empty_frame_stream_returns_empty():
    out = collect_utterance([], ScriptedVad([]), _cfg(), SR)
    assert out.shape[0] == 0 and out.dtype == np.float32


def test_normalization_scales_int16_to_unit_float():
    vad = ScriptedVad([True] * 2 + [False] * 3)
    out = collect_utterance(_frames(5, value=16384), vad, _cfg(), SR)
    assert np.allclose(out[:960], 0.5, atol=1e-4)     # 16384/32768 == 0.5


def test_silence_ms_below_one_frame_still_ends_the_utterance():
    # silence_ms < frame_ms would floor to 0 silent frames; guarded to >= 1 so a
    # misconfigured VAD can't spin forever waiting to end.
    vad = ScriptedVad([True] * 2 + [False])
    out = collect_utterance(_frames(30), vad, _cfg(silence_ms=10), SR)
    assert out.shape[0] == 3 * 480
    assert vad.calls == 3
