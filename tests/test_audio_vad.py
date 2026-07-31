"""The shared VAD end-of-utterance state machine (`voice.audio.collect_utterance`).

Every mic backend — local sounddevice (USB/built-in), ESP32 serial, network
stream — funnels through this one function, so these pure tests cover the
end-of-utterance rules for all of them. A scripted fake VAD stands in for
webrtcvad, so no microphone, model, or hardware is needed.
"""
import numpy as np

from chessmachine.config import AudioConfig, VadConfig
from chessmachine.voice.audio import (
    AudioCapture,
    _lead_in,
    collect_utterance,
    make_beep,
    resample,
)

SR = 16000


# -- 48 kHz capture -> 16 kHz for Whisper ------------------------------------ #
def _tone(freq, seconds, rate):
    t = np.arange(int(rate * seconds), dtype="float32") / rate
    return np.sin(2 * np.pi * freq * t).astype("float32")


def test_resample_48k_to_16k_length_and_dtype():
    out = resample(_tone(440, 1.0, 48000), 48000, 16000)
    assert out.dtype == np.float32
    assert out.shape[0] == 16000                 # exactly 1 s at the new rate


def test_resample_preserves_a_speech_band_tone():
    """A 440 Hz tone must survive 48k -> 16k intact (it's well inside the band)."""
    out = resample(_tone(440, 0.5, 48000), 48000, 16000)
    mid = out[800:-800]                          # skip FIR edge transients
    assert 0.6 < float(np.max(np.abs(mid))) < 1.05
    # dominant frequency is still ~440 Hz
    spec = np.abs(np.fft.rfft(mid))
    peak_hz = float(np.fft.rfftfreq(mid.size, 1 / 16000)[int(np.argmax(spec))])
    assert abs(peak_hz - 440) < 25


def test_resample_rejects_out_of_band_content_instead_of_aliasing_it():
    """THE reason this isn't plain linear interpolation: a 15 kHz tone is above the
    16 kHz Nyquist and must be filtered out, not folded back into the speech band
    as a phantom ~1 kHz tone that Whisper would try to transcribe."""
    aliased = resample(_tone(15000, 0.5, 48000), 48000, 16000)
    clean = resample(_tone(440, 0.5, 48000), 48000, 16000)
    mid_a, mid_c = aliased[800:-800], clean[800:-800]
    assert float(np.sqrt(np.mean(mid_a**2))) < 0.05 * float(np.sqrt(np.mean(mid_c**2)))


def test_resample_is_a_no_op_at_the_same_rate_or_when_empty():
    x = _tone(440, 0.1, 16000)
    assert resample(x, 16000, 16000) is x or np.array_equal(resample(x, 16000, 16000), x)
    assert resample(np.zeros(0, dtype="float32"), 48000, 16000).shape[0] == 0


def test_capture_rate_falls_back_when_webrtcvad_cannot_handle_it():
    """webrtcvad only accepts 8/16/32/48 kHz, so a 44.1 kHz capture rate would
    raise on every frame -- fall back to the Whisper rate rather than going deaf."""
    cap = AudioCapture(AudioConfig(capture_rate=44100, sample_rate=16000))
    assert cap._capture_rate() == 16000
    assert AudioCapture(AudioConfig(capture_rate=48000))._capture_rate() == 48000
    # 0 means "just use the whisper rate"
    assert AudioCapture(AudioConfig(capture_rate=0, sample_rate=16000))._capture_rate() == 16000


def test_keep_alive_buffer_is_inaudibly_quiet_but_not_silent():
    """The whole point: a signal-detect amp needs actual SIGNAL to stay awake
    (silence is what puts it to sleep), yet it must not be audible as hiss."""
    from chessmachine.voice.audio import SpeakerKeepAlive

    buf = np.frombuffer(SpeakerKeepAlive(sample_rate=8000, level=0.002)._buffer(),
                        dtype="<i2")
    assert buf.size == 8000                       # ~1 s at that rate
    assert buf.any()                              # NOT digital silence
    assert int(np.abs(buf).max()) <= int(0.002 * 32767) + 1   # ~-54 dBFS, inaudible


def test_keep_alive_is_a_noop_without_pw_play(monkeypatch):
    """Dev boxes have no pw-play; start() must decline rather than raise."""
    from chessmachine.voice import audio as audio_mod

    monkeypatch.setattr(audio_mod.shutil, "which", lambda name: None)
    ka = audio_mod.SpeakerKeepAlive()
    assert ka.start() is False
    ka.stop()                                      # must be safe even if never started


def test_make_beep_is_bounded_faded_and_the_right_length():
    b = make_beep(sample_rate=24000, ms=160, volume=0.22)
    assert b.dtype == np.float32
    assert b.shape[0] == int(24000 * 160 / 1000)
    assert float(np.max(np.abs(b))) <= 0.22 + 1e-6       # never louder than asked
    assert abs(float(b[0])) < 1e-3 and abs(float(b[-1])) < 1e-3   # faded in and out


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


def test_scattered_false_positives_still_end_the_utterance():
    """THE 7.4-second bug. webrtcvad mislabels some room noise as speech (measured:
    38/200 frames of an empty room at aggressiveness 2). The old rule needed
    silence_ms of CONSECUTIVE silence and reset on any single speech frame, so one
    blip per second held the mic open until the max_utterance_s cap and Whisper got
    the whole window. A mostly-quiet trailing window must end it."""
    # 3 speech frames, then a tail with a blip every 3rd frame: silence_ms==3 frames
    # is NEVER satisfied consecutively, yet the window is plainly not speech.
    tail = [False, False, True] * 19            # 57 frames -> scripts all 60
    vad = ScriptedVad([True] * 3 + tail)
    out = collect_utterance(_frames(60), vad, _cfg(end_tolerance=0.34), SR)
    assert 0 < out.shape[0] < 10 * 480          # ended promptly, not at the cap


def test_strict_tolerance_restores_the_old_consecutive_rule():
    """end_tolerance=0.0 is the documented escape hatch back to the old behaviour --
    and demonstrates the bug: with blips it never ends, consuming the whole window."""
    tail = [False, False, True] * 19
    vad = ScriptedVad([True] * 3 + tail)
    out = collect_utterance(_frames(60), vad, _cfg(end_tolerance=0.0), SR)
    assert out.shape[0] == 60 * 480             # never ends: consumes every frame


def test_tolerance_does_not_end_the_utterance_while_speech_continues():
    """The tolerant window must not cut someone off mid-sentence: continuous speech
    fills the window with True and can never look like silence."""
    vad = ScriptedVad([True] * 40 + [False] * 3)
    out = collect_utterance(_frames(60), vad, _cfg(end_tolerance=0.34), SR)
    # all 40 speech frames kept, then it stops within a frame or two of the silence
    assert 40 * 480 <= out.shape[0] <= 43 * 480


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
