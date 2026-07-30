"""The voice backends mirror the conversation to the terminal.

Both objects are built with `__new__` to skip their heavy model-loading
`__init__`, so these run without faster-whisper / Kokoro installed.
"""
import io
import sys
import types

from chessmachine.voice.console import safe_print
from chessmachine.voice.stt import DistilWhisperSTT
from chessmachine.voice.tts import KokoroTTS


def _stt_cfg(**overrides):
    # normalize/pad default OFF here so the behavioural tests below exercise the
    # transcribe/guard logic in isolation; _prep_audio has its own tests.
    base = dict(language="en", beam_size=1, prompt="", max_new_tokens=48,
               temperature=0.0, repetition_penalty=1.3, no_repeat_ngram_size=3,
               max_chars=200, normalize=False, target_rms=0.12, pad_ms=0,
               vad_filter=False)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _stt(transcribe_result, **cfg_overrides):
    stt = DistilWhisperSTT.__new__(DistilWhisperSTT)
    stt.cfg = _stt_cfg(**cfg_overrides)
    stt.capture = types.SimpleNamespace(record_utterance=lambda: [0.1, 0.2])
    stt.model = types.SimpleNamespace(transcribe=lambda samples, **kw: transcribe_result)
    return stt


def test_whisper_listen_echoes_transcript(capsys):
    seg = types.SimpleNamespace(text=" e4 ")
    stt = _stt(([seg], None))
    assert stt.listen() == "e4"
    assert "you> e4" in capsys.readouterr().out


def test_whisper_listen_silent_prints_nothing(capsys):
    stt = _stt(([], None))
    stt.capture = types.SimpleNamespace(record_utterance=lambda: [])
    assert stt.listen() == ""
    assert "you>" not in capsys.readouterr().out       # nothing heard -> no echo


def test_transcribe_passes_hallucination_guards_to_faster_whisper():
    """A near-silent/noisy mic can send Whisper into a repetition hallucination
    that keeps decoding for tens of seconds; these params bound that: one greedy
    pass (temperature=0.0, no 6-step fallback ladder) capped at max_new_tokens."""
    captured = {}

    def fake_transcribe(samples, **kw):
        captured.update(kw)
        return ([types.SimpleNamespace(text="e4")], None)

    stt = _stt((None, None))          # placeholder; model swapped below
    stt.model = types.SimpleNamespace(transcribe=fake_transcribe)
    stt.transcribe([0.1, 0.2])
    assert captured["max_new_tokens"] == 48
    assert captured["temperature"] == 0.0
    assert captured["repetition_penalty"] == 1.3
    assert captured["no_repeat_ngram_size"] == 3
    # The capture layer already VAD-gates, so Whisper's internal Silero VAD is off
    # by default (double-VADing a short clip re-trimmed and mangled it).
    assert captured["vad_filter"] is False
    assert captured["beam_size"] == 1          # from this test's cfg (prod default is 5)


def test_prep_audio_normalizes_quiet_input_and_pads():
    """A real mic is far quieter than the synthetic TTS the offline tests use, and
    Whisper hallucinates on quiet input. _prep_audio scales the utterance up toward
    target_rms and frames it with silence so a short command keeps clean edges."""
    import numpy as np

    stt = _stt((None, None), normalize=True, target_rms=0.2, pad_ms=100)
    quiet = np.full(1600, 0.01, dtype="float32")     # rms 0.01 -> should scale ~20x
    out = stt._prep_audio(quiet)
    assert out.shape[0] == 1600 + 2 * int(16000 * 100 / 1000)   # padded both sides
    core = out[1600:1600 + 1600]
    assert float(np.sqrt(np.mean(core * core))) > 0.1            # lifted toward target
    assert float(np.max(np.abs(out))) <= 0.98                    # and never clips


def test_prep_audio_gain_is_capped_so_a_transient_never_clips():
    """RMS-normalizing quiet speech that contains one loud click must not push the
    click past 1.0 -- the gain is capped by the peak, not just the RMS."""
    import numpy as np

    stt = _stt((None, None), normalize=True, target_rms=0.5, pad_ms=0)
    x = np.full(1600, 0.02, dtype="float32")
    x[0] = 0.9                                        # a lone transient near full scale
    out = stt._prep_audio(x)
    assert float(np.max(np.abs(out))) <= 0.98


def test_prep_audio_leaves_digital_silence_alone():
    import numpy as np

    stt = _stt((None, None), normalize=True, target_rms=0.2, pad_ms=0)
    z = np.zeros(800, dtype="float32")
    assert np.array_equal(stt._prep_audio(z), z)     # no divide-by-zero blow-up


def test_transcribe_truncates_runaway_output(caplog):
    """Bounds the returned text itself, independent of the token-count guards
    above -- a hard backstop against a hallucinated wall of text reaching the
    SLM/pipeline."""
    junk = ", ".join(["four"] * 100)      # far longer than max_chars=20 below
    seg = types.SimpleNamespace(text=junk)
    stt = _stt(([seg], None), max_chars=20)
    with caplog.at_level("WARNING"):
        text = stt.transcribe([0.1, 0.2])
    assert len(text) == 20
    assert text == junk[:20]
    assert "truncating" in caplog.text.lower()


def test_transcribe_leaves_short_output_untouched():
    seg = types.SimpleNamespace(text="knight to f3")
    stt = _stt(([seg], None), max_chars=20)
    assert stt.transcribe([0.1, 0.2]) == "knight to f3"


def _tts():
    tts = KokoroTTS.__new__(KokoroTTS)
    tts.output_device = None
    tts.cfg = types.SimpleNamespace(lead_in_s=0.5, lead_in_primer=False)
    return tts


def test_kokoro_say_echoes_text(capsys, monkeypatch):
    tts = _tts()
    monkeypatch.setattr(tts, "synth", lambda text: ([0.0], 24000))
    monkeypatch.setattr("chessmachine.voice.audio.play", lambda *a, **k: None)
    tts.say("Check.")
    assert "[speaker] Check." in capsys.readouterr().out


def test_kokoro_say_passes_lead_in_from_config(monkeypatch):
    """The amp-wake lead-in is config-driven, so say() must forward it to play()."""
    tts = _tts()
    tts.cfg = types.SimpleNamespace(lead_in_s=0.8, lead_in_primer=True)
    seen = {}
    monkeypatch.setattr(tts, "synth", lambda text: ([0.0], 24000))
    monkeypatch.setattr("chessmachine.voice.audio.play",
                        lambda *a, **k: seen.update(k))
    tts.say("Your move.")
    assert seen["lead_in_s"] == 0.8 and seen["lead_in_primer"] is True


def test_kokoro_say_empty_is_silent(capsys, monkeypatch):
    tts = _tts()
    monkeypatch.setattr("chessmachine.voice.audio.play", lambda *a, **k: None)
    tts.say("")
    assert capsys.readouterr().out == ""                # nothing to say


def test_play_via_pipewire_prepends_lead_in(monkeypatch):
    """The WAV handed to pw-play must carry the amp-wake lead-in ahead of the speech
    (subprocess is stubbed so nothing actually plays)."""
    import wave

    import numpy as np

    from chessmachine.voice import audio

    seen = {}

    def fake_run(cmd, check=False):
        with wave.open(cmd[1], "rb") as w:            # cmd == [pw_play, wav_path]
            seen["frames"], seen["rate"] = w.getnframes(), w.getframerate()

    monkeypatch.setattr("subprocess.run", fake_run)
    sr = 24000
    speech = np.ones(sr // 10, dtype="float32")       # 0.1 s of "speech"
    audio._play_via_pipewire("pw-play", speech, sr, lead_in_s=0.5)
    assert seen["rate"] == sr
    assert seen["frames"] == sr // 10 + int(sr * 0.5)  # speech + 0.5 s lead-in


def test_safe_print_survives_unencodable_chars(monkeypatch):
    # A cp1252/ascii console (or redirected stdout) must not crash on a non-ASCII
    # transcript — the offending characters are replaced, not raised on.
    class AsciiStream(io.StringIO):
        encoding = "ascii"

    stream = AsciiStream()
    monkeypatch.setattr(sys, "stdout", stream)
    safe_print("mate soon — a sharp line")          # em dash isn't ASCII
    assert "mate soon" in stream.getvalue()
    assert "—" not in stream.getvalue()
