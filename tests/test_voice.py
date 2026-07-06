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


def test_whisper_listen_echoes_transcript(capsys):
    stt = DistilWhisperSTT.__new__(DistilWhisperSTT)
    stt.cfg = types.SimpleNamespace(language="en", beam_size=1)
    stt.capture = types.SimpleNamespace(record_utterance=lambda: [0.1, 0.2])
    seg = types.SimpleNamespace(text=" e4 ")
    stt.model = types.SimpleNamespace(transcribe=lambda samples, **kw: ([seg], None))
    assert stt.listen() == "e4"
    assert "you> e4" in capsys.readouterr().out


def test_whisper_listen_silent_prints_nothing(capsys):
    stt = DistilWhisperSTT.__new__(DistilWhisperSTT)
    stt.cfg = types.SimpleNamespace(language="en", beam_size=1)
    stt.capture = types.SimpleNamespace(record_utterance=lambda: [])
    stt.model = types.SimpleNamespace(transcribe=lambda samples, **kw: ([], None))
    assert stt.listen() == ""
    assert "you>" not in capsys.readouterr().out       # nothing heard -> no echo


def test_kokoro_say_echoes_text(capsys, monkeypatch):
    tts = KokoroTTS.__new__(KokoroTTS)
    tts.output_device = None
    monkeypatch.setattr(tts, "synth", lambda text: ([0.0], 24000))
    monkeypatch.setattr("chessmachine.voice.audio.play", lambda *a, **k: None)
    tts.say("Check.")
    assert "[speaker] Check." in capsys.readouterr().out


def test_kokoro_say_empty_is_silent(capsys, monkeypatch):
    tts = KokoroTTS.__new__(KokoroTTS)
    tts.output_device = None
    monkeypatch.setattr("chessmachine.voice.audio.play", lambda *a, **k: None)
    tts.say("")
    assert capsys.readouterr().out == ""                # nothing to say


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
