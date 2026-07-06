import shutil

from chessmachine.chess_engine.engine import resolve_stockfish_path


def test_resolve_prefers_existing_explicit_path(tmp_path):
    """An explicit path that exists is used verbatim (the Pi/winget case)."""
    binary = tmp_path / "stockfish.exe"
    binary.write_text("", encoding="utf-8")
    assert resolve_stockfish_path(str(binary)) == str(binary)


def test_resolve_falls_back_to_path(monkeypatch, tmp_path):
    """A bare command name resolves via PATH, so config can just say 'stockfish'."""
    fake = tmp_path / "stockfish"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(shutil, "which",
                        lambda cmd: str(fake) if cmd == "stockfish" else None)
    assert resolve_stockfish_path("stockfish") == str(fake)


def test_resolve_returns_configured_when_nothing_found(monkeypatch):
    """When nothing is found anywhere, hand back the configured value unchanged
    so the caller can raise a clear, named error."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.setattr(
        "chessmachine.chess_engine.engine._wellknown_stockfish_paths",
        lambda: [],
    )
    assert resolve_stockfish_path("nope-not-here") == "nope-not-here"
