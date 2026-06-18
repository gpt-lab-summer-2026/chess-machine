"""CLI argument handling: overrides and graceful failure."""
from chessmachine.app import main, _apply_overrides, _parse_args
from chessmachine.config import Config


def test_cli_difficulty_accepts_elo():
    cfg = _apply_overrides(Config(), _parse_args(["--difficulty", "1600"]))
    assert cfg.engine.default_difficulty == "1600"
    assert cfg.engine.presets["1600"].elo == 1600


def test_cli_difficulty_accepts_named_preset():
    cfg = _apply_overrides(Config(), _parse_args(["--difficulty", "hard"]))
    assert cfg.engine.default_difficulty == "hard"


def test_cli_difficulty_ignores_garbage():
    cfg = _apply_overrides(Config(), _parse_args(["--difficulty", "banana"]))
    assert cfg.engine.default_difficulty == "medium"   # default kept


def test_main_returns_2_on_bad_config(capsys):
    rc = main(["--config", "does/not/exist.yaml"])
    assert rc == 2
    assert "configuration error" in capsys.readouterr().err.lower()
