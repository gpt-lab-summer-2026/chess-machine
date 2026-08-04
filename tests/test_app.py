"""CLI argument handling: overrides and graceful failure."""
from chessmachine.app import _apply_overrides, _parse_args, main
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


def test_main_exits_cleanly_on_keyboard_interrupt(monkeypatch, capsys):
    """Ctrl-C is the documented way to stop deploy/boot_chessmachine.sh's restart
    loop, so it must print a short message and return 130, not a raw traceback."""
    import chessmachine.app as app_mod

    class BoomMachine:
        def start(self, home=True):
            pass

        def run(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(app_mod, "ChessMachine", lambda **kw: BoomMachine())
    # random/rule_based: the fully-offline recipe from this module's own docstring,
    # so the test builds no real Stockfish subprocess or llama-server client.
    rc = main(["--dev", "--mock", "--engine", "random", "--slm", "rule_based"])
    assert rc == 130
    assert "stopped" in capsys.readouterr().err.lower()
