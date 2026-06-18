import pytest

from chessmachine.config import Config, load_config


def test_defaults():
    c = Config()
    assert c.app.play_as == "black"
    assert c.engine.default_difficulty == "medium"
    assert set(c.engine.presets) >= {"easy", "medium", "hard"}
    assert c.engine.presets["medium"].elo == 1500


def test_deep_merge(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "app:\n  play_as: white\nmotion:\n  geometry:\n    square_pitch_mm: 20.0\n",
        encoding="utf-8",
    )
    c = load_config(p)
    assert c.app.play_as == "white"
    assert c.motion.geometry.square_pitch_mm == 20.0
    # untouched values keep their defaults
    assert c.app.auto_reply is True
    assert c.motion.geometry.origin_x_mm == 10.0


def test_partial_preset_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("engine:\n  presets:\n    easy:\n      elo: 600\n", encoding="utf-8")
    c = load_config(p)
    assert c.engine.presets["easy"].elo == 600
    assert c.engine.presets["easy"].skill == 3       # other easy fields preserved
    assert c.engine.presets["medium"].elo == 1500    # other presets preserved


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("app:\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_scalar_where_section_expected_rejected(tmp_path):
    # A scalar where a whole section is expected must error loudly, not silently
    # replace the dataclass.
    p = tmp_path / "c.yaml"
    p.write_text("app: 5\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(p)
