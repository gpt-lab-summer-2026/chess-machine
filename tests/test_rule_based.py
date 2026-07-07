"""Tests for the deterministic keyword NLU (the dev/offline backend & fallback)."""
import chess
import pytest

from chessmachine.nlu.rule_based import RuleBasedNLU, _looks_like_move


def _ctx(board=None):
    return {"board": board or chess.Board(), "machine_color": "black",
            "turn": "white", "difficulty": "medium"}


@pytest.mark.parametrize("text,action", [
    ("let's start a new game", "new_game"),
    ("take that back", "undo"),
    ("retake my turn", "undo"),
    ("I resign", "resign"),
    ("say again please", "repeat"),
    ("what can you do", "help"),
    ("okay your move", "engine_move"),
    ("whose turn is it", "status"),
    ("who is winning", "analyze"),
    ("knight to f3", "opponent_move"),
    ("the weather is nice today", "unknown"),
])
def test_interpret_routing(text, action):
    assert RuleBasedNLU().interpret(text, _ctx())[0].action == action


def test_difficulty_extraction():
    nlu = RuleBasedNLU()
    assert nlu.interpret("set difficulty to hard", _ctx())[0].difficulty == "hard"
    assert nlu.interpret("make it easy", _ctx())[0].difficulty == "easy"
    assert nlu.interpret("set elo to 1600", _ctx())[0].difficulty == "1600"


def test_color_switch():
    nlu = RuleBasedNLU()
    # "let me play black" -> the user wants black, so the MACHINE takes white.
    i = nlu.interpret("let me play black", _ctx())[0]
    assert i.action == "set_side" and i.color == "white"
    # "switch sides" is a swap: set_side with no explicit color.
    swap = nlu.interpret("switch sides", _ctx())[0]
    assert swap.action == "set_side" and swap.color is None


def test_looks_like_move():
    assert _looks_like_move("e4")
    assert _looks_like_move("castle kingside")
    assert not _looks_like_move("hello there")


def test_phrase_analysis_best_move_branch():
    facts = {"verdict": "the position is roughly equal", "in_check": False, "turn": "white",
             "material": {"white": 39, "black": 39, "diff": 0, "leader": "even"},
             "best_move_san": "Nf3", "pv_sans": ["Nf3"], "game_over": False}
    out = RuleBasedNLU().phrase_analysis("what's the best move", facts)
    assert "Nf3" in out


def test_phrase_analysis_material_branch():
    facts = {"verdict": "White is clearly better", "in_check": False, "turn": "white",
             "material": {"white": 39, "black": 34, "diff": 5, "leader": "white"},
             "best_move_san": None, "pv_sans": [], "game_over": False}
    out = RuleBasedNLU().phrase_analysis("am I ahead in material", facts)
    assert "up 5 points" in out
