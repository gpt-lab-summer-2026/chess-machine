import chess

from chessmachine.config import SlmConfig
from chessmachine.nlu import create_nlu
from chessmachine.nlu.intents import intent_from_json


def _nlu():
    return create_nlu(SlmConfig(backend="rule_based"))


def _ctx(board=None):
    return {"board": board or chess.Board(), "machine_color": "black",
            "turn": "white", "difficulty": "medium"}


def test_routing():
    n, c = _nlu(), _ctx()
    assert n.interpret("okay your move", c).action == "engine_move"
    assert n.interpret("who is winning right now", c).action == "analyze"
    assert n.interpret("let's start a new game", c).action == "new_game"
    assert n.interpret("take that back", c).action == "undo"
    assert n.interpret("knight to f3", c).action == "opponent_move"
    assert n.interpret("e2 to e5", c).action == "opponent_move"   # illegal but move-like
    assert n.interpret("the weather is nice", c).action == "unknown"


def test_difficulty_extraction():
    n, c = _nlu(), _ctx()
    assert n.interpret("set difficulty to hard", c).difficulty == "hard"
    assert n.interpret("make it easy", c).difficulty == "easy"
    assert n.interpret("set elo to 1600", c).difficulty == "1600"


def test_intent_from_json():
    assert intent_from_json({"action": "opponent_move", "move": "Nf3"}, "x").move == "Nf3"
    assert intent_from_json({"action": "frobnicate"}, "x").action == "unknown"
    assert intent_from_json("not a dict", "x").action == "unknown"
    assert intent_from_json({"action": "engine_move", "move": "  "}, "x").move is None
    assert intent_from_json({"action": "set_color", "color": "white"}, "x").color == "white"


def test_set_color_routing():
    n, c = _nlu(), _ctx()
    i = n.interpret("can I play black instead", c)
    assert i.action == "set_color" and i.color == "black"


def test_phrase_analysis_grounded():
    n = _nlu()
    facts = {
        "verdict": "White is winning", "in_check": False, "turn": "black",
        "material": {"white": 10, "black": 5, "diff": 5, "leader": "white"},
        "best_move_san": "Qd5", "pv_sans": ["Qd5"], "game_over": False,
    }
    out = n.phrase_analysis("who is winning", facts)
    assert "White" in out
