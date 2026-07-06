import chess

from chessmachine.config import SlmConfig
from chessmachine.nlu import create_nlu
from chessmachine.nlu.intents import intent_from_json, intents_from_json


def _nlu():
    return create_nlu(SlmConfig(backend="rule_based"))


def _ctx(board=None):
    return {"board": board or chess.Board(), "machine_color": "black",
            "turn": "white", "difficulty": "medium"}


def _one(n, text, c):
    """interpret() now returns a list; a simple utterance yields exactly one."""
    out = n.interpret(text, c)
    assert len(out) == 1
    return out[0]


def test_routing():
    n, c = _nlu(), _ctx()
    assert _one(n, "okay your move", c).action == "engine_move"
    assert _one(n, "who is winning right now", c).action == "analyze"
    assert _one(n, "let's start a new game", c).action == "new_game"
    assert _one(n, "take that back", c).action == "undo"
    assert _one(n, "knight to f3", c).action == "opponent_move"
    assert _one(n, "e2 to e5", c).action == "opponent_move"   # illegal but move-like
    assert _one(n, "the weather is nice", c).action == "unknown"


def test_set_side_routing():
    n, c = _nlu(), _ctx()
    # explicit colors, perspective-aware (color slot = the MACHINE's color)
    assert _one(n, "let me play black", c).action == "set_side"
    assert _one(n, "let me play black", c).color == "white"
    assert _one(n, "you take white", c).color == "white"
    assert _one(n, "I want to play white", c).color == "black"
    assert _one(n, "give me black", c).action == "set_side"
    assert _one(n, "give me black", c).color == "white"
    # swap leaves the color unset
    swap = _one(n, "switch sides", c)
    assert swap.action == "set_side" and swap.color is None
    # not a side request
    assert _one(n, "is white winning", c).action != "set_side"


def test_compound_splits_into_multiple_intents():
    n, c = _nlu(), _ctx()
    out = n.interpret("give me black and set difficulty to hard", c)
    assert [i.action for i in out] == ["set_side", "set_difficulty"]
    assert out[0].color == "white" and out[1].difficulty == "hard"


def test_difficulty_extraction():
    n, c = _nlu(), _ctx()
    assert _one(n, "set difficulty to hard", c).difficulty == "hard"
    assert _one(n, "make it easy", c).difficulty == "easy"
    assert _one(n, "set elo to 1600", c).difficulty == "1600"


def test_intent_from_json():
    assert intent_from_json({"action": "opponent_move", "move": "Nf3"}, "x").move == "Nf3"
    assert intent_from_json({"action": "frobnicate"}, "x").action == "unknown"
    assert intent_from_json("not a dict", "x").action == "unknown"
    assert intent_from_json({"action": "engine_move", "move": "  "}, "x").move is None
    assert intent_from_json({"action": "set_side", "color": "white"}, "x").color == "white"


def test_intents_from_json_handles_object_list_and_wrapper():
    # bare object -> one intent
    assert [i.action for i in intents_from_json({"action": "undo"})] == ["undo"]
    # list -> several
    got = intents_from_json([{"action": "set_side", "color": "white"},
                             {"action": "set_difficulty", "difficulty": "hard"}])
    assert [i.action for i in got] == ["set_side", "set_difficulty"]
    # {"actions": [...]} wrapper (the SLM's output shape)
    wrapped = intents_from_json({"actions": [{"action": "engine_move"}]})
    assert [i.action for i in wrapped] == ["engine_move"]


def test_phrase_analysis_grounded():
    n = _nlu()
    facts = {
        "verdict": "White is winning", "in_check": False, "turn": "black",
        "material": {"white": 10, "black": 5, "diff": 5, "leader": "white"},
        "best_move_san": "Qd5", "pv_sans": ["Qd5"], "game_over": False,
    }
    out = n.phrase_analysis("who is winning", facts)
    assert "White" in out
