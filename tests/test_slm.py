"""Tests for the SLM NLU: JSON salvaging and graceful fallback.

A `FakeClient` stands in for the llama.cpp HTTP/in-proc client, returning a
canned string or raising. The fallback is the real `RuleBasedNLU`, so we also
confirm the SLM degrades to deterministic behavior on any failure.
"""
import json
import urllib.error

import chess
import pytest

from chessmachine.config import SlmConfig
from chessmachine.nlu.intents import INTENT_ACTIONS_SCHEMA
from chessmachine.nlu.rule_based import RuleBasedNLU
from chessmachine.nlu.slm import LlamaCppClient, SlmNLU, _extract_json


class FakeClient:
    def __init__(self, reply="", raises=False):
        self.reply = reply
        self.raises = raises
        self.calls = 0

    def chat(self, messages, json_mode=False, temperature=None, max_tokens=None, schema=None):
        self.calls += 1
        if self.raises:
            raise RuntimeError("model unavailable")
        return self.reply


def _ctx():
    return {"board": chess.Board(), "machine_color": "black",
            "turn": "white", "difficulty": "medium"}


FACTS = {
    "verdict": "White is winning", "in_check": False, "turn": "black",
    "material": {"white": 39, "black": 30, "diff": 9, "leader": "white"},
    "best_move_san": "Qd5", "pv_sans": ["Qd5"], "game_over": False,
}


# -- _extract_json ----------------------------------------------------------- #
def test_extract_json_plain():
    assert _extract_json('{"action": "engine_move"}') == {"action": "engine_move"}


def test_extract_json_salvages_from_prose():
    assert _extract_json('Sure thing: {"action": "undo"} hope that helps') == {"action": "undo"}


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        _extract_json("there is no json here")


# -- interpret --------------------------------------------------------------- #
def test_interpret_uses_model_json():
    nlu = SlmNLU(FakeClient('{"action": "engine_move"}'), RuleBasedNLU())
    assert nlu.interpret("your move", _ctx())[0].action == "engine_move"


def test_interpret_falls_back_when_model_says_unknown():
    nlu = SlmNLU(FakeClient('{"action": "unknown"}'), RuleBasedNLU())
    # rule-based reads "your move" as a request for the engine to move
    assert nlu.interpret("your move", _ctx())[0].action == "engine_move"


def test_interpret_falls_back_on_exception():
    client = FakeClient(raises=True)
    nlu = SlmNLU(client, RuleBasedNLU())
    assert nlu.interpret("take that back", _ctx())[0].action == "undo"
    assert client.calls == 2                     # initial try + one retry, then fallback


def test_interpret_falls_back_on_garbage_output():
    nlu = SlmNLU(FakeClient("no json at all"), RuleBasedNLU())
    assert nlu.interpret("let's start a new game", _ctx())[0].action == "new_game"


class FlakyClient:
    """Raises on the first call, returns `good` after that."""
    def __init__(self, good):
        self.good = good
        self.calls = 0

    def chat(self, messages, json_mode=False, temperature=None, max_tokens=None, schema=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return self.good


def test_interpret_retries_once_then_uses_model():
    client = FlakyClient('{"actions": [{"action": "engine_move"}]}')
    nlu = SlmNLU(client, RuleBasedNLU())
    assert nlu.interpret("your move", _ctx())[0].action == "engine_move"
    assert client.calls == 2


def test_interpret_downgrades_question_to_analyze():
    # SLM misclassifies a question as a move; the guard rewrites it to analyze.
    reply = '{"actions": [{"action": "opponent_move", "move": "g1f3"}]}'
    nlu = SlmNLU(FakeClient(reply), RuleBasedNLU())
    assert nlu.interpret("what is threatening my knight", _ctx())[0].action == "analyze"


def test_interpret_keeps_real_move():
    reply = '{"actions": [{"action": "opponent_move", "move": "e2e4"}]}'
    nlu = SlmNLU(FakeClient(reply), RuleBasedNLU())
    out = nlu.interpret("e4", _ctx())
    assert out[0].action == "opponent_move"


# -- phrasing / small talk / commentary -------------------------------------- #
def test_phrase_analysis_uses_model_text():
    nlu = SlmNLU(FakeClient("  White is much better.  "), RuleBasedNLU())
    assert nlu.phrase_analysis("who is winning", FACTS) == "White is much better."


def test_phrase_analysis_falls_back_to_deterministic_summary():
    nlu = SlmNLU(FakeClient(raises=True), RuleBasedNLU())
    out = nlu.phrase_analysis("who is winning", FACTS)
    assert "White is winning" in out             # facts_to_summary, grounded


def test_small_talk_falls_back_on_exception():
    nlu = SlmNLU(FakeClient(raises=True), RuleBasedNLU())
    out = nlu.small_talk("hello there", _ctx())
    assert isinstance(out, str) and out


def test_comment_on_move_falls_back_on_exception():
    nlu = SlmNLU(FakeClient(raises=True), RuleBasedNLU())
    out = nlu.comment_on_move({"label": "blunder", "motifs": [], "difficulty": "medium"})
    assert "blunder" in out.lower()


# -- LlamaCppClient HTTP path (urllib mocked) -------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_chat_server_builds_request_and_parses_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LlamaCppClient(SlmConfig(mode="server", server_url="http://x:8080"))
    out = client.chat([{"role": "user", "content": "hi"}], json_mode=True)

    assert out == "hello"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_intent_actions_schema_wraps_per_action_schema():
    from chessmachine.nlu.intents import INTENT_JSON_SCHEMA
    assert INTENT_ACTIONS_SCHEMA["properties"]["actions"]["items"] is INTENT_JSON_SCHEMA
    assert INTENT_ACTIONS_SCHEMA["required"] == ["actions"]


def test_chat_server_sends_json_schema_and_enables_cache_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LlamaCppClient(SlmConfig(mode="server", server_url="http://x:8080"))
    client.chat([{"role": "user", "content": "hi"}], json_mode=True,
                schema=INTENT_ACTIONS_SCHEMA)

    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["cache_prompt"] is True   # prefix-cache on (speeds real moves)


def test_chat_server_falls_back_to_json_object_on_400(monkeypatch):
    seen = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        seen.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            raise urllib.error.HTTPError(req.full_url, 400, "bad schema", {}, None)
        return _FakeResp({"choices": [{"message": {"content": '{"actions": []}'}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LlamaCppClient(SlmConfig(mode="server", server_url="http://x:8080"))
    out = client.chat([{"role": "user", "content": "hi"}], json_mode=True,
                      schema={"type": "object"})

    assert seen == ["json_schema", "json_object"]
    assert out == '{"actions": []}'


# -- warmup (absorbs SLM cold-start at startup) ------------------------------ #
def test_warmup_warms_the_real_intent_path_with_long_timeout():
    import types
    seen = {}

    class WarmFake:
        cfg = types.SimpleNamespace(warmup_timeout_s=123.0)

        def chat(self, messages, json_mode=False, temperature=None, max_tokens=None,
                 schema=None, timeout=None):
            seen.update(timeout=timeout, schema=schema, json_mode=json_mode, messages=messages)
            return '{"actions": []}'

    SlmNLU(WarmFake(), RuleBasedNLU()).warmup(
        {"machine_color": "black", "turn": "white", "difficulty": "medium"})
    assert seen["timeout"] == 123.0                     # cold-load-tolerant, not the 30 s default
    # Warms the SAME path a real move takes — a bare "hello" wouldn't:
    assert seen["json_mode"] is True and seen["schema"] is not None
    assert len(seen["messages"]) > 2   # full prompt + few-shots, not a bare line


def test_warmup_is_non_fatal_when_server_down():
    import types

    class Boom:
        cfg = types.SimpleNamespace(warmup_timeout_s=1.0)

        def chat(self, *a, **k):
            raise RuntimeError("server down")

    SlmNLU(Boom(), RuleBasedNLU()).warmup()             # must NOT raise (falls back at runtime)
