"""Tests for the SLM NLU: JSON salvaging and graceful fallback.

A `FakeClient` stands in for the llama.cpp HTTP/in-proc client, returning a
canned string or raising. The fallback is the real `RuleBasedNLU`, so we also
confirm the SLM degrades to deterministic behavior on any failure.
"""
import json

import chess
import pytest

from chessmachine.config import SlmConfig
from chessmachine.nlu.rule_based import RuleBasedNLU
from chessmachine.nlu.slm import LlamaCppClient, SlmNLU, _extract_json


class FakeClient:
    def __init__(self, reply="", raises=False):
        self.reply = reply
        self.raises = raises
        self.calls = 0

    def chat(self, messages, json_mode=False, temperature=None, max_tokens=None):
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
    assert client.calls == 1                     # the model was tried first


def test_interpret_falls_back_on_garbage_output():
    nlu = SlmNLU(FakeClient("no json at all"), RuleBasedNLU())
    assert nlu.interpret("let's start a new game", _ctx())[0].action == "new_game"


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


import urllib.error

from chessmachine.nlu.intents import INTENT_ACTIONS_SCHEMA


def test_intent_actions_schema_wraps_per_action_schema():
    from chessmachine.nlu.intents import INTENT_JSON_SCHEMA
    assert INTENT_ACTIONS_SCHEMA["properties"]["actions"]["items"] is INTENT_JSON_SCHEMA
    assert INTENT_ACTIONS_SCHEMA["required"] == ["actions"]


def test_chat_server_sends_json_schema_and_disables_cache(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LlamaCppClient(SlmConfig(mode="server", server_url="http://x:8080"))
    client.chat([{"role": "user", "content": "hi"}], json_mode=True,
                schema=INTENT_ACTIONS_SCHEMA)

    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["cache_prompt"] is False


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
