# SLM Intent Path Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SLM intent classifier stop crashing llama-server, stop silently falling back on bad JSON, stop turning questions into moves, and never actuate a move it can't ground in the user's actual words.

**Architecture:** Constrain the intent call's output with a llama.cpp `json_schema` response format (reusing the schema already in `intents.py`), disable prompt caching on every request, shrink the few-shot block, add a punctuation-free "questions are never moves" guard, and make `_do_opponent_move` ground moves in the literal transcript (confirming an SLM-only guess instead of playing it).

**Tech Stack:** Python 3.10+, `python-chess`, llama.cpp `llama-server` (OpenAI-compatible HTTP), `pytest`, `unittest.mock`/`monkeypatch`. No new dependencies.

## Global Constraints

- Python ≥ 3.10; **no new third-party dependencies**.
- All tests run **offline** — no hardware, no live model, no network (mock `urllib.request.urlopen`, use `FakeClient`).
- `json_schema` support is version-dependent: **every schema request must fall back to `{"type":"json_object"}` on HTTP 400**.
- **`"cache_prompt": false` on every server request body** (the direct fix for the `n_past == task.n_tokens` crash).
- The question-guard **only ever downgrades `opponent_move` → `analyze`**, never the reverse.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Guard/move logic operates on the **normalized** transcript (`normalize_spoken`) and must not rely on punctuation (Whisper output has none).

---

### Task 1: `looks_like_analysis` question-guard helper

**Files:**
- Modify: `src/chessmachine/nlu/move_parsing.py` (add function near `normalize_spoken`)
- Modify: `src/chessmachine/nlu/__init__.py` (export it)
- Test: `tests/test_move_parsing.py`

**Interfaces:**
- Consumes: `normalize_spoken` (already in this module).
- Produces: `looks_like_analysis(transcript: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_move_parsing.py`:

```python
from chessmachine.nlu.move_parsing import looks_like_analysis


@pytest.mark.parametrize("text", [
    "what is threatening my knight",     # punctuation-free, Whisper style
    "is my knight on c4 good",
    "is my knight on c4 in a good position",
    "should i take the pawn",
    "who is winning",
    "how is my position",
    "what's the best move",
    "is my king safe",
])
def test_looks_like_analysis_true(text):
    assert looks_like_analysis(text) is True


@pytest.mark.parametrize("text", [
    "e4",
    "knight to f3",
    "castle kingside",
    "bishop takes d5",
    "e2 to e4",
    "",
])
def test_looks_like_analysis_false(text):
    assert looks_like_analysis(text) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_move_parsing.py -k looks_like_analysis -v`
Expected: FAIL — `ImportError: cannot import name 'looks_like_analysis'`.

- [ ] **Step 3: Implement the function**

In `src/chessmachine/nlu/move_parsing.py`, add after `normalize_spoken`:

```python
# Cues that mark an utterance as a question / analysis request rather than a
# move. Kept STT-robust: matched against the NORMALIZED string (no punctuation),
# because Whisper output usually has no '?'.
_INTERROGATIVES = {
    "what", "whats", "which", "where", "why", "how", "hows", "who",
    "is", "are", "am", "do", "does", "did", "can", "could", "should",
    "would", "will",
}
_ANALYSIS_STEMS = (
    "threat", "attack", "hang", "defend", "protect", "safe", "danger",
    "win", "better", "worse", "good", "best", "weak", "strong",
    "advantage", "worth", "should i", "is it", "how is",
)


def looks_like_analysis(transcript: str) -> bool:
    """True if the utterance reads as a question / analysis request, not a move.

    Used only to DOWNGRADE a move classification to analyze, so a false positive
    costs a re-route (the machine answers instead of moving), never a wrong move.
    Operates on the normalized transcript so it never depends on a '?' that STT
    tends to drop.
    """
    t = normalize_spoken(transcript)
    if not t:
        return False
    if t.split()[0] in _INTERROGATIVES:
        return True
    return any(stem in t for stem in _ANALYSIS_STEMS)
```

In `src/chessmachine/nlu/__init__.py`, add `looks_like_analysis` to both the `from .move_parsing import (...)` block and `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_move_parsing.py -k looks_like_analysis -v`
Expected: PASS (all parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/chessmachine/nlu/move_parsing.py src/chessmachine/nlu/__init__.py tests/test_move_parsing.py
git commit -m "feat(nlu): add STT-robust looks_like_analysis question guard" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Shrink few-shots + add analyze examples + "never a move" rule

**Files:**
- Modify: `src/chessmachine/nlu/prompts.py` (`INTENT_EXAMPLES`, `INTENT_SYSTEM`)
- Test: `tests/test_prompts.py` (new)

**Interfaces:**
- Produces: trimmed `INTENT_EXAMPLES` (≤ 9 pairs, including two piece-question `analyze` examples); `INTENT_SYSTEM` containing an explicit "a question is never a move" rule. No signature changes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompts.py`:

```python
from chessmachine.nlu.prompts import INTENT_EXAMPLES, INTENT_SYSTEM, build_intent_messages


def test_examples_are_trimmed():
    assert len(INTENT_EXAMPLES) <= 9        # was 16


def test_examples_cover_piece_questions_as_analyze():
    users = [u for u, _ in INTENT_EXAMPLES]
    threat = next((a for u, a in INTENT_EXAMPLES if "threat" in u), None)
    placed = next((a for u, a in INTENT_EXAMPLES if "well placed" in u), None)
    assert threat is not None and '"analyze"' in threat
    assert placed is not None and '"analyze"' in placed


def test_system_prompt_forbids_question_as_move():
    assert "never a move" in INTENT_SYSTEM.lower()


def test_build_intent_messages_still_ends_with_user_turn():
    msgs = build_intent_messages("e4", {"machine_color": "black", "turn": "white",
                                         "difficulty": "medium"})
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "e4"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: FAIL — `test_examples_are_trimmed` (16 > 9) and `test_system_prompt_forbids_question_as_move`.

- [ ] **Step 3: Implement the prompt changes**

In `src/chessmachine/nlu/prompts.py`, replace `INTENT_EXAMPLES` with:

```python
# A small set of few-shots: the output shape is enforced by the json_schema
# grammar, so examples only need to teach the tricky MAPPINGS — UCI moves, a
# bare black reply, the side-swap perspective flip, a compound request, and
# (critically) that piece/position QUESTIONS are analyze, not moves.
INTENT_EXAMPLES = [
    ("knight to f3", '{"actions": [{"action": "opponent_move", "move": "g1f3"}]}'),
    ("e5", '{"actions": [{"action": "opponent_move", "move": "e5"}]}'),            # black reply
    ("okay, your move", '{"actions": [{"action": "engine_move"}]}'),
    ("let me play black", '{"actions": [{"action": "set_side", "color": "white"}]}'),
    ("switch sides", '{"actions": [{"action": "set_side"}]}'),
    ("give me black and set difficulty to hard",
     '{"actions": [{"action": "set_side", "color": "white"}, '
     '{"action": "set_difficulty", "difficulty": "hard"}]}'),
    ("what is threatening my knight",
     '{"actions": [{"action": "analyze", "question": "what is threatening my knight"}]}'),
    ("is my knight on c4 well placed",
     '{"actions": [{"action": "analyze", "question": "is my knight on c4 well placed"}]}'),
]
```

Then, in `INTENT_SYSTEM`, immediately before the final `Context: {context_line}` line, insert:

```
A question — asking what/which/why/how, or whether a piece or move is good, \
safe, or threatened, or who is winning — is NEVER a move; use "analyze", never \
"opponent_move".
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/chessmachine/nlu/prompts.py tests/test_prompts.py
git commit -m "feat(nlu): trim few-shots, add piece-question analyze examples + no-move-question rule" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Schema-constrained JSON + cache_prompt:false + json_object fallback

**Files:**
- Modify: `src/chessmachine/nlu/intents.py` (add `INTENT_ACTIONS_SCHEMA`)
- Modify: `src/chessmachine/nlu/slm.py` (`LlamaCppClient.chat`, `_chat_server`, `_chat_inproc`)
- Test: `tests/test_slm.py`

**Interfaces:**
- Consumes: `INTENT_JSON_SCHEMA` (already in `intents.py`).
- Produces: `INTENT_ACTIONS_SCHEMA: dict`; `LlamaCppClient.chat(messages, json_mode=False, temperature=None, max_tokens=None, schema=None) -> str` (new trailing `schema` arg, backward compatible).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_slm.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_slm.py -k "actions_schema or json_schema or json_object_on_400" -v`
Expected: FAIL — `ImportError: cannot import name 'INTENT_ACTIONS_SCHEMA'`.

- [ ] **Step 3: Implement**

In `src/chessmachine/nlu/intents.py`, after `INTENT_JSON_SCHEMA`, add:

```python
# Wrapper schema for the full {"actions": [...]} reply. Passed to llama.cpp's
# json_schema response format so the model cannot emit malformed / off-enum JSON.
INTENT_ACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": INTENT_JSON_SCHEMA},
    },
    "required": ["actions"],
}
```

In `src/chessmachine/nlu/slm.py`, add `import urllib.error` at the top if not present, then replace `chat`, `_chat_server`, and `_chat_inproc`:

```python
    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float | None = None,
             max_tokens: int | None = None,
             schema: dict | None = None) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        maxt = self.cfg.max_tokens if max_tokens is None else max_tokens
        if self.cfg.mode == "inproc":
            return self._chat_inproc(messages, json_mode, temp, maxt, schema)
        return self._chat_server(messages, json_mode, temp, maxt, schema)

    def _chat_server(self, messages, json_mode, temperature, max_tokens,
                     schema=None) -> str:
        url = self.cfg.server_url.rstrip("/") + "/v1/chat/completions"

        def _post(response_format) -> str:
            body = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "cache_prompt": False,   # avoid the n_past==n_tokens caching crash
            }
            if response_format is not None:
                body["response_format"] = response_format
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as resp:
                payload = json.loads(resp.read().decode())
            return payload["choices"][0]["message"]["content"]

        if schema is not None:
            rf = {"type": "json_schema",
                  "json_schema": {"name": "intent", "schema": schema, "strict": True}}
            try:
                return _post(rf)
            except urllib.error.HTTPError as exc:
                if exc.code != 400:
                    raise
                log.warning("server rejected json_schema (%s); retrying with json_object", exc)
                return _post({"type": "json_object"})
        return _post({"type": "json_object"} if json_mode else None)

    def _chat_inproc(self, messages, json_mode, temperature, max_tokens,
                     schema=None) -> str:
        if self._llm is None:
            from llama_cpp import Llama  # lazy: heavy dependency
            self._llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads,
                n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False,
            )
        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        out = self._llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_slm.py -v`
Expected: PASS — the new tests plus the existing `test_chat_server_builds_request_and_parses_response` (its `response_format == {"type":"json_object"}` assertion still holds on the no-schema path).

- [ ] **Step 5: Commit**

```bash
git add src/chessmachine/nlu/intents.py src/chessmachine/nlu/slm.py tests/test_slm.py
git commit -m "feat(nlu): schema-constrained intent JSON with cache_prompt:false and json_object fallback" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `SlmNLU.interpret` — pass schema, retry once, apply the guard

**Files:**
- Modify: `src/chessmachine/nlu/slm.py` (`SlmNLU.interpret`, imports)
- Modify: `tests/test_slm.py` (`FakeClient` signature; update one assertion; add tests)

**Interfaces:**
- Consumes: `looks_like_analysis` (Task 1), `INTENT_ACTIONS_SCHEMA` (Task 3), `chat(..., schema=)` (Task 3).
- Produces: no new public signature; `interpret` behavior = schema-constrained call, one retry on exception/unparse, then question-guard downgrade, then rule-based fallback.

- [ ] **Step 1: Update `FakeClient` and write the failing tests**

In `tests/test_slm.py`, update `FakeClient.chat` to accept the new arg:

```python
    def chat(self, messages, json_mode=False, temperature=None, max_tokens=None, schema=None):
        self.calls += 1
        if self.raises:
            raise RuntimeError("model unavailable")
        return self.reply
```

Update the existing exception test's call-count assertion (retry now means two attempts):

```python
def test_interpret_falls_back_on_exception():
    client = FakeClient(raises=True)
    nlu = SlmNLU(client, RuleBasedNLU())
    assert nlu.interpret("take that back", _ctx())[0].action == "undo"
    assert client.calls == 2                     # initial try + one retry, then fallback
```

Add new tests:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_slm.py -k "retries_once or downgrades_question or keeps_real_move or falls_back_on_exception" -v`
Expected: FAIL — `test_interpret_retries_once_then_uses_model` (no retry yet → `RuntimeError`/fallback, `calls == 1`) and `test_interpret_downgrades_question_to_analyze` (returns `opponent_move`).

- [ ] **Step 3: Implement**

In `src/chessmachine/nlu/slm.py`, update the imports:

```python
from .intents import Intent, INTENT_ACTIONS_SCHEMA, intents_from_json
from .move_parsing import looks_like_analysis
```

Replace `SlmNLU.interpret` and add the guard helper:

```python
    def interpret(self, transcript: str, context: dict) -> list[Intent]:
        last_exc: Exception | None = None
        for attempt in range(2):          # initial try + one retry
            try:
                raw = self.client.chat(
                    build_intent_messages(transcript, context),
                    json_mode=True, temperature=0.0, max_tokens=192,
                    schema=INTENT_ACTIONS_SCHEMA,
                )
                intents = intents_from_json(_extract_json(raw), transcript)
                intents = [self._guard(i, transcript) for i in intents]
                known = [i for i in intents if i.action != "unknown"]
                if known:
                    return known
                log.info("SLM returned no known action; using rule-based fallback")
                break                     # parsed OK but nothing usable -> don't retry
            except Exception as exc:      # noqa: BLE001 - degrade gracefully
                last_exc = exc
                log.warning("SLM interpret attempt %d failed (%s)", attempt + 1, exc)
        if last_exc is not None:
            log.warning("SLM interpret failed after retry; using rule-based fallback")
        return self.fallback.interpret(transcript, context)

    @staticmethod
    def _guard(intent: Intent, transcript: str) -> Intent:
        """Downgrade a misclassified question to analyze. STT-robust; only ever
        turns opponent_move into analyze, so real moves are never touched."""
        if intent.action == "opponent_move" and looks_like_analysis(transcript):
            return Intent(action="analyze", question=transcript, text=transcript)
        return intent
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_slm.py -v`
Expected: PASS (all, including the updated `calls == 2`).

- [ ] **Step 5: Commit**

```bash
git add src/chessmachine/nlu/slm.py tests/test_slm.py
git commit -m "feat(nlu): schema-constrained interpret with one retry and question-to-analyze guard" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Ground opponent moves in the literal transcript + confirm SLM guesses

**Files:**
- Modify: `src/chessmachine/pipeline.py` (`_do_opponent_move`, `_resolve_pending`, add `_play_opponent_move`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `parse_move`, `describe_candidates`, `explain_move_failure` (already imported), `_is_affirmative`, `_play_move`, `_do_engine_move`, `_safe_park`, `chess`.
- Produces: `_play_opponent_move(self, move: chess.Move) -> str`; a new `_pending` value form `"confirm_move:<uci>"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_opponent_move_confirms_when_only_slm_resolves():
    m, _ = make(auto_reply=False)
    # Transcript lost its destination ("knight"); the SLM recovered a legal move.
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move="g1f3"))
    assert m._pending == "confirm_move:g1f3"
    assert len(m.game.history) == 0              # nothing played yet
    m.handle("yes")
    assert m.game.history[0][1] == "Nf3"


def test_opponent_move_declined_confirmation_plays_nothing():
    m, _ = make(auto_reply=False)
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move="g1f3"))
    m.handle("no")
    assert len(m.game.history) == 0
    assert m._pending is None


def test_opponent_move_ungrounded_without_slm_move_reasks():
    m, tts = make(auto_reply=False)
    m._do_opponent_move(Intent(action="opponent_move", text="knight", move=None))
    assert m._pending is None
    assert len(m.game.history) == 0
    assert tts.lines                              # it said something (a re-ask)


def test_normal_move_still_plays_directly():
    m, _ = make(auto_reply=False)
    m.handle("e4")
    assert m.game.history[0][1] == "e4"
    assert m._pending is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -k "confirms_when_only_slm or declined_confirmation or ungrounded_without" -v`
Expected: FAIL — the current `_do_opponent_move` plays `parse_move(intent.move)` directly, so `_pending` is never set and the knight move is played immediately.

- [ ] **Step 3: Implement**

In `src/chessmachine/pipeline.py`, replace `_do_opponent_move` with:

```python
    def _do_opponent_move(self, intent: Intent) -> str:
        if self.game.is_game_over():
            return self._say(self.game.result_text())
        board = self.game.board
        # Ground the move in what the user LITERALLY said. The SLM's `move` field
        # is only a tie-breaker among these candidates, never a source on its own,
        # so a mangled transcript or a hallucinated move can't actuate a piece.
        candidates = parse_move(intent.text or "", board)
        if not candidates:
            # Literal words didn't resolve. If the SLM proposed a single legal
            # move, CONFIRM it (STT may have dropped a word) rather than guessing.
            slm_moves = parse_move(intent.move, board) if intent.move else []
            if len(slm_moves) == 1:
                self._pending = f"confirm_move:{slm_moves[0].uci()}"
                return self._say(f"Did you mean {board.san(slm_moves[0])}? Say yes.")
            return self._say(explain_move_failure(intent.text or intent.move or "", board))
        # Ambiguous literal words: let the SLM's move break the tie, but only when
        # it resolves to exactly one of the candidates we already found.
        if len(candidates) > 1 and intent.move:
            slm_moves = parse_move(intent.move, board)
            if len(slm_moves) == 1 and slm_moves[0] in candidates:
                candidates = slm_moves
        if len(candidates) > 1:
            return self._say(f"ambiguous move: did you mean "
                             f"{describe_candidates(candidates, board)}?")
        return self._play_opponent_move(candidates[0])

    def _play_opponent_move(self, move: chess.Move) -> str:
        """Actuate the human's move, then auto-reply with the engine if it's our turn."""
        spoken = self._play_move(move, "Okay,")   # _play_move speaks internally
        if (self.cfg.app.auto_reply and not self.game.is_game_over()
                and self.game.is_machine_turn()):
            try:
                reply = self._do_engine_move("My move:")
            except Exception:  # noqa: BLE001 - the opponent's move already stuck
                log.exception("Auto-reply failed after the opponent's move")
                self._safe_park()
                note = self._say("I couldn't make my reply just now; please check the board.")
                return spoken + " " + note
            return spoken + " " + reply
        return spoken
```

Replace `_resolve_pending` with (adds the `confirm_move:` branch):

```python
    def _resolve_pending(self, transcript: str) -> str:
        """Resolve a yes/no answer to a pending confirmation (new game, undo, or
        a low-confidence move recovered from the SLM)."""
        pending, self._pending = self._pending, None
        if pending.startswith("confirm_move:"):
            if not _is_affirmative(transcript):
                return self._say("Okay, ignoring that. Say your move again?")
            move = chess.Move.from_uci(pending.split(":", 1)[1])
            if move not in self.game.board.legal_moves:
                return self._say("That move isn't legal now; say it again?")
            return self._play_opponent_move(move)
        if _is_affirmative(transcript):
            if pending == "undo":
                return self._really_undo()
            return self._really_new_game()          # pending == "new_game"
        return self._say("Okay, keeping the move." if pending == "undo"
                         else "Okay, keeping the current game.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS — new tests plus existing pipeline tests (`test_opponent_move_actuated_with_autoreply`, `test_illegal_move_is_explained`, etc.), since grounded moves and the auto-reply path are unchanged in behavior.

- [ ] **Step 5: Commit**

```bash
git add src/chessmachine/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): ground opponent moves in the transcript, confirm SLM-only guesses" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full-suite regression + lint/type gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS — all tests, including the pre-existing suite. If any pre-existing test fails, it is a regression from Tasks 1–5; fix inline and re-run.

- [ ] **Step 2: Lint + type-check (matches CI)**

Run: `ruff check . && mypy src`
Expected: clean. Fix any findings inline (e.g. the added `import urllib.error`, unused imports).

- [ ] **Step 3: Commit any fixups (only if needed)**

```bash
git add -A
git commit -m "chore(nlu): lint/type fixups for the intent-path robustness work" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Whole-flow behavior after this plan:** `transcript → SLM (json_schema, cache_prompt:false) →[retry once] → parse → guard downgrade → dispatch`; `opponent_move → ground in literal words → confirm if only the SLM resolved → actuate, else re-ask`.
- **Out of scope (do not add):** answering piece/square questions with real facts, move-quality attribution changes, conversational memory, or a model swap — those are separate specs (B/C/D). After this plan, "what is threatening my knight" correctly routes to `analyze` and receives the existing whole-board summary; that's expected.
- **Manual Pi check (optional, post-merge):** with `llama-server` running, `python -m chessmachine --text --config config/config.yaml`, play ~20 moves and confirm no `n_past == task.n_tokens` error and that "what is threatening my knight" no longer moves a piece.
