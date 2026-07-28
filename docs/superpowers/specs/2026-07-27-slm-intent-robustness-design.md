# Design: Robust SLM intent path (STT-noise-aware)

**Date:** 2026-07-27
**Status:** Approved for planning
**Scope:** "A" of a larger NLU/coaching improvement effort. B (positional Q&A
answering), C (commentary honesty), and D (fluidity / model choice) are separate
specs and out of scope here.

## Problem

Observed on the real Pi (`--text`, real SLM via `llama-server` b11058, Whisper STT):

1. **"Context window can't play ~15 moves."** `llama-server` errors with
   `need to evaluate at least 1 token for each active slot, n_past = 997,
   task.n_tokens = 997`. This is **not** a game-length problem — every SLM call
   in this codebase is stateless and sends a fixed prompt. The `997` is the
   intent prompt itself (system message + 16 few-shot examples), which is
   byte-identical on every call. With `cache_prompt` on (the llama-server
   default) and a single slot, the fully-cached identical prompt trips the
   "nothing new to decode" condition and the request errors.
2. **JSON fallback.** `_extract_json` raises `No JSON object in model output`
   because the intent call only requests `response_format: {type:"json_object"}`
   (loose) — the 3B model still sometimes emits prose or off-shape JSON, and the
   call silently drops to the rule-based NLU.
3. **Questions become moves.** "what is threatening my knight?" and "is my
   knight on c4 in a good position" get classified as `opponent_move` and the
   machine physically moves a piece. Root cause: the few-shot examples are
   move-heavy, the only `analyze` examples are generic ("who is winning"), and
   `_do_opponent_move` will play the SLM's hallucinated `move` field whenever the
   literal transcript doesn't itself parse to a move.
4. **STT noise.** Transcripts arrive from Whisper: lowercase, usually
   **without punctuation** (no reliable "?"), with words occasionally dropped or
   mangled. Any guard or move decision must tolerate this and fail safe.

Points 1 and 2 are the same failure (a brittle, oversized intent call) surfacing
two ways. Point 3 is the sharpest user-visible symptom. Point 4 is a constraint
on the whole design.

## Goal

The intent classifier: (a) never crashes llama-server on the repeated prompt,
(b) returns schema-valid JSON without silent rule-based fallback, (c) never turns
a question into a move, and (d) never actuates a move it cannot ground in the
user's actual (normalized) words. Answering piece/square questions *well* is
explicitly deferred to B — after A, such a question routes to `analyze` and
receives the existing whole-board summary.

## Design

### 1. Schema-constrained JSON (`nlu/intents.py`, `nlu/slm.py`)

- Add a wrapper schema in `intents.py` that reuses the existing per-action
  `INTENT_JSON_SCHEMA`:
  `INTENT_ACTIONS_SCHEMA = {"type":"object","properties":{"actions":{"type":"array","items":INTENT_JSON_SCHEMA}},"required":["actions"]}`.
- `LlamaCppClient.chat` gains an optional `schema: dict | None`. When set (server
  mode), it sends `response_format: {"type":"json_schema","json_schema":
  {"name":"intent","schema":INTENT_ACTIONS_SCHEMA,"strict":true}}`. llama.cpp
  compiles this to a grammar, so the model **cannot** emit malformed or off-enum
  output. In-process mode passes the schema through llama-cpp-python's equivalent
  `response_format`.
- **Version safety:** if the server rejects `json_schema` (HTTP 400 on older
  builds), fall back once to the current `{"type":"json_object"}` request. So the
  change is safe on any llama.cpp build; b11058 supports json_schema.

### 2. Crash fix + prompt shrink (`nlu/slm.py`, `nlu/prompts.py`)

- Add `"cache_prompt": false` to every request body. This is the direct fix for
  the `n_past == task.n_tokens` crash and is a no-op field on non-llama servers.
  Cost is negligible once the prompt is small (below).
- Shrink `INTENT_EXAMPLES` from 16 to ~6, keeping only the mappings a constrained
  model still needs taught: a UCI move (`knight to f3`→`g1f3`), a bare Black reply
  (`e5`), the side-swap perspective flip (`let me play black`→`set_side white`),
  and one compound request. **Add two piece-question examples**:
  `what is threatening my knight` → `analyze`, and
  `is my knight on c4 well placed` → `analyze`.
- Add one line to `INTENT_SYSTEM`: *"A question — asking what/which/why/how or
  whether something is good/safe/threatened, or about who is winning — is NEVER a
  move; classify it as analyze, not opponent_move."*

### 3. STT-robust question-guard (`nlu/`, new pure function)

- `looks_like_analysis(transcript: str) -> bool` operating on the **normalized**
  string (via `normalize_spoken`), keying on cues that survive punctuation-free
  Whisper output:
  - a leading interrogative token: `what/whats/which/where/why/how/hows/who/is/
    are/am/do/does/did/can/could/should/would/will`, or
  - an analysis stem anywhere: `threat`, `attack`, `hang`, `defend`, `protect`,
    `safe`, `danger`, `win`, `better`, `worse`, `good`, `best`, `weak`, `strong`,
    `advantage`, `worth`, `should i`, `is it`, `how is`.
- Applied as a **post-classification downgrade** in `SlmNLU.interpret`: for any
  returned intent whose `action == "opponent_move"`, if `looks_like_analysis` is
  true, rewrite that intent to `analyze(question=transcript)`. It only ever
  downgrades opponent_move→analyze (other actions are left untouched); a bare
  "e4" / "knight f3" (no interrogative, no stem) is untouched too, so real moves
  are unaffected.

### 4. Move grounding — the STT-safety rule (`pipeline._do_opponent_move`)

- A move is actuated **only if the user's literal (normalized) words resolve via
  `parse_move`**. The SLM's `move` field remains **only a tie-breaker** among
  literal candidates (the existing block that accepts the SLM move only when it is
  exactly one of the already-found candidates) — it is never the sole source.
- Remove the current fallback that plays `parse_move(intent.move)` when
  `parse_move(intent.text)` is empty.
- When the literal words don't resolve **but** the SLM proposed a *legal* move,
  **confirm** it via the existing `_pending` yes/no mechanism:
  "Did you mean knight f3? Say yes." A "yes" plays it; anything else keeps the
  turn. This salvages a mangled transcript without a full repeat.
- When nothing plausible resolves, `explain_move_failure` re-asks as today.
- Net effect: a mangled or question-shaped transcript **fails safe** (confirm or
  re-ask) instead of moving the wrong piece — the "double-check instead of guess"
  behavior.

### 5. Retry then fallback (`nlu/slm.py`)

- `SlmNLU.interpret` retries the SLM call once (temperature 0) on any exception or
  unparseable output before dropping to the rule-based fallback, so a single
  transient hiccup doesn't silently downgrade the whole turn.

### 6. Data flow

```
transcript
  -> SLM interpret (json_schema-constrained, cache_prompt:false)   [retry once]
  -> parse actions
  -> question-guard downgrade (opponent_move -> analyze if looks_like_analysis)
  -> dispatch
       opponent_move: ground in literal words
                      -> confirm if only the SLM proposed a legal move
                      -> actuate, else re-ask
```

## Files touched

- `src/chessmachine/nlu/intents.py` — add `INTENT_ACTIONS_SCHEMA`.
- `src/chessmachine/nlu/slm.py` — `schema` arg + json_schema request with
  json_object fallback; `cache_prompt:false`; retry-once; apply question-guard.
- `src/chessmachine/nlu/prompts.py` — shrink examples, add piece-question
  examples, add the "questions are never moves" system line.
- `src/chessmachine/nlu/move_parsing.py` (or a small `nlu` module) —
  `looks_like_analysis`.
- `src/chessmachine/pipeline.py` — `_do_opponent_move` grounding + confirm path;
  reuse `_pending` (new pending kind, e.g. `"confirm_move:<uci>"`).

## Testing (no hardware / no model required)

- `looks_like_analysis` truth table, including punctuation-free Whisper-style
  strings ("what is threatening my knight", "is my knight on c4 good",
  "should i take", plus negatives "e4", "knight f3", "castle kingside").
- Question-guard: a stubbed SLM returning `opponent_move` for
  "what is threatening my knight" is downgraded to `analyze`.
- `_do_opponent_move`: an SLM-only legal move (literal text doesn't parse) sets a
  pending confirmation and does **not** actuate; "yes" then plays it; garbage
  re-asks; a normally-parsing move still plays directly; the SLM tie-breaker among
  ambiguous literal candidates still works.
- Schema path: mock client whose json_schema request 400s falls back to
  json_object and still parses.
- Retry: mock client failing once then succeeding returns the SLM intent (no
  rule-based fallback); failing twice falls back.
- Existing intent/parse/pipeline tests stay green.

## Risks / notes

- `json_schema` support is version-dependent; mitigated by the json_object
  fallback.
- The confirm path adds one interaction on a mangled move — intended, per the
  "double-check" preference; if it proves chatty in practice, the threshold can
  be tuned in a follow-up.
- `cache_prompt:false` slightly increases per-call compute (prefix re-processed);
  negligible with the shrunk prompt and acceptable given the speed headroom.

## Out of scope (future specs)

- **B:** answer piece/square questions with real facts (attackers/defenders,
  hanging, outpost, mobility) instead of the generic whole-board summary.
- **C:** correct move attribution + "only annotate my blunder if you punish it."
- **D:** conversational fluidity and a larger/slower model.
