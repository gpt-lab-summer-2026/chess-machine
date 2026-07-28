"""llama.cpp-backed NLU.

`LlamaCppClient` talks to either a running `llama-server` over its
OpenAI-compatible HTTP endpoint (default, recommended on the Pi) or an
in-process model via llama-cpp-python. `SlmNLU` uses it for intent JSON and
answer phrasing, falling back to the rule-based NLU on any failure.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from ..config import SlmConfig
from .base import NLU
from .intents import INTENT_ACTIONS_SCHEMA, Intent, intents_from_json
from .move_parsing import looks_like_analysis
from .prompts import build_analysis_messages, build_intent_messages, build_move_comment_messages

if TYPE_CHECKING:
    from ..chess_engine.analysis import PositionFacts

log = logging.getLogger(__name__)


class LlamaCppClient:
    def __init__(self, cfg: SlmConfig):
        self.cfg = cfg
        self._llm: Any = None  # lazily created in-process model

    def chat(self, messages: list[dict], json_mode: bool = False,
             temperature: float | None = None,
             max_tokens: int | None = None,
             schema: dict | None = None,
             timeout: float | None = None) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        maxt = self.cfg.max_tokens if max_tokens is None else max_tokens
        to = self.cfg.request_timeout_s if timeout is None else timeout
        if self.cfg.mode == "inproc":
            return self._chat_inproc(messages, json_mode, temp, maxt, schema)
        return self._chat_server(messages, json_mode, temp, maxt, schema, to)

    def _chat_server(self, messages, json_mode, temperature, max_tokens,
                     schema=None, timeout=None) -> str:
        url = self.cfg.server_url.rstrip("/") + "/v1/chat/completions"
        to = self.cfg.request_timeout_s if timeout is None else timeout

        def _post(response_format) -> str:
            body = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "cache_prompt": True,    # reuse the shared system-prompt prefix across calls
            }
            if response_format is not None:
                body["response_format"] = response_format
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=to) as resp:
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


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Salvage the first balanced {...} block from a chatty reply.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON object in model output: {text!r}")


class SlmNLU(NLU):
    def __init__(self, client: LlamaCppClient, fallback: NLU):
        self.client = client
        self.fallback = fallback

    def warmup(self, context: dict | None = None) -> None:
        """Warm the REAL intent path before the game starts, so the user's first
        move doesn't race a cold server and time out.

        A bare "hello" does NOT work: with cache_prompt on, real moves are fast
        because the big system-prompt+few-shot PREFIX and the json_schema grammar
        are already cached — and a trivial request warms none of those. So we run
        an actual throwaway move ("e2e4") through the exact same chat call
        `interpret` uses (same messages, schema, json_mode), which primes the
        model, the prefix cache, and the grammar. Long timeout absorbs cold model
        load on the Pi; failures are non-fatal (server down -> rule-based)."""
        ctx = context or {"machine_color": "black", "turn": "white", "difficulty": "medium"}
        try:
            self.client.chat(
                build_intent_messages("e2e4", ctx),
                json_mode=True, temperature=0.0, max_tokens=96,
                schema=INTENT_ACTIONS_SCHEMA,
                timeout=self.client.cfg.warmup_timeout_s,
            )
            log.info("SLM warmed up (real intent path)")
        except Exception as exc:  # noqa: BLE001 - warmup must never block startup
            log.info("SLM warmup skipped (%s); will use rule-based until it responds", exc)

    def interpret(self, transcript: str, context: dict) -> list[Intent]:
        last_exc: Exception | None = None
        for attempt in range(2):          # initial try + one retry
            try:
                raw = self.client.chat(
                    build_intent_messages(transcript, context),
                    json_mode=True, temperature=0.0, max_tokens=96,
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

    def phrase_analysis(self, question: str, facts: PositionFacts) -> str:
        try:
            text = self.client.chat(
                build_analysis_messages(question, facts),
                temperature=0.3, max_tokens=120,
            )
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("SLM phrasing failed (%s); using deterministic summary", exc)
            return self.fallback.phrase_analysis(question, facts)

    def small_talk(self, transcript: str, context: dict) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a concise, friendly chess robot. "
                 "Reply in one short sentence suitable for speech."},
                {"role": "user", "content": transcript},
            ]
            return self.client.chat(messages, temperature=0.5, max_tokens=60).strip()
        except Exception:  # noqa: BLE001
            return self.fallback.small_talk(transcript, context)

    def comment_on_move(self, info: dict) -> str:
        try:
            text = self.client.chat(
                build_move_comment_messages(info), temperature=0.4, max_tokens=80,
            ).strip().strip('"').strip()      # small models like to wrap replies in quotes
            return text or self.fallback.comment_on_move(info)
        except Exception as exc:  # noqa: BLE001
            log.warning("SLM move comment failed (%s); using deterministic", exc)
            return self.fallback.comment_on_move(info)
