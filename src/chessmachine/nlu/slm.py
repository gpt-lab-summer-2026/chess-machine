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
from .intents import Intent, intent_from_json
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
             max_tokens: int | None = None) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        maxt = self.cfg.max_tokens if max_tokens is None else max_tokens
        if self.cfg.mode == "inproc":
            return self._chat_inproc(messages, json_mode, temp, maxt)
        return self._chat_server(messages, json_mode, temp, maxt)

    def _chat_server(self, messages, json_mode, temperature, max_tokens) -> str:
        url = self.cfg.server_url.rstrip("/") + "/v1/chat/completions"
        body = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_s) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]

    def _chat_inproc(self, messages, json_mode, temperature, max_tokens) -> str:
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
        if json_mode:
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

    def interpret(self, transcript: str, context: dict) -> Intent:
        try:
            raw = self.client.chat(
                build_intent_messages(transcript, context),
                json_mode=True, temperature=0.0, max_tokens=128,
            )
            intent = intent_from_json(_extract_json(raw), transcript)
            if intent.action != "unknown":
                return intent
            log.info("SLM returned 'unknown'; trying rule-based fallback")
        except Exception as exc:  # noqa: BLE001 - any failure should degrade gracefully
            log.warning("SLM interpret failed (%s); using rule-based fallback", exc)
        return self.fallback.interpret(transcript, context)

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
