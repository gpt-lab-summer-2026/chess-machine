"""Build the configured NLU backend."""
from __future__ import annotations

from ..config import SlmConfig
from .base import NLU
from .rule_based import RuleBasedNLU


def create_nlu(cfg: SlmConfig) -> NLU:
    fallback = RuleBasedNLU()
    if cfg.backend == "rule_based":
        return fallback
    if cfg.backend == "llama_cpp":
        from .slm import LlamaCppClient, SlmNLU
        return SlmNLU(LlamaCppClient(cfg), fallback=fallback)
    raise ValueError(f"Unknown slm backend: {cfg.backend!r}")
