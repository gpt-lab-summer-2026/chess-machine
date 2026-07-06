"""Natural-language understanding (intent + slots) and generation (phrasing).

The SLM is confined to language: it classifies what the user wants and phrases
answers. Move legality and evaluations are never trusted from the model — they
are resolved by `move_parsing` against python-chess and by the engine.
"""
from .base import NLU
from .factory import create_nlu
from .intents import ACTIONS, Intent, intent_from_json
from .move_parsing import (
    describe_candidates,
    explain_move_failure,
    normalize_spoken,
    parse_move,
)
from .rule_based import resolve_side_request

__all__ = [
    "Intent",
    "ACTIONS",
    "intent_from_json",
    "NLU",
    "parse_move",
    "normalize_spoken",
    "describe_candidates",
    "explain_move_failure",
    "resolve_side_request",
    "create_nlu",
]
