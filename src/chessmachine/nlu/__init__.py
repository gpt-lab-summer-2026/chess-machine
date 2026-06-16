"""Natural-language understanding (intent + slots) and generation (phrasing).

The SLM is confined to language: it classifies what the user wants and phrases
answers. Move legality and evaluations are never trusted from the model — they
are resolved by `move_parsing` against python-chess and by the engine.
"""
from .intents import Intent, ACTIONS, intent_from_json
from .base import NLU
from .move_parsing import parse_move, normalize_spoken, describe_candidates
from .factory import create_nlu

__all__ = [
    "Intent",
    "ACTIONS",
    "intent_from_json",
    "NLU",
    "parse_move",
    "normalize_spoken",
    "describe_candidates",
    "create_nlu",
]
