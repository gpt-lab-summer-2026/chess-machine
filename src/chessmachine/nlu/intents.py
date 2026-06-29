"""Intent schema shared by the SLM and the rule-based fallback."""
from __future__ import annotations

from dataclasses import dataclass

# The full set of actions the controller can dispatch.
ACTIONS = {
    "set_difficulty",   # difficulty: easy|medium|hard or an elo number
    "analyze",          # question: free-text question about the position
    "engine_move",      # make and actuate the machine's own move
    "opponent_move",    # move: the human's move, to validate and actuate
    "new_game",         # reset the board
    "undo",             # take back the last move
    "resign",           # machine or player resigns
    "status",           # speak score / whose turn
    "repeat",           # repeat the last spoken line
    "help",             # explain capabilities
    "chitchat",         # general conversation
    "unknown",          # could not be understood
}

# JSON schema handed to the SLM (also documents the contract).
INTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "move": {"type": ["string", "null"],
                 "description": "the chess move in UCI or SAN, e.g. 'e2e4' or 'Nf3'"},
        "difficulty": {"type": ["string", "null"],
                       "description": "easy, medium, hard, or an Elo number"},
        "question": {"type": ["string", "null"],
                     "description": "the user's question about the position"},
    },
    "required": ["action"],
}


@dataclass
class Intent:
    action: str
    move: str | None = None
    difficulty: str | None = None
    question: str | None = None
    text: str | None = None            # original transcript
    raw: dict | None = None            # raw model output, for debugging

    def __post_init__(self):
        if self.action not in ACTIONS:
            self.action = "unknown"


def _clean(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def intent_from_json(data: dict, transcript: str = "") -> Intent:
    """Build an Intent from a (possibly messy) model JSON object."""
    if not isinstance(data, dict):
        return Intent(action="unknown", text=transcript, raw={"_invalid": data})
    action = _clean(data.get("action")) or "unknown"
    return Intent(
        action=action,
        move=_clean(data.get("move")),
        difficulty=_clean(data.get("difficulty")),
        question=_clean(data.get("question")),
        text=transcript,
        raw=data,
    )
