"""Prompt construction for the SLM (intent classification + answer phrasing)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chess_engine.analysis import PositionFacts
INTENT_SYSTEM = """You control a voice-operated chess robot. Convert the user's \
utterance into JSON of the form {"actions": [ ... ]} and output nothing else. \
Put ONE object per thing the user asked for, in order — usually one, but a \
compound request like "give me black and make it harder" becomes two.

Actions (each object has an "action" plus only the fields it needs):
- "opponent_move": the human states THEIR move. Copy EXACTLY what they said into \
"move" as UCI (e2e4) or SAN (Nf3, O-O) — do not substitute, "correct", or invent a \
different move, and remember the human may be playing black (e.g. they say "e5", \
"c5", "knight f6").
- "engine_move": the user asks YOU (the robot) to make your move.
- "set_difficulty": put easy, medium, hard, or an Elo number in "difficulty".
- "set_side": the user chooses who plays which color, mid-game. Put the color \
YOU (the robot) will play in "color" as "white" or "black"; for "switch sides" \
omit "color". Remember: if the user wants a color for THEMSELVES, you take the \
other one.
- "analyze": a question about the position (who is winning, best move, threats, \
evaluation). Put the question text in "question".
- "new_game", "undo", "resign", "status", "repeat", "help", "chitchat".

INTENT_SYSTEM = """You control a voice-operated chess robot. The user speaks; you \
convert ONE utterance into ONE JSON object and output nothing else (no prose).

Pick exactly one "action":
- "opponent_move": the user states THEIR own move. Put it in "move" as UCI \
(e2e4, d5e6) or SAN (Nf3, exd5, O-O). Use the squares they actually say — \
"d5 takes e6" is move "d5e6". Never invent or change squares.
- "engine_move": the user explicitly asks YOU to move ("your move", "you go", \
"make your move"). A bare "okay", "hmm", or "sure" is NOT this — that is chitchat.
- "set_difficulty": set strength; put easy, medium, hard, or an Elo number in "difficulty".
- "set_color": the user wants to switch sides or pick a colour ("let me play \
black", "I'll take white", "switch sides"). Put the colour THEY want in "color".
- "analyze": a question about the position (who's winning, best move, threats). \
Put the question in "question".
- "undo": take back the last move ("undo", "take that back", "retake my turn", \
"let me redo that").
- "new_game": start over / reset the board.
- "resign": the user gives up.
- "status": whose turn it is, or the score.
- "repeat": say the last thing again.
- "help": what can you do.
- "chitchat": anything else, including bare acknowledgements ("okay", "thanks", "cool").

Fill only the slot for the chosen action ("move", "difficulty", "color", or \
"question"); omit the others.
Context: {context_line}"""

# Few-shot pairs steer a small model toward strict, correct JSON. They cover the
# tricky cases: captures, undo phrasings, colour switches, and fillers.
INTENT_EXAMPLES = [
    ("knight to f3", '{"actions": [{"action": "opponent_move", "move": "Nf3"}]}'),
    ("I'll play e4", '{"actions": [{"action": "opponent_move", "move": "e2e4"}]}'),
    ("e5", '{"actions": [{"action": "opponent_move", "move": "e5"}]}'),            # black replies
    ("knight to f6", '{"actions": [{"action": "opponent_move", "move": "Nf6"}]}'),  # black, verbatim
    ("castle kingside", '{"actions": [{"action": "opponent_move", "move": "O-O"}]}'),
    ("okay, your move", '{"actions": [{"action": "engine_move"}]}'),
    ("make it harder", '{"actions": [{"action": "set_difficulty", "difficulty": "hard"}]}'),
    ("set elo to 1600", '{"actions": [{"action": "set_difficulty", "difficulty": "1600"}]}'),
    ("let me play black", '{"actions": [{"action": "set_side", "color": "white"}]}'),
    ("you take white from here", '{"actions": [{"action": "set_side", "color": "white"}]}'),
    ("switch sides", '{"actions": [{"action": "set_side"}]}'),
    ("give me black and set difficulty to hard",
     '{"actions": [{"action": "set_side", "color": "white"}, '
     '{"action": "set_difficulty", "difficulty": "hard"}]}'),
    ("who is winning right now?", '{"actions": [{"action": "analyze", "question": "who is winning"}]}'),
    ("what's the best move here", '{"actions": [{"action": "analyze", "question": "best move"}]}'),
    ("let's start a new game", '{"actions": [{"action": "new_game"}]}'),
    ("take that back", '{"actions": [{"action": "undo"}]}'),
]

PHRASE_SYSTEM = """You are a friendly chess opponent speaking out loud. Answer the \
user's question in one or two short, natural sentences suitable for text-to-speech. \
Use ONLY the facts below — never invent an evaluation, move, or material count. \
Refer to moves simply (e.g. "knight to f3").

Facts:
{facts}"""


def _context_line(context: dict) -> str:
    machine = context.get("machine_color", "?")
    user = {"white": "black", "black": "white"}.get(machine, "?")
    return (f"you (the robot) play {machine}, the human plays {user}, "
            f"{context.get('turn', '?')} to move, "
            f"difficulty {context.get('difficulty', '?')}.")


def build_intent_messages(transcript: str, context: dict) -> list[dict]:
    messages = [
        {"role": "system", "content": INTENT_SYSTEM.format(context_line=_context_line(context))}
    ]
    for user, assistant in INTENT_EXAMPLES:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": transcript})
    return messages


def facts_to_text(facts: PositionFacts) -> str:
    lines = [f"evaluation: {facts.get('verdict', 'unknown')}"]
    mat = facts.get("material", {})
    if mat:
        lines.append(
            f"material: white {mat.get('white')} vs black {mat.get('black')} "
            f"(difference {mat.get('diff')} in "
            f"{'white' if mat.get('diff', 0) >= 0 else 'black'}'s favor)"
        )
    best = facts.get("best_move_san")
    if best:
        line = facts.get("pv_sans") or [best]
        lines.append("best line: " + " ".join(line))
    turn = facts.get("turn", "?")
    lines.append(f"{turn} to move" + (", and is in check" if facts.get("in_check") else ""))
    lines.append(f"phase: {facts.get('phase', 'unknown')}")
    if facts.get("last_move_san"):
        lines.append(f"last move played: {facts['last_move_san']}")
    return "\n".join(lines)


def build_analysis_messages(question: str, facts: PositionFacts) -> list[dict]:
    return [
        {"role": "system", "content": PHRASE_SYSTEM.format(facts=facts_to_text(facts))},
        {"role": "user", "content": question or "How does the position look?"},
    ]


# Proactive reaction to a move just played (blunder/best/brilliant + tactics).
COMMENT_SYSTEM = """You are a chess coach reacting OUT LOUD to a move, for a {tier} player. \
Speak {length}, natural and suitable for text-to-speech. Use ONLY the facts below — never \
invent a tactic, evaluation, or piece location, and don't restate the move's notation.
{teach}
Facts:
- move played: {san} by {mover}
- assessment: {label}
- tactics found: {motifs}"""

_LABEL_WORD = {
    "blunder": "a blunder (loses material or the advantage)",
    "mistake": "a mistake (clearly inferior)",
    "best": "the engine's top choice",
    "great": "the only move that holds the position",
    "brilliant": "a brilliant, sound sacrifice",
    "normal": "a reasonable move",
    "forced": "forced",
}
_TIER_LENGTH = {"easy": "one or two short sentences", "medium": "one short sentence",
                "hard": "one terse sentence, expert tone"}
_TIER_TEACH = {"easy": "If you name a tactic (fork, pin, outpost), briefly explain what it means.",
               "medium": "", "hard": ""}


def build_move_comment_messages(info: dict) -> list[dict]:
    tier = info.get("tier", "medium")
    motifs = info.get("motifs") or []
    system = COMMENT_SYSTEM.format(
        tier=tier,
        length=_TIER_LENGTH.get(tier, _TIER_LENGTH["medium"]),
        teach=_TIER_TEACH.get(tier, ""),
        san=info.get("san", "the move"),
        mover=info.get("mover", "a player"),
        label=_LABEL_WORD.get(info.get("label", "normal"), "a move"),
        motifs="; ".join(motifs) if motifs else "none",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Give your spoken reaction now."},
    ]
