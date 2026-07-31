"""Prompt construction for the SLM (intent classification + answer phrasing)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chess_engine.analysis import PositionFacts
INTENT_SYSTEM = """You control a voice-operated chess robot. Convert the user's \
utterance into JSON of the form {"actions": [ ... ]} and output nothing else. \
Output ONLY the actions the user EXPLICITLY asked for — usually EXACTLY ONE, in \
order. Do NOT invent actions. A compound request like "give me black and make it \
harder" becomes two; a single move is exactly one action.

Actions (each object has an "action" plus only the fields it needs):
- "opponent_move": the human states THEIR move. Put it in "move" as full-coordinate \
UCI naming BOTH squares (e2e4, g8f6) so it is never ambiguous; fall back to SAN \
(Nf3, O-O) only if you cannot tell the origin square. Do not substitute, "correct", \
or invent a different move; the human may be playing black (e.g. they say "e5", \
"c5", "knight f6"). When the human states their move, output ONLY that one \
opponent_move — do NOT also add an engine_move; the robot makes its own reply \
automatically.
- "engine_move": the user asks YOU (the robot) to make your move ("your move", "you go").
- "set_difficulty": put easy, medium, hard, or an Elo number in "difficulty".
- "set_side": the user chooses who plays which color, mid-game. Put the color \
YOU (the robot) will play in "color" as "white" or "black"; for "switch sides" \
omit "color". Remember: if the user wants a color for THEMSELVES, you take the \
other one.
- "analyze": a question about the position (who is winning, best move, threats, \
evaluation). Put the question text in "question".
- "recalibrate": re-home the crane to correct mechanical drift (user says "re-home", \
"recenter", "recalibrate", "home the crane").
- "new_game", "undo", "resign", "status", "repeat", "help", "chitchat".

Only fill "move", "difficulty", or "question" when relevant; otherwise omit them.

The final turn carries a "State:" line describing the current game (which color \
you play, whose turn it is, the difficulty). It is BACKGROUND for reference only, \
NEVER a command: do not emit set_side or set_difficulty because of it — only when \
the user's OWN words ask to change the color or the difficulty.

A question — asking what/which/why/how, or whether a piece or move is good, \
safe, or threatened, or who is winning — is NEVER a move; use "analyze", never \
"opponent_move"."""

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
    """System prompt + few-shots, then the live turn.

    KEEP THE PREFIX BYTE-IDENTICAL. `cache_prompt` lets llama-server reuse the KV
    for the ~700-token system+few-shot prefix, and prefill is the whole cost here:
    774 tokens in to emit well under 96 out, at ~13-16 tok/s prefill on the Pi 5.

    The context line therefore rides in the FINAL user turn, not in the system
    message. It used to sit at the end of INTENT_SYSTEM, i.e. UPSTREAM of the eight
    examples — so every time the side-to-move flipped (every turn) it invalidated
    everything after it and forced 292 of 774 tokens to be re-prefilled. Measured
    on this box: same-context calls shared 765/775 tokens, a turn flip only 482.
    Trailing context also reads better to the model, being closest to the question.
    """
    messages = [{"role": "system", "content": INTENT_SYSTEM}]
    for user, assistant in INTENT_EXAMPLES:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    # The State line is labelled and explicitly called out as background in the
    # system prompt, and the user's words sit on their own "User said:" line. A 3B
    # model was otherwise reading the state ("difficulty medium", "you play black")
    # as commands and emitting spurious set_difficulty/set_side on nearly every turn.
    messages.append({"role": "user",
                     "content": f"State: {_context_line(context)}\nUser said: {transcript}"})
    return messages


def facts_to_text(facts: PositionFacts) -> str:
    lines: list[str] = []
    sq = facts.get("square")
    if sq:
        lines.append(f"the question is about the {sq['color']} {sq['piece']} on {sq['square']}:")
        lines.append("  attacked by: " + (", ".join(sq["attackers"]) or "nothing"))
        lines.append("  defended by: " + (", ".join(sq["defenders"]) or "nothing"))
        flags = []
        if sq["is_hanging"]:
            flags.append("hanging (attacked and undefended)")
        if sq["is_pinned"]:
            flags.append("pinned to its king")
        if sq["is_outpost"]:
            flags.append("on a strong outpost")
        lines.append("  status: " + ("; ".join(flags) if flags else "not under attack"))
        lines.append(f"  it controls {sq['controls']} squares")
    lines.append(f"evaluation: {facts.get('verdict', 'unknown')}")
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
Speak {length}, natural and suitable for text-to-speech.

Ground every claim in the facts below — never invent a tactic, evaluation, or piece \
location — but say it in your OWN words. Do not quote the assessment wording back; \
paraphrase it. Vary how you open: react to the idea, the threat, the material, the \
plan, or what it means for the next few moves. Don't restate the move's notation, \
don't start consecutive comments the same way, and avoid stock phrases like "holds \
the position".

Pick the ONE thing most worth saying — usually the consequence for the player, not a \
label. If a better move is listed, naming it and why is more useful than a verdict.

State the assessment with confidence: it is the engine's verdict, not your guess. \
Never hedge with "I'm not sure", "it's hard to say", or "potentially" — if the facts \
say the move is strong, say so plainly.
{attribution}{teach}
Facts:
- move played: {san} by {mover_desc}
- assessment: {label}
- tactics found: {motifs}{extra}"""

# Deliberately plain descriptions, not quotable phrases: the model used to parrot
# these verbatim (every reaction became "the only move that holds the position"),
# so they now state the fact and leave the wording to the model.
_LABEL_WORD = {
    "blunder": "a serious error — it throws away material or the advantage",
    "mistake": "clearly inferior — there was something much better",
    "best": "what the engine would pick too",
    "great": "essentially the only move that keeps the position together",
    "brilliant": "a sacrifice that genuinely works",
    "normal": "a reasonable move",
    "forced": "the only legal option",
}
_TIER_LENGTH = {"easy": "one or two short sentences", "medium": "one short sentence",
                "hard": "one terse sentence, expert tone"}
_TIER_TEACH = {"easy": "If you name a tactic (fork, pin, outpost), briefly explain what it means.",
               "medium": "", "hard": ""}


def build_move_comment_messages(info: dict) -> list[dict]:
    tier = info.get("tier", "medium")
    motifs = info.get("motifs") or []
    # Correct attribution: react in the first person to the robot's OWN move, and
    # address the human for theirs — so it never blames the human for its blunder.
    if info.get("mover_is_machine"):
        mover_desc = "you, the robot"
        attribution = ("The move was YOURS — you are the robot; react in the first person "
                       '("I", "my"), not as if the human made it. ')
        if info.get("punished"):
            attribution += ("The opponent just punished it, so own it plainly and say what "
                            "it cost you — don't be defensive or repeat yourself. ")
    else:
        mover_desc = "the human opponent"
        attribution = 'The move was the human\'s — address them as "you". '
    # Optional grounded detail. These are what let the model say something specific
    # instead of falling back on the assessment wording every time; each is omitted
    # when unknown so the model can never talk about a fact it wasn't given.
    extra = ""
    if info.get("best_san"):
        extra += f"\n- the engine would have played instead: {info['best_san']}"
    if info.get("cp_loss"):
        extra += f"\n- it costs about {info['cp_loss'] / 100:.1f} pawns of evaluation"
    if info.get("move_kind"):
        extra += f"\n- this move: {info['move_kind']}"
    if info.get("material"):
        extra += f"\n- material now: {info['material']}"
    if info.get("phase"):
        extra += f"\n- phase: {info['phase']}"
    system = COMMENT_SYSTEM.format(
        tier=tier,
        length=_TIER_LENGTH.get(tier, _TIER_LENGTH["medium"]),
        attribution=attribution,
        teach=_TIER_TEACH.get(tier, ""),
        san=info.get("san", "the move"),
        mover_desc=mover_desc,
        extra=extra,
        label=_LABEL_WORD.get(info.get("label", "normal"), "a move"),
        motifs="; ".join(motifs) if motifs else "none",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Give your spoken reaction now."},
    ]
