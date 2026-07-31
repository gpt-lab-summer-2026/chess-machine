from chessmachine.nlu.prompts import (
    INTENT_EXAMPLES,
    INTENT_SYSTEM,
    build_intent_messages,
    build_move_comment_messages,
)


def test_examples_are_trimmed():
    assert len(INTENT_EXAMPLES) <= 9        # was 16


def test_examples_cover_piece_questions_as_analyze():
    threat = next((a for u, a in INTENT_EXAMPLES if "threat" in u), None)
    placed = next((a for u, a in INTENT_EXAMPLES if "well placed" in u), None)
    assert threat is not None and '"analyze"' in threat
    assert placed is not None and '"analyze"' in placed


def test_system_prompt_forbids_question_as_move():
    assert "never a move" in INTENT_SYSTEM.lower()


def test_system_prompt_marks_state_line_as_background():
    """The State line must be flagged as background so the model stops echoing
    "difficulty medium" / "you play black" back as set_difficulty / set_side."""
    s = INTENT_SYSTEM.lower()
    assert "state:" in s and "background" in s


def test_system_prompt_forbids_extra_engine_move_after_a_move():
    assert "do not also add an engine_move" in INTENT_SYSTEM.lower()


def test_live_turn_carries_state_and_user_words():
    msgs = build_intent_messages("e4", {"machine_color": "black", "turn": "white",
                                        "difficulty": "medium"})
    content = msgs[-1]["content"]
    assert content.startswith("State:")
    assert "User said: e4" in content


def test_build_intent_messages_still_ends_with_user_turn():
    msgs = build_intent_messages("e4", {"machine_color": "black", "turn": "white",
                                         "difficulty": "medium"})
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"].endswith("e4")


def test_intent_prefix_is_identical_across_turns():
    """The system+few-shot prefix must not vary, or llama-server's prompt cache
    misses and re-prefills ~774 tokens at ~15 tok/s. The volatile context line
    belongs in the last user turn only."""
    white = build_intent_messages("e4", {"machine_color": "black", "turn": "white",
                                         "difficulty": "medium"})
    black = build_intent_messages("e5", {"machine_color": "white", "turn": "black",
                                         "difficulty": "hard"})
    assert white[:-1] == black[:-1]          # prefix byte-identical
    assert "{context_line}" not in INTENT_SYSTEM
    assert "Context:" not in INTENT_SYSTEM
    # and the context still actually reaches the model
    assert "black" in white[-1]["content"] and "medium" in white[-1]["content"]


def test_intent_system_json_braces_are_literal():
    """INTENT_SYSTEM is no longer .format()ted, so its braces must be single."""
    assert '{"actions": [ ... ]}' in INTENT_SYSTEM
    assert "{{" not in INTENT_SYSTEM


def test_comment_prompt_attributes_machine_move_first_person():
    msgs = build_move_comment_messages(
        {"label": "blunder", "mover_is_machine": True, "san": "Qd1", "tier": "medium"})
    system = msgs[0]["content"].lower()
    assert "you are the robot" in system and "first person" in system


def test_comment_prompt_addresses_human_move():
    msgs = build_move_comment_messages(
        {"label": "blunder", "mover_is_machine": False, "san": "Qd1", "tier": "medium"})
    assert "the human" in msgs[0]["content"].lower()
