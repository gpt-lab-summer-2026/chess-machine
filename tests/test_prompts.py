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


def test_build_intent_messages_still_ends_with_user_turn():
    msgs = build_intent_messages("e4", {"machine_color": "black", "turn": "white",
                                         "difficulty": "medium"})
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "e4"}


def test_comment_prompt_attributes_machine_move_first_person():
    msgs = build_move_comment_messages(
        {"label": "blunder", "mover_is_machine": True, "san": "Qd1", "tier": "medium"})
    system = msgs[0]["content"].lower()
    assert "you are the robot" in system and "first person" in system


def test_comment_prompt_addresses_human_move():
    msgs = build_move_comment_messages(
        {"label": "blunder", "mover_is_machine": False, "san": "Qd1", "tier": "medium"})
    assert "the human" in msgs[0]["content"].lower()
