from chessmachine.nlu.prompts import INTENT_EXAMPLES, INTENT_SYSTEM, build_intent_messages


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
