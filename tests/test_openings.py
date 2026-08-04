from chessmachine.chess_engine.openings import _norm, identify_opening


def test_identifies_common_openings():
    assert identify_opening(["e4", "c5"]) == ("the Sicilian Defense", 2)
    assert identify_opening(["e4", "e5", "Nf3", "Nc6", "Bb5"]) == ("the Ruy Lopez", 5)
    assert identify_opening(["d4", "d5", "c4", "e6"]) == ("the Queen's Gambit Declined", 4)


def test_longest_matching_line_wins():
    najdorf = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]
    assert identify_opening(najdorf) == ("the Sicilian, Najdorf Variation", 10)
    # a shallow prefix still resolves only to the shallower name
    assert identify_opening(["e4", "c5", "Nf3"]) == ("the Sicilian Defense", 2)


def test_check_and_annotation_marks_are_stripped():
    assert _norm("Bb5+") == "Bb5"
    assert _norm("Qxf7#") == "Qxf7"
    assert _norm("Nf3!?") == "Nf3"
    # a check-annotated history still matches
    assert identify_opening(["e4", "e5", "Nf3+", "Nc6", "Bb5"]) == ("the Ruy Lopez", 5)


def test_offbeat_or_empty_is_unnamed():
    assert identify_opening(["a4", "h5"]) is None
    assert identify_opening([]) is None
    assert identify_opening(["e4"]) == ("the King's Pawn opening", 1)  # depth 1; caller ignores <2
