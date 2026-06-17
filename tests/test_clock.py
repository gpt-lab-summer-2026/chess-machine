"""The one-sided match clock: charges the human's think time only."""
from chessmachine.clock import MatchClock


def _fake_clock():
    t = {"v": 0.0}
    return t, MatchClock(now=lambda: t["v"])


def test_accumulates_only_while_running():
    t, clk = _fake_clock()
    clk.start_user(); t["v"] = 5.0; clk.stop_user()
    assert clk.user_seconds == 5.0
    # Machine's turn (clock stopped): elapsed time is NOT charged to the user.
    t["v"] = 20.0
    clk.start_user(); t["v"] = 23.0; clk.stop_user()
    assert clk.user_seconds == 8.0          # 5 + 3, the 12s machine turn excluded


def test_double_start_is_idempotent():
    t, clk = _fake_clock()
    clk.start_user(); t["v"] = 2.0; clk.start_user()   # second start ignored
    t["v"] = 5.0
    assert clk.stop_user() == 5.0
    assert clk.user_seconds == 5.0


def test_stop_without_start_is_noop():
    _, clk = _fake_clock()
    assert clk.stop_user() == 0.0
    assert clk.user_seconds == 0.0


def test_format_seconds():
    assert MatchClock.format_seconds(5) == "5 seconds"
    assert MatchClock.format_seconds(1) == "1 second"
    assert MatchClock.format_seconds(65) == "1 minute 5 seconds"
    assert MatchClock.format_seconds(120) == "2 minutes"
    assert MatchClock.format_seconds(61) == "1 minute 1 second"
