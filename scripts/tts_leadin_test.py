#!/usr/bin/env python3
"""Stop the BT speaker clipping the first syllable of every reply.

    python scripts/tts_leadin_test.py --keepalive   # RUN THIS ONE FIRST
    python scripts/tts_leadin_test.py               # lead-in lengths (keep_alive off)

Bluetooth speakers power their amplifier down between utterances and swallow the
first ~350 ms when it wakes ("Chess machine ready" -> "ess machine ready").

A silent lead-in often does NOT fix this, because silence is exactly what puts a
signal-detect amp to sleep — it naps straight through the pad. So the real fix is
`--keepalive`: stream inaudible noise continuously and the amp never sleeps, at no
per-utterance latency. That mode finds the quietest level that keeps it awake.

The default mode is the fallback for amps that DO wake on a silent link: it plays
one phrase at several lead-in lengths (idling ~6 s before each, like a real turn)
and you take the smallest value that comes through clean.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config                     # noqa: E402
from chessmachine.voice.audio import SpeakerKeepAlive, play     # noqa: E402
from chessmachine.voice.tts import KokoroTTS                    # noqa: E402

PHRASE = "Chess machine ready."
IDLE_S = 6.0        # let the amp sleep like it does while whisper/SLM run a turn
TRIALS = [
    (0.15, False),  # the old default — expect this to clip
    (0.35, False),
    (0.50, False),
    (0.80, False),
    (0.50, True),   # primer: inaudible noise wakes signal-detect amps
    (0.80, True),
]
# Quietest first: take the first level where the leading "Ch" is intact AND you
# cannot hear hiss in the gaps.
KEEP_ALIVE_LEVELS = [0.0005, 0.002, 0.005, 0.02]


def run_keepalive(samples, sr) -> int:
    print("\nKEEP-ALIVE SWEEP — for each level, listen for TWO things:")
    print("  (a) is the leading 'Ch' of 'Chess' intact?")
    print("  (b) can you hear hiss during the 6 s gap?")
    print("Take the QUIETEST level where (a) is yes and (b) is no.\n")
    print("  [baseline] keep-alive OFF, minimal lead-in — expect this to clip:", flush=True)
    time.sleep(IDLE_S)
    play(samples, sr, lead_in_s=0.05)
    for level in KEEP_ALIVE_LEVELS:
        ka = SpeakerKeepAlive(level=level)
        if not ka.start():
            print("  pw-play not found — keep-alive is unavailable on this box.")
            return 1
        try:
            print(f"\n  keep_alive_level={level} — {IDLE_S:.0f}s gap, then the phrase...",
                  flush=True)
            time.sleep(IDLE_S)
            play(samples, sr, lead_in_s=0.05)     # tiny lead-in: the amp should be awake
            time.sleep(0.5)
        finally:
            ka.stop()
    print("\nSet the winner in config/config.yaml:")
    print("  tts:\n    keep_alive: true\n    keep_alive_level: <value>\n    lead_in_s: 0.15")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keepalive", action="store_true",
                    help="sweep keep-alive levels (the real fix) instead of lead-in lengths")
    args = ap.parse_args()

    cfg = load_config("config/config.yaml")
    print(f"default sink should be the BT speaker. Loading Kokoro ({cfg.tts.model_path})...")
    tts = KokoroTTS(cfg.tts)
    samples, sr = tts.synth(PHRASE)

    if args.keepalive:
        return run_keepalive(samples, sr)

    print(f"\nPhrase: {PHRASE!r}. Listen for the leading 'Chess' each time.")
    print("(set tts.keep_alive: false for this mode to mean anything.)\n")
    for lead, primer in TRIALS:
        print(f"  idle {IDLE_S:.0f}s, then lead_in_s={lead:<4} primer={primer} ...", flush=True)
        time.sleep(IDLE_S)
        play(samples, sr, lead_in_s=lead, lead_in_primer=primer)
    print("\nPick the SMALLEST clean row and set it in config/config.yaml:")
    print("  tts:\n    lead_in_s: <value>\n    lead_in_primer: <true if a primer row won>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
