#!/usr/bin/env python3
"""Tune the TTS lead-in until the first syllable stops clipping on YOUR speaker.

    python scripts/tts_leadin_test.py

Bluetooth speakers power their amplifier down between utterances and swallow the
first ~350 ms when it wakes ("Chess machine ready" -> "ess machine ready"). The
right lead-in depends on the speaker, so this plays one phrase at several lead-in
lengths — pausing ~6 s before each so the amp idles exactly like it does during a
chess turn — and you pick the SMALLEST value that comes through clean.

It tries pure-silence lead-ins first, then two "primer" variants (inaudible
low-level noise) for amps that only wake on actual signal. Set the winner in
config/config.yaml as tts.lead_in_s (and tts.lead_in_primer: true if a primer
row was the first clean one).
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from chessmachine.config import load_config          # noqa: E402
from chessmachine.voice.audio import play             # noqa: E402
from chessmachine.voice.tts import KokoroTTS          # noqa: E402

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


def main() -> int:
    cfg = load_config("config/config.yaml")
    print(f"default sink should be the BT speaker. Loading Kokoro ({cfg.tts.model_path})...")
    tts = KokoroTTS(cfg.tts)
    samples, sr = tts.synth(PHRASE)
    print(f"\nPhrase: {PHRASE!r}. Listen for the leading 'Chess' each time.\n")
    for lead, primer in TRIALS:
        print(f"  idle {IDLE_S:.0f}s, then lead_in_s={lead:<4} primer={primer} ...", flush=True)
        time.sleep(IDLE_S)
        play(samples, sr, lead_in_s=lead, lead_in_primer=primer)
    print("\nPick the SMALLEST clean row and set it in config/config.yaml:")
    print("  tts:\n    lead_in_s: <value>\n    lead_in_primer: <true if a primer row won>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
