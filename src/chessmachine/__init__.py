"""chessmachine — voice-controlled, self-actuating chess board.

Pipeline:  audio -> STT -> SLM (intent) -> {Stockfish, motion} -> TTS -> audio

The SLM is used only for natural-language *understanding* (intent + slots) and
*generation* (phrasing answers). All chess truth (legality, evaluation, best
moves) comes from python-chess + Stockfish, never from the language model.
"""

__version__ = "0.1.0"
