"""Console output helpers for the voice pipeline."""
from __future__ import annotations

import sys


def safe_print(text: str) -> None:
    """Print `text`, replacing characters the console can't encode rather than
    raising `UnicodeEncodeError`.

    The voice backends echo arbitrary text — a whisper transcript or an SLM
    reply — which may contain non-ASCII (em dashes, smart quotes, accents). On a
    non-UTF-8 console (e.g. Windows cp1252) or when stdout is redirected to a
    file, a bare `print` of that would raise and kill the listen loop.
    """
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc))
