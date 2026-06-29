"""Abstract motor-control transport.

A backend exposes only low-level primitives (home / move / pulley / magnet).
All chess-aware sequencing lives in `choreography.Choreographer`, so swapping
ESP32-over-serial for a mock (or a future transport) changes nothing upstream.

Positions are planar (x,z) millimetres in the crane-pivot frame; a backend may
realize them however it likes (the serial backend converts to polar radial +
rotary axes for the crane).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class MotionController(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Open the transport and verify the controller responds."""

    @abstractmethod
    def close(self) -> None:
        """Release the transport."""

    @abstractmethod
    def home(self) -> None:
        """Home all axes against their endstops; defines the zero reference."""

    @abstractmethod
    def move_xz(self, x_mm: float, z_mm: float, feed: Optional[int] = None) -> None:
        """Move the head to planar (x,z) mm (pivot frame). Blocks until done."""

    @abstractmethod
    def set_pulley(self, height_mm: float, feed: Optional[int] = None) -> None:
        """Raise/lower the electromagnet to `height_mm` above the board."""

    @abstractmethod
    def magnet(self, on: bool) -> None:
        """Energize (True) or release (False) the electromagnet."""

    @abstractmethod
    def status(self) -> dict:
        """Return controller state as a dict (mock: x/z; serial: r/a + height,
        magnet, endstops)."""

    def estop(self) -> None:
        """Emergency stop. Backends should override; default is a no-op."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
