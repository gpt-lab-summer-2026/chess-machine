"""Abstract motor-control transport.

A backend exposes only low-level primitives (home / move / pulley / magnet).
All chess-aware sequencing lives in `choreography.Choreographer`, so swapping
ESP32-over-serial for a mock (or a future transport) changes nothing upstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class MotionController(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Open the transport and verify the controller responds."""

    @abstractmethod
    def close(self) -> None:
        """Release the transport."""

    @abstractmethod
    def home(self) -> None:
        """Home all axes against their endstops; defines (0,0) and zero height."""

    @abstractmethod
    def move_xz(self, x_mm: float, z_mm: float, feed: int | None = None) -> None:
        """Move the gantry to (x,z) in mm. Blocks until the move completes."""

    @abstractmethod
    def set_pulley(self, height_mm: float, feed: int | None = None) -> None:
        """Raise/lower the electromagnet to `height_mm` above the board."""

    @abstractmethod
    def magnet(self, on: bool) -> None:
        """Energize (True) or release (False) the electromagnet."""

    @abstractmethod
    def status(self) -> dict:
        """Return {x, z, height, magnet, ...} as reported by the controller."""

    def estop(self) -> None:
        """Emergency stop. Backends should override; default is a no-op."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
