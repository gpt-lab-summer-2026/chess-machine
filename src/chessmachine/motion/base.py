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
    def move_xz(self, x_mm: float, z_mm: float, feed: int | None = None) -> None:
        """Move the head to planar (x,z) mm (pivot frame). Blocks until done."""

    @abstractmethod
    def set_pulley(self, height_mm: float, feed: int | None = None) -> None:
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

    def pick_dip(self, steps: int) -> None:
        """Dip the winch `steps` PAST the current pick depth to guarantee contact
        when the calibrated pick height is slightly high, then hold at that depth.
        Default no-op; backends with absolute winch control (stepmap) override it."""

    def set_pulley_for_drop(self, height_mm: float, feed: int | None = None) -> None:
        """Lower for a DROP (release). Default = same as set_pulley; backends with an
        absolute winch (stepmap) stop short of the mapped depth so the piece is set
        down from just above the board rather than pressed into it."""
        self.set_pulley(height_mm, feed)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
