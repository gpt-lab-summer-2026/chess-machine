"""The winch's lateral offset correction in the serial transport.

`SerialMotion._magnet_to_cart` maps a desired MAGNET position to the CART's polar
target, since the magnet hangs a fixed distance to the side of the arm. These
tests are pure math (no hardware / no pyserial)."""
import math

import pytest

from chessmachine.motion.serial_esp32 import SerialMotion


def _cart_to_magnet(r: float, a_deg: float, offset: float) -> tuple[float, float]:
    """Forward model: where an offset magnet lands for a cart at (r, a_deg).

    Cart rides the arm ray; the magnet sits `offset` along the arm's right
    perpendicular (sin a, -cos a)."""
    a = math.radians(a_deg)
    cx, cz = r * math.cos(a), r * math.sin(a)
    return cx + offset * math.sin(a), cz - offset * math.cos(a)


def test_offset_zero_is_plain_polar():
    r, a = SerialMotion._magnet_to_cart(200.0, 50.0, 0.0)
    assert r == pytest.approx(math.hypot(200.0, 50.0))
    assert a == pytest.approx(math.degrees(math.atan2(50.0, 200.0)))


def test_offset_roundtrip_places_magnet_on_target():
    offset = 33.0
    # a few board squares + a graveyard-ish point
    for x, z in [(200.0, 0.0), (282.5, 94.5), (93.5, -94.5), (150.0, 60.0)]:
        r, a = SerialMotion._magnet_to_cart(x, z, offset)
        assert _cart_to_magnet(r, a, offset) == pytest.approx((x, z), abs=1e-6)


def test_offset_shrinks_radius_and_leads_angle():
    x, z, offset = 200.0, 0.0, 33.0
    r, a = SerialMotion._magnet_to_cart(x, z, offset)
    m = math.hypot(x, z)
    assert r == pytest.approx(math.sqrt(m * m - offset * offset))  # cart is closer
    assert a == pytest.approx(math.degrees(math.asin(offset / m)))  # arm leads target


def test_negative_offset_leads_the_other_way():
    _, a_pos = SerialMotion._magnet_to_cart(200.0, 0.0, 33.0)
    _, a_neg = SerialMotion._magnet_to_cart(200.0, 0.0, -33.0)
    assert a_pos == pytest.approx(-a_neg)


def test_target_closer_than_offset_falls_back_to_plain():
    r, a = SerialMotion._magnet_to_cart(10.0, 0.0, 33.0)
    assert (r, a) == pytest.approx((10.0, 0.0))
