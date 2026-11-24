import pytest

from scripts.experimental.tt39_opportunity_map import projected_height


def test_projected_height_uses_inbound_time():
    crossing = dict(xyz=[0.0, -1.1, 1.0], velocity=[1.0, -4.0, 0.5])
    t = 0.43 / 3.92
    assert projected_height(crossing) == pytest.approx(1.0 + t * 0.5 - 4.905 * t * t)
    crossing["velocity"][1] = 1.0
    with pytest.raises(ValueError):
        projected_height(crossing)
