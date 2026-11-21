import numpy as np

from scripts.experimental.tt18_surface import SurfaceMemory
from scripts.experimental.tt18_surface_report import bounces


def test_measured_bounce_influences_nearby_trajectories_and_survives_reload():
    learned = SurfaceMemory(.12, 8., [[-.3,-.6,.55], [.3,-.6,.96]] * 4)
    assert learned.predict([-.3, -.6], .86) < .62
    assert learned.predict([.3, -.6], .86) > .92
    twin = SurfaceMemory(**learned.state())
    np.testing.assert_equal(twin.predict([-.3,-.6],.86),learned.predict([-.3,-.6],.86))


def test_diagnostic_uses_departure_speed_only_for_an_incoming_near_side_bounce():
    events = [
        dict(kind='bounce', side=1, incoming=[0.,-2.,-3.], step=8, location=[-.22,-.8,0.]),
        dict(kind='separation', collider='table', step=9, velocity=[0.,-1.8,2.4]),
        dict(kind='bounce', side=-1, incoming=[0.,2.,-3.], step=15, location=[.2,.8,0.]),
        dict(kind='separation', collider='table', step=16, velocity=[0.,1.8,2.4]),
    ]
    [(tick, xy, ratio)] = list(bounces(events))
    assert (tick, xy) == (8, [-.22, -.8])
    np.testing.assert_allclose(ratio,.8)
