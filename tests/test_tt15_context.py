from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt15_back_transfer import signed_fit
from scripts.experimental.tt15_context import MotorEvidence
from scripts.experimental.tt15_discovery import DiscoveryMemory


def test_geometry_selection_separates_rest_from_informative_movement():
    original = np.diag([.09, .1, .12])
    changed = original[:, [2, 0, 1]] * [-1, 1, -1]
    evidence = MotorEvidence([original, changed])
    motor = SimpleNamespace(requested=np.zeros(3), curve=np.ones(3), history=np.zeros((3, 3)))
    active = 0
    for tick in range(45):
        active = evidence.suggest(motor, np.zeros(3), active, tick)
        assert active == 0
    rng = np.random.default_rng(8739)
    for tick in range(45, 105):
        motor.requested += rng.normal(0., .14, 3)
        active = evidence.suggest(motor, changed @ motor.requested, active, tick)
    assert active == 1


def test_discovery_validates_on_later_motion_and_projects_new_port_directions():
    rng = np.random.default_rng(312709)
    native = np.diag([.08, .1, .11])
    order = [1, 2, 0]
    signs = [-1, 1, -1]
    changed = native[:, order] * signs
    positions = np.cumsum(rng.normal(0,.16,(70,3)),axis=0)
    times = np.arange(12, len(positions),2)
    movement = (positions[times]-positions[times-6]) @ changed.T
    _, held_error, jac = DiscoveryMemory._fit_delay(native, positions, times, movement,
                                                     len(times)//2, True)
    assert held_error < .01
    assert signed_fit(native, jac) == (order, signs)


def test_search_motion_fit_recovers_earlier_port_geometry_from_recent_displacements():
    from scripts.experimental.tt15_motion_transfer import match_motion

    rng = np.random.default_rng(991)
    original = rng.normal(0,.07,(3,4))
    order = [2,0,3,1]
    signs = [-1,1,1,-1]
    actions = rng.normal(0,.17,(150,4))
    positions = np.vstack([np.zeros(4),np.cumsum(actions,axis=0)])[:-1]
    seen = positions[np.maximum(0,np.arange(len(actions))-2)] @ (original[:,order]*signs).T
    fitted, error = match_motion(original,actions,seen,np.ones(4))
    assert fitted == (order,signs)
    assert error < 1e-8
