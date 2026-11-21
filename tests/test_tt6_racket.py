import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt4_arena import TT4Arena
from scripts.experimental.tt6_racket import (
    DRAG,
    FRAME_DT,
    OpticalImpact,
    RacketAdaptation,
    optical_impact,
    practiced_gain,
    set_racket_response,
)


def flight(incoming, outgoing):
    path = []
    for tick in range(-5, 6):
        velocity = -incoming if tick < 0 else outgoing
        path.append((0., -1.49 + FRAME_DT * tick * velocity, .35) if abs(tick) > 1 else None)
    return path


def test_optical_impact_tracks_a_reversal_through_occlusion():
    impact = optical_impact(flight(2.1, 3.2))
    assert impact.tick == 5
    assert impact.incoming == pytest.approx(2.1)
    assert impact.outgoing == pytest.approx(3.2)
    assert optical_impact([None] * 20) is None


def test_varied_practice_changes_the_correction_rate_from_identical_evidence():
    reference = OpticalImpact(10, 3., 3.)
    ordinary = practiced_gain([(reference, reference)])
    varied = practiced_gain([(reference, OpticalImpact(10, 3., 1.6))])
    assert ordinary['gain'] < varied['gain']
    learners = [RacketAdaptation(gain=data['gain']) for data in (ordinary, varied)]
    frozen = RacketAdaptation(gain=varied['gain'], enabled=False)
    for learner in [*learners, frozen]:
        learner.finish(flight(2.1, 4.4), 0.)
        learner.finish(flight(2.0, 4.5), 1.)
        learner.finish(flight(2.1, 2.8), -1.)
    assert frozen.speed == pytest.approx(.049)
    assert frozen.speed < learners[0].speed < learners[1].speed
    previous = learners[1].speed
    assert learners[1].finish([None] * 20, -1.)['next_speed'] == previous


def test_racket_intervention_changes_only_learner_contact_damping():
    arena = TT4Arena()
    try:
        solref = arena.model.geom_solref.copy()
        geom = arena.model.geom('learner_racket')
        set_racket_response(arena, 'medium')
        solref[geom.id, 1] = -135.
        np.testing.assert_array_equal(arena.model.geom_solref, solref)
        with pytest.raises(ValueError, match='unknown racket response'):
            set_racket_response(arena, 'unrecognized')
        set_racket_response(arena, 'stock')
        assert geom.solref[1] == -25.
    finally:
        arena.close()


def test_first_changed_impact_preserves_the_reference_after_camera_occlusion():
    learner = RacketAdaptation(gain=.73)
    assert learner.finish([None] * 20, 0.)['reference'] is None
    baseline = learner.finish(flight(2.1, 4.4), 0.)['reference']
    changed = learner.finish(flight(2.1, 2.8), -1.)
    assert changed['reference'] == baseline
    assert changed['next_speed'] > changed['used_speed']


def test_delayed_outgoing_observations_can_recover_the_first_impact():
    path = []
    for tick in range(-5, 11):
        time = FRAME_DT * tick
        if tick < 0:
            path.append((0., -1.39 - 2.1 * time, .35))
        elif tick < 6:
            path.append(None)
        else:
            distance = np.log1p(DRAG * 4.8 * time) / DRAG
            path.append((0., -1.39 + distance, .35))
    assert optical_impact(path) is None
    found = optical_impact(path, reacquisition=True)
    assert found.tick == 5
    assert found.incoming == pytest.approx(2.1)
    assert found.outgoing == pytest.approx(4.8)
