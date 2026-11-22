import numpy as np

from scripts.experimental.tt22_branch_motor import BranchMotor, MotorCue, OutcomeScorer


def example(origin, leg, winning):
    return dict(features=[origin]+[0.]*9, leg=leg,
                outcomes=[[i, i == winning, True, [0., .59] if i == winning else None] for i in (0, 1)])


def test_physical_outcome_contrasts_change_the_selected_motor_only_with_context():
    corpus = [example(-.3, 1, 0), example(.3, 1, 1), example(.3, 0, 0)]
    left, right = (np.array([origin]+[0.]*9) for origin in (-.3, .3))
    conditional = OutcomeScorer(corpus)
    baseline = OutcomeScorer(corpus, contextual=False)
    motors, errors = np.arange(2), np.zeros(2)
    assert np.argmax(conditional.ranking(left, True, motors, errors)) == 0
    assert np.argmax(conditional.ranking(right, True, motors, errors)) == 1
    assert np.argmax(conditional.ranking(right, False, motors, errors)) == 0
    assert np.array_equal(baseline.ranking(left, True, motors, errors), baseline.ranking(right, True, motors, errors))


def test_public_visual_gate_rearms_only_after_outbound_sightings():
    cue = MotorCue((1., 0.))
    cue.observe(0, [0., -.22, .11])
    estimate = cue.observe(1, [0., -.31, .11])
    assert estimate is not None
    cue.consume()
    assert cue.observe(2, [0., -.40, .11]) is None
    assert not cue.armed
    cue.observe(3, [0., .10, .20])
    cue.observe(4, [0., .17, .20])
    assert cue.armed
    assert cue.previous == 1
    cue.reset()
    assert cue.previous == 0 and cue.armed and not cue.history
    action = BranchMotor((1., 0.), OutcomeScorer([example(-.3, 0, 0)])).command(0, None)
    assert action.shape == (6,) and np.max(np.abs(action)) <= 1.
