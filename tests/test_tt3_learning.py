import numpy as np

from scripts.experimental.tt3_learning import ReturnMemory, TableLearner
from scripts.experimental.tt3_learning_stroke import stroke_learner_factory
from scripts.experimental.tt3_visual import Binding, RoleModel, Track


def test_rebinding_restores_early_visual_pose_and_replaces_wrong_motor_samples():
    learner = TableLearner(RoleModel.untrained(), probe_frames=4)

    class ChangingAssociation:
        def reset(self):
            pass

        def update(self, tick, _action):
            moving = Track(0, (.15 - .01 * tick, 0., 0.), (-.01, 0., 0.))
            controlled = Track(1, (-.30 + .01 * tick, 0., 0.), (.01, 0., 0.))
            return Binding(moving if tick < 10 else controlled, None, (moving, controlled))

    learner.binder = ChangingAssociation()
    learner.reset(option=2)
    for tick in range(10):
        learner.act(tick)
    assert np.isclose(learner.ready[0], .15)
    assert learner.motor.samples > 0
    assert np.all(learner.motor.responses[:learner.motor.samples, 0] < 0)

    learner.act(10)

    assert np.isclose(learner.ready[0], -.30)
    assert learner.motor.samples > 0
    assert np.all(learner.motor.responses[:learner.motor.samples, 0] > 0)


def test_outcome_permutation_preserves_the_observations_and_original_checkpoint():
    memory = ReturnMemory()
    for index in range(6):
        for _ in range(index + 1):
            memory.observe(index, 1. if index % 2 else -1.)
    before = memory.state()
    order = [2, 4, 0, 5, 1, 3]

    disrupted = ReturnMemory.restore(before).permuted(order)

    assert disrupted.counts.tolist() == [before['counts'][index] for index in order]
    assert disrupted.scores.tolist() == [before['scores'][index] for index in order]
    assert memory.state() == before


def test_combined_stroke_agent_initializes_both_learned_models():
    from scripts.experimental.tt3_flight import ForecastHistory
    from scripts.experimental.tt3_motor import IntegratedMotor

    factory = stroke_learner_factory(forecast=True, integrated_motor=True)
    learner = factory(RoleModel.untrained())
    learner.reset(option=0)

    assert isinstance(learner.motor, IntegratedMotor)
    assert isinstance(learner.flight, ForecastHistory)
    assert learner.flight.evidence is learner.observed_motion
