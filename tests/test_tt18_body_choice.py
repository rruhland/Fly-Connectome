import numpy as np

from scripts.experimental.tt18_body_choice import BodyChoice
from scripts.experimental.tt18_online import blank_body


def test_checkpoint_keeps_comparisons_for_the_next_point():
    learner = BodyChoice(blank_body(4), seed=9)
    learner.window = [[.06, .055, .010], [.08, .06, .012]]
    resumed = BodyChoice.restore(learner.state())
    for model in (learner, resumed):
        model.select()
        model.begin_point()
        assert model.active == 'relearn'
        assert model.window == []
        np.testing.assert_equal(model.explore(np.ones(4),np.zeros(4),None), np.ones(4))
    assert learner.state() == resumed.state()
