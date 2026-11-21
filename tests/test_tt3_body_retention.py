import numpy as np

from scripts.experimental.tt3_body_retention import ResponseConditionedMemory, response_rank
from scripts.experimental.tt3_learning import RELATIONS


def test_visual_response_context_keeps_shared_updates_without_replacing_source_values():
    source = dict(relations=list(RELATIONS), scores=[0., 0., 0., 0., 9., 0.], counts=[10] * len(RELATIONS))
    memory = ResponseConditionedMemory(source, observed_rank=1)
    memory.prepare_stroke([0., -1., 0.])
    broad = np.eye(3)
    narrow = np.array([[0.], [1.]])
    assert response_rank(broad) == 2
    assert response_rank(narrow) == 1
    memory.observe_response(broad)
    for _ in range(20):
        memory.observe(4, -1.)
        memory.observe(1, 1.)
    rng = np.random.default_rng(52)
    assert memory.choose_for_response(broad, rng) == 1
    assert memory.choose_for_response(narrow, rng) == 4
    assert memory.state()['return_memory']['counts'][4] == 30
    assert memory.state()['by_response_rank']['1'] == source
    assert memory.state()['strokes_by_response_rank'] == {'2': [0., -1., 0.]}
    recovered = ResponseConditionedMemory.restore(memory.state())
    assert recovered.state() == memory.state()

    arguments = (4, [0., 0.], [1., .1], [-1., .1], [0., 0.])
    target = recovered.goal_for_response(narrow, *arguments, response_axis=narrow[:, 0])
    np.testing.assert_allclose(target, [0., .23])
