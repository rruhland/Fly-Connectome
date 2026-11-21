import numpy as np

from scripts.experimental.tt22_spatial_context import ContextMemory, signature


def test_observed_motion_signature_ignores_origin_orientation_and_scale():
    increments = np.array([[1., .0], [1., .5], [0., .8], [-1., .0], [-1., -.25], [.1, -.7], [.7, 0.]])
    original = np.vstack((np.zeros((1, 2)), np.cumsum(increments, axis=0)))
    rotated = 4.*original[:, ::-1]*[-1., 1.] + [20., -102.]
    result = signature(list(enumerate(original)))
    transformed = signature(list(enumerate(rotated)))
    assert result['displacements'] == transformed['displacements'] == 7
    assert np.isclose(result['value'], transformed['value'])


def test_shared_store_retrieves_scores_by_observed_motion():
    prior = dict(coefficients=[[0., 1., 1.], [0., .9, -1.]], prior_total=[.50, 0.],
                 prior_count=[8, 1], ties=[0., 1.])
    store = ContextMemory(prior)
    body = dict(value=-.17, displacements=17)
    store.observe_context(1, 1, 2, body)
    stored = ContextMemory.restore(store.state())
    assert stored.choose_context(body) == 1
    side = dict(value=-6.2, displacements=7)
    assert stored.choose_context(side) == 0
    assert len(stored.evidence) == 1
