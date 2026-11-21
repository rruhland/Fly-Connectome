import numpy as np
import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt8_transfer_continuity import generic_controls
from scripts.experimental.tt8_transfer_model import RESTITUTION
from scripts.experimental.tt9_transfer_memory import retrieve


def test_scrambling_source_pairs_keeps_the_unconditional_catch_prior():
    generic = generic_controls()['generic_mixture']
    low, high = np.eye(len(RESTITUTION))[[10, 55]]
    examples = [dict(source=high, target=low), dict(source=low, target=high)]
    mismatched = [dict(source=low, target=low), dict(source=high, target=high)]
    selected, marginal = retrieve(examples, low, generic)
    shuffled, other_marginal = retrieve(mismatched, low, generic)
    assert selected @ RESTITUTION > shuffled @ RESTITUTION
    np.testing.assert_equal(marginal, other_marginal)
    np.testing.assert_allclose([selected.sum(), shuffled.sum(), marginal.sum()], 1.)
