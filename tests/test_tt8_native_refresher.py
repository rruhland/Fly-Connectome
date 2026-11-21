import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt8_native_refresher import _revalidate


def test_native_promotion_needs_independent_paired_public_scores():
    def outcomes(challenger):
        return [dict(match={'serve': start}, point=point, relation=choice, score=score)
                for start in range(8) for point in (1, 2)
                for choice, score in ((2, 0), (3, challenger))]

    assert _revalidate(outcomes(1), 3, 2, seed=8) == dict(starts=8, mean=1., lower90=1., promoted=True)
    assert not _revalidate(outcomes(0), 3, 2, seed=8)['promoted']
    assert not _revalidate(outcomes(-1), 3, 2, seed=8)['promoted']


def test_native_promotion_rejects_duplicate_policy_trials():
    row = dict(match={'serve': 42}, point=1, relation=2, score=1)
    with pytest.raises(ValueError, match='duplicate'):
        _revalidate([row, row], 3, 2, seed=8)
