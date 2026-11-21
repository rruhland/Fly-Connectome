import numpy as np

from scripts.experimental.tt22_spatial import SpatialChoice, SpatialPlayer, broad_hypotheses, portable_contact
from scripts.experimental.tt22_spatial_trial import paired_point


def test_geometric_contact_and_relative_relocation():
    actor, mover, forecast = (-.12, .08, .113)
    assert np.isclose(portable_contact(actor, mover, forecast, (.1, 1., 1.)), forecast)
    gains = (.1, .65, -.40)
    goal = portable_contact(actor, mover, forecast, gains)
    assert np.isclose(portable_contact(actor+7, mover+7, forecast+7, gains), goal+7)
    assert np.isclose(portable_contact(-actor, -mover, -forecast, gains), -goal)


def test_scratch_and_score_permutation_share_original_signed_candidates():
    broad = broad_hypotheses(23)
    assert (np.min(broad, axis=0) < 0).all() and (np.max(broad, axis=0) > 0).all()
    finalists = [[0., 1.03, 1.12], [0., 1.01, 1.03], [0., .93, .98], [0., .96, 1.09]]
    native = dict(coefficients=finalists, totals=[7, 5, -4, -4], counts=[104]*4)
    modes = [SpatialChoice(broad, native, name, np.arange(10)) for name in ('pong', 'scratch', 'score_permuted')]
    assert [memory.choose() for memory in modes] == [6, 5, 8]
    for learner in modes:
        assert np.array_equal(learner.coefficients[:6], broad)
        previous = learner.choose()
        learner.update(previous, -1, 0)
        assert learner.count.sum() == 1
        assert learner.total.sum() == -1


def test_exact_pair_cache_does_not_share_mutable_policy_receipts():
    class Fixture:
        calls = 0

        def point(self, seed, lane, coefficients, *, flat, geometry):
            self.calls += 1
            return {'actions': [[.2, 0.]], 'rgb': 'native'}

    plant, cache = Fixture(), {}
    initial = paired_point(plant, cache, 'before', 10, .2, [0., 1., 1.])
    initial['actions'][0][0] = .8
    repeated = paired_point(plant, cache, 'equal', 10, .2, [0., 1., 1.])
    assert repeated['cached_from'] == 'before' and plant.calls == 1
    assert repeated['actions'] == [[.2, 0.]]
    paired_point(plant, cache, 'different', 10, .2, [0., 1., 1.], flat=True)
    assert plant.calls == 2


def test_direct_target_can_oppose_geometric_contact_and_has_a_neutral_fallback():
    player = SpatialPlayer.__new__(SpatialPlayer)
    player.direct, player.flat_center, player.geometric_override = True, False, False
    player.frames, player.actor_seen = 5, 5
    player.applied, player.neutral_fallbacks, player.max_correction = 0, 0, 0.
    player.actor, player.coefficients = [0., 0., 0.], [0., -1., -1.]
    center = [.30, -.9, .2]
    estimate = ([.20, -.5, .3], [0., -.08, -.02])
    assert np.isclose(player.lateral_contact(estimate, center), -.30)
    assert np.isclose(player.max_correction, .60)
    player.frames = 10
    assert player.lateral_contact(estimate, center) == 0.
    assert player.neutral_fallbacks == 1
    player.geometric_override = True
    assert player.lateral_contact(estimate, center) == .30
