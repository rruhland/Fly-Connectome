from contextlib import closing

import numpy as np

from scripts.experimental.tt24_world_prior import WEIGHTS
from scripts.experimental.tt24_world_prior_body import AlignedWorlds, SCALE, identify
from scripts.experimental.tt24_world_prior_train import data


def test_ball_free_alignment_recovers_delay_and_ignores_audit_columns(tmp_path):
    rng = np.random.default_rng(445)
    excitation = np.repeat(rng.uniform(-.7, .7, (6, 14, 5)), 6, axis=1)
    seen = np.zeros_like(excitation)
    for t in range(2, excitation.shape[1] - 1):
        current = seen[:, t]
        prior = current - seen[:, t - 1]
        seen[:, t + 1] = current + .16 * (excitation[:, t - 2] - current) + .4 * prior
    path = tmp_path / 'public_motor_play.npz'
    np.savez(path, goal=excitation * SCALE, measured=seen * SCALE, true_diagnostic=np.full_like(seen, np.nan))
    bank, report = identify(path, tmp_path / 'body.json')
    assert report['selected']['delay'] == 2
    assert report['withheld_runs'] == [4, 5]
    assert report['held_model']['18']['normalized'] < .01

    with closing(AlignedWorlds(bank, workers=1, frames=100)) as world:
        zero = np.zeros(WEIGHTS)
        starts = data(810, 6)
        played = world.play(zero, starts)
        assert played.shape == (1, 6, 6)
        assert np.isfinite(played).all()
        np.testing.assert_array_equal(played, world.play(zero, starts))
