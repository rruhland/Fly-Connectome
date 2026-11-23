from contextlib import closing

import numpy as np

from scripts.experimental.tt24_redundant import MarkerBounce, RedundantBounce


def test_distal_motor_is_independent_and_physically_pinnable():
    with closing(RedundantBounce(view='none', calibration=True)) as moving, closing(
            RedundantBounce(view='none', calibration=True, pinned=True)) as pinned:
        assert moving.action_axes[:5] == MarkerBounce.action_axes
        assert moving.action_size == pinned.action_size == 6
        first = moving.reset(seed=61155)
        locked = pinned.reset(seed=61155)
        assert first['joints'].shape == locked['joints'].shape == (6,)
        moving_at_start = first['joints'][5]
        locked_at_start = locked['joints'][5]
        command = np.r_[np.zeros(5), .27]
        for _ in range(10):
            first, reward, done = moving.step(command, render=False)
            locked, compared_reward, compared_done = pinned.step(command, render=False)
            assert reward == compared_reward == 0.
            assert not done and not compared_done
        assert first['joints'][5] - moving_at_start > .14
        assert abs(locked['joints'][5] - locked_at_start) < .002


def test_original_pitch_jams_at_each_seeded_reset_angle():
    with closing(RedundantBounce(view='none', calibration=True, jammed=True)) as jammed:
        for seed in (31611, 36291):
            obs = jammed.reset(seed=seed, split='held')
            expected = jammed.judge_metrics()['drop']['pose'][3]
            assert abs(obs['joints'][3] - expected) < .002
            for _ in range(11):
                obs, _, _ = jammed.step(np.array((0., 0., 0., -.67, 0., .22)), render=False)
            assert abs(obs['joints'][3] - expected) < .002
            assert obs['joints'][5] > .15
