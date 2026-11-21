"""Source reward and paired blind camera checks for the portable visual roles."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experimental import tt3_arena
from scripts.experimental.tt3_pong import source_episodes
from scripts.experimental.tt3_visual import RoleBinder, RoleModel


@pytest.fixture(scope='module')
def native_practice():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        practice = tuple(source_episodes(range(1000, 1024)))
    finally:
        torch.set_num_threads(threads)
    return practice, RoleModel.fit(practice)


def test_relevant_role_requires_visual_alignment_with_source_reward(native_practice):
    practice, model = native_practice
    assert model.source_control_examples > 20
    assert model.source_reward_examples > 20
    relocated = []
    for episode in practice:
        anchor = min(18, len(episode) - 1)
        reward = sum(sample.reward for sample in episode)
        relocated.append([replace(sample, reward=reward if index == anchor else 0.)
                          for index, sample in enumerate(episode)])
    with pytest.raises(ValueError, match='visible reward events'):
        RoleModel.fit(relocated)


def _camera_position(point):
    # The evaluator has the object locations. The binder gets camera images only.
    horizontal, top = tt3_arena._project(point, 0)
    _, side = tt3_arena._project(point, 1)
    width = tt3_arena.IMAGE_WIDTH
    height = tt3_arena.IMAGE_HEIGHT
    return (np.array((horizontal, top, side)) + .5 - (width / 2, height / 2, height / 2)) / width


def _matches(track, position):
    return track is not None and np.linalg.norm(np.asarray(track.position) - _camera_position(position)) < .019


def test_frozen_source_roles_in_registered_views_with_independent_moving_distractor(native_practice):
    _, model = native_practice
    observed = []
    for seed in range(12):
        rows = []
        for distractor in (False, True):
            arena = tt3_arena.TT3Arena(seed, view_seed=seed + 842, distractor=distractor)
            binder = RoleBinder(model)
            unpaired = RoleBinder(model)
            camera = arena.reset()
            binder.update(camera)
            unpaired.update(camera)
            rng = np.random.default_rng(seed)
            unrelated = np.random.default_rng(seed + 8172)
            for frame in range(24):
                if frame % 4 == 0:
                    request = rng.choice((-1, 0, 1), size=3)
                    null_request = unrelated.choice((-1, 0, 1), size=3)
                camera, _, done, _ = arena.step(request)
                assert not done
                roles = binder.update(camera, request)
                null_roles = unpaired.update(camera, null_request)
            rows.append((_matches(roles.controlled, arena.paddle), _matches(roles.relevant, arena.ball),
                         _matches(null_roles.controlled, arena.paddle)))
            assert binder.model == model
        observed.append(rows)
    counts = np.asarray(observed).sum(axis=0)
    assert counts[0, 0] >= 10
    assert counts[0, 1] >= 10
    assert counts[1, 0] >= 9
    assert counts[1, 1] >= 8
    assert counts[1, 0] > counts[1, 2]


def test_zero_prior_can_update_roles_from_receiving_commands_and_reward():
    binder = RoleBinder(RoleModel.untrained(), learn=True)
    for seed in range(6):
        arena = tt3_arena.TT3Arena(seed)
        binder.reset()
        camera = arena.reset()
        roles = binder.update(camera)
        rng = np.random.default_rng(seed)
        assert roles.controlled is None and roles.relevant is None
        for frame in range(arena.max_frames):
            if frame % 4 == 0:
                request = rng.choice((-1, 0, 1), size=3)
            camera, reward, done, _ = arena.step(request)
            roles = binder.update(camera, request)
            if done:
                binder.observe_reward(reward)
                break
        assert done
    assert binder.model.source_control_examples == 0
    assert binder.model.source_reward_examples == 0
    assert binder.model.receiving_control_examples > 0
    assert binder.model.receiving_reward_examples > 0
    assert roles.relevant is not None
    updated = binder.model
    binder.reset()
    assert binder.model == updated
