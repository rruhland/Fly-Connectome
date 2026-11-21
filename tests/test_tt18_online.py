import json
from dataclasses import replace

import numpy as np

from scripts.experimental.tt18_online import (BodyLearning, RecallLearning, ServeChoices, blank_body,
                                              carry_body, signed_basis)


def example_body():
    body = blank_body(4)
    coefficients = body.coefficients.copy()
    coefficients[1:5] = np.diag([.16, .12, .14, .55])
    coefficients[5, 1] = .034
    coefficients[8, 3] = -.021
    return replace(body, coefficients=coefficients)


def test_carried_quadratic_and_recalled_signed_basis_match_camera_motion():
    source = example_body()
    transform = np.eye(4)[:, [2, 0, 3, 1]] * [-1, 1, -1, 1]
    carried = carry_body(source, transform)
    states = np.random.default_rng(903).uniform(-1., 1., (20, 4))
    assert np.allclose(carried.pose(states), source.pose(states @ transform.T))
    assert np.array_equal(signed_basis(source, carried), transform)


def test_online_samples_are_causal_and_checkpoint_continues_same_exploration():
    truth = example_body()
    initial = blank_body(truth.action_size)
    learner = BodyLearning(initial, enabled=True, seed=38)
    states = np.random.default_rng(13).uniform(-1., 1., (240, 4))
    errors = [learner.observe(state, truth.pose(state)) for state in states]
    assert errors[0] == np.linalg.norm(initial.pose(states[0])[:3] - truth.pose(states[0])[:3])
    assert np.median(errors[-40:]) < .008 < np.median(errors[:12])
    copy = BodyLearning.restore(json.loads(json.dumps(learner.state())))
    for memory in (learner, copy):
        memory.begin_point()
    planned = np.array([.95, -.5, .1, .2])
    for state in states[-10:]:
        outputs = [memory.explore(planned, state, None) for memory in (learner, copy)]
        assert np.array_equal(*outputs)
    assert learner.state() == copy.state()


def test_passive_updates_without_dither_and_frozen_does_neither():
    body = blank_body(4)
    truth = example_body()
    command = np.array([.2, -.3, .4, 0.])
    for enabled in (True, False):
        learner = BodyLearning(body, enabled=enabled, directed=False, seed=3)
        for state in np.eye(4).repeat(3, axis=0):
            assert np.array_equal(learner.explore(command, state, 3.), command)
            learner.observe(state, truth.pose(state))
        assert (learner.updates > 0) == enabled


def test_basis_recall_carries_a_pending_model_through_checkpoint():
    body = example_body()
    pending = carry_body(body, np.diag([-1., 1., 1., 1.]))
    learner = RecallLearning(body, enabled=True, seed=42)
    learner.pending = learner.pending_model = pending
    learner = RecallLearning.restore(json.loads(json.dumps(learner.state())))
    learner.begin_point()
    for offset in np.linspace(.4, 1., 12):
        state = np.array([offset, 0., 0., 0.])
        learner.observe(state, pending.pose(state))
    learner.end_point()
    assert learner.decision['adopted']
    assert np.array_equal(learner.prior.coefficients, pending.coefficients)


def test_service_variation_starts_from_inherited_stroke_and_uses_earned_credit():
    choices = ServeChoices()
    assert choices.choose() == 0
    choices.observations.extend([dict(choice=0, score=0.)] * 4)
    alternate = choices.choose()
    assert alternate != 0
    choices.observations.extend([dict(choice=alternate, score=.8)] * 3)
    assert choices.choose() == alternate
