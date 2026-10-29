"""Frozen generic feedback semantics; no policy fitting or native games."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_relational_feedback import RelationalPolicy, RelationalController, structural_choices
from fly_connectome.control.durable import fingerprint


def scene():
    return np.array([[.1, -.3, .2, -.4, 1, 1], [.2, .3, -.1, .5, 1, 1]])


def policy(coefficients=(0, 0, 0, 0, 0), axis=0, sign=1):
    return RelationalPolicy(2, 0, 1, axis, sign, coefficients, np.ones(4))


def test_generic_pair_is_permutation_and_common_translation_invariant():
    actual = policy((.2, -.3, .1, .01, .02))
    first = actual.inspect(scene(), 1)
    assert first == actual.inspect(scene()[::-1], 1)
    translated = scene(); translated[:, :2] += [1., -2.]
    second = actual.inspect(translated, 1)
    assert second['action'] == first['action']
    assert second['score'] == pytest.approx(first['score'])
    assert len(structural_choices()) == len(set(structural_choices())) == 40


def test_axis_polarity_velocity_and_sensor_scaling_are_explicit():
    actual = scene(); actual[1, 0] = -.1
    assert policy().inspect(actual)['action'] == 0
    assert policy(axis=1).inspect(actual)['action'] == 2
    assert policy(sign=-1).inspect(actual)['action'] == 2
    feedback = policy((2., 3., 0., 0., 0.)).inspect(scene())
    assert feedback['score'] == pytest.approx(.1 + 2 * .2 + 3 * -.1)
    scaled = RelationalPolicy(2, 0, 1, 0, 1, [2, 3, 0, 0, 0], [2, 2, 4, 4])
    assert scaled.inspect(scene())['score'] == pytest.approx(.1 / 2 + 2 * .2 / 4 + 3 * -.1 / 4)


def test_deadzone_and_hysteresis_use_previous_actual_drive():
    actual = policy((0, 0, 0, .2, .2))
    assert actual.inspect(scene(), 1)['action'] == 1
    assert actual.inspect(scene(), 2)['action'] == 2
    assert actual.inspect(scene(), 0)['action'] == 1
    exact = np.array([[0, -.3, 0, 0, 1, 1], [.25, .3, 0, 0, 1, 1]])
    assert policy((0, 0, 0, .25, 0)).inspect(exact)['action'] == 1


def test_only_observed_entities_count_and_missing_velocity_is_zero():
    actual = scene(); actual[0, 5] = 0
    assert policy((10, 0, 0, 0, 0)).inspect(actual)['velocity_i'] == 0
    ghost = np.array([[100, 100, 100, 100, 0, 1]])
    assert policy().inspect(np.r_[scene(), ghost]) == policy().inspect(scene())
    actual[0, 4] = 0
    assert not policy().inspect(actual)['supported']
    invalid = scene(); invalid[0, 0] = np.nan
    assert policy().inspect(invalid)['mode'] == 'baseline_invalid_scene'


def test_opening_fallback_and_actual_action_memory():
    controller = RelationalController(policy((0, 0, 0, .2, .2)))
    action, decision = controller.choose(scene(), 2)
    assert action == 2 and decision['mode'] == 'baseline_before_hit'
    assert controller.previous_action == 1
    controller.observe(1)
    assert controller.previous_action == 2 and controller.past_hit
    action, _ = controller.choose(scene(), 0)
    assert action == 2
    controller.observe(0)
    action, decision = controller.choose(scene()[:1], 0)
    assert action == 0 and decision['mode'] == 'baseline_cardinality_mismatch'
    controller.observe(0)
    assert controller.previous_action == 0


def test_serialization_and_fingerprint_cover_all_choices_coefficients_scales(tmp_path):
    actual = policy((.2, -.3, .1, .2, .3)); path = tmp_path / 'policy.npz'
    actual.save(path); restored = RelationalPolicy.load(path)
    assert actual.inspect(scene(), 2) == restored.inspect(scene(), 2)
    assert fingerprint(actual.frozen_parameters()) == fingerprint(restored.frozen_parameters())
    assert not restored.coefficients.flags.writeable and not restored.scales.flags.writeable
    baseline = fingerprint(restored.frozen_parameters())
    restored.axis = 1
    assert fingerprint(restored.frozen_parameters()) != baseline
    with pytest.raises(ValueError):
        RelationalPolicy(2, 0, 1, 0, 1, [0, 0, 0, -1, 0], np.ones(4))
