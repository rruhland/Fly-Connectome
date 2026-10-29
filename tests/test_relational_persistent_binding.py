"""Public tracking continuity around frozen feedback; no native games or fit."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import (
    PersistentRelationalPolicy, PersistentRelationalController, public_rows,
)
from fly_connectome.control.durable import fingerprint


def entities():
    return [entity(10, .0, -.3), entity(11, .2, .0), entity(12, -.1, .3)]


def entity(identity, y, x, observed=True, known=True, age=0., strength=1.):
    return dict(id=identity, normalized_position=np.array([y, x]),
        observed_velocity=np.array([.1, -.2]), observed=observed,
        velocity_known=known, observation_age_seconds=age,
        velocity_age_seconds=age if known else None, association_strength=strength)


def model(gated=False, coefficients=(0, 0, 0, 0, 0)):
    return PersistentRelationalPolicy(RelationalPolicy(3, 0, 1, 0, -1,
        coefficients, np.ones(4)), require_first_hit=gated)


def test_full_observation_scores_and_actions_exactly_match_original():
    rng = np.random.default_rng(713)
    for _ in range(240):
        n = int(rng.integers(2, 5)); pair = sorted(rng.choice(n, 2, replace=False))
        axis = int(rng.integers(2)); polarity = int(rng.choice([-1, 1]))
        original = RelationalPolicy(n, *map(int, pair), axis, polarity,
            np.r_[rng.normal(size=3), rng.uniform(size=2)], rng.uniform(.1, 1, 4))
        current = [entity(i, *rng.normal(size=2), known=bool(rng.integers(2))) for i in range(n)]
        for entry in current: entry['observed_velocity'] = rng.normal(size=2)
        current += [entity(99, -10, -10, observed=False, age=1.2, strength=0)]
        rng.shuffle(current)
        controller = PersistentRelationalController(PersistentRelationalPolicy(original, False))
        controller.previous_action = int(rng.integers(3))
        action, decision = controller.choose(current, 1)
        rows, _ = public_rows(current)
        expected = original.inspect(rows, controller.previous_action)
        assert action == expected['action']
        for key in ('score', 'position_difference', 'velocity_i', 'velocity_j', 'previous_drive'):
            assert decision[key] == expected[key]
        assert decision['binding_mode'] == 'observed_pair_reanchor'


def test_partial_scene_uses_same_published_ids_and_supplied_memory_positions():
    controller = PersistentRelationalController(model())
    controller.choose(entities(), 1); controller.observe(0)
    current = entities()
    current[1] = entity(11, -.3, .1, observed=False, known=False, age=1.28, strength=0.)
    current[1]['observed_velocity'][:] = 1000
    current += [entity(40, 999, -.9, observed=False, age=1.28, strength=0.)]
    action, decision = controller.choose(current, 1)
    assert action == 2 and decision['bound_ids'] == [10, 11]
    assert decision['position_difference'] == -.3
    assert decision['velocity_j'] == 0
    assert decision['binding_mode'] == 'persistent_visual_carry'
    assert decision['selected_entities'][1]['observation_age_seconds'] == 1.28
    assert decision['selected_entities'][1]['association_strength'] == 0
    # Neither a new age cutoff nor ghost-rank reassignment is allowed.
    controller.observe(0)
    current[1]['observation_age_seconds'] = 100.
    assert controller.choose(current, 1)[1]['supported']


def test_public_id_renaming_and_input_permutation_preserve_actions():
    left = PersistentRelationalController(model())
    right = PersistentRelationalController(model())
    renamed = entities()[::-1]
    for entry in renamed: entry['id'] += 1000
    assert left.choose(entities(), 1)[0] == right.choose(renamed, 1)[0]
    left.observe(0); right.observe(0)
    a = entities(); a[1]['observed'] = False
    b = a[::-1]
    b = [dict(entry, id=entry['id'] + 1000) for entry in b]
    assert left.choose(a, 1)[0] == right.choose(b, 1)[0]
    assert tuple(x + 1000 for x in left.bound_ids) == right.bound_ids


def test_missing_or_retired_id_clears_and_requires_new_complete_anchor():
    controller = PersistentRelationalController(model())
    controller.choose(entities(), 1); controller.observe(0)
    partial = entities(); partial[1] = entity(111, .2, 0, observed=False, age=.02)
    action, decision = controller.choose(partial, 2)
    assert action == 2 and controller.bound_ids is None
    assert decision['mode'] == 'baseline_selected_public_id_absent'
    controller.observe(0)
    partial[1] = entity(11, .2, 0, observed=False, age=.04)
    assert controller.choose(partial, 1)[1]['mode'] == 'baseline_no_public_binding'
    controller.observe(0)
    full = entities(); full[0]['normalized_position'][1] = .1
    full[1]['normalized_position'][1] = -.4
    action, decision = controller.choose(full, 1)
    assert decision['bound_ids'] == [11, 10]
    assert action == controller.policy.original.inspect(public_rows(full)[0], controller.previous_action)['action']


@pytest.mark.parametrize('kind', ['duplicate', 'nonfinite', 'excess', 'incomplete_metadata'])
def test_ambiguous_or_invalid_metadata_clears_binding_and_uses_actual_baseline(kind):
    controller = PersistentRelationalController(model())
    controller.choose(entities(), 1); controller.observe(0)
    current = entities()
    if kind == 'duplicate': current[1]['id'] = current[0]['id']
    elif kind == 'nonfinite': current[1]['normalized_position'][0] = np.nan
    elif kind == 'excess': current.append(entity(999, 0, 0))
    else: del current[1]['observation_age_seconds']
    action, decision = controller.choose(current, 2)
    assert action == 2 and not decision['supported'] and controller.bound_ids is None
    controller.observe(0)
    assert controller.previous_action == 2


def test_gate_only_changes_output_and_bindings_update_before_actual_hit():
    gated = PersistentRelationalController(model(True))
    opening = PersistentRelationalController(model(False))
    assert gated.choose(entities(), 2)[0] == 2
    assert opening.choose(entities(), 2)[0] == 0
    assert gated.bound_ids == opening.bound_ids == (10, 11)
    gated.observe(1); opening.observe(0)
    assert gated.past_hit and not opening.past_hit
    current = entities(); current[1]['observed'] = False
    for controller in (gated, opening):
        assert controller.choose(current, 2)[0] == 0
        controller.observe(0)
        controller.reset()
        assert controller.bound_ids is None and not controller.past_hit
        assert controller.previous_action == 1 and controller.ordinal == 0


def test_hysteresis_uses_previous_executed_command_and_order_is_enforced():
    controller = PersistentRelationalController(model(True, (0, 0, 0, .2, .5)))
    with pytest.raises(ValueError): controller.observe(0)
    controller.choose(entities(), 2)
    with pytest.raises(ValueError): controller.choose(entities(), 0)
    assert controller.previous_action == 1
    controller.observe(1)
    _, decision = controller.choose(entities(), 0)
    assert decision['previous_actual_action'] == 2 and decision['previous_drive'] == 1
    assert decision['score'] == pytest.approx(.3)


def test_serialization_and_fingerprint_cover_policy_and_gate(tmp_path):
    for gated in (False, True):
        actual = model(gated, (.2, -.3, .1, .05, .1)); path = tmp_path / f'{gated}.npz'
        actual.save(path); restored = PersistentRelationalPolicy.load(path)
        assert fingerprint(actual.frozen_parameters()) == fingerprint(restored.frozen_parameters())
        a, b = PersistentRelationalController(actual), PersistentRelationalController(restored)
        for current, hit in ((entities(), 1), (entities()[:1], 0), (entities(), 0)):
            assert a.choose(current, 2) == b.choose(current, 2)
            a.observe(hit); b.observe(hit)
        assert not restored.original.coefficients.flags.writeable
        old = fingerprint(restored.frozen_parameters())
        restored.require_first_hit = not restored.require_first_hit
        assert fingerprint(restored.frozen_parameters()) != old
    original = model(); before = fingerprint(original.frozen_parameters())
    original.original.axis = 1
    assert fingerprint(original.frozen_parameters()) != before


def test_actual_public_encoder_output_types_and_carried_fields_are_accepted():
    # Exercise the production public serializer, without an environment, image
    # acquisition, native transition, or reconstruction of a hidden identity.
    import torch
    from fly_connectome.vision.interface import VisualStateEncoder
    from pong_continuous_visual import canonical_scene

    encoder = VisualStateEncoder(64, 64, .02)
    controller = PersistentRelationalController(model(False, (.2, -.3, 0, 0, 0)))
    for sample in range(3):
        rows = [dict(id=identity, position=torch.tensor([y + sample, x]),
            observed=not (sample == 2 and identity == 11),
            observation_age=int(sample == 2 and identity == 11), association_strength=.35)
            for identity, y, x in ((10, 10., 8.), (11, 20., 31.), (12, 30., 55.))]
        encoded = encoder.encode(dict(sample=sample, entities=rows, forecasts=[]), None)
        action, decision = controller.choose(encoded['entities'], 1)
        if sample < 2:
            assert decision['score'] == controller.policy.original.inspect(
                canonical_scene(encoded), controller.previous_action)['score']
        else:
            assert decision['mode'] == 'persistent_visual_carry'
            assert decision['selected_entities'][1]['observation_age_seconds'] == .02
            assert decision['selected_entities'][1]['velocity_known']
        assert action in (0, 1, 2)
        controller.observe(0)
