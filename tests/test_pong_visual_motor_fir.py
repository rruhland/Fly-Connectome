"""Independent synthetic checks; no native Pong environment or cameras.

The synthetic motor intentionally differs from native Pong. Its state is used
only to generate public quantized coordinates in these tests, never passed to
the learner. These tests establish causal plumbing and a useful learnable plant,
not native-game qualification.
"""
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_motor_fir import DEFAULTS, VisualMotorController, VisualMotorFIR
from public_entity_pong_native import save_public_entities, snapshot
import run_pong_visual_motor_learning as runner


def settings(**changes):
    return dict(DEFAULTS, **changes)


def inherited():
    return PersistentRelationalPolicy(RelationalPolicy(3, 0, 1, 0, -1,
        [0.] * 5, np.ones(4)), require_first_hit=False)


def entity(identity, y, x, observed=True):
    return dict(id=identity, normalized_position=[float(y), float(x)],
        observed_velocity=[17., -19.], observed=observed, velocity_known=True,
        observation_age_seconds=0. if observed else .02,
        velocity_age_seconds=0. if observed else .02, association_strength=1.)


def scene(own, target=.1, *, motor_id=10, own_observed=True, rename=0):
    return [entity(motor_id + rename, own, -.35, own_observed),
            entity(11 + rename, target, 0.), entity(12 + rename, -.1, .35)]


def feedback(dt=.02, **changes):
    return dict(dict(own_hits=0, reward=0., terminated=False, truncated=False,
        elapsed_seconds=dt), **changes)


def force_actual(controller, action):
    """A reporting comparison must advance the actual command, not its choice."""
    controller.pending_action = int(action)
    controller.inherited.pending_action = int(action)


def test_forecast_delay_index_and_actual_queue_are_causal():
    learner = VisualMotorFIR(settings=settings(lags=8, horizon=6))
    # Only a three-interval delayed actuator response and a constant drift.
    learner.theta[0] = .03
    learner.theta[4] = -.8
    history = [1, -1, -1, 1, 0]
    for drive, prediction in zip((-1, 0, 1), learner.candidate_displacements(history)):
        # Six future intervals see three known old requests, then three copies
        # of this candidate. There is no hidden delivered-command queue.
        expected = .02 * (6 * .03 - .8 * (sum(history[-3:]) + 3 * drive))
        assert prediction == pytest.approx(expected, abs=1e-15)
    old = learner.future_features(history, 1)
    history[0] = 100  # Older than this response support: prediction is unchanged.
    assert learner.predict(old) == learner.predict(learner.future_features(history, 1))


def test_known_delayed_numeric_response_is_learned_from_weighted_actual_intervals():
    fixed = settings(lags=6, window=4, initial_covariance=1e7,
        forgetting_seconds=10000., initial_dither_probability=0.,
        continuing_dither_probability=0.)
    learner = VisualMotorFIR(901, settings=fixed)
    controller = VisualMotorController(inherited(), learner)
    # This response law is synthetic and not copied from the learner or Pong.
    true = np.array([.015, -.31, 0., 0., -.17, 0., 0.])
    rng = np.random.default_rng(816)
    requests = deque([0.] * 6, maxlen=6)
    own = 0.
    errors = []
    for i in range(800):
        controller.choose(scene(own), 1)
        action = int(rng.integers(3))
        force_actual(controller, action)
        requests.append(action - 1)
        dt = 1 / 60 if i % 2 else 1 / 40
        own += dt * (true[0] + true[1] * requests[-1] + true[4] * requests[-4])
        before = learner.theta.copy()
        update = controller.observe(**feedback(dt), public_entities=scene(own))
        if update is not None and i > 300:
            errors.append(update['residual'])
        if update is not None:
            # The archived prediction belongs to the pre-observation model.
            summed_features = np.sum(controller.features, axis=0)
            assert update['prediction'] == float(before @ summed_features)
    np.testing.assert_allclose(learner.theta, true, atol=2e-5, rtol=0.)
    assert np.sqrt(np.mean(np.square(errors))) < 2e-7
    assert sum(learner.command_counts) == 800
    assert learner.elapsed_training_seconds == pytest.approx(sum(
        1 / 60 if i % 2 else 1 / 40 for i in range(3, 800)))


def test_missing_observations_are_not_zero_labels_and_identity_jump_is_not_fitted():
    learner = VisualMotorFIR(settings=settings(window=3,
        initial_dither_probability=0., continuing_dither_probability=0.))
    controller = VisualMotorController(inherited(), learner)
    observed = []
    original_observe = learner.observe

    def record(x, displacement, dt):
        observed.append(displacement)
        return original_observe(x, displacement, dt)

    learner.observe = record
    current = scene(0.)
    for i, (own, visible) in enumerate(((99., False), (-99., False), (.03, True))):
        controller.choose(current, 1)
        current = scene(own, own_observed=visible)
        controller.observe(**feedback(), public_entities=current)
        assert len(observed) == int(i == 2)
    assert observed == pytest.approx([.03])
    # A new public ID appears one coordinate unit away. No target spans IDs.
    controller.choose(current, 1)
    current = scene(1.03, motor_id=110)
    assert controller.observe(**feedback(), public_entities=current) is None
    for i in range(3):
        controller.choose(current, 1)
        update = controller.observe(**feedback(), public_entities=current)
        assert (update is not None) == (i == 2)
    assert observed == pytest.approx([.03, 0.])
    assert controller.motor_id == 110


def test_public_id_renaming_permutation_and_unused_velocity_do_not_change_learning():
    left = VisualMotorFIR(51)
    right = VisualMotorFIR.from_state(json.loads(json.dumps(left.state())))
    a, b = VisualMotorController(inherited(), left), VisualMotorController(inherited(), right)
    for i in range(260):
        current = scene(.08 * math.sin(i * .029), .12 * math.sin(i * .017))
        renamed = copy.deepcopy(current[::-1])
        for row in renamed:
            row['id'] += 1000
            row['observed_velocity'] = [-1000., 1000.]
        aa, da = a.choose(current, 1)
        ab, db = b.choose(renamed, 1)
        assert aa == ab
        assert da['predictions'] == db['predictions']
        next_current = scene(.08 * math.sin((i + 1) * .029), .12 * math.sin((i + 1) * .017))
        next_renamed = copy.deepcopy(next_current[::-1])
        for row in next_renamed:
            row['id'] += 1000
            row['observed_velocity'] = [1000., -1000.]
        assert a.observe(**feedback(), public_entities=next_current) == b.observe(
            **feedback(), public_entities=next_renamed)
    assert left.state() == right.state()


def test_game_boundary_json_restore_preserves_model_rng_and_future_updates():
    live = VisualMotorFIR(105)
    first = VisualMotorController(inherited(), live)
    for i in range(340):
        first.choose(scene(.1 * math.sin(i * .03)), 1)
        first.observe(**feedback(), public_entities=scene(.1 * math.sin((i + 1) * .03)))
    restored = VisualMotorFIR.from_state(json.loads(json.dumps(live.state())))
    a, b = VisualMotorController(inherited(), live), VisualMotorController(inherited(), restored)
    count = live.updates
    # A new game's unrelated coordinate cannot become a cross-game label.
    for i in range(220):
        current = scene(-.27 + .06 * math.sin(i * .04))
        assert a.choose(current, 1) == b.choose(current, 1)
        next_current = scene(-.27 + .06 * math.sin((i + 1) * .04))
        assert a.observe(**feedback(), public_entities=next_current) == b.observe(
            **feedback(), public_entities=next_current)
        if i < live.settings['window'] - 1:
            assert live.updates == count
    assert live.state() == restored.state()
    assert live.updates > count and live.updates > 0


def test_frozen_numerical_model_keeps_runtime_and_excitation_running():
    trained = VisualMotorFIR(63)
    rng = np.random.default_rng(107)
    for _ in range(400):
        x = rng.normal(size=len(trained.theta)) * .02
        trained.observe(x, -.4 * x[3], .02)
    frozen = trained.frozen_branch()
    theta, covariance = frozen.theta.copy(), frozen.covariance.copy()
    # Frozen updates do not silence or reseed independent exploration.
    assert [trained.perturb(2, True) for _ in range(1000)] == [
        frozen.perturb(2, True) for _ in range(1000)]
    controller = VisualMotorController(inherited(), frozen)
    for i in range(30):
        controller.choose(scene(i / 200), 1)
        action = i % 3
        force_actual(controller, action)
        controller.observe(**feedback(), public_entities=scene((i + 1) / 200))
        assert controller.commands[-1] == action - 1
        assert controller.inherited.previous_action == action
        assert controller.positions[-1] == pytest.approx((i + 1) / 200)
    np.testing.assert_array_equal(frozen.theta, theta)
    np.testing.assert_array_equal(frozen.covariance, covariance)
    assert frozen.updates == 400 and frozen.observations > 400


class QuantizedMotor:
    """Different delay, gains and time constants from native Pong."""
    def __init__(self):
        self.queue = deque([0.] * 6)
        self.activation = self.velocity = self.position = 0.

    def step(self, request):
        self.queue.append(float(request))
        delivered = self.queue.popleft()
        self.activation += (delivered - self.activation) * (1 - math.exp(-.02 / .14))
        self.velocity += (-.65 * self.activation - self.velocity) * (1 - math.exp(-.02 / .19))
        self.position = float(np.clip(self.position + .02 * self.velocity, -.38, .38))

    def observation(self, step):
        target = .18 * math.sin(step * .02 * 2 * math.pi / 8.)
        return scene(round(self.position * 128) / 128, round(target * 128) / 128)


def test_cold_continuous_learning_improves_quantized_delayed_synthetic_tracking():
    learner = VisualMotorFIR(731)
    controller = VisualMotorController(inherited(), learner)
    # A control with zero learned evidence executes the same inherited servo,
    # without probe commands. This favors the frozen inherited comparison.
    baseline = VisualMotorController(inherited(), VisualMotorFIR(731,
        settings=settings(initial_dither_probability=0., continuing_dither_probability=0.),
        updates_enabled=False))
    plants = [QuantizedMotor(), QuantizedMotor()]
    squared = [[], []]
    for i in range(3000):
        for j, (control, plant) in enumerate(zip((controller, baseline), plants)):
            current = plant.observation(i)
            action, _ = control.choose(current, 1)
            plant.step(action - 1)
            after = plant.observation(i + 1)
            control.observe(**feedback(), public_entities=after)
            if i >= 1000:
                squared[j].append((after[0]['normalized_position'][0]
                    - after[1]['normalized_position'][0]) ** 2)
    learned_rmse, inherited_rmse = [float(np.sqrt(np.mean(s))) for s in squared]
    assert learned_rmse < .045
    assert learned_rmse < .4 * inherited_rmse
    assert controller.numeric_actions > 2000
    assert controller.changed_from_inherited > 500
    assert np.linalg.norm(learner.theta[1:]) > .1
    assert np.isfinite(learner.theta).all() and np.isfinite(learner.covariance).all()


def test_full_public_replay_reconstructs_numeric_choices_and_updates_without_native(tmp_path, monkeypatch):
    learner = VisualMotorFIR(413)
    before = learner.state()
    controller = VisualMotorController(inherited(), learner)
    plant = QuantizedMotor()
    inputs, actions, feedbacks = [snapshot(plant.observation(0))], [], []
    digest = hashlib.sha256()
    for i in range(550):
        action, decision = controller.choose(inputs[-1], 1)
        plant.step(action - 1)
        after = snapshot(plant.observation(i + 1))
        public = feedback()
        public['own_hits'] = int(i in (125, 340))
        public['truncated'] = i == 549
        update = controller.observe(**public, public_entities=after)
        runner.digest_row(digest, 1, action, decision, public, update)
        inputs.append(after)
        actions.append(action)
        feedbacks.append(public)
    public_path, tape_path = tmp_path / 'public.npz', tmp_path / 'events.npz'
    save_public_entities(public_path, inputs)
    np.savez_compressed(tape_path, action=np.array(actions), baseline_action=np.ones(len(actions), int),
        hits=np.array([p['own_hits'] for p in feedbacks]),
        reward=np.array([p['reward'] for p in feedbacks]),
        elapsed_seconds=np.array([p['elapsed_seconds'] for p in feedbacks]),
        terminated=np.array([p['terminated'] for p in feedbacks]),
        truncated=np.array([p['truncated'] for p in feedbacks]))
    receipt = dict(learner_before=before, learner_after_sha256=runner.state_hash(learner.state()),
        public_file=public_path, tape_file=tape_path, transitions=len(actions),
        public_decision_update_digest=digest.hexdigest())

    def forbidden_native(*args, **kwargs):
        raise AssertionError('offline replay must never create a native evaluator')

    monkeypatch.setattr(runner, 'NumericEvaluator', forbidden_native)
    result = runner.replay(receipt, inherited())
    assert result['passed'] and result['native_cameras'] == 0
    assert result['exact_actions_decisions_updates_and_final_state']
    assert result['within_game_update_changed_unperturbed_actions'] > 50
    damaged = dict(receipt, learner_after_sha256='0' * 64)
    with pytest.raises(AssertionError):
        runner.replay(damaged, inherited())


def test_summary_preserves_not_started_and_partial_camera_accounting():
    rows = [dict(complete=False, not_started=True, transitions=0, initial_cameras=0,
        transition_camera_upper_bound=0, initial_camera_upper_bound=0),
        dict(complete=False, transitions=7, initial_cameras=1,
            transition_camera_upper_bound=8, initial_camera_upper_bound=1,
            report=dict(hits=2), stop_reason='native_failure')]
    summary = runner.summarize(rows)
    assert summary['saved_games'] == 2 and summary['completed_games'] == 0
    assert summary['transitions'] == 7 and summary['initial_cameras'] == 1
    assert summary['transition_camera_upper_bound'] == 8
    assert summary['initial_camera_upper_bound'] == 1
    assert summary['own_hits'] == [None, 2]
