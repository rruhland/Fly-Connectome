import numpy as np

import pytest

pytest.importorskip('mujoco')

from scripts.experimental.tt3_visual import Track
from scripts.experimental.tt5_player_body import FittedArm, centers
from scripts.experimental.tt6_timing import DelayedWiring, Disturbance, TimingMemory, VisualDelay, advance_stroke
from scripts.experimental.tt6_timing_gated import GatedTimingServo


class CommandReceiver:
    action_size = 5

    def __init__(self):
        self.arena = self

    def reset(self, **kwargs):
        return None

    def step(self, command, *, render):
        return command.copy()


def test_private_transport_holds_and_releases_commands_on_latency_change():
    wire = DelayedWiring(CommandReceiver(), Disturbance(delays=((0, 1), (2, 2), (4, 1))))
    wire.reset(seed=11)
    sent = [np.full(5, rate) for rate in (.05, .10, .15, .20, .25)]
    received = [wire.step(command) for command in sent]
    expected = [0., .05, 0., .10, .35]
    np.testing.assert_allclose(received, np.repeat(np.array(expected)[:, None], 5, axis=1))

    wire.reset(seed=12)
    np.testing.assert_array_equal(wire.step(sent[0]), np.zeros(5))


def test_noise_is_paired_by_private_seed_and_actuator_commands_remain_bounded():
    disturbed = Disturbance(noise=((0, .3), (3, .1)))
    wire = DelayedWiring(CommandReceiver(), disturbed, seed=21)
    wire.reset(seed=7)
    commands = np.full((6, 5), .85)
    first = np.array([wire.step(command) for command in commands])
    wire.reset(seed=7)
    second = np.array([wire.step(command) for command in commands])
    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first)) <= 1.
    assert not np.allclose(first, commands)


def test_latency_drop_preserves_queued_actuation_across_rate_saturation():
    wire = DelayedWiring(CommandReceiver(), Disturbance(delays=((0, 2), (2, 0))))
    wire.reset(seed=17)
    sent = [np.full(5, rate) for rate in (.8, .8, .8, 0., 0.)]
    received = [wire.step(command) for command in sent]
    expected = [0., 0., 1., 1., .4]
    np.testing.assert_allclose(received, np.repeat(np.array(expected)[:, None], 5, axis=1))
    np.testing.assert_allclose(np.sum(sent, axis=0), np.sum(received, axis=0))


def test_visual_delay_recovers_after_a_hidden_switch_and_retains_across_points():
    rng = np.random.default_rng(52)
    initial = np.zeros(3)
    positions = np.cumsum(rng.uniform(-.09, .09, (100, 3)), axis=0)
    memory = TimingMemory(max_delay=5)
    observer = VisualDelay(memory, initial, .012)
    for tick, position in enumerate(positions):
        observer.advance(position)
        latency = 1 if tick < 50 else 4
        previous = positions[tick - latency] if tick >= latency else initial
        noisy = previous + rng.normal(0., .005, len(previous))
        noisy[1] = np.nan if tick % 5 == 0 else noisy[1]
        observer.observe(None if tick % 3 == 0 else noisy)

    np.testing.assert_allclose(np.mean(observer.trajectory[35:50]), 1., atol=.25)
    np.testing.assert_allclose(np.mean(observer.trajectory[80:]), 4., atol=.25)
    observer.finish(carry=True)
    resumed = VisualDelay(TimingMemory(memory.state()), initial, .012)
    np.testing.assert_allclose(resumed.delay, 4., atol=.35)

    frozen = VisualDelay(memory, initial, .012, learn=False)
    saved = memory.state()
    for position in positions:
        frozen.advance(position)
        frozen.observe(position)
    frozen.finish(carry=True)
    assert memory.state() == saved
    assert frozen.informative == 0
    assert frozen.delay == frozen.trajectory[0]


def test_timing_leads_the_stroke_through_the_bounded_windup():
    goal = np.array([.15, -1.5, .18])
    lead = goal + [0., .025, 0.]
    now, following = advance_stroke(goal, lead, 4., .049, 2.)
    np.testing.assert_array_equal(now[[0, 2]], goal[[0, 2]])
    np.testing.assert_array_equal(following[[0, 2]], lead[[0, 2]])
    np.testing.assert_allclose(now[1], goal[1] + .145 - .098)
    np.testing.assert_allclose(following[1], lead[1] + .145 - .049)


def test_gated_servo_applies_sustained_motion_evidence_across_repeated_switches():
    rng = np.random.default_rng(833)
    fit = FittedArm(np.array([.02, -1.94, .22, .38]), np.arange(4), np.array([0., -.58, 1.37, .26]),
                    np.array([.18, .25, .30, .25]), np.array([.45, .52, .44, .30]), .012, 120, 100)
    servo = GatedTimingServo(fit, TimingMemory([1., 0., 0., 0., 0., 0.]))
    predictions, committed = [], []
    for tick, action in enumerate(rng.uniform(-.65, .65, (165, 4))):
        servo.advance(action)
        predictions.append(centers(servo.nominal, fit.geometry))
        delay = 5 if 45 <= tick < 105 else 0
        observed = predictions[tick - delay] + rng.normal(0., .0025, 3)
        servo.observe(Track(0, tuple(observed), (0., 0., 0.)), None)
        committed.append(servo.confirmed_delay)
    np.testing.assert_array_equal(committed[:3], 0.)
    assert np.mean(committed[85:105]) > 4.4
    assert np.mean(committed[-15:]) < .6
