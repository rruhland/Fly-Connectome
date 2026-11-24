"""Far-racket anticipation uses causal stereo pixels and an optical handoff."""

import numpy as np

from scripts.experimental.tt18_view import CameraFit
from scripts.experimental.tt29_vision import Flight
from scripts.experimental.tt37_cue_audit import matching_delivery
from scripts.experimental.tt37_cue_capture import match
from scripts.experimental.tt37_cue_fit import predict, solve
from scripts.experimental.tt37_cue_motor import EarlyJoint, to_plane
from scripts.experimental.tt37_cue_vision import CueGate, PaddleEye, plane_sample


def flight(t, y, vy):
    return Flight(np.array([.1, y, 1.04]), np.array([.1, vy, .25]),
                  t, .40, .9, .045, 4, False)


def test_gate_is_one_shot_and_requires_a_fresh_outbound_exposure():
    gate = CueGate()
    assert gate.update(flight(.20, .8, 3.6), .20) is None
    assert gate.update(flight(.22, 1.1, 3.6), .23) is None
    assert gate.update(flight(.24, 1.1, 3.6), .24) is not None
    assert gate.update(flight(.26, 1.2, 3.6), .26) is None
    assert gate.update(flight(.44, 1.0, -4.), .44) is None
    assert gate.update(flight(1.44, 1.0, 3.6), 1.44) is not None


def test_paddle_tracker_recovers_gray_blob_motion_from_rgb_only():
    # Two parallel surveyed pinhole cameras with 1.4 m baseline.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    views = {k: CameraFit(np.array([x, -2.5, 1.6]), rotation, 265., 384, 288)
             for k, x in [('left', -.7), ('right', .7)]}
    eye = PaddleEye(views)
    yy, xx = np.ogrid[:288, :384]
    target = np.array([0., 1.50, 1.10])
    for tick in range(5):
        t = tick / 30.
        target[0] = .6*t
        frames = {}
        for k, cam in views.items():
            x, y, _ = cam.project(target)
            frames[k] = np.full((288, 384, 3), [19, 49, 75], np.uint8)
            frames[k][(xx-x)**2+(yy-y)**2 <= 25.] = [55, 58, 63]
        guess = eye.update(t, frames)
    assert guess is not None
    np.testing.assert_allclose(guess[:3], target, atol=.035)
    np.testing.assert_allclose(guess[3:6], [.6, 0., 0.], atol=.16)
    assert eye.update(.30, None) is None


def test_visual_label_plane_and_contact_time_fence():
    f = flight(.9, -.36, -4.)
    distance, capture, label = plane_sample(f, .9)
    assert 0. < distance < .10 and capture > .9
    p, v = f.advance(capture)
    assert abs(p[1] + .40) < 1e-10
    assert np.allclose(label, np.r_[p[[0, 2]], v])
    assert to_plane(np.r_[f.position, f.velocity][None]).shape == (1, 5)
    events = [dict(kind='contact', time=.7, side=-1), dict(kind='contact', time=1.1, side=1)]
    gates = [(.5, list(np.arange(6.)), list(np.arange(12.))), (.7, [], []), (.8, [], [])]
    held, n = match(events, gates, [(.9, distance, capture, label.tolist())], [dict(hit=.7)])
    assert n == 1 and held[0]['tick_s'] == .5
    assert held[0]['label_capture_s'] > held[0]['contact_s']
    assert not match(events, gates[1:], [(.9, distance, capture, label.tolist())], [dict(hit=.7)])[0]


def test_group_balancing_for_uniform_baseline():
    f = solve(np.empty((4, 0)), np.array([[0.], [0.], [0.], [2.]]), [0, 0, 0, 1], 0.)
    np.testing.assert_allclose(predict(f, np.empty((2, 0))), [[1.], [1.]])


def test_first_delivery_pair_allows_solver_roundoff_but_not_physical_change():
    first = dict(target=[.2], far_contact={'time': .8},
                 near_bounce={'time': 1.4, 'step': 84, 'ball': [.2, -.72, .78]})
    rounding = dict(first, near_bounce=dict(first['near_bounce'], ball=[.2, -.72+1e-12, .78]))
    different = dict(first, near_bounce=dict(first['near_bounce'], ball=[.2, -.71, .78]))
    assert matching_delivery(first, rounding)
    assert not matching_delivery(first, different)
    assert not matching_delivery(first, None)


def test_early_joint_releases_the_arm_at_first_inbound_estimate():
    class Body:
        def bias(self, tick, optical, q, frames):
            return None

    class Pupil:
        home = np.zeros(6)
        options = {'recovery': .5}

        def drive(self, goal, q, limit):
            return goal - q

        def command(self, optical, q, offset):
            return self.home - q

    class Prep:
        def goal(self, forecast, home):
            return np.ones(6) * .1, .5

    class Rig:
        fixed = {name: CameraFit(np.zeros(3), np.eye(3), 265., 384, 288)
                 for name in ('left', 'right')}

    class Paddle:
        def update(self, now, frames):
            return np.ones(12)

    checkpoint = {'chosen': 'motion', 'models': {
        'uniform': dict(mean=[], spread=[], ys=[0., 0., 0., 0., 0.], beta=[])}}
    actor = EarlyJoint(checkpoint, 'uniform', Pupil(), Rig(), Prep(), Body())
    actor.paddle = Paddle()
    q = np.zeros(6)
    prepared = actor.command(1, .30, flight(.3, 1.1, 4.), None, q, {'left': None})
    assert np.max(prepared) == .1
    stopped = actor.command(2, .32, flight(.32, 1.40, -4.), np.ones(6), q, {'left': None})
    assert not np.any(stopped)
    assert actor.pending is None and actor.driven == len(actor.handoffs) == 1
