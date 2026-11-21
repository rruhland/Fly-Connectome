import json

import numpy as np

from scripts.experimental.tt7_flight import FlightMemory
from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt19_online import Court, OpticalPlayer, Stroke
from scripts.experimental.tt19_online_body import InverseModel, kernel_for


def test_models_fit_scored_camera_movement_without_joint_labels():
    width = 6
    jac = np.diag([.075, .092, .069, .032, .041])
    jac = np.column_stack([jac, np.zeros(5)])
    origin = np.array([0., -.91, .20, .01, 0.])
    histories = np.random.default_rng(917).uniform(-.32, .32, size=(5, 90, width))
    for mode in ('secant', 'pose'):
        model = InverseModel(width, kernel_for(), mode, 2)
        for actions in histories:
            model.reset()
            model.measure(origin)
            for action in actions:
                model.advance(action)
                model.measure(origin + jac @ model.q[0])
        assert model.updates > 0
        assert np.max(abs(model.jacobian(np.zeros(width)) - jac)) < .006


def test_motor_solver_bounds_and_checkpoint_rng():
    model = InverseModel(6, kernel_for(), 'secant', 721, solver='drive')
    state = json.loads(json.dumps(model.state()))
    recovered = InverseModel(6, state['kernel'], state['mode'], 4, state=state)
    for candidate in (model, recovered):
        candidate.measure(np.array([0., -.91, .2, .02, 0.]))
        candidate.weights[:, :5] = np.eye(5)[None] * .08
    commands = [candidate.control(lambda t: np.array([1., -.65, .35, .02, .01]), 4.,
                                  explore=True, fixed=(-1,), strike_rate=.04)
                for candidate in (model, recovered)]
    assert np.array_equal(*commands)
    assert np.max(abs(commands[0])) <= .85
    assert commands[0][-1] == 0
    assert model.state() == recovered.state()


def test_service_uses_first_descending_crossing_behind_end_line():
    body = InverseModel(6, kernel_for(), 'secant', 33)
    player = OpticalPlayer(body, lambda: None, Court(.8, .424, .1725), FlightLaw(), FlightMemory('impulse'))
    player.reset()
    player.tick = 11
    ready = np.array([0., -.907, .2, .004, 0.])
    player.ready = ready
    player.serving = True
    player.body.measure(ready)
    position, velocity = np.array([.02, -.9, .67]), np.array([.001, .001, -.045])
    stroke = player._prepare((position, velocity))
    assert stroke.role == 'serve'
    assert 16 < stroke.tick < 19
    assert np.isclose(stroke.center[2], ready[2] + .054)
    assert stroke.center[1] < -player.court.half_length
    assert isinstance(stroke, Stroke)


def test_pitch_registration_preserves_contact_point_and_updates_on_reward():
    body = InverseModel(6, kernel_for(), 'secant', 4, solver='register')
    player = OpticalPlayer(body, lambda: None, Court(.8, .424, .1725), FlightLaw(), FlightMemory('impulse'))
    player.reset()
    stroke = Stroke(18., np.array([.04, -.90, .24]), np.zeros(3), .30, .06, .04, 'receive')
    player.attempt['receive'] = dict(intended=.30, measured=.10)
    evidence = player.finish(-1.)['pitch_registration']
    bias = evidence['receive']['next']
    assert np.isclose(bias, .11)
    before, lifted = stroke.pose(17.), stroke.pose(17., bias)
    assert np.array_equal(before[:3], lifted[:3])
    assert lifted[3] > before[3]
    player.reset()
    assert player.registration['receive'] == bias
    player.attempt['receive'] = dict(intended=.30, measured=.10)
    assert player.finish(1.)['pitch_registration']['receive']['next'] < .92 * bias + .11
