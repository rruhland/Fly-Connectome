import numpy as np

from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt6_continuous import BodyLibrary
from scripts.experimental.tt13_live_ports import effective
from scripts.experimental.tt14_gain_hardware import perturbation
from scripts.experimental.tt14_gain_model import GainMemory
from scripts.experimental.tt15_motor_guard import GuardedMemory
from scripts.experimental.tt15_motor_probe import ProbeMemory, paired_observation, pulse_commands
from scripts.experimental.tt15_motor_trial import restore_memory


def camera_body():
    coefficients = np.zeros((15, 4))
    coefficients[1:5] = np.diag([.14, .15, .13, .61])
    kernels = 1. - np.exp(-np.outer([.24, .37, .49, .63], np.arange(14) + .4))
    return CameraBody(coefficients, np.ones(4), kernels, 0., 0, 0, True)


def movement(model, matrix, seed):
    actions = np.random.default_rng(seed).uniform(-.9, .9, (155, 4))
    ordered = np.vstack([np.zeros(4), actions]) @ matrix.T
    state = effective(ordered, model.kernels)
    poses = model.pose(state[np.maximum(np.arange(len(state)) - 2, 0)])
    return actions, [dict(blade=row[:3], blade_pitch=row[3]) for row in poses]


def test_one_mismatched_optical_history_does_not_change_the_map():
    model = camera_body()
    memory = GuardedMemory(GainMemory(model, 'hybrid'))
    identity = np.eye(4)
    matrix = perturbation(4, 18583119, 'coupling')
    for seed, wiring in [(1, identity), (2, matrix), (3, identity), (4, identity)]:
        assert memory.observe(*movement(model, wiring, seed))['active'] == 0
    assert len(memory.maps) == 1


def test_independent_points_confirm_change_and_restore_with_checkpoint():
    model = camera_body()
    memory = GuardedMemory(GainMemory(model, 'hybrid'))
    identity = np.eye(4)
    matrix = perturbation(4, 18583119, 'coupling')
    assert memory.observe(*movement(model, matrix, 4201))['active'] == 0
    state = dict(series=dict(model=0), motor_guard=dict(version=1),
                 gain_memory=dict(active=0, maps=[dict(model=0, transform=identity.tolist(), kind='original')],
                                  previous=None, support=[obs.state() for obs in memory.support]))
    resumed = restore_memory(state, BodyLibrary([model]))
    second = movement(model, matrix, 4202)
    assert memory.observe(*second) == resumed.observe(*second)
    assert resumed.maps[resumed.active].kind == 'coupling'
    assert resumed.observe(*movement(model, identity, 4203))['active'] == 1
    assert resumed.observe(*movement(model, identity, 4204))['active'] == 0


def test_repeated_response_distinguishes_a_stroke_error_from_a_wiring_change():
    model = camera_body()
    identity = np.eye(4)
    matrix = perturbation(4, 18583119, 'coupling')
    commands = np.vstack([np.zeros(4), pulse_commands(4)])
    assert len(commands) == 29
    np.testing.assert_allclose(commands.sum(axis=0), 0., atol=1e-12)
    reference = model.pose(effective(commands, model.kernels))
    changed = model.pose(effective(commands @ matrix.T, model.kernels))
    shifted = reference + [.04, -.12, .05, .03]
    stationary = paired_observation(reference, shifted, 4)
    np.testing.assert_allclose(stationary.poses, 0., atol=1e-12)

    memory = ProbeMemory(GainMemory(model, 'hybrid'), reference)
    surprising_stroke = movement(model, matrix, 4201)
    untouched = memory.observe(*surprising_stroke, lambda: shifted)
    assert untouched['reason'] == 'original response' and untouched['active'] == 0
    correction = memory.observe(*surprising_stroke, lambda: changed)
    assert correction['active'] > 0
    restored = memory.observe(*movement(model, identity, 4203), lambda: shifted)
    assert restored['reason'] == 'original response' and restored['active'] == 0
