import numpy as np

from scripts.experimental.tt6_body_control import CameraBody
from scripts.experimental.tt12_motor_recall import rewire
from scripts.experimental.tt13_live_ports import PortMemory, effective


def test_recall_and_restore_with_optical_history_only():
    n = 4
    coefficients = np.zeros((1+n+n*(n+1)//2, 4))
    coefficients[1:n+1] = np.diag([.10, .09, .11, .12])
    body = CameraBody(coefficients, np.ones(n), np.ones((n, 4)), 0., 0, 0, True)
    key = ((1,3,0,2), (-1,1,-1,1))
    changed = rewire(body, *key)
    commands = np.random.default_rng(71821).normal(0,.16,(119,n))
    memory = PortMemory(body, 'remap')

    def select(model):
        moves = np.vstack([np.zeros(n), commands])
        poses = model.pose(effective(moves,model.kernels))[np.maximum(np.arange(len(moves))-2,0)]
        trace = [dict(blade=p[:3],blade_pitch=p[3]) for p in poses]
        return memory.observe(commands,trace)['active']

    assert select(body) == 0
    assert select(changed) == 1
    assert memory.keys[1] == key
    assert select(changed) == 1
    assert select(body) == 0
    assert len(memory.models) == 2
