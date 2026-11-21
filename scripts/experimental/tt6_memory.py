"""Portable, between-installation memory for the continuously adapting table player."""
from dataclasses import asdict

import numpy as np

from .tt3_visual import RoleModel
from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import _read, _write
from .tt6_body_control import CameraBody
from .tt6_continuous import BodyLibrary
from .tt6_racket import RacketAdaptation
from .tt6_timing import TimingMemory


def _serializable_body(model):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(model).items()}


def save(path, memory, prior, response, timing, *, entry_memory=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema='tt6-installation-memory-v1', models=[_serializable_body(m) for m in memory.models],
                 rng=prior.rng.bit_generator.state, roles=asdict(prior.roles), clock=prior.clock,
                 numeric=[a.tolist() for a in prior.numeric], timing_choices=[a.tolist() for a in prior.delay],
                 bounce=prior.bounce_values, delay_belief=timing.state(),
                 racket=dict(gain=response.gain, enabled=response.enabled, default_speed=response.default_speed,
                             dt=response.dt, reacquisition=response.reacquisition),
                 speed=response.speed, reference=response.reference)
    if entry_memory is not None:
        state['entry_memory'] = entry_memory.state()
    _write(path, state)


def load(path):
    state = _read(path)
    if state['schema'] != 'tt6-installation-memory-v1':
        raise ValueError('unsupported table-tennis memory')
    models = []
    for item in state['models']:
        item = dict(item)
        for key in ('coefficients', 'lags', 'kernels'):
            item[key] = np.asarray(item[key])
        models.append(CameraBody(**item))
    prior = TransferPrior('fresh', {}, 0)
    prior.roles = RoleModel(**state['roles'])
    prior.clock = state['clock']
    prior.numeric = tuple(np.asarray(a) for a in state['numeric'])
    prior.delay = tuple(np.asarray(a) for a in state['timing_choices'])
    prior.bounce_values = state['bounce']
    prior.rng.bit_generator.state = state['rng']
    response = RacketAdaptation(**state['racket'])
    response.speed = state['speed']
    response.reference = state['reference']
    timing = TimingMemory(state['delay_belief'])
    # Preserve the exact posterior used by the preceding point across serialization.
    if not np.isclose(sum(state['delay_belief']), 1.):
        raise ValueError('stored latency belief must sum to one')
    timing.probabilities = np.asarray(state['delay_belief'], dtype=float)
    return BodyLibrary(models), prior, response, timing
