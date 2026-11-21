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


def save(path, memory, prior, response, timing, *, entry_memory=None, shot_memory=None, service_memory=None,
         service_drive=None, receiving_lift=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema='tt6-installation-memory-v1', models=[_serializable_body(m) for m in memory.models],
                 rng=prior.rng.bit_generator.state, roles=asdict(prior.roles), clock=prior.clock,
                 numeric=[a.tolist() for a in prior.numeric], timing_choices=[a.tolist() for a in prior.delay],
                 bounce=prior.bounce_values, delay_belief=timing.state(),
                 racket=dict(gain=response.gain, enabled=response.enabled, default_speed=response.default_speed,
                             dt=response.dt, reacquisition=response.reacquisition),
                 speed=response.speed, reference=response.reference)
    if prior.span_memory is not None:
        state['motor_span'] = prior.span_memory.state()
    if entry_memory is not None:
        state['entry_memory'] = entry_memory.state()
    if shot_memory is not None:
        state['shot_memory'] = shot_memory.state()
    if service_memory is not None:
        state['service_memory'] = service_memory.state()
    if service_drive is not None:
        state['service_drive'] = asdict(service_drive)
    if receiving_lift is not None:
        state['receiving_lift'] = receiving_lift
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
    if 'motor_span' in state:
        from .tt3_cycle import ConditionedNumeric
        prior.span_memory = ConditionedNumeric.restore(state['motor_span'])
        shared = prior.span_memory.shared
        if any(not np.array_equal(stored, routed) for stored, routed in
               zip(prior.numeric, (shared.coefficients, shared.scores, shared.counts), strict=True)):
            raise ValueError('shared motor-span evidence differs from the portable numeric state')
        prior.numeric = shared.coefficients, shared.scores, shared.counts
        prior.span_memory.active = None
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
