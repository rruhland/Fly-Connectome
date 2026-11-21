"""Evaluate a carried hall checkpoint in native Pong on the previously qualified starts."""
import argparse
from pathlib import Path

import torch

from .tt3_numeric_retention import _hashes, _native
from .tt3_pong import assess_retention, state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_memory import load


def portable(path):
    checkpoint = _read(path)
    _, prior, _, _ = load(path)
    state = dict(memory=prior.state()['numeric'], roles=checkpoint['roles'])
    if prior.span_memory is not None:
        state['motor_span'] = prior.span_memory.state()
    return state


def evaluate(schedule, state):
    if 'motor_span' not in state:
        return _native(schedule, state)
    from .tt3_cycle import ConditionedNumeric
    from .tt3_pong import PongCase, evaluate_pong, summarize_pong
    from .tt3_visual import RoleModel
    memory = ConditionedNumeric.restore(state['motor_span'])
    rows = evaluate_pong([PongCase(**case) for case in schedule['evaluation']], RoleModel(**state['roles']),
                         memory, points_per_match=schedule['evaluation_points'],
                         opponent_speed=schedule['opponent_speed'], max_ticks=schedule['max_ticks'])
    return dict(summary=summarize_pong(rows), points=rows)


def run(before, after, reference, output):
    old = _read(reference)
    original, carried = portable(before), portable(after)
    if old['states']['after'] != original:
        raise ValueError('starting player differs from the reference Pong player')
    for name, digest in _hashes().items():
        if old['implementation_sha256'][name] != digest:
            raise ValueError(f'Pong implementation changed at {name}')
    end = evaluate(old['schedule'], carried)
    initial_points = old['runs']['after']['points']
    differences = sum(x['action_sha256'] != y['action_sha256'] for x, y in
                      zip(initial_points, end['points'], strict=True))
    retention = assess_retention(initial_points, end['points'])
    _write(output, dict(schema='tt8-native-pong-retention-v1', reference_sha256=_hash(reference),
                        before_sha256=_hash(before), after_sha256=_hash(after),
                        portable_before_sha256=state_sha256(original), portable_after_sha256=state_sha256(carried),
                        changed_action_tapes=differences, retention=retention, run=end))
    print(retention['established'], differences, end['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('after', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--before', type=Path, default=ROOT/'artifacts/tt6/continual-player.json.gz')
    p.add_argument('--reference', type=Path, default=ROOT/'artifacts/tt6/pong-retention.json.gz')
    a = p.parse_args()
    torch.set_num_threads(1)
    run(a.before, a.after, a.reference, a.output)
