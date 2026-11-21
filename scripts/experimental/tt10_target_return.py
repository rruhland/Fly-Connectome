"""Use public cursor motion to update the translational part of a table player's motor model."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import CameraServo
from .tt6_memory import _serializable_body, load
from .tt6_timing import DelayedWiring, Disturbance
from .tt10_target import BLOCK, TargetArena, read_markers
from .tt10_target_trial import SOURCES


def motion_samples(arena, row, model):
    disturbance = Disturbance(delays=((0, 0), (48, 2), (120, 1)), noise=((0, .0), (48, .025)))
    interface = DelayedWiring(TT6Wiring(arena, row['wiring']), disturbance, seed=row['session'])
    interface.reset(seed=row['session'], render=False)
    observer = CameraServo(model)
    states, motion = [], []
    for t, action in enumerate(row['runs']['evidence_carry']['actions'], 1):
        interface.step(action, render=False)
        observer.advance(np.asarray(action))
        if t % BLOCK == 0:
            cursor, _ = read_markers(arena.frame())
            states.append(observer.effective.copy())
            motion.append(cursor)
    return np.asarray(states), np.asarray(motion)


def error(model, states, observed):
    return np.linalg.norm(model.pose(states)[:, :3] - model.ready - observed, axis=1)


def run(source, practice, output, report):
    data = _read(practice)
    name, game = SOURCES[source]
    source_file = ROOT / 'artifacts/tt9' / f'{name}.json.gz'
    state = _read(source_file)
    library, _, _, _ = load(source_file)
    previous = library.models[-1]
    settings = _read(ROOT / 'artifacts/tt9' / f'{game}.json.gz')['arena']
    arena = TargetArena(**dict(settings, render_mode='projected'))
    rows = [r for r in data['rows'] if not r['rewired'] and r['site'] == source]
    try:
        pairs = [motion_samples(arena, r, previous) for r in rows]
    finally:
        arena.close()
    if len(rows) < 3:
        raise ValueError('need two practice sessions and at least one held session')
    train_x, train_y = (np.concatenate(values) for values in zip(*pairs[:2], strict=True))
    residual = train_y - (previous.pose(train_x)[:, :3] - previous.ready)
    correction = np.linalg.solve(train_x.T @ train_x + 12. * np.eye(previous.action_size), train_x.T @ residual)
    coef = previous.coefficients.copy()
    coef[1:previous.action_size + 1, :3] += correction
    adapted = replace(previous, coefficients=coef)
    state['models'][-1] = _serializable_body(adapted)
    _write(output, state)
    held_x, held_y = (np.concatenate(values) for values in zip(*pairs[2:], strict=True))
    _write(report, dict(source_sha256=_hash(source_file), practice_sha256=_hash(practice),
                        checkpoint_sha256=_hash(output), site=source,
                        training=[r['session'] for r in rows[:2]], held=[r['session'] for r in rows[2:]],
                        training_poses=len(train_x), held_poses=len(held_x),
                        held_error_before_m=float(np.mean(error(previous, held_x, held_y))),
                        held_error_after_m=float(np.mean(error(adapted, held_x, held_y))),
                        correction=correction.tolist()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('practice', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('report', type=Path)
    a = p.parse_args()
    run(a.site, a.practice, a.checkpoint, a.report)
