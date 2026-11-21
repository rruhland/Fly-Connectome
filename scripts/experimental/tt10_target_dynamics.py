"""Identify transient arm response from a cursor trial and return it to the rally controller."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_body_control import _response_bank
from .tt6_memory import _serializable_body, load
from .tt10_target import BLOCK, BLOCKS_PER_TARGET, PULSE, CursorPlayer, TargetArena, read_markers
from .tt10_target_trial import SOURCES, targets


def observe(arena, wiring_seed, model, seed):
    """Collect requested pulses and metric positions decoded from the public display."""
    goals = targets(arena, seed, 8)
    wiring = TT6Wiring(arena, wiring_seed)
    wiring.reset(seed=seed, render=False)
    jacobian = model.jacobian(np.zeros(model.action_size))[:3]
    player = CursorPlayer(jacobian, exploration='evidence')
    actions = [np.zeros(model.action_size)]
    positions = [read_markers(arena.frame())[0]]
    scores = []
    for goal in goals:
        arena.set_target(goal)
        score = 0
        for _ in range(BLOCKS_PER_TARGET):
            command = player.act(arena.frame())
            for tick in range(BLOCK):
                action = command if tick < PULSE else np.zeros_like(command)
                rgb, reward, _, _, _ = wiring.step(action)
                score += reward
                actions.append(action.copy())
                positions.append(read_markers(rgb)[0])
        scores.append(score)
    return np.array(actions), np.array(positions), scores


def prediction(model, effective):
    return model.pose(effective)[:, :, :3] - model.ready


def fit_kernels(model, actions, measured):
    bank, lags = _response_bank()
    accum = np.cumsum(actions, axis=1)
    effective = np.stack([accum + lfilter(kernel - 1., [1.], actions, axis=1) for kernel in bank])
    selected = np.argmin(np.linalg.norm(model.kernels[:, None, :] - bank, axis=2), axis=1)
    state = np.stack([effective[k, :, :, i] for i,k in enumerate(selected)], axis=-1)
    # Absolute position includes display/fixture origin error. Short-term
    # changes directly test a pulse's timing, independent of that offset.
    actual_motion = measured[:, 2:] - measured[:, :-2]
    for _ in range(2):
        for axis in range(model.action_size):
            costs = []
            for i, kernel in enumerate(bank):
                state[:, :, axis] = effective[i, :, :, axis]
                predicted = prediction(model, state)
                motion_error = predicted[:, 2:] - predicted[:, :-2] - actual_motion
                regularizer = .001**2 * np.mean((kernel - model.kernels[axis])**2)
                costs.append(np.mean(np.minimum(motion_error**2, .035**2))
                             + .08 * np.mean(np.minimum((predicted - measured)**2, .08**2)) + regularizer)
            selected[axis] = np.argmin(costs)
            state[:, :, axis] = effective[selected[axis], :, :, axis]
    return replace(model, kernels=bank[selected], lags=lags[selected])


def errors(model, actions, measured):
    effective = np.cumsum(actions, axis=1)
    for i, kernel in enumerate(model.kernels):
        effective[:, :, i] += lfilter(kernel - 1., [1.], actions[:, :, i], axis=1)
    predicted = prediction(model, effective)
    delta = (predicted[:, 2:] - predicted[:, :-2]) - (measured[:, 2:] - measured[:, :-2])
    return dict(mean_position_m=float(np.mean(np.linalg.norm(predicted - measured, axis=-1))),
                mean_two_frame_motion_m=float(np.mean(np.linalg.norm(delta, axis=-1))),
                mean_forward_speed_error_mps=float(np.mean(abs(delta[:, :, 1])) * 15.))


def run(site, output, report):
    name, game = SOURCES[site]
    folder = ROOT/'artifacts/tt9'
    source = folder/f'{name}.json.gz'
    receipt = _read(folder / f'{game}.json.gz')
    original = load(source)[0].models[-1]
    arena = TargetArena(**dict(receipt['arena'], render_mode='projected'))
    rng = np.random.default_rng([913371057, list(SOURCES).index(site)])
    seeds = rng.integers(0, 2**31 - 1, size=8).tolist()
    try:
        attempts = [observe(arena, receipt['installation_seed'], original, s) for s in seeds]
    finally:
        arena.close()
    actions, motion, outcomes = map(np.asarray, zip(*attempts, strict=True))
    updated = fit_kernels(original, actions[:2], motion[:2])
    state = _read(source)
    state['models'][-1] = _serializable_body(updated)
    _write(output, state)
    _write(report, dict(schema='tt10-cursor-pulse-identification-v1', source=_hash(source), checkpoint=_hash(output),
                        site=site, training=seeds[:2], held=seeds[2:], scores=outcomes.tolist(),
                        source_errors=errors(original, actions[2:], motion[2:]),
                        trained_errors=errors(updated, actions[2:], motion[2:]),
                        source_lags=original.lags.tolist(), trained_lags=updated.lags.tolist(),
                        tape=dict(requested=actions.tolist(), observed_xyz=motion.tolist())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('site', choices=SOURCES)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('report', type=Path)
    a = p.parse_args()
    run(a.site, a.checkpoint, a.report)
