"""Audit a frozen hinged-donor pupil and assess independent shifted physics."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena, mujoco
from .tt24_donor_trial import (DESIGNS, ROOT, SHAPE, Design, DonorBounce, Native, _paired, _seeds, _summary, _utility)


TRAINING = ROOT / 'tt24_donor-refined.json.gz'
CHECKPOINT = ROOT / 'tt24_donor-refined.npz'
ARCHIVE = ROOT / 'tt24_donor-development-weights.npz'
OUTPUT = ROOT / 'tt24_donor-extended.json.gz'
SHIFTED_RNG = 1111181509
LONG_RNG = 1648917313
SHIFTED_DESIGN = Design((.025, .135), 1.18, .058, .88, .89)
GRAVITY = .84
DRAG = 1.35
RISING = .055


class ShiftedBounce(DonorBounce):
    def __init__(self):
        super().__init__(SHIFTED_DESIGN, view='wrist', width=192, height=144)
        self._model.opt.gravity[2] *= GRAVITY

    def reset(self, **options):
        observation = super().reset(**options)
        self._drag_force_scale *= DRAG
        return observation


def _champions(report, frozen, archive):
    if not report['settings']['refine']:
        raise ValueError('the timeline expects the refined donor-only selection rule')

    def score(data):
        data = np.asarray(data).copy()
        data[..., 0] = .55 * data[..., 0] + .45 * data[..., 1]
        return _utility(data, 300)

    initial = np.asarray(report['neutral_development'])[:, None]
    wins, generations, utilities = [np.zeros(SHAPE)], [0], [float(score(initial)[0])]
    exposure = [dict(training_drops=0, training_frames=0, validation_drops=int(np.prod(initial.shape[:-1])),
                     validation_frames=int(initial[..., 2].sum()))]
    total = exposure[0].copy()
    for row in report['updates']:
        practise, checked = (np.asarray(row[key]) for key in ('training_results', 'development'))
        for label, data in [('training', practise), ('validation', checked)]:
            total[label + '_drops'] += int(np.prod(data.shape[:-1]))
            total[label + '_frames'] += int(data[..., 2].sum())
        performance = score(checked)
        chosen = int(np.argmax(performance))
        if performance[chosen] > utilities[-1]:
            wins.append(np.asarray(row['development_weights'][chosen]))
            generations.append(row['generation'])
            utilities.append(float(performance[chosen]))
            exposure.append(total.copy())
    if not np.array_equal(wins[-1], frozen):
        raise ValueError('the donor validation ledger does not reproduce the frozen readout')
    pending = archive.with_name(archive.stem + '.tmp.npz')
    np.savez_compressed(pending, weights=np.asarray(wins), generations=generations, utility=utilities,
                        **{key: [row[key] for row in exposure] for key in total})
    pending.replace(archive)
    return dict(file=archive.name, sha256=_hash(archive), generations=generations, utilities=utilities, exposure=exposure)


def _impact_speeds(arena, case):
    metrics = case['metrics']
    arena.reset(seed=metrics['seed'], split=metrics['split'], regime=metrics['regime'], render=False)
    original = arena._judge_substep
    impacts = {}

    def record(incoming):
        touches = arena._paddle_contacts
        result = original(incoming)
        if arena._paddle_contacts != touches and arena._contact is not None:
            data = arena._data
            last = [item for item in data.contact if item.efc_address >= 0 and
                    set(item.geom) == {arena._ball_geom, arena._blade}][-1]
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(arena._model, data, mujoco.mjtObj.mjOBJ_GEOM, arena._blade, velocity, 0)
            offset = last.pos - data.geom_xpos[arena._blade]
            at_impact = velocity[3:] + np.cross(velocity[:3], offset)
            impacts[arena._contact.impact_index] = at_impact.tolist()
        return result

    arena._judge_substep = record
    try:
        for action in case['actions']:
            _, _, done = arena.step(action, render=False)
    finally:
        arena._judge_substep = original
    if not done or digest_json(metrics) != digest_json(arena.judge_metrics()):
        raise ValueError(f'impact-speed audit changed the original replay on {metrics["seed"]}')
    return [dict(impact_index=event['impact_index'], rise_m=event['rise_m'], velocity_m_s=impacts[event['impact_index']])
            for event in metrics['rebound_events']]


def _power(arena, cases):
    evidence = [{mode: _impact_speeds(arena, row) for mode, row in case['conditions'].items()}
                for case in cases]
    summary = {}
    for mode in evidence[0]:
        speeds = [[event['velocity_m_s'][2] for event in case[mode]] for case in evidence]
        vertical = [sample for drop in speeds for sample in drop]
        summary[mode] = dict(scored_hits=len(vertical), median_upward_speed=float(np.median(vertical)) if vertical else None,
                             above_055_m_s=sum(sample > RISING for sample in vertical),
                             drops_with_at_least_ten_rising_hits=sum(sum(sample > RISING for sample in drop) >= 10
                                                                   for drop in speeds))
    return dict(summary=summary, per_case=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=TRAINING)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    report = _read(args.training)
    if not report['donor_gate_passed'] or _hash(args.checkpoint) != report['frozen_sha256'] or 'held_stock' not in report:
        raise ValueError('complete the frozen stock-arm assessment before the independent extended comparisons')
    with np.load(args.checkpoint) as values:
        weights = values['weights'].copy()
    file_sha = _hash(args.checkpoint)
    source = dict(schema='tt24-donor-extended-1', frozen_sha256=file_sha, training_sha256=_hash(args.training),
                  no_new_motor_training=True, clean_and_public_use_same_weights=True,
                  long_test_gate=dict(minimum_public_mean=5., minimum_public_ten_runs=4))
    source['checkpoints'] = _champions(report, weights, args.archive)
    source['donor_power'] = []
    for design, cases in zip(DESIGNS, report['held_donor'], strict=True):
        arena = DonorBounce(design)
        try:
            source['donor_power'].append(_power(arena, cases))
        finally:
            arena.close()
    arena = BounceArena(view='none', width=192, height=144)
    try:
        source['stock_power'] = _power(arena, report['held_stock'])
    finally:
        arena.close()
    _write(args.output, source)
    print('donor upward-speed summaries', [row['summary'] for row in source['donor_power']], flush=True)
    native = Native()
    shifted = ShiftedBounce()
    try:
        starts = _seeds(SHIFTED_RNG, 1, 12)[0]
        rows = _paired(shifted, native, weights, starts, ('public', 'clean', 'hold'), 'changed')
        source['shifted'] = dict(design=asdict(SHIFTED_DESIGN), gravity_scale=GRAVITY, changed_regime_drag_multiplier=DRAG,
                                 public_observer='original stock observer, still frozen on altered arm', cases=rows,
                                 summary=_summary(rows), physical_replays=len(starts) * 3)
        _write(args.output, source)
        print('shifted', source['shifted']['summary'], flush=True)
    finally:
        shifted.close()
    stock = report['held_stock_summary']['public']
    if stock['mean'] >= 5. and stock['at_least_ten'] >= 4:
        arena = BounceArena(view='wrist', width=192, height=144, max_steps=900)
        try:
            seeds = _seeds(LONG_RNG, 1, 12)[0]
            rows = _paired(arena, native, weights, seeds, ('public', 'clean', 'hold'), 'standard')
            source['long_stock'] = dict(seconds=15., cases=rows, summary=_summary(rows), physical_replays=len(seeds) * 3)
            _write(args.output, source)
            print('long stock', source['long_stock']['summary'], flush=True)
        finally:
            arena.close()
    if _hash(args.checkpoint) != file_sha:
        raise ValueError('extended assessment changed the frozen policy')
    native.working.cleanup()


if __name__ == '__main__':
    main()
